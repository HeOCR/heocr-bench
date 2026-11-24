"""Measure OCR throughput, resident memory and estimated cost on public images."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic, sleep

from panel import DATASET_REVISION, HERE, LEADERBOARD_REVISION, READERS, panel, prepare, sha256

TESSERACT_ENV = dict(OMP_THREAD_LIMIT="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
CAPTURES = {"phone-a": "iphone", "phone-b": "pixel", "archive": "scanner",
            "scanner": "scanner", "issuer-render": "issuer-render"}
MODEL = {"iphone": "heocr_modern_v1", "pixel": "heocr_modern_v1",
         "scanner": "heb_rivok", "issuer-render": "heb_rivok"}
SCHEMA = "heocr.bench.efficiency.v1"


def tesseract(root, settings, paths):
    assets = root / "assets" / Path(settings["assets"][0]["path"]).parent
    with TemporaryDirectory() as folder:
        stem = Path(folder) / "recognized"
        for path, carrier in paths:
            subprocess.run(
                ["tesseract", str(path), str(stem), "--tessdata-dir", str(assets),
                 *settings["argv"], "-c", "tessedit_create_txt=1", "-c", "tessedit_create_tsv=1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=180,
            )
            for suffix in (".txt", ".tsv"):
                stem.with_suffix(suffix).unlink()
            yield carrier


def paddle(root, settings, paths):
    import cv2
    from paddleocr_hebrew import HebrewOCR

    options = settings["options"]
    engine = HebrewOCR.word(root / "assets/paddle", providers=[options["provider"]], **{
        key: value for key, value in options.items() if key not in ("provider", "pipeline", "rec_mode", "order")
    })
    for path, carrier in paths:
        raster = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raster is None:
            raise ValueError("Paddle could not decode a public raster")
        result = engine.read_array(raster, rec_mode=options["rec_mode"])
        "\n".join(line["text"] for line in result["lines"])
        yield carrier


def router(model_root, settings, paths):
    import heocr

    os.environ["HEOCR_MODELS"] = str(model_root)
    grouped = defaultdict(list)
    for path, carrier in paths:
        grouped[carrier].append(path)
    v6_seen = 0
    for carrier, sources in grouped.items():
        capture = CAPTURES[carrier]
        v6 = model_root / "v6.traineddata" if carrier in ("scanner", "issuer-render") else None
        model = MODEL[capture]
        expected = "sha256:" + settings["models"][model]
        options = dict(ocr_router="v2", ocr_capture_type=capture, ocr_line_v6=v6, convert=False)
        for doc in heocr.extract_images(sources, workers=settings["workers"], **options):
            if len(doc.pages) != 1 or "ocr_call_failed" in doc.flags:
                raise RuntimeError("router failed to complete a public raster")
            if doc.pages[0].route != "ocr" or doc.components.get(f"tesseract:{model}.traineddata") != expected:
                raise RuntimeError("router used an unexpected page route or model")
            v6_sha = doc.components.get("tesseract:line_v6.traineddata")
            if v6_sha is not None:
                if not v6 or v6_sha != "sha256:" + settings["models"]["v6"]:
                    raise RuntimeError("router used an unexpected line model")
                v6_seen += 1
            doc.to_json(indent=None)
            yield carrier
    if not v6_seen:
        raise RuntimeError("router did not exercise its line-model branch")


def worker(root, name, model_root):
    paths = [(root / name, carrier) for name, _, carrier in panel()]
    settings = READERS[name]
    if settings["runtime"] == "tesseract":
        results = tesseract(root, settings, paths)
    elif settings["runtime"] == "paddle-cpu":
        results = paddle(root, settings, paths)
    else:
        results = router(model_root, settings, paths)
    completed = dict(Counter(results))
    if completed != Counter(carrier for _, carrier in paths):
        raise RuntimeError("reader coverage is incomplete")
    print(json.dumps(completed))


def tree_rss(process):
    import psutil

    rss = 0
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    for item in (process, *children):
        try:
            rss += item.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def measure(root, reader, model_root):
    command = [sys.executable, "-B", __file__, str(root), "--worker", reader]
    if model_root:
        command.extend(["--models", str(model_root)])
    timings = sample(command, reader)
    return {"completed_by_carrier": timings.pop("worker_result"), **timings}


def sample(command, reader):
    import psutil

    start = monotonic()
    with TemporaryFile() as errors, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=errors, env={**os.environ, **TESSERACT_ENV},
    ) as job:
        try:
            parent = psutil.Process(job.pid)
        except psutil.NoSuchProcess:
            parent = None
        peak = 0
        while job.poll() is None:
            if parent is not None:
                peak = max(peak, tree_rss(parent))
            sleep(0.02)
        output, _ = job.communicate()
        if job.returncode:
            errors.seek(0)
            raise RuntimeError(f"{reader} failed with exit {job.returncode}: {errors.read()[-2000:].decode(errors='replace')}")
    if peak <= 0:
        raise RuntimeError("RSS sampler did not observe the worker")
    return {"seconds": round(monotonic() - start, 6), "peak_rss_bytes": peak,
            "worker_result": json.loads(output)}


def metrics(trials, hourly_rate):
    count = sum(trials[0]["completed_by_carrier"].values())
    hours = median(run["seconds"] for run in trials) * 1000 / count / 3600
    return {
        "pages": count,
        "pages_per_minute": round(1000 / hours / 60, 4),
        "peak_rss_mib": round(max(run["peak_rss_bytes"] for run in trials) / 1024**2, 2),
        "marginal_usd_per_1000": 0,
        "runner_hours_per_1000": round(hours, 6),
        "runner_usd_per_1000": round(hours * hourly_rate, 6) if hourly_rate is not None else None,
    }


def environment(readers):
    from importlib.metadata import version

    info = dict(platform=platform.platform(), machine=platform.machine(), logical_cpus=os.cpu_count(),
                python=platform.python_version(), runner_image=os.getenv("ImageOS"),
                runner_image_version=os.getenv("ImageVersion"))
    if platform.system() == "Darwin":
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    else:
        info["cpu"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")), None)
    if any(READERS[name]["runtime"] != "paddle-cpu" for name in readers):
        from shutil import which

        binary = which("tesseract")
        if binary is None:
            raise RuntimeError("Tesseract is required for these readers")
        info["tesseract"] = subprocess.check_output([binary, "--version"], text=True).splitlines()
        info["tesseract_sha256"] = sha256(Path(binary))
    if "paddle" in readers:
        info["paddle_runtime"] = {pkg: version(pkg) for pkg in ("paddleocr-hebrew", "onnxruntime", "opencv-python", "numpy", "pyclipper")}
    if "router-v2" in readers:
        info["heocr_runtime"] = {pkg: version(pkg) for pkg in ("heocr", "Pillow", "pypdfium2", "defusedxml")}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--models", type=Path, help="local directory containing the three exact router model files")
    parser.add_argument("--heocr-revision", help="HeOCR source commit for a router benchmark")
    parser.add_argument("--reader", choices=READERS, action="append")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--runner-usd-hour", type=float)
    parser.add_argument("--worker", choices=READERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    model_root = args.models.resolve() if args.models else None
    if args.worker:
        worker(root, args.worker, model_root)
        return
    readers = args.reader or [reader for reader in READERS if reader != "router-v2"]
    if len(set(readers)) != len(readers) or args.repeat < 1:
        parser.error("request each reader once and at least one repetition")
    if args.runner_usd_hour is not None and not 0 <= args.runner_usd_hour < float("inf"):
        parser.error("the hourly rate must be finite and nonnegative")
    if "router-v2" in readers:
        if not model_root or not args.heocr_revision:
            parser.error("router v2 requires --models and --heocr-revision")
        for name, digest in READERS["router-v2"]["models"].items():
            if sha256(model_root / f"{name}.traineddata") != digest:
                raise ValueError(f"router model digest differs for {name}")
    prepare(root, readers, args.download)
    runtime = environment(readers)
    trials = defaultdict(list)
    for turn in range(args.repeat):
        pivot = turn % len(readers)
        for reader in readers[pivot:] + readers[:pivot]:
            trials[reader].append(measure(root, reader, model_root))
            print(f"{reader} repetition {turn + 1} complete", file=sys.stderr, flush=True)
    print(json.dumps({
        "schema": SCHEMA,
        "dataset_revision": DATASET_REVISION,
        "image_manifest_sha256": sha256(HERE / "images.sha256"),
        "reader_settings_sha256": sha256(HERE / "readers.json"),
        "leaderboard_revision": LEADERBOARD_REVISION,
        "heocr_revision": args.heocr_revision,
        "runtime": runtime,
        "runner_usd_hour": args.runner_usd_hour,
        "readers": {name: {"measurements": runs, **metrics(runs, args.runner_usd_hour)}
                    for name, runs in trials.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
