"""Reuse the existing CPU adapters or simulate provider requests for DEV."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import local
from zipfile import ZipFile

from c1_protocol import PROTOCOL
from panel import READERS
from run import paddle, paddle_reader, router, router_options, router_output, tesseract, tesseract_page
from telemetry import Telemetry


def ordered_cpu(telemetry, assets, settings):
    """Read with two CPU slots while keeping submission and output order on the roster."""
    backend = settings["runtime"]
    layout = telemetry.job["ocr_layout"]
    state = local()
    if backend == "heocr":
        import heocr

        os.environ["HEOCR_MODELS"] = str(assets)

    def read(page):
        path = Path(page["path"])
        if backend == "heocr":
            options = router_options(assets, page["carrier"], layout="v1" if layout == "v1" else None)
            options.update(repairs=True, ocr=True, ocr_lang="heb", ocr_psm=3,
                           force_ocr=True, check_numbers="suspect", max_pages=1, time_limit=None)
            text, used_v6 = router_output(heocr.extract(path, **options), settings, options)
            return text.encode(), used_v6
        if backend == "paddle-cpu":
            if not hasattr(state, "read"):
                state.read = paddle_reader(assets, settings)
            return state.read(path).encode(), False
        with TemporaryDirectory() as folder:
            stem = Path(folder) / "recognized"
            tesseract_page(assets, settings, path, stem)
            return stem.with_suffix(".txt").read_bytes(), False

    v6_seen = False
    with ThreadPoolExecutor(max_workers=telemetry.job["concurrency"]) as workers:
        for page, (serialized, used_v6) in zip(telemetry.pages, workers.map(read, telemetry.pages), strict=True):
            telemetry.save_first_output(page["id"], serialized)
            telemetry.complete(page["id"])
            v6_seen |= used_v6
    if backend == "heocr" and any(page["carrier"] in ("scanner", "issuer-render") for page in telemetry.pages) and not v6_seen:
        raise RuntimeError("router did not exercise its line-model branch")


def verify_heocr_wheel(path):
    """Require the importing interpreter's HeOCR package to match its pinned wheel."""
    import heocr

    root = Path(heocr.__file__).resolve().parent
    with ZipFile(path) as wheel:
        members = [name for name in wheel.namelist() if name.startswith("heocr/") and not name.endswith("/")]
        if "heocr/__init__.py" not in members:
            raise ValueError("wheel does not contain a HeOCR package")
        for name in members:
            local_path = (root / name.removeprefix("heocr/")).resolve()
            if not local_path.is_relative_to(root) or local_path.read_bytes() != wheel.read(name):
                raise ValueError("installed HeOCR differs from the pinned wheel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-api", action="store_true")
    parser.add_argument("--reader", choices=READERS)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--verify-heocr-wheel", type=Path)
    args = parser.parse_args()
    if args.verify_heocr_wheel is not None:
        verify_heocr_wheel(args.verify_heocr_wheel)
        return
    telemetry = Telemetry()
    if args.mock_api:
        for page in telemetry.pages:
            with telemetry.request(page["id"]) as usage:
                Path(page["path"]).read_bytes()
                usage({key: 1 if key == "units" else 100 for key in telemetry.job["billing_categories"]})
            telemetry.complete(page["id"])
        return
    if not (args.reader and args.assets):
        parser.error("CPU adapter needs --reader and --assets")
    settings = READERS[args.reader]
    if telemetry.job.get("protocol") == PROTOCOL:
        ordered_cpu(telemetry, args.assets, settings)
        return
    pages = telemetry.pages
    if args.reader == "router-v2":
        pages = sorted(pages, key=lambda item: item["carrier"])
        settings = {**settings, "workers": telemetry.job["concurrency"]}
    elif telemetry.job["concurrency"] != 1:
        parser.error("this adapter is serial")
    run = {"tesseract": tesseract, "paddle-cpu": paddle, "heocr": router}[settings["runtime"]]
    paths = [(Path(page["path"]), page["carrier"]) for page in pages]
    options = {"require_v6": telemetry.job["phase"] != "warmup"} if args.reader == "router-v2" else {}
    for page, carrier in zip(pages, run(args.assets, settings, paths, **options), strict=True):
        if page["carrier"] != carrier:
            raise ValueError("adapter completion order differs")
        telemetry.complete(page["id"])


if __name__ == "__main__":
    main()
