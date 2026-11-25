"""Run local readers on native page images and common BiblIA line crops."""

from hashlib import file_digest
import json
from math import ceil, floor
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


TESSERACT_READERS = ("rivok-full", "rivok-lite", "tessdata-best", "tessdata-fast")
HEOCR_READERS = ("heocr", "heocr-router-v2")
READERS = (*HEOCR_READERS, *TESSERACT_READERS, "paddle")
HEOCR_OPTIONS = {
    "ocr_model", "ocr_capture_type", "ocr_line_v6", "ocr_layout", "ocr_old_print_model",
    "crnn_model", "crnn_sha256", "ocr_phone_geometry", "photo_preprocess", "check_numbers",
}
TESSERACT_ENV = {
    "OMP_THREAD_LIMIT": "1", "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
}


def read_pins(reader, spec, assets):
    if spec is None or assets is None:
        raise ValueError(f"{reader} requires --reader-spec and --assets")
    settings = json.loads(spec.read_text(encoding="utf-8"))[reader]
    for asset in settings["assets"]:
        path = Path(asset["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("reader asset path must be relative to --assets")
        with (assets / path).open("rb") as stream:
            sha = file_digest(stream, "sha256").hexdigest()
        if sha != asset["sha256"]:
            raise ValueError(f"reader asset SHA-256 mismatch: {asset['path']}")
    return settings


def tesseract_reader(tesseract, settings, assets, psm):
    directory = (assets / settings["assets"][0]["path"]).parent
    argv = list(settings["argv"])
    argv[argv.index("--psm") + 1] = str(psm)

    def read(path):
        result = subprocess.run(
            [tesseract, str(path), "stdout", "--tessdata-dir", str(directory), *argv],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180,
            env={**os.environ, **TESSERACT_ENV},
        )
        return result.stdout.decode("utf-8")

    return read


def paddle_reader(assets, settings):
    import cv2
    from paddleocr_hebrew import HebrewOCR

    options = settings["options"]
    engine = HebrewOCR.word(assets / "paddle", providers=[options["provider"]], **{
        key: value for key, value in options.items()
        if key not in ("provider", "pipeline", "rec_mode", "order")
    })

    def read(path):
        raster = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raster is None:
            raise ValueError("Paddle could not decode the input image")
        result = engine.read_array(raster, rec_mode=options["rec_mode"])
        return "\n".join(line["text"] for line in result["lines"])

    return read


def heocr_reader(reader, options_file, tesseract, psm):
    import heocr

    options = json.loads(options_file.read_text(encoding="utf-8")) if options_file else {}
    if not isinstance(options, dict) or options.keys() - HEOCR_OPTIONS:
        raise ValueError("--heocr-options must be an object containing only local reader options")
    if reader == "heocr-router-v2":
        options["ocr_router"] = "v2"
    options.update(convert=False, tesseract=tesseract, ocr_psm=psm)

    def read(path):
        doc = heocr.extract(path, **options)
        failure_flags = {"bundled_model_unavailable", "routed_model_unavailable", "ocr_call_failed"}
        if (len(doc.pages) != 1 or doc.pages[0].route != "ocr"
                or failure_flags.intersection(doc.flags + doc.pages[0].flags)):
            raise RuntimeError("heocr did not complete the configured OCR route")
        return doc.text

    return read


def make_reader(args):
    psm = 7 if args.set_id == "biblia" else 3
    if args.reader in HEOCR_READERS:
        return heocr_reader(args.reader, args.heocr_options, args.tesseract, psm)
    if args.heocr_options:
        raise ValueError("--heocr-options requires a HeOCR reader")
    settings = read_pins(args.reader, args.reader_spec, args.assets)
    if args.reader == "paddle":
        return paddle_reader(args.assets, settings)
    return tesseract_reader(args.tesseract, settings, args.assets, psm)


def line_crop(image, polygon):
    from PIL import Image, ImageDraw

    xs, ys = zip(*polygon, strict=True)
    x0, y0, x1, y1 = floor(min(xs)), floor(min(ys)), ceil(max(xs)), ceil(max(ys))
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("ALTO line crop is outside the source raster")
    crop = image.crop((x0, y0, x1, y1))
    mask = Image.new("1", crop.size)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in polygon], fill=1)
    white = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, white, mask)


def predict(units, read):
    """Give the reader one raster at a time; retain only the current ALTO carrier."""
    from PIL import Image

    page_path = page = None
    with TemporaryDirectory(prefix="heocr-bench-external-") as directory:
        root = Path(directory)
        for unit in units:
            if unit.polygon:
                if unit.image != page_path:
                    with Image.open(unit.image) as original:
                        page = original.convert("RGB")
                    page_path = unit.image
                if page.size != unit.dimensions:
                    raise ValueError(f"ALTO page dimensions differ from raster for {unit.id!r}")
                path = root / "line.png"
                line_crop(page, unit.polygon).save(path)
            elif isinstance(unit.image, bytes):
                path = root / "page.img"
                path.write_bytes(unit.image)
            else:
                path = unit.image
            yield {"id": unit.id, "text": read(path)}
