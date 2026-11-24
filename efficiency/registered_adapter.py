"""Reuse the existing CPU adapters or simulate provider requests for DEV."""

import argparse
from pathlib import Path

from panel import READERS
from run import paddle, router, tesseract
from telemetry import Telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-api", action="store_true")
    parser.add_argument("--reader", choices=READERS)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
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
