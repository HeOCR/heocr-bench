"""Time sequential Vertex requests from an authorized client on the public image panel."""

import argparse
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

from panel import DATASET_REVISION, HERE, panel, prepare, sha256
from quote import quote
from run import SCHEMA, environment, metrics, sample
from vertex import USAGE, price, read

READER = "gemini-vertex-v0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--budget-usd", type=Decimal, required=True)
    parser.add_argument("--runner-usd-hour", type=float)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.budget_usd.is_finite() or args.budget_usd <= 0:
        parser.error("budget must be a finite positive dollar amount")
    if not os.environ.get("HEOCR_BENCH_VERTEX_ACCESS_TOKEN"):
        parser.error("set HEOCR_BENCH_VERTEX_ACCESS_TOKEN in the runner environment")
    paths = [(args.root / name, carrier) for name, _, carrier in panel()]
    if args.worker:
        telemetry = read(paths, args.project, args.prompt, args.model_version, args.budget_usd)
        print(json.dumps(telemetry))
        return
    prepare(args.root, [], args.download)
    telemetry = sample([sys.executable, "-B", __file__, *sys.argv[1:], "--worker"], READER)
    telemetry.update(telemetry.pop("worker_result"))
    if telemetry["completed_by_carrier"] != Counter(carrier for _, carrier in paths):
        raise RuntimeError("Vertex coverage is incomplete")
    runs = [telemetry]
    summary = metrics(runs, args.runner_usd_hour)
    usage = telemetry["vertex_usage"]
    summary["marginal_usd_per_1000"] = round(
        float(quote(len(paths), *(usage[key] for key in USAGE), price())), 6
    )
    report = {
        "schema": SCHEMA,
        "dataset_revision": DATASET_REVISION,
        "image_manifest_sha256": sha256(HERE / "images.sha256"),
        "reader_settings_sha256": sha256(HERE / "readers.json"),
        "vertex_pricing_sha256": sha256(HERE / "api-pricing.json"),
        "vertex_prompt_sha256": sha256(args.prompt),
        "vertex_model_version": args.model_version,
        "runtime": environment([]),
        "runner_usd_hour": args.runner_usd_hour,
        "readers": {READER: {"measurements": runs, **summary}},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
