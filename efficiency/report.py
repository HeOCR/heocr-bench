"""Export the standalone efficiency table and its external-only projection."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

from panel import DATASET_REVISION, HERE, panel, sha256
from registered import read_registered
from run import SCHEMA, metrics

PLATFORMS = ("m4-cpu", "github-ubuntu-cpu")
COLUMNS = ("reader_id", "platform", "execution", "status", "pages", "repetitions",
           "pages_per_minute", "peak_rss_mib", "marginal_usd_per_1000",
           "runner_hours_per_1000", "runner_usd_per_1000",
           "api_input_usd_per_million", "api_output_usd_per_million", "api_usd_per_1000_units",
           "api_registration", "api_pages_per_minute", "api_mean_request_latency_seconds",
           "api_timed_pages", "api_timing_scope", "api_timing_label")
REGISTERED_COLUMNS = COLUMNS + ("pages_per_minute_ci95", "peak_rss_mib_ci95", "api_pages_per_minute_ci95",
                                "fully_loaded_usd_per_1000", "api_price_key", "api_service_region")


def export_registered(path, output, *, allow_dry_run=False):
    from registered_run import SCHEMA as REGISTERED_SCHEMA

    receipt = json.loads(path.read_text())
    if receipt["schema"] != REGISTERED_SCHEMA or (receipt["dry_run"] and not allow_dry_run):
        raise ValueError("expected a registered efficiency receipt; DEV is excluded by default")
    csv_output = csv.DictWriter(output, fieldnames=REGISTERED_COLUMNS, lineterminator="\n")
    csv_output.writeheader()
    for row in receipt["table"]:
        csv_output.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                             for key, value in row.items()})


def read_receipt(path, label):
    data = json.loads(path.read_text())
    if (data["schema"] != SCHEMA or data["dataset_revision"] != DATASET_REVISION
            or data["image_manifest_sha256"] != sha256(HERE / "images.sha256")
            or data["reader_settings_sha256"] != sha256(HERE / "readers.json")):
        raise ValueError(f"receipt pins differ for {path.name}")
    runtime = data["runtime"]
    if label == "m4-cpu":
        if runtime["machine"] != "arm64" or not runtime["cpu"].startswith("Apple M4"):
            raise ValueError("expected Apple M4 CPU")
    elif runtime["runner_image"] != "ubuntu24":
        raise ValueError("expected a GitHub Ubuntu 24.04 runner")
    expected_coverage = Counter(carrier for _, _, carrier in panel())
    for reader, result in data["readers"].items():
        trials = result["measurements"]
        if not trials or any(r["completed_by_carrier"] != expected_coverage or
                             not math.isfinite(r["seconds"]) or r["seconds"] <= 0 or
                             r["peak_rss_bytes"] <= 0 for r in trials):
            raise ValueError(f"incomplete receipt for {reader}")
        expected = metrics(trials, data["runner_usd_hour"])
        if any(result[key] != value for key, value in expected.items()):
            raise ValueError(f"receipt metrics differ for {reader}")
    return data["readers"]


def rows(receipts, api, external_only=False):
    roster = json.loads((HERE / "leaderboard.json").read_text())
    readers = [row["id"] for row in roster["readers"]]
    if not api.keys() <= set(readers):
        raise ValueError("registered API receipt reader missing from the leaderboard roster")
    if not external_only:
        readers.append("router-v2")
    for reader in readers:
        if reader in api:
            yield {"reader_id": reader, **api[reader]}
            continue
        for host in PLATFORMS:
            result = receipts[host].get(reader)
            mode = {"paddle": "onnx-cpu", "router-v2": "2-image-workers"}.get(reader, "native-tesseract")
            if result:
                status = "measured-cpu-variant" if reader == "paddle" else "measured"
            else:
                status = "pending"
            fields = {key: result[key] for key in COLUMNS if result and key in result}
            yield dict(reader_id=reader, platform=host, execution=mode, status=status,
                       repetitions=len(result["measurements"]) if result else None, **fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--receipt-dir", type=Path, default=HERE / "receipts")
    parser.add_argument("--registered-run", type=Path, help="export a decision 0030 JSON receipt")
    parser.add_argument("--allow-dry-run", action="store_true", help="export a DEV receipt for inspection")
    args = parser.parse_args()
    if args.registered_run:
        if args.external_only:
            parser.error("registered runs already carry their exact reader roster")
        export_registered(args.registered_run, sys.stdout, allow_dry_run=args.allow_dry_run)
        return
    receipts = {host: {} for host in PLATFORMS}
    for host in PLATFORMS:
        for path in sorted(args.receipt_dir.glob(f"{host}*.json")):
            data = read_receipt(path, host)
            if receipts[host].keys() & data.keys():
                raise ValueError(f"multiple receipts for the same reader on {host}")
            receipts[host].update(data)
    output = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    output.writeheader()
    api_path = args.receipt_dir / "registered-api.json"
    if not api_path.is_file():
        api_path = HERE / "receipts/registered-api.json"
    api = read_registered(api_path)
    output.writerows(rows(receipts, api, args.external_only))


if __name__ == "__main__":
    main()
