"""Export the standalone efficiency table and its external-only projection."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

from panel import DATASET_REVISION, HERE, panel, sha256
from run import SCHEMA, metrics

PLATFORMS = ("m4-cpu", "github-ubuntu-cpu")
SCOPE = "public-v0-image-only"
COLUMNS = ("reader_id", "platform", "execution", "status", "pages", "repetitions",
           "pages_per_minute", "peak_rss_mib", "marginal_usd_per_1000",
           "runner_hours_per_1000", "runner_usd_per_1000")


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
        if any(result[key] != value for key, value in metrics(trials, data["runner_usd_hour"]).items()):
            raise ValueError(f"receipt metrics differ for {reader}")
    return data["readers"]


def rows(receipts, external_only=False):
    roster = json.loads((HERE / "leaderboard.json").read_text())
    readers = [row["id"] for row in roster["readers"]]
    if not external_only:
        readers.append("router-v2")
    for reader in readers:
        for host in PLATFORMS:
            result = receipts[host].get(reader)
            mode = {"paddle": "onnx-cpu", "gemini-vertex-v0": "vertex-global-https",
                    "router-v2": "2-image-workers"}.get(reader, "native-tesseract")
            if result:
                status = "measured-cpu-variant" if reader == "paddle" else "measured"
            elif reader == "gemini-vertex-v0":
                status = "list-price-only"
            else:
                status = "pending"
            fields = {key: result[key] for key in COLUMNS if result and key in result}
            yield dict(reader_id=reader, platform=host, execution=mode, status=status,
                       repetitions=len(result["measurements"]) if result else None, **fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--receipt-dir", type=Path, default=HERE / "receipts")
    args = parser.parse_args()
    receipts = {host: {} for host in PLATFORMS}
    for host in PLATFORMS:
        for path in sorted(args.receipt_dir.glob(f"{host}*.json")):
            data = read_receipt(path, host)
            if receipts[host].keys() & data.keys():
                raise ValueError(f"multiple receipts for the same reader on {host}")
            receipts[host].update(data)
    output = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    output.writeheader()
    output.writerows(rows(receipts, args.external_only))


if __name__ == "__main__":
    main()
