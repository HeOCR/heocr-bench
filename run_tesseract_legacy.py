"""Run the registered Tesseract legacy diagnostic on the exact v0 OCR-only roster."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

TESSERACT_SHA256 = "d3d15b31535c2dd88cbbb53ec54a16b9a739f4146e1648b4b6ad57cf5ce79f1d"
MODEL_SHA256 = "7da6ea6b7a2620ec8e8b41de2967a13d429635a56657a0b30b622501a573d3e1"
TRANSPORT_SHA256 = "cd878a90dd144d7e57b76dc040352ce3170f7c05994ff111247af860044f67d9"
MANIFEST_SHA256 = "f173417a49f960f8bdd2bf9950a9e676d2070aeb6d6a6a3b0066887af6a11190"
MODEL_REVISION = "4767ea922bcc460e70b87b1d303ebdfed0897da8"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError("A pinned input failed SHA-256 verification")


def write_json(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("tesseract", "tessdata", "transport", "manifest", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    checked(args.tesseract, TESSERACT_SHA256)
    checked(args.tessdata / "heb.traineddata", MODEL_SHA256)
    checked(args.transport, TRANSPORT_SHA256)
    checked(args.manifest, MANIFEST_SHA256)

    members = json.loads(args.transport.read_bytes())["carriers"]
    source = json.loads(args.manifest.read_bytes())
    expected = {
        carrier["id"]: carrier["scored_raster_sha256"]
        for page in source["pages"] for carrier in page["carriers"]
    }
    observed = {member["id"]: member["sha256"] for member in members}
    if len(members) != 65 or len(observed) != 65 or observed != expected:
        raise ValueError("OCR-only transport differs from the scored raster roster")

    input_paths = []
    for member in members:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", member["id"]):
            raise ValueError("Invalid carrier ID")
        image = args.transport.parent / member["path"]
        checked(image, member["sha256"])
        input_paths.append((member["id"], image))

    args.out.mkdir(mode=0o700)
    submission = args.out / "submission"
    submission.mkdir(mode=0o700)
    write_json(args.out / "claim.json", {
        "manifest_sha256": MANIFEST_SHA256, "transport_sha256": TRANSPORT_SHA256,
        "driver_sha256": sha256(Path(__file__)), "carriers": len(members),
    })
    write_json(submission / "metadata.json", {
        "reader": "Tesseract legacy Hebrew", "version": "5.5.3",
        "license": "Apache-2.0", "weights": "open", "text_format": "plain",
        "params": {"lang": "heb", "psm": 3, "oem": 0, "threads": 1,
                   "tesseract_sha256": TESSERACT_SHA256,
                   "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256},
    })

    for carrier_id, image in input_paths:
        result = subprocess.run(
            [args.tesseract, image, "stdout", "--tessdata-dir", args.tessdata,
             "-l", "heb", "--oem", "0", "--psm", "3"],
            env=os.environ | {"OMP_THREAD_LIMIT": "1"}, capture_output=True, timeout=45,
        )
        if result.returncode:
            raise RuntimeError("A Tesseract carrier attempt failed")
        result.stdout.decode("utf-8", errors="strict")
        with (submission / f"{carrier_id}.txt").open("xb") as stream:
            stream.write(result.stdout)

    ledger = "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(submission.iterdir()))
    (args.out / "submission.sha256").write_text(ledger, encoding="ascii")
    write_json(args.out / "terminal.json", {
        "status": "complete", "carriers": len(members),
        "submission_sha256": sha256(args.out / "submission.sha256"),
        "claim_sha256": sha256(args.out / "claim.json"),
    })


if __name__ == "__main__":
    main()
