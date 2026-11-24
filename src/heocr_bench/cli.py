"""Entry point for heocr-bench."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from .data import load_carriers, load_submission, split_descriptor
from .metrics import LANES
from .scoring import replay_comparison, score_submission


def main(argv=None):
    parser = argparse.ArgumentParser(prog="heocr-bench", description=__doc__)
    parser.add_argument("--version", action="version", version=version("heocr-bench"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (("score", "Score one complete submission"),
                              ("replay", "Reproduce the ten-reader v0 comparison")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--set", dest="set_id", required=True)
        command.add_argument("--split", default="test")
        command.add_argument("--preds", type=Path, required=True)
        command.add_argument("--data", type=Path, help="Use a local copy of the same pinned public bundle")
    args = parser.parse_args(argv)
    try:
        descriptor = split_descriptor(args.set_id, args.split)
        carriers = load_carriers(descriptor, args.data)
        if args.command == "score":
            result = score_submission(carriers, load_submission(args.preds, carriers))
            view = descriptor["headline_view"]
            result["headline"] = {"metric": "micro_page_cer", "view": view,
                                  "value": result["views"][view]["pooled"]["page_cer"]}
        else:
            lanes = {lane: load_submission(args.preds / lane, carriers) for lane in LANES}
            result = replay_comparison(carriers, lanes)
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        parser.exit(2, f"heocr-bench: {exc}\n")
    result.update(set=args.set_id, split=args.split, manifest_sha256=descriptor["manifest_sha256"],
                  normalization=descriptor["normalization"], scorer_version=version("heocr-bench"))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
