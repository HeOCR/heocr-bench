"""Predict and score without a dataset download or a public-registry entry."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import subprocess
from xml.etree.ElementTree import ParseError

from rapidfuzz.distance import Levenshtein

from .churro_metric import normalize_churro
from .external_data import SETS, load_units
from .external_readers import READERS, make_reader, predict
from .metrics import FOLDS, normalize


def load_predictions(path):
    predictions = {}
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            row = json.loads(line)
            if (not isinstance(row, dict) or set(row) != {"id", "text"}
                    or not all(isinstance(value, str) for value in row.values())
                    or not row["id"] or row["id"] in predictions):
                raise ValueError(f"invalid or duplicate prediction at JSONL row {i}")
            predictions[row["id"]] = row["text"]
    return predictions


def normalized_levenshtein(ref, pred):
    ref, pred = normalize_churro(ref), normalize_churro(pred)
    length = max(len(ref), len(pred))
    return 1 - Levenshtein.distance(ref, pred) / length if length else 1.0


def score(units, predictions, set_id, *, heocr_cell_delimited=False):
    chars = edits = raw_chars = raw_edits = words = word_edits = count = 0
    similarities = defaultdict(list)
    remaining = predictions.copy()
    for unit in units:
        if unit.id not in remaining:
            raise ValueError(f"missing prediction for {unit.id!r}")
        hypothesis = remaining.pop(unit.id)
        if heocr_cell_delimited:
            hypothesis = hypothesis.replace(" | ", " ")
        reference = unit.reference
        ref = normalize(reference, FOLDS)
        pred = normalize(hypothesis, FOLDS)
        chars += len(ref)
        edits += Levenshtein.distance(ref, pred)
        count += 1
        if set_id == "churro":
            similarities[unit.document_type].append(normalized_levenshtein(reference, hypothesis))
        elif set_id == "biblia":
            raw_ref, raw_pred = (" ".join(text.split()) for text in (reference, hypothesis))
            raw_chars += len(raw_ref)
            raw_edits += Levenshtein.distance(raw_ref, raw_pred)
            rw, pw = raw_ref.split(), raw_pred.split()
            words += len(rw)
            word_edits += Levenshtein.distance(rw, pw)
    if remaining:
        raise ValueError(f"{len(remaining)} unexpected prediction IDs")
    if not chars:
        raise ValueError("no reference characters under line_fold_v1")

    official = {"status": "not_applicable", "metric": "word_spotting_map"}
    if set_id == "churro":
        official = {
            "metric": "normalized_levenshtein_similarity",
            "value": mean(value for scores in similarities.values() for value in scores),
            "by_document_type": {
                name: {"units": len(scores), "value": mean(scores)}
                for name, scores in sorted(similarities.items())
            },
        }
    elif set_id == "biblia":
        cer = raw_edits / raw_chars
        official = {"metric": "character_accuracy", "value": 1 - cer, "cer": cer,
                    "reference_chars": raw_chars, "edits": raw_edits,
                    "wer": word_edits / words, "reference_words": words, "word_edits": word_edits}
    return {
        "schema": "heocr.bench.external-score.v1",
        "set": set_id,
        "unit": "line" if set_id == "biblia" else "page",
        "units": count,
        "line_fold_v1": {"cer": edits / chars, "reference_chars": chars, "edits": edits},
        "publisher_metric": official,
    }


def add_command(commands):
    command = commands.add_parser("external", help="Run local external benchmark adapters")
    stages = command.add_subparsers(dest="stage", required=True)
    for stage in ("predict", "score"):
        parser = stages.add_parser(stage)
        parser.add_argument("--set", dest="set_id", choices=SETS, required=True)
        parser.add_argument("--data", type=Path, required=True, help="Local extracted directory or CHURRO Parquet")
        parser.add_argument("--pages", type=Path, help="BiblIA list of admitted relative ALTO XML paths")
        if stage == "predict":
            parser.add_argument("--reader", choices=READERS, required=True)
            parser.add_argument("--assets", type=Path, help="Local root for reader weights")
            parser.add_argument("--reader-spec", type=Path, help="Pinned comparator settings and SHA-256 manifest")
            parser.add_argument("--heocr-options", type=Path, help="JSON object of local HeOCR settings")
            parser.add_argument("--tesseract", default="tesseract", help="Path to the registered Tesseract binary")
        else:
            parser.add_argument("--preds", type=Path, required=True)
            parser.add_argument("--text-format", choices=("plain", "heocr-cell-delimited"), default="plain")


def run(args, parser):
    try:
        predicting = args.stage == "predict"
        units = load_units(args.set_id, args.data, args.pages, with_images=predicting)
        if predicting:
            for row in predict(units, make_reader(args)):
                print(json.dumps(row, ensure_ascii=False))
        else:
            result = score(units, load_predictions(args.preds), args.set_id,
                           heocr_cell_delimited=args.text_format == "heocr-cell-delimited")
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ImportError, ValueError, KeyError, TypeError, RuntimeError,
            ParseError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"heocr-bench external: {exc}\n")
