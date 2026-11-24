"""Aggregate registered page metrics and supplementary line diagnostics."""

from collections import Counter, defaultdict

from rapidfuzz.distance import Levenshtein

from .data import CELLS
from .metrics import FOLDS, LANES, STRATA, compare, measures, normalize, page_text, project


GROUPS = (*STRATA, "pooled")
VIEWS = ("all", "filtered")
LINE_CER = {"status": "unavailable", "reason": "no verified reference-to-prediction line joins"}


def carrier_metrics(carrier, prediction, metadata):
    cell_delimiters = metadata.get("text_format") == CELLS
    ref, hyp, kept = page_text(carrier.lines, carrier.retained, FOLDS, prediction, cell_delimiters)
    if not ref:
        raise ValueError(f"Empty registered reference on {carrier.id}")
    return {"all": (ref, hyp), "filtered": project(ref, hyp, kept)}


def extra_counts(ref, hyp):
    edits = Counter(operation.tag for operation in Levenshtein.editops(ref, hyp))
    return {"char_substitutions": edits["replace"], "char_deletions": edits["delete"],
            "char_insertions": edits["insert"]}


def line_counts(carrier, prediction, metadata):
    if metadata.get("text_format") == CELLS:
        prediction = prediction.replace(" | ", " ")
    expected = [text for line in carrier.lines if (text := normalize(line, FOLDS))]
    got = [text for line in prediction.splitlines() if (text := normalize(line, FOLDS))]
    return {"reference_lines": len(expected), "prediction_lines": len(got),
            "line_edits": Levenshtein.distance(expected, got)}


def score_submission(carriers, submission):
    metadata, predictions = submission
    by_view = {view: defaultdict(Counter) for view in VIEWS}
    for key, carrier in carriers.items():
        prediction = predictions[key]
        for view, (ref, hyp) in carrier_metrics(carrier, prediction, metadata).items():
            counts = measures(ref, hyp)
            if counts is None:
                continue
            counts.update(extra_counts(ref, hyp))
            if view == "all":
                counts.update(line_counts(carrier, prediction, metadata))
            counts["carriers"] = 1
            for group in (carrier.stratum, "pooled"):
                by_view[view][group].update(counts)
    views = {view: {} for view in VIEWS}
    for view in VIEWS:
        for group in GROUPS:
            counts = by_view[view][group]
            cer = counts["edits"] / counts["chars"] if counts["chars"] else None
            wer = counts["word_edits"] / counts["words"] if counts["words"] else None
            ler = counts["line_edits"] / counts["reference_lines"] if counts["reference_lines"] else None
            views[view][group] = {**counts, "carriers": counts["carriers"],
                                  "page_cer": cer, "wer": wer, "line_error_rate": ler}
    return {"metadata": metadata, "views": views, "line_cer": LINE_CER}


def replay_comparison(carriers, submissions):
    if set(submissions) != set(LANES):
        raise ValueError("Replay needs the ten registered baseline directories")
    views = {view: defaultdict(list) for view in VIEWS}
    omitted = Counter()
    for key, carrier in carriers.items():
        page = {view: {} for view in VIEWS}
        for lane in LANES:
            metadata, predictions = submissions[lane]
            for view, (ref, hyp) in carrier_metrics(carrier, predictions[key], metadata).items():
                page[view][lane] = measures(ref, hyp)
        for view in VIEWS:
            for group in (carrier.stratum, "pooled"):
                if all(counts is None for counts in page[view].values()):
                    omitted[group] += 1
                else:
                    views[view][group].append((carrier.issue, page[view]))
    return {"views": {view: {group: compare(views[view][group]) for group in GROUPS} for view in VIEWS},
            "scoped_carriers": {group: sum(c.stratum == group for c in carriers.values()) for group in STRATA},
            "filtered_no_retained_characters": dict(omitted), "line_cer": LINE_CER}
