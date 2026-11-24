"""Deterministic page-text projection and aggregate statistics for C1 v0."""

from collections import Counter, defaultdict
import math
import random
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

PROFILE_SHA = "2e2627104be4f27dc3204787dd3ff4f8591d95d1d96294c65833d2773041d317"
BIDI = re.compile(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
NUM = re.compile(r"\d+(?:[.,/]\d+)*%?")
STRATA = ("archive", "issuer-render", "issuer-scan", "scanner", "phone-a", "phone-b")
LANES = ("stock-heb", "modern-v1", "v4-120k", "v6-40k", "crnn-v2-640k",
         "best", "fast", "rivok-full", "rivok-lite", "paddle")


# Transcription folds and mask behavior from the frozen C1 v0 scorer.
FOLDS = str.maketrans({
    "\u05be": "-", "\u05f3": "'", "\u05f4": '"',
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2014": "\u2013",
})


def normalize(text, translations):
    """Retain supplied logical order and join whitespace to ASCII spaces."""
    nfc = unicodedata.normalize("NFC", BIDI.sub("", text))
    return " ".join(nfc.translate(translations).split())


def page_text(lines, flags, translations, prediction, heocr_layout=False):
    if len(lines) != len(flags) or any(type(flag) is not bool for flag in flags):
        raise ValueError("invalid reference mask")
    chunks = []
    retained = []
    for line, flag in zip(lines, flags, strict=True):
        chunk = normalize(line, translations)
        if not chunk:
            continue
        if chunks:
            # An inter-line space is masked only when both adjacent lines are.
            retained.append(retained[-1] or flag)
        chunks.append(chunk)
        retained.extend([flag] * len(chunk))
    ref = " ".join(chunks)
    # The canonical renderer uses this exact structural delimiter for cells.
    if heocr_layout:
        prediction = prediction.replace(" | ", " ")
    hyp = normalize(prediction, translations)
    return ref, hyp, retained


def project(ref, hyp, retained):
    """Remove the masked reference chars and the hypothesis chars assigned to them."""
    if len(ref) != len(retained):
        raise ValueError("mask length differs")
    if all(retained):
        return ref, hyp
    keep_hyp = []
    i = j = 0
    for op, src, dst in Levenshtein.editops(ref, hyp):
        if src - i != dst - j or ref[i:src] != hyp[j:dst]:
            raise ValueError("noncanonical edit script")
        keep_hyp.extend(c for c, keep in zip(hyp[j:dst], retained[i:src], strict=True) if keep)
        i, j = src, dst
        if op == "replace":
            if retained[i]:
                keep_hyp.append(hyp[j])
            i += 1
            j += 1
        elif op == "delete":
            i += 1
        elif op == "insert":
            # Charge page edges and mask boundaries to retained content.
            if i == 0 or i == len(ref) or retained[i - 1] or retained[i]:
                keep_hyp.append(hyp[j])
            j += 1
        else:
            raise ValueError("unexpected edit operation")
    if ref[i:] != hyp[j:]:
        raise ValueError("edit script tail differs")
    keep_hyp.extend(c for c, keep in zip(hyp[j:], retained[i:], strict=True) if keep)
    return (" ".join("".join(c for c, keep in zip(ref, retained, strict=True) if keep).split()),
            " ".join("".join(keep_hyp).split()))


def measures(ref, hyp):
    if not ref:
        return None
    numbers = Counter(NUM.findall(ref))
    hits = numbers & Counter(NUM.findall(hyp))
    rw = ref.split()
    return {"chars": len(ref), "edits": Levenshtein.distance(ref, hyp), "words": len(rw),
            "word_edits": Levenshtein.distance(rw, hyp.split()), "ref_numbers": numbers.total(),
            "missed_numbers": numbers.total() - hits.total()}


def quantile_higher(values, level):
    return sorted(values)[math.ceil((len(values) - 1) * level)]


def compare(captures):
    """Each item is (issue cluster, fixed arm -> page counts) on the common scope."""
    clusters = defaultdict(list)
    for issue, counts in captures:
        if set(counts) != set(LANES):
            raise ValueError("comparison lanes differ")
        if len({count["chars"] for count in counts.values()}) != 1:
            raise ValueError("comparison reference denominators differ")
        clusters[issue].append(counts)
    if not clusters:
        return {"status": "no_reference_characters", "arms": {}, "comparisons": {}}
    arms = {lane: Counter() for lane in LANES}
    cluster_characters = []
    cluster_edits = []
    for issue in sorted(clusters):
        issue_counts = {lane: Counter() for lane in LANES}
        for capture in clusters[issue]:
            for lane in LANES:
                issue_counts[lane].update(capture[lane])
                arms[lane].update(capture[lane])
        cluster_edits.append([issue_counts[lane]["edits"] for lane in LANES])
        cluster_characters.append(issue_counts[LANES[0]]["chars"])
    n = len(clusters)
    rng = random.Random(0)
    comparative = ("v6-40k", "crnn-v2-640k")
    comparators = ("stock-heb", "best", "fast", "rivok-full", "rivok-lite", "paddle")
    columns = {lane: i for i, lane in enumerate(LANES)}
    samples = {(a, c): [] for a in comparative for c in comparators}
    for _ in range(2000):
        selected = [cluster_edits[rng.randrange(n)] for _ in range(n)]
        edits = list(map(sum, zip(*selected, strict=True)))
        for (a, c), values in samples.items():
            test, baseline = edits[columns[a]], edits[columns[c]]
            values.append(test / baseline - 1 if baseline else math.inf)
    result = {}
    for (a, c), values in samples.items():
        cer_a = arms[a]["edits"] / arms[a]["chars"]
        cer_c = arms[c]["edits"] / arms[c]["chars"]
        upper = quantile_higher(values, 0.975)
        denom = arms[c]["chars"]
        guard = 50 * arms[c]["edits"] >= denom or 500 * (arms[c]["edits"] - arms[a]["edits"]) >= denom
        result.setdefault(a, {})[c] = {
            "relative_change": cer_a / cer_c - 1 if cer_c else None,
            "upper_97_5": upper if math.isfinite(upper) else None,
            "undefined_bootstrap_draws": sum(math.isinf(x) for x in values),
            "absolute_guard_pass": guard, "passes": upper < -0.03 and guard,
        }
    return {"status": "complete", "issue_clusters": n, "carriers": len(captures),
            "arms": {lane: {**counts, "cer": counts["edits"] / counts["chars"],
                            "wer": counts["word_edits"] / counts["words"]} for lane, counts in arms.items()},
            "comparisons": result}
