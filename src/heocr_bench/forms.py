"""Form-structure metrics (0032): field/value F1 and section-tree similarity.

Form format (``hdoc_form.v1``, the gold schema heocr-eval approved on 2026-11-25)::

    {"schema": "hdoc_form.v1", "sections": [
       {"title": "...", "pages": [1, 1], "children": [
          {"type": "field", "id": "f1", "label": "...", "value": "...", "kind": "text", "slot": "line"},
          {"type": "group", "columns": 2, "children": [...]},
          {"type": "section", "title": "...", "children": [...]},
          {"type": "table", ...}]}]}

``kind`` is one of text, number, date, checkbox, choice, signature. Checkbox values are ``checked`` or
``unchecked``; a choice value is the printed option text; an unfilled field is ``""``. Groups are layout only and
are flattened into their section. Tables inside a form are scored by the table metrics, not here.
Definitions: docs/forms.md.
"""

import math
import random

from rapidfuzz.distance import Indel, Levenshtein

from .tables import _Node, _assign, cell_text, tree_edit_distance

KINDS = ("text", "number", "date", "checkbox", "choice", "signature")


def label_text(text):
    """Cell normaliser, then one trailing colon or dash (and the space before it) removed."""
    t = cell_text(text)
    if t and t[-1] in ":-–":
        t = t[:-1].rstrip()
    return t


def value_text(text):
    return cell_text(text)


def fields(form):
    """All fields of a form in document order, groups and nested sections flattened."""
    out = []

    def walk(children):
        for ch in children:
            kind = ch.get("type")
            if kind == "field":
                if ch.get("value") is None:
                    raise ValueError("a field value is a string; an unfilled field is \"\"")
                if ch.get("kind", "text") not in KINDS:
                    raise ValueError(f"unknown field kind {ch.get('kind')!r}")
                out.append((label_text(ch.get("label", "")), value_text(ch["value"])))
            elif kind in ("group", "section"):
                walk(ch.get("children", []))
            elif kind != "table":
                raise ValueError(f"unknown form node {kind!r}")

    for section in form.get("sections", []):
        walk(section.get("children", []))
    return out


def field_counts(gold, pred):
    """One-to-one matching within a form: a Hungarian assignment that takes exact (label, value) pairs first, then
    label similarity; a pair counts as matched only if the normalised label and value are both exact.
    Returns gold, predicted and matched counts."""
    g, p = fields(gold), fields(pred)
    if not g or not p:
        return {"gold_fields": len(g), "pred_fields": len(p), "matched": 0}
    # exact (label, value) pairs first (weight 2 + similarity), then label similarity alone: duplicate labels
    # ("date" twice) never steal an exact match from each other
    w = [[(2.0 if a == b else 0.0) + Indel.normalized_similarity(a[0], b[0]) for b in p] for a in g]
    pairs = _assign(w)
    matched = sum(1 for i, j in pairs.items() if g[i] == p[j])
    return {"gold_fields": len(g), "pred_fields": len(p), "matched": matched}


def _tree(form):
    """form > section > (section | field-label) nodes; groups flattened; values and tables excluded."""

    def section(sec):
        node = _Node("section", cell_text(sec.get("title", "")))
        for ch in _flat(sec.get("children", [])):
            if ch.get("type") == "section":
                node.children.append(section(ch))
            else:
                node.children.append(_Node("field", label_text(ch.get("label", ""))))
        return node

    return _Node("form", children=[section(s) for s in form.get("sections", [])])


def _flat(children):
    for ch in children:
        if ch.get("type") == "group":
            yield from _flat(ch.get("children", []))
        elif ch.get("type") in ("field", "section"):
            yield ch


def _rename(a, b):
    if a.label != b.label:
        return 1.0
    if not a.text and not b.text:
        return 0.0
    return Levenshtein.distance(a.text, b.text) / max(len(a.text), len(b.text))


def _size(node):
    return 1 + sum(_size(c) for c in node.children)


def section_tree_similarity(gold, pred):
    """TEDS over form > section > field-label nodes (section titles and labels compared, values excluded)."""
    t1, t2 = _tree(gold), _tree(pred)
    return 1.0 - tree_edit_distance(t1, t2, rename=_rename) / max(_size(t1), _size(t2))


def score_form(gold, pred):
    return {**field_counts(gold, pred), "section_tree": section_tree_similarity(gold, pred)}


def _prf(m, g, p):
    precision = m / p if p else None
    recall = m / g if g else None
    f1 = 2 * m / (g + p) if g + p else None
    return precision, recall, f1


def pool_forms(documents, seed=0, draws=2000):
    """Pooled field precision, recall and F1 (summed counts over documents) and the mean section-tree similarity
    per gold form, with a percentile bootstrap that resamples documents (sorted ids, random.Random(seed)).

    ``documents`` maps document id -> list of score_form() results (one per gold form; an unmatched gold form is
    scored against an empty prediction).
    """
    ids = sorted(documents)
    rows = []
    for d in ids:
        forms = documents[d]
        rows.append((sum(f["matched"] for f in forms), sum(f["gold_fields"] for f in forms),
                     sum(f["pred_fields"] for f in forms), [f["section_tree"] for f in forms]))
    if not rows or not sum(r[1] for r in rows):
        return {"status": "no_gold_fields"}

    def stats(sample):
        m, g, p = (sum(r[k] for r in sample) for k in range(3))
        trees = [x for r in sample for x in r[3]]
        return (*_prf(m, g, p), sum(trees) / len(trees) if trees else None)

    point = stats(rows)
    rng = random.Random(seed)
    boot = [stats([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(draws)]
    names = ("precision", "recall", "f1", "section_tree")
    out = {"documents": len(ids), "gold_fields": sum(r[1] for r in rows), "pred_fields": sum(r[2] for r in rows)}
    for k, name in enumerate(names):
        vals = sorted(b[k] for b in boot if b[k] is not None)
        lo = vals[math.floor((len(vals) - 1) * 0.025)] if vals else None
        hi = vals[math.ceil((len(vals) - 1) * 0.975)] if vals else None
        out[name] = {"value": point[k], "ci95": [lo, hi]}
    return out
