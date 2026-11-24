"""Form metrics on invented fixtures only."""

import unittest

from heocr_bench.forms import (field_counts, fields, label_text, pool_forms, score_form,
                               section_tree_similarity)


def field(label, value, kind="text"):
    return {"type": "field", "label": label, "value": value, "kind": kind, "slot": "line"}


GOLD = {"schema": "hdoc_form.v1", "sections": [
    {"title": "פרטים", "children": [
        field("שם:", "דנה"),
        {"type": "group", "columns": 2, "children": [
            field("תאריך", "01.02.2026", "date"),
            field("תאריך", "", "date")]},
        field("מאשר", "checked", "checkbox")]},
    {"title": "חתימה", "children": [field("חתימה –", "", "signature")]}]}


class FormTests(unittest.TestCase):
    def test_labels_drop_one_trailing_colon_or_dash(self):
        self.assertEqual(label_text("שם:"), "שם")
        self.assertEqual(label_text("שם –"), "שם")
        self.assertEqual(label_text("שם::"), "שם:")

    def test_groups_flatten_and_unfilled_is_empty_string(self):
        self.assertEqual(len(fields(GOLD)), 5)
        bad = {"sections": [{"children": [{"type": "field", "label": "x", "value": None}]}]}
        with self.assertRaisesRegex(ValueError, "unfilled"):
            fields(bad)

    def test_identical_form_scores_perfectly(self):
        self.assertEqual(score_form(GOLD, GOLD),
                         {"gold_fields": 5, "pred_fields": 5, "matched": 5, "section_tree": 1.0})

    def test_duplicate_labels_keep_their_exact_matches(self):
        swapped = {"sections": [{"title": "", "children": [
            field("תאריך", ""), field("תאריך", "01.02.2026")]}]}
        gold = {"sections": [{"title": "", "children": [
            field("תאריך", "01.02.2026"), field("תאריך", "")]}]}
        self.assertEqual(field_counts(gold, swapped)["matched"], 2)

    def test_exact_pair_beats_a_chain_of_similar_labels(self):
        # Keeping the exact pair (g1, p1) leaves two zero-similarity pairs (total = bonus + 1); breaking it gives three
        # similarity-1 pairs (g1-p2, g2-p3, g3-p1). A small bonus ties or loses the exact match; W must dominate.
        gold = {"sections": [{"title": "", "children": [field("ab", "1"), field("cd", "2"), field("ab", "7")]}]}
        pred = {"sections": [{"title": "", "children": [field("ab", "1"), field("ab", "9"), field("cd", "8")]}]}
        self.assertEqual(field_counts(gold, pred)["matched"], 1)

    def test_matched_count_is_the_maximum_number_of_exact_pairs(self):
        import itertools
        import random
        rng = random.Random(5)
        labels, values = ["ab", "ac", "cd", "x"], ["1", "2"]
        for _ in range(150):
            g = [(rng.choice(labels), rng.choice(values)) for _ in range(rng.randint(1, 4))]
            p = [(rng.choice(labels), rng.choice(values)) for _ in range(rng.randint(1, 4))]
            best = max(sum(1 for i, j in zip(rows, cols) if g[i] == p[j])
                       for k in range(min(len(g), len(p)) + 1)
                       for rows in itertools.permutations(range(len(g)), k)
                       for cols in itertools.permutations(range(len(p)), k))
            form = lambda fs: {"sections": [{"title": "", "children": [field(a, b) for a, b in fs]}]}
            self.assertEqual(field_counts(form(g), form(p))["matched"], best)

    def test_wrong_value_is_unmatched_but_tree_ignores_values(self):
        pred = {"sections": [dict(s) for s in GOLD["sections"]]}
        pred["sections"][0] = dict(pred["sections"][0], children=[field("שם", "אחר")]
                                   + pred["sections"][0]["children"][1:])
        result = score_form(GOLD, pred)
        self.assertEqual(result["matched"], 4)
        self.assertEqual(result["section_tree"], 1.0)

    def test_flat_prediction_loses_tree_structure(self):
        flat = {"sections": [{"title": "", "children": [c for s in GOLD["sections"] for c in s["children"]]}]}
        self.assertLess(section_tree_similarity(GOLD, flat), 1.0)
        self.assertEqual(field_counts(GOLD, flat)["matched"], 5)

    def test_pool_counts_and_determinism(self):
        docs = {"a": [score_form(GOLD, GOLD)], "b": [score_form(GOLD, {"sections": []})]}
        out = pool_forms(docs, draws=300)
        self.assertEqual((out["gold_fields"], out["pred_fields"]), (10, 5))
        self.assertAlmostEqual(out["recall"]["value"], 0.5)
        self.assertAlmostEqual(out["precision"]["value"], 1.0)
        self.assertAlmostEqual(out["f1"]["value"], 2 * 5 / 15)
        self.assertEqual(out, pool_forms(docs, draws=300))
        self.assertEqual(pool_forms({}), {"status": "no_gold_fields"})


if __name__ == "__main__":
    unittest.main()
