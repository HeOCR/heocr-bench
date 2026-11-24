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
