"""Table metrics on invented fixtures only (no benchmark or DEV text)."""

import unittest

from heocr_bench.metrics import FOLDS, PROFILE_SHA, normalize
from heocr_bench.tables import (_Node, _assign, cell_text, exact_match, grits, load_table, pool,
                                score_document, score_table, teds, tree_edit_distance)


def table(rows, cols, texts, header_rows=0, spans=()):
    """texts: row-major list of strings for 1x1 cells; spans: (r, c, rs, cs, text) cells first."""
    cells, covered = [], set()
    for r, c, rs, cs, text in spans:
        cells.append({"r": r, "c": c, "rowspan": rs, "colspan": cs, "text": text})
        covered |= {(i, j) for i in range(r, r + rs) for j in range(c, c + cs)}
    it = iter(texts)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in covered:
                cells.append({"r": r, "c": c, "text": next(it)})
    return {"rows": rows, "cols": cols, "header_rows": header_rows, "cells": cells}


# Invented RTL fixture: column 0 is the logical first (rightmost) column.
GOLD = table(3, 3, ["שם", "כמות", "מחיר",
                    "תפוח", "2", "10.50",
                    "אגס", "5", "3.20"], header_rows=1)


def mirrored(t):
    cols = t["cols"]
    out = dict(t)
    out["cells"] = [dict(c, c=cols - c["c"] - c.get("colspan", 1)) for c in t["cells"]]
    return out


class TableFormatTests(unittest.TestCase):
    def test_uncovered_positions_are_filled_and_overlaps_rejected(self):
        rows, cols, _h, cells = load_table({"rows": 1, "cols": 2, "cells": [{"r": 0, "c": 0, "text": "x"}]})
        self.assertEqual(cells, [(0, 0, 1, 1, "x", False), (0, 1, 1, 1, "", False)])
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_table({"rows": 1, "cols": 2, "cells": [{"r": 0, "c": 0, "colspan": 2}, {"r": 0, "c": 1}]})
        with self.assertRaisesRegex(ValueError, "outside"):
            load_table({"rows": 1, "cols": 1, "cells": [{"r": 0, "c": 0, "colspan": 2}]})

    def test_cell_text_is_normalised(self):
        a = table(1, 1, ["‏א  ב"])
        b = table(1, 1, ["א ב"])
        self.assertTrue(exact_match(a, b))


class TedsTests(unittest.TestCase):
    def test_identical_tables_score_one(self):
        self.assertEqual(score_table(GOLD, GOLD),
                         {"teds": 1.0, "teds_struct": 1.0, "grits_con": 1.0, "grits_top": 1.0, "exact": True,
                          "masked_cells": 0})

    def test_zhang_shasha_on_small_trees(self):
        # a(b, c) vs a(b): one deletion.
        t1 = _Node("a", children=[_Node("b"), _Node("c")])
        t2 = _Node("a", children=[_Node("b")])
        self.assertEqual(tree_edit_distance(t1, t2), 1)
        # a(b(c, d)) vs a(c, d): delete b, children move up.
        t3 = _Node("a", children=[_Node("b", children=[_Node("c"), _Node("d")])])
        t4 = _Node("a", children=[_Node("c"), _Node("d")])
        self.assertEqual(tree_edit_distance(t3, t4), 1)

    def test_one_wrong_cell_text_costs_its_normalised_edit_distance(self):
        pred = table(3, 3, ["שם", "כמות", "מחיר",
                            "תפוח", "2", "10.60",
                            "אגס", "5", "3.20"], header_rows=1)
        # Tree: table, thead, tbody, 3 tr, 9 td = 15 nodes; one char of five differs.
        self.assertAlmostEqual(teds(GOLD, pred), 1 - (1 / 5) / 15)
        self.assertEqual(teds(GOLD, pred, content=False), 1.0)

    def test_visual_order_output_is_penalised(self):
        flipped = mirrored(GOLD)
        self.assertLess(teds(GOLD, flipped), 1.0)
        self.assertEqual(teds(GOLD, flipped, content=False), 1.0)
        self.assertLess(grits(GOLD, flipped, "con"), 1.0)

    def test_span_mismatch_is_structural(self):
        gold = table(2, 2, ["a", "b"], spans=[(0, 0, 1, 2, "h")])
        pred = table(2, 2, ["h", "", "a", "b"])
        self.assertLess(teds(gold, pred, content=False), 1.0)
        self.assertLess(grits(gold, pred, "top"), 1.0)

    def test_missing_header_split_costs_structure(self):
        no_header = dict(GOLD, header_rows=0)
        self.assertLess(teds(GOLD, no_header, content=False), 1.0)
        self.assertEqual(grits(GOLD, no_header, "con"), 1.0)


def brute_ted(a, b):
    """Exact TED by the recursive forest definition (rightmost-root decomposition)."""
    from functools import lru_cache
    from heocr_bench.tables import _rename_cost

    @lru_cache(None)
    def forest(f1, f2):
        if not f1 and not f2:
            return 0.0
        if not f1:
            return forest(f1, f2[:-1] + tuple(f2[-1].children)) + 1
        if not f2:
            return forest(f1[:-1] + tuple(f1[-1].children), f2) + 1
        v, w = f1[-1], f2[-1]
        return min(forest(f1[:-1] + tuple(v.children), f2) + 1,
                   forest(f1, f2[:-1] + tuple(w.children)) + 1,
                   forest(tuple(v.children), tuple(w.children)) + forest(f1[:-1], f2[:-1])
                   + _rename_cost(v, w, True))

    return forest((a,), (b,))


class ZhangShashaReferenceTests(unittest.TestCase):
    def test_matches_brute_force_on_seeded_random_trees(self):
        import random
        rng = random.Random(20261125)

        def rnd(depth):
            node = _Node(rng.choice(["td:1:1", "td:1:2", "tr"]), rng.choice(["", "ab", "abc", "b"]))
            if depth < 3:
                node.children = [rnd(depth + 1) for _ in range(rng.randint(0, 3))]
            return node

        for _ in range(300):
            a, b = rnd(0), rnd(0)
            self.assertAlmostEqual(tree_edit_distance(a, b), brute_ted(a, b))


class GritsTests(unittest.TestCase):
    def test_extra_row_scores_by_grid_size(self):
        gold = table(2, 2, ["a", "b", "c", "d"])
        pred = table(3, 2, ["a", "b", "x", "y", "c", "d"])
        # 4 matched positions; |G| = 4, |P| = 6.
        self.assertAlmostEqual(grits(gold, pred, "con"), 2 * 4 / 10)
        self.assertAlmostEqual(grits(gold, pred, "top"), 2 * 4 / 10)

    def test_merged_cell_topology_iou(self):
        gold = table(1, 2, [], spans=[(0, 0, 1, 2, "ab")])
        pred = table(1, 2, ["ab", ""])
        # Each gold position holds a 1x2 box; each pred position a 1x1 box: IoU 1/2.
        self.assertAlmostEqual(grits(gold, pred, "top"), 0.5)

    def test_empty_tables(self):
        empty = {"rows": 0, "cols": 0, "cells": []}
        self.assertEqual(grits(empty, empty), 1.0)
        self.assertEqual(grits(GOLD, empty), 0.0)

    def test_deterministic(self):
        pred = mirrored(GOLD)
        self.assertEqual(grits(GOLD, pred, "con"), grits(GOLD, pred, "con"))


def masked(t, r, c):
    out = dict(t)
    out["cells"] = [dict(x, mask=True) if (x["r"], x["c"]) == (r, c) else x for x in t["cells"]]
    return out


class MaskTests(unittest.TestCase):
    def test_masked_cell_text_is_ignored_but_structure_counts(self):
        pred = table(3, 3, ["\u05e9\u05dd", "\u05db\u05de\u05d5\u05ea", "\u05de\u05d7\u05d9\u05e8",
                            "\u05ea\u05e4\u05d5\u05d7", "2", "99.99",
                            "\u05d0\u05d2\u05e1", "5", "3.20"], header_rows=1)
        gold = masked(GOLD, 1, 2)
        self.assertLess(teds(GOLD, pred), 1.0)
        self.assertEqual(score_table(gold, pred),
                         {"teds": 1.0, "teds_struct": 1.0, "grits_con": 1.0, "grits_top": 1.0, "exact": True,
                          "masked_cells": 1})
        wider = table(3, 4, ["\u05e9\u05dd", "\u05db\u05de\u05d5\u05ea", "\u05de\u05d7\u05d9\u05e8", "",
                             "\u05ea\u05e4\u05d5\u05d7", "2", "99.99", "",
                             "\u05d0\u05d2\u05e1", "5", "3.20", ""], header_rows=1)
        self.assertFalse(exact_match(gold, wider))
        self.assertLess(teds(gold, wider, content=False), 1.0)

    def test_grits_con_excludes_masked_positions(self):
        gold = {"rows": 1, "cols": 2, "cells": [{"r": 0, "c": 0, "text": "ab"},
                                                {"r": 0, "c": 1, "text": "zz", "mask": True}]}
        right = {"rows": 1, "cols": 2, "cells": [{"r": 0, "c": 0, "text": "ab"}, {"r": 0, "c": 1, "text": "qq"}]}
        wrong = {"rows": 1, "cols": 2, "cells": [{"r": 0, "c": 0, "text": "xy"}, {"r": 0, "c": 1, "text": "qq"}]}
        # The masked column is aligned and dropped from both sides: 2 * 1 / (1 + 1).
        self.assertAlmostEqual(grits(gold, right, "con"), 1.0)
        # No wildcard credit: with the only unmasked cell wrong, the score is 0, not 0.5.
        self.assertAlmostEqual(grits(gold, wrong, "con"), 0.0)
        extra = {"rows": 1, "cols": 3, "cells": [{"r": 0, "c": 0, "text": "ab"}, {"r": 0, "c": 1, "text": "qq"},
                                                 {"r": 0, "c": 2, "text": "ww"}]}
        # Unaligned extra predicted cells still count in |P|.
        self.assertLess(grits(gold, extra, "con"), 1.0)
        everything_masked = {"rows": 1, "cols": 1, "cells": [{"r": 0, "c": 0, "text": "a", "mask": True}]}
        self.assertEqual(grits(everything_masked, {"rows": 1, "cols": 1, "cells": [{"r": 0, "c": 0}]}, "con"), 1.0)

    def test_mask_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_table({"rows": 1, "cols": 1, "cells": [{"r": 0, "c": 0, "mask": "yes"}]})


class NormaliserPinTests(unittest.TestCase):
    """Cell text uses the registered page scorer's line_fold_v1 implementation."""

    def test_folds_are_the_registered_line_fold_v1_table(self):
        self.assertEqual(PROFILE_SHA[:8], "2e262710")
        self.assertEqual({chr(k): v for k, v in FOLDS.items()}, {
            "\u05be": "-", "\u05f3": "'", "\u05f4": '"', "\u201c": '"', "\u201d": '"', "\u201e": '"',
            "\u2014": "\u2013"})

    def test_cell_text_equals_page_normalize(self):
        samples = ["\u200f\u05d0\u05be\u05d1", "\u05d2\u05f4\u05d3 \u2014 1\u20132",
                   "\u05e9\u05b8\u05c1 x\n y", "\u201c\u05d0\u201d  \u202b\u05d1\u202c", ""]
        for text in samples:
            self.assertEqual(cell_text(text), normalize(text, FOLDS))
        # Line breaks inside a cell fold to one space; en dash and hyphen stay distinct.
        self.assertEqual(cell_text("a\nb"), "a b")
        self.assertNotEqual(cell_text("1\u20132"), cell_text("1-2"))


class DocumentTests(unittest.TestCase):
    def test_hungarian_is_optimal_on_small_matrices(self):
        import itertools
        import random
        rng = random.Random(7)
        for _ in range(200):
            n, m = rng.randint(1, 4), rng.randint(1, 4)
            w = [[rng.choice([0.0, 0.2, 0.5, 0.9, 1.0]) for _ in range(m)] for _ in range(n)]
            got = sum(w[i][j] for i, j in _assign(w).items())
            best = max(sum(w[i][j] for i, j in zip(rows, cols))
                       for k in range(min(n, m) + 1)
                       for rows in itertools.permutations(range(n), k)
                       for cols in itertools.permutations(range(m), k))
            self.assertAlmostEqual(got, best)

    def test_matching_respects_pages_and_scores_missing_tables_zero(self):
        a = dict(table(1, 2, ["\u05d0", "1"]), pages=[1, 1])
        b = dict(table(1, 2, ["\u05d1", "2"]), pages=[2, 3])
        pred_b = dict(b, pages=[3, 3])
        stray = dict(table(1, 2, ["\u05d0", "1"]), pages=[5, 5])
        result = score_document([a, b], [stray, pred_b])
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["unmatched_predictions"], 1)
        self.assertEqual(result["tables"][0]["teds"], 0.0)
        self.assertEqual(result["tables"][1]["teds"], 1.0)

    def test_pool_is_table_weighted_and_deterministic(self):
        perfect = {"teds": 1.0, "teds_struct": 1.0, "grits_con": 1.0, "grits_top": 1.0, "exact": True}
        zero = {"teds": 0.0, "teds_struct": 0.0, "grits_con": 0.0, "grits_top": 0.0, "exact": False}
        docs = {"d1": {"tables": [perfect, perfect, perfect], "unmatched_predictions": 0},
                "d2": {"tables": [zero], "unmatched_predictions": 2}}
        out = pool(docs, seed=0, draws=500)
        self.assertEqual(out["gold_tables"], 4)
        self.assertEqual(out["unmatched_predictions"], 2)
        self.assertEqual(out["masked_cells"], 0)
        self.assertAlmostEqual(out["teds"]["mean"], 0.75)
        self.assertAlmostEqual(out["exact"]["mean"], 0.75)
        lo, hi = out["teds"]["ci95"]
        self.assertTrue(0.0 <= lo <= 0.75 <= hi <= 1.0)
        self.assertEqual(out, pool(docs, seed=0, draws=500))
        self.assertEqual(pool({}), {"status": "no_gold_tables"})


if __name__ == "__main__":
    unittest.main()
