"""Table metrics on invented fixtures only (no benchmark or DEV text)."""

import unittest

from heocr_bench.tables import (_Node, exact_match, grits, load_table, score_table, teds,
                                tree_edit_distance)


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
        self.assertEqual(cells, [(0, 0, 1, 1, "x"), (0, 1, 1, 1, "")])
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
                         {"teds": 1.0, "teds_struct": 1.0, "grits_con": 1.0, "grits_top": 1.0, "exact": True})

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


if __name__ == "__main__":
    unittest.main()
