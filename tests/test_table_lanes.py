"""Pure parts of the table comparator harness (no third-party tool needed). Invented text only."""

import json
import unittest

from heocr_bench import table_lanes as tl
from heocr_bench.tables import load_table


class FromHtmlTest(unittest.TestCase):
    def test_spans_are_placed_on_an_occupancy_grid(self):
        html = ("<table><tr><th rowspan='2'>שם</th><th colspan='2'>סכום</th></tr>"
                "<tr><th>א</th><th>ב</th></tr><tr><td>קוד</td><td>1</td><td>2<br>3</td></tr></table>")
        (t,) = tl.from_html(html, 4)
        self.assertEqual((t["rows"], t["cols"], t["header_rows"], t["pages"]), (3, 3, 2, [4, 4]))
        cells = {(c["r"], c["c"]): c for c in t["cells"]}
        self.assertEqual(cells[(0, 0)]["rowspan"], 2)
        self.assertEqual(cells[(0, 1)]["colspan"], 2)
        self.assertEqual(cells[(1, 1)]["text"], "א")   # placed after the rowspan's covered position
        self.assertEqual(cells[(2, 2)]["text"], "2\n3")
        load_table(t)

    def test_nested_tables_stay_inside_their_cell_and_overlong_spans_are_clipped(self):
        html = "<table><tr><td rowspan='5'>x<table><tr><td>y</td></tr></table></td><td>z</td></tr></table>"
        (t,) = tl.from_html(html, 1)
        self.assertEqual((t["rows"], t["cols"]), (1, 2))
        self.assertEqual(t["cells"][0]["rowspan"], 1)
        load_table(t)

    def test_no_table_gives_nothing_and_bad_spans_count_as_one(self):
        self.assertEqual(tl.from_html("<p>טקסט</p>", 1), [])
        (t,) = tl.from_html("<table><tr><td colspan='x'>a</td><td>b</td></tr></table>", 1)
        self.assertEqual(t["cols"], 2)


class FromGridTest(unittest.TestCase):
    def test_ragged_rows_and_none(self):
        t = tl.from_grid([["א", None], ["ב"]], 2)
        self.assertEqual((t["rows"], t["cols"]), (2, 2))
        self.assertEqual([c["text"] for c in t["cells"]], ["א", "", "ב"])


class RunnerTest(unittest.TestCase):
    def test_a_failing_page_is_recorded_and_the_run_continues(self):
        import tempfile
        from pathlib import Path

        def fake():
            def run(pdf, page, folder):
                if page == 2:
                    raise RuntimeError("boom")
                return [tl.from_grid([["א"]], page)]
            return run, lambda: {"versions": {}}
        tl.LANES["fake"] = fake
        try:
            with tempfile.TemporaryDirectory() as d:
                m = Path(d) / "m.jsonl"
                m.write_text("\n".join(json.dumps({"doc_id": f"d{p}", "pdf": "x.pdf", "page": p}) for p in (1, 2)))
                rec = tl.run_lane("fake", m, Path(d) / "out")
                self.assertEqual(list(rec["errors"]), ["d2"])
                self.assertEqual(len(json.loads((Path(d) / "out/fake/d1.json").read_text())), 1)
                self.assertIn("error", json.loads((Path(d) / "out/fake/d2.json").read_text()))
                self.assertTrue((Path(d) / "out/fake/run.json").exists())
        finally:
            del tl.LANES["fake"]


if __name__ == "__main__":
    unittest.main()
