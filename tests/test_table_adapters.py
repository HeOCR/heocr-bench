"""Comparator adapters on invented fixtures only."""

import unittest

from heocr_bench.table_adapters import adapter, mirror_columns, reverse_words, visual_to_logical
from heocr_bench.tables import exact_match

LOGICAL = {"rows": 1, "cols": 3, "cells": [
    {"r": 0, "c": 0, "text": "שם מלא"},
    {"r": 0, "c": 1, "text": "(א) 12.5"},
    {"r": 0, "c": 2, "text": "abc"}]}


def visual(t):
    """What a visual-order reader emits: columns LTR, each Hebrew line stored as drawn."""
    return visual_to_logical(mirror_columns(t))  # the line transform is its own inverse


class AdapterTests(unittest.TestCase):
    def test_mirror_is_an_involution_and_keeps_spans(self):
        t = {"rows": 1, "cols": 3, "cells": [{"r": 0, "c": 0, "colspan": 2, "text": "x"}, {"r": 0, "c": 2, "text": "y"}]}
        m = mirror_columns(t)
        self.assertEqual([(c["c"], c.get("colspan", 1)) for c in m["cells"]], [(1, 2), (0, 1)])
        self.assertEqual(mirror_columns(m), t)

    def test_reverse_words_only_touches_hebrew_lines(self):
        t = {"rows": 1, "cols": 1, "cells": [{"r": 0, "c": 0, "text": "מלא שם\nab cd"}]}
        self.assertEqual(reverse_words(t)["cells"][0]["text"], "שם מלא\nab cd")

    def test_visual_to_logical_keeps_numbers_and_bracket_order(self):
        # Logical "(א) 12.5" drawn RTL is stored left to right as "12.5 )א(".
        drawn = {"rows": 1, "cols": 1, "cells": [{"r": 0, "c": 0, "text": "12.5 )\u05d0("}]}
        self.assertEqual(visual_to_logical(drawn)["cells"][0]["text"], "(\u05d0) 12.5")

    def test_composed_adapter_recovers_a_visual_reader(self):
        recovered = adapter("mirror_columns+visual_to_logical")(visual(LOGICAL))
        self.assertTrue(exact_match(LOGICAL, recovered))
        self.assertTrue(exact_match(LOGICAL, adapter("identity")(LOGICAL)))

    def test_unknown_adapter_is_rejected(self):
        with self.assertRaises(KeyError):
            adapter("mirror_columns+nope")


if __name__ == "__main__":
    unittest.main()
