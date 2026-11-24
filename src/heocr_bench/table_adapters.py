"""Fixed comparator adapters for table predictions (0032).

Some readers emit RTL tables in visual order: columns left to right, and/or
cell text in visual order. Each comparator lane gets ONE adapter, chosen on
DEV and pinned in the registration; it is applied uniformly to every table of
that lane, never per table. The scorer itself never mirrors (docs/tables.md).

Adapters compose left to right, e.g. "mirror_columns+reverse_words".
"""

import re

HEBREW = re.compile(r"[֐-׿]")
LTR_RUN = re.compile(r"[0-9A-Za-z.,:/%\-+]+")


def mirror_columns(table):
    """Reverse the column index: visual left-to-right grid -> logical RTL grid."""
    cols = int(table["cols"])
    return dict(table, cells=[dict(c, c=cols - int(c["c"]) - int(c.get("colspan", 1))) for c in table["cells"]])


def _map_text(table, fn):
    return dict(table, cells=[dict(c, text=fn(c.get("text") or "")) for c in table["cells"]])


def _reverse_words_line(line):
    return " ".join(line.split()[::-1]) if HEBREW.search(line) else line


def reverse_words(table):
    """Characters already logical, words in visual order: reverse word order in each Hebrew line."""
    return _map_text(table, lambda t: "\n".join(_reverse_words_line(x) for x in t.split("\n")))


def _visual_to_logical_line(line):
    if not HEBREW.search(line):
        return line
    flipped = line[::-1]
    return LTR_RUN.sub(lambda m: m.group(0)[::-1], flipped)


def visual_to_logical(table):
    """Whole line stored in visual order: reverse it and keep digit/Latin runs LTR.

    Visual-order layers store a line as the mirror image of its logical string (")1(" for "(1)"),
    so reversing the line restores bracket order without remapping bracket code points.
    """
    return _map_text(table, lambda t: "\n".join(_visual_to_logical_line(x) for x in t.split("\n")))


ADAPTERS = {"identity": lambda t: t, "mirror_columns": mirror_columns, "reverse_words": reverse_words,
            "visual_to_logical": visual_to_logical}


def adapter(spec):
    """Build a composed adapter from 'a+b+c'."""
    parts = [ADAPTERS[name] for name in spec.split("+")]

    def run(table):
        for fn in parts:
            table = fn(table)
        return table

    return run
