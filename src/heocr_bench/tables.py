"""Table-structure metrics: TEDS, TEDS-struct, GriTS-Con, GriTS-Top and exact match.

Definitions are spelled out in docs/tables.md; this module is the reference
implementation. Everything here is deterministic and dependency-free apart
from rapidfuzz.

Table format (``heocr_table.v1``)::

    {"rows": 3, "cols": 2, "header_rows": 1, "direction": "rtl",
     "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 2, "text": "..."}, ...]}

``c`` is the logical column index: column 0 is the first column in reading
order, so in a right-to-left table it is the rightmost one on the page. The
scorers never mirror a table; a prediction written in visual left-to-right
order is scored as written, and is penalised. Every grid position must be
covered by exactly one cell; positions no cell covers are filled with empty
1x1 cells by ``load_table``.
"""

from rapidfuzz.distance import Indel, Levenshtein

from .metrics import FOLDS, normalize

SCHEMA = "heocr_table.v1"


def cell_text(text):
    """NFC, bidi controls removed, registered folds, whitespace collapsed."""
    return normalize(text or "", FOLDS)


def load_table(obj):
    """Validate a table dict and return (rows, cols, header_rows, cells).

    ``cells`` is a list of (r, c, rowspan, colspan, text) tuples with
    normalised text, sorted by (r, c), with uncovered positions filled.
    """
    rows, cols = int(obj["rows"]), int(obj["cols"])
    header_rows = int(obj.get("header_rows", 0))
    if rows < 0 or cols < 0 or not 0 <= header_rows <= rows:
        raise ValueError("invalid table shape")
    owner = [[None] * cols for _ in range(rows)]
    cells = []
    for cell in obj.get("cells", []):
        r, c = int(cell["r"]), int(cell["c"])
        rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
        if rs < 1 or cs < 1 or r < 0 or c < 0 or r + rs > rows or c + cs > cols:
            raise ValueError(f"cell ({r}, {c}) outside the grid")
        for i in range(r, r + rs):
            for j in range(c, c + cs):
                if owner[i][j] is not None:
                    raise ValueError(f"cells overlap at ({i}, {j})")
                owner[i][j] = len(cells)
        cells.append((r, c, rs, cs, cell_text(cell.get("text", ""))))
    for i in range(rows):
        for j in range(cols):
            if owner[i][j] is None:
                owner[i][j] = len(cells)
                cells.append((i, j, 1, 1, ""))
    cells.sort(key=lambda x: (x[0], x[1]))
    return rows, cols, header_rows, cells


# ---------------------------------------------------------------- TEDS


class _Node:
    __slots__ = ("label", "text", "children")

    def __init__(self, label, text="", children=None):
        self.label = label
        self.text = text
        self.children = children or []


def _tree(table):
    """HTML-like tree: table > (thead, tbody) > tr > td.

    A cell belongs to the row it starts in. Rows below ``header_rows`` go to
    tbody; thead is present only when header_rows > 0. A td label carries its
    spans, so a span mismatch costs a full substitution.
    """
    rows, _cols, header_rows, cells = load_table(table)
    trs = [_Node("tr") for _ in range(rows)]
    for r, _c, rs, cs, text in cells:
        trs[r].children.append(_Node(f"td:{rs}:{cs}", text))
    parts = []
    if header_rows:
        parts.append(_Node("thead", children=trs[:header_rows]))
    parts.append(_Node("tbody", children=trs[header_rows:]))
    return _Node("table", children=parts)


def _postorder(root):
    nodes, lmd = [], []

    def walk(node):
        first = None
        for child in node.children:
            leftmost = walk(child)
            if first is None:
                first = leftmost
        nodes.append(node)
        lmd.append(len(nodes) - 1 if first is None else first)
        return lmd[-1]

    walk(root)
    return nodes, lmd


def _rename_cost(a, b, content):
    if a.label != b.label:
        return 1.0
    if not content or not a.label.startswith("td"):
        return 0.0
    if not a.text and not b.text:
        return 0.0
    return Levenshtein.distance(a.text, b.text) / max(len(a.text), len(b.text))


def tree_edit_distance(t1, t2, content=True):
    """Zhang-Shasha tree edit distance; insert and delete cost 1."""
    n1, l1 = _postorder(t1)
    n2, l2 = _postorder(t2)
    keyroots1 = sorted({max(i for i in range(len(n1)) if l1[i] == v) for v in set(l1)})
    keyroots2 = sorted({max(j for j in range(len(n2)) if l2[j] == v) for v in set(l2)})
    td = [[0.0] * len(n2) for _ in range(len(n1))]
    for i in keyroots1:
        for j in keyroots2:
            li, lj = l1[i], l2[j]
            w, h = i - li + 2, j - lj + 2
            fd = [[0.0] * h for _ in range(w)]
            for x in range(1, w):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, h):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, w):
                a = li + x - 1
                for y in range(1, h):
                    b = lj + y - 1
                    if l1[a] == li and l2[b] == lj:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[x - 1][y - 1] + _rename_cost(n1[a], n2[b], content))
                        td[a][b] = fd[x][y]
                    else:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[l1[a] - li][l2[b] - lj] + td[a][b])
    return td[-1][-1]


def _size(node):
    return 1 + sum(_size(c) for c in node.children)


def teds(gold, pred, content=True):
    """1 - TED / max(|T_gold|, |T_pred|). ``content=False`` gives TEDS-struct."""
    t1, t2 = _tree(gold), _tree(pred)
    return 1.0 - tree_edit_distance(t1, t2, content) / max(_size(t1), _size(t2))


# ---------------------------------------------------------------- GriTS


def _grid(table, kind):
    rows, cols, _h, cells = load_table(table)
    grid = [[None] * cols for _ in range(rows)]
    for r, c, rs, cs, text in cells:
        for i in range(r, r + rs):
            for j in range(c, c + cs):
                if kind == "con":
                    grid[i][j] = text
                else:  # the parent cell's box relative to this position
                    grid[i][j] = (c - j, r - i, c + cs - j, r + rs - i)
    return grid


def _sim_con(a, b):
    if not a and not b:
        return 1.0
    return Indel.normalized_similarity(a, b)  # 2*LCS / (|a| + |b|)


def _sim_top(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def _align(n, m, score):
    """Max-weight monotone alignment of two sequences (gaps cost 0).

    Returns (total, pairs). Ties prefer the diagonal, then skipping in the
    first sequence, so the result is deterministic.
    """
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j - 1] + score(i - 1, j - 1), dp[i - 1][j], dp[i][j - 1])
    pairs, i, j = [], n, m
    while i and j:
        if dp[i][j] == dp[i - 1][j - 1] + score(i - 1, j - 1):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return dp[n][m], pairs[::-1]


def grits(gold, pred, kind="con"):
    """GriTS (Smock et al. 2023), factored 2D most-similar-substructure form.

    Rows are aligned by the 1D alignment score of their cells, columns
    likewise, and the score is 2 * sum f(aligned grid cells) / (|G| + |P|),
    where |G| and |P| count grid positions.
    """
    if kind not in ("con", "top"):
        raise ValueError("kind is 'con' or 'top'")
    f = _sim_con if kind == "con" else _sim_top
    g, p = _grid(gold, kind), _grid(pred, kind)
    ng = sum(len(row) for row in g)
    np_ = sum(len(row) for row in p)
    if ng + np_ == 0:
        return 1.0
    if not ng or not np_:
        return 0.0
    gc, pc = len(g[0]), len(p[0])

    def row_score(i, j):
        return _align(gc, pc, lambda a, b: f(g[i][a], p[j][b]))[0]

    def col_score(a, b):
        return _align(len(g), len(p), lambda i, j: f(g[i][a], p[j][b]))[0]

    _t, row_pairs = _align(len(g), len(p), row_score)
    _t, col_pairs = _align(gc, pc, col_score)
    total = sum(f(g[i][a], p[j][b]) for i, j in row_pairs for a, b in col_pairs)
    return 2 * total / (ng + np_)


# ---------------------------------------------------------------- exact match


def exact_match(gold, pred):
    """True when shape, header rows, spans and normalised cell texts all agree."""
    return load_table(gold) == load_table(pred)


def score_table(gold, pred):
    return {"teds": teds(gold, pred), "teds_struct": teds(gold, pred, content=False),
            "grits_con": grits(gold, pred, "con"), "grits_top": grits(gold, pred, "top"),
            "exact": exact_match(gold, pred)}
