"""Table-structure metrics: TEDS, TEDS-struct, GriTS-Con, GriTS-Top and exact match.

Definitions are spelled out in docs/tables.md; this module is the reference
implementation. Everything here is deterministic and dependency-free apart
from rapidfuzz.

Table format (``heocr_table.v1``)::

    {"rows": 3, "cols": 2, "header_rows": 1, "direction": "rtl",
     "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 2, "text": "..."}, ...]}

A cell may carry ``"mask": true``: its text is excluded from scoring (the
cell's position and spans still count). See docs/tables.md.

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

    ``cells`` is a list of (r, c, rowspan, colspan, text, masked) tuples with
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
        masked = cell.get("mask", False)
        if type(masked) is not bool:
            raise ValueError("mask must be a boolean")
        cells.append((r, c, rs, cs, cell_text(cell.get("text", "")), masked))
    for i in range(rows):
        for j in range(cols):
            if owner[i][j] is None:
                owner[i][j] = len(cells)
                cells.append((i, j, 1, 1, "", False))
    cells.sort(key=lambda x: (x[0], x[1]))
    return rows, cols, header_rows, cells


# ---------------------------------------------------------------- TEDS


class _Node:
    __slots__ = ("label", "text", "children", "masked")

    def __init__(self, label, text="", children=None, masked=False):
        self.label = label
        self.text = text
        self.children = children or []
        self.masked = masked


def _tree(table):
    """HTML-like tree: table > (thead, tbody) > tr > td.

    A cell belongs to the row it starts in. Rows below ``header_rows`` go to
    tbody; thead is present only when header_rows > 0. A td label carries its
    spans, so a span mismatch costs a full substitution.
    """
    rows, _cols, header_rows, cells = load_table(table)
    trs = [_Node("tr") for _ in range(rows)]
    for r, _c, rs, cs, text, masked in cells:
        trs[r].children.append(_Node(f"td:{rs}:{cs}", text, masked=masked))
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
    if not content or not a.label.startswith("td") or a.masked or b.masked:
        return 0.0
    if not a.text and not b.text:
        return 0.0
    return Levenshtein.distance(a.text, b.text) / max(len(a.text), len(b.text))


def tree_edit_distance(t1, t2, content=True, rename=None):
    """Zhang-Shasha tree edit distance; insert and delete cost 1.

    ``rename(a, b)`` overrides the substitution cost (default: the TEDS td rule).
    """
    rename = rename or (lambda a, b: _rename_cost(a, b, content))
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
                                       fd[x - 1][y - 1] + rename(n1[a], n2[b]))
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
    for r, c, rs, cs, text, masked in cells:
        for i in range(r, r + rs):
            for j in range(c, c + cs):
                if kind == "con":
                    grid[i][j] = None if masked else text
                else:  # the parent cell's box relative to this position
                    grid[i][j] = (c - j, r - i, c + cs - j, r + rs - i)
    return grid


def _sim_con(a, b):
    if a is None:  # a masked gold position contributes nothing (excluded, see grits)
        return 0.0
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
    where |G| and |P| count grid positions. In GriTS-Con a masked gold
    position scores 0 during alignment and is excluded afterwards: it leaves
    |G|, and the predicted position aligned to it (if any) leaves |P|.
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
    if kind == "con":
        # Masked gold positions are excluded: out of the numerator (f = 0), out of |G|, and the predicted
        # positions aligned to them out of |P|.
        ng -= sum(1 for row in g for x in row if x is None)
        np_ -= sum(1 for i, _j in row_pairs for a, _b in col_pairs if g[i][a] is None)
        if ng + np_ == 0:
            return 1.0
    return 2 * total / (ng + np_)


# ---------------------------------------------------------------- exact match


def exact_match(gold, pred):
    """True when shape, header rows, spans and normalised unmasked cell texts all agree."""
    g, p = load_table(gold), load_table(pred)
    if g[:3] != p[:3] or len(g[3]) != len(p[3]):
        return False
    for a, b in zip(g[3], p[3]):
        if a[:4] != b[:4] or (not a[5] and a[4] != b[4]):
            return False
    return True


def score_table(gold, pred):
    return {"teds": teds(gold, pred), "teds_struct": teds(gold, pred, content=False),
            "grits_con": grits(gold, pred, "con"), "grits_top": grits(gold, pred, "top"),
            "exact": exact_match(gold, pred), "masked_cells": masked_cells(gold)}


def masked_cells(table):
    """Number of gold cells whose text is masked."""
    return sum(1 for cell in load_table(table)[3] if cell[5])


# ---------------------------------------------------------------- documents and pooling

METRICS = ("teds", "teds_struct", "grits_con", "grits_top", "exact")
ZERO = {"teds": 0.0, "teds_struct": 0.0, "grits_con": 0.0, "grits_top": 0.0, "exact": False}


def _pages(t):
    pages = t.get("pages")
    return None if pages is None else set(range(int(pages[0]), int(pages[1]) + 1))


def _assign(weights):
    """Maximum-weight assignment (Hungarian, O(n^3)) on a rectangular matrix.

    Returns {row: col}. Ties resolve to the lowest indices, deterministically.
    """
    n_r = len(weights)
    n_c = len(weights[0]) if n_r else 0
    n = max(n_r, n_c)
    if n == 0:
        return {}
    big = max((w for row in weights for w in row), default=0.0)
    cost = [[(big - weights[i][j]) if i < n_r and j < n_c else big for j in range(n)] for i in range(n)]
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return {p[j] - 1: j - 1 for j in range(1, n + 1) if p[j] and p[j] - 1 < n_r and j - 1 < n_c}


def score_document(gold_tables, pred_tables):
    """Match predicted to gold tables in one document and score each gold table.

    A pair is eligible only if both tables' ``pages`` ranges ([first, last],
    inclusive) overlap; a table without ``pages`` is eligible with every table.
    The matching maximises the summed TEDS over eligible pairs with TEDS > 0.
    An unmatched gold table scores 0 on every measure. Returns per-gold-table
    scores (in gold order) and the number of unmatched predicted tables.
    """
    w = [[0.0] * len(pred_tables) for _ in gold_tables]
    for i, g in enumerate(gold_tables):
        for j, p in enumerate(pred_tables):
            gp, pp = _pages(g), _pages(p)
            if gp is None or pp is None or gp & pp:
                w[i][j] = teds(g, p)
    match = {i: j for i, j in _assign(w).items() if w[i][j] > 0}
    per_table = [score_table(g, pred_tables[match[i]]) if i in match else dict(ZERO, masked_cells=masked_cells(g))
                 for i, g in enumerate(gold_tables)]
    return {"tables": per_table, "matched": len(match),
            "unmatched_predictions": len(pred_tables) - len(match)}


def pool(documents, seed=0, draws=2000):
    """Mean over gold tables, with a percentile bootstrap that resamples documents.

    ``documents`` maps document id -> score_document() result. The point value
    of each measure is the mean over all gold tables (exact match as a
    fraction). The 95% interval resamples documents with replacement
    (``draws`` draws, ``random.Random(seed)``, documents in sorted-id order)
    and takes the lower and upper 2.5% quantiles of the table-weighted means.
    """
    import math
    import random

    ids = sorted(documents)
    per_doc = []
    for d in ids:
        tables = documents[d]["tables"]
        per_doc.append((len(tables), {m: sum(float(t[m]) for t in tables) for m in METRICS}))
    total = sum(n for n, _ in per_doc)
    if not total:
        return {"status": "no_gold_tables"}
    out = {"documents": len(ids), "gold_tables": total,
           "masked_cells": sum(int(t.get("masked_cells", 0)) for d in ids for t in documents[d]["tables"]),
           "unmatched_predictions": sum(documents[d]["unmatched_predictions"] for d in ids)}
    rng = random.Random(seed)
    samples = {m: [] for m in METRICS}
    for _ in range(draws):
        pick = [per_doc[rng.randrange(len(per_doc))] for _ in per_doc]
        n = sum(k for k, _ in pick)
        for m in METRICS:
            samples[m].append(sum(s[m] for _, s in pick) / n if n else math.nan)
    for m in METRICS:
        vals = sorted(x for x in samples[m] if not math.isnan(x))
        lo = vals[math.floor((len(vals) - 1) * 0.025)] if vals else None
        hi = vals[math.ceil((len(vals) - 1) * 0.975)] if vals else None
        out[m] = {"mean": sum(s[m] for _, s in per_doc) / total, "ci95": [lo, hi]}
    return out
