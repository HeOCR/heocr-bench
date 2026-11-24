# Table-structure metrics

Module: `heocr_bench.tables`. Status: scorer only. No set is registered yet; the tables and forms benchmark (decision 0032) will pin this scorer by commit before any held-out pass.

## Table format: `heocr_table.v1`

```json
{"rows": 3, "cols": 3, "header_rows": 1, "direction": "rtl",
 "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 2, "text": "..."}]}
```

- **Logical column order.** `c = 0` is the first column in reading order. In a right-to-left table that is the rightmost column on the page. The scorer never mirrors a table, so a prediction written in visual left-to-right order is scored as written and loses points. `direction` is informational.
- **Coverage.** Every grid position belongs to exactly one cell. Overlapping or out-of-grid cells are an error. Positions that no cell covers are filled with empty 1×1 cells.
- **Masked cells.** A cell may carry `"mask": true` (for example an illegible or redacted cell). Its text is never scored; its position and spans still are:
  - TEDS: a substitution touching a masked `td` costs 0 for content (a span difference still costs 1);
  - GriTS-Con: a masked gold position has similarity 1 with whatever it is aligned to; GriTS-Top is unchanged;
  - exact match: the masked cell's text is ignored, everything else must agree.
- **Page range.** A table may carry `"pages": [first, last]` (inclusive), used only for document matching.
- **Header rows.** `header_rows` counts the leading rows that form the column header (the `thead`).
- **Cell text normalisation.** The same as the page scorer (`metrics.normalize` with the registered `FOLDS`):
  - NFC;
  - bidi control characters removed;
  - maqaf → hyphen, geresh/gershayim and curly quotes → ASCII, em dash → en dash;
  - runs of whitespace (line breaks included) → one space, trimmed.

  Hebrew pointing is kept.

### Pin to `line_fold_v1`

`cell_text` is `metrics.normalize(text, metrics.FOLDS)`: the same function and fold table the registered page scorer uses for `line_fold_v1` (profile sha256 `2e262710…`, checked by `data.split_descriptor`). The tests pin the fold table one for one and check `cell_text` against `normalize` on samples. Differences from the page protocol, all structural rather than character folds:

1. A cell has no line structure: line breaks inside a cell become one space (the page protocol joins lines with a space too, but also keeps them for line diagnostics).
2. The ` | ` cell-delimiter rule of `text_format: heocr-cell-delimited` doesn't apply: cells come as structure.
3. Masking is per cell (above), not the page protocol's per-character mask projection.

## TEDS and TEDS-struct

This follows Zhong et al. (2020, PubTabNet), with HeOCR's choices stated explicitly below.

- **Tree.** `table → [thead] → tbody → tr → td`.
  - `thead` holds the first `header_rows` rows and is present only when `header_rows > 0`.
  - A cell belongs to the row it starts in.
  - Each `td` is labelled with its `rowspan` and `colspan`.
- **Costs.**
  - Insertion and deletion cost 1.
  - Substituting nodes with different labels costs 1, and a different span counts as a different label.
  - Substituting two `td` nodes with the same spans costs the normalised Levenshtein distance of their texts: `lev(a, b) / max(|a|, |b|)`, or 0 if both are empty.
  - In TEDS-struct every same-label substitution costs 0.
- **Score.** `TEDS = 1 − TED(G, P) / max(|G|, |P|)`, where `|T|` counts nodes.
- **TED algorithm.** Exact Zhang–Shasha. The test suite checks it against a brute-force recursive definition.
  - In our check of 3,000 random small trees, the `apted` Python package returned a non-optimal distance in 9 cases.
  - Third-party TEDS figures computed with that package can therefore be slightly lower than ours on the same pair.

## GriTS-Con and GriTS-Top

This follows Smock et al. (2023), in the factored "2D most similar substructure" form.

- **Grid.** Each grid position takes its cell's value, so a spanning cell is repeated at every position it covers.
  - In GriTS-Con the value is the cell's normalised text.
  - In GriTS-Top the value is the cell's box relative to that position: `(c0 − c, r0 − r, c0 + colspan − c, r0 + rowspan − r)`.
- **Position similarity `f`.**
  - Con: `2·LCS(a, b) / (|a| + |b|)`, which is 1 when both are empty.
  - Top: the IoU of the two boxes.
- **Alignment.**
  - Rows are aligned by a monotone maximum-weight alignment with zero gap cost. The weight of a row pair is the same 1D alignment score over the cells of the two rows.
  - Columns are aligned the same way, with columns in place of rows.
  - When two alignments score the same, the diagonal step wins, then skipping a gold row or column.
- **Score.** `GriTS = 2 · Σ f(G[i][a], P[j][b]) / (|G| + |P|)`, summed over aligned row pairs × aligned column pairs, where `|·|` counts grid positions.
  - Two empty tables score 1. An empty table against a non-empty one scores 0.
- **Not included.** GriTS-Loc needs cell boxes, which a CSV/XLSX export doesn't carry.

## Exact match

A table matches exactly when its shape, `header_rows`, every cell's position and spans, and every normalised cell text all agree.

## Documents and pooling

- **Matching (`score_document`).** Within one document, predicted tables are matched one-to-one to gold tables by a maximum-weight assignment (Hungarian) on TEDS. A pair is eligible only if the two `pages` ranges overlap (a table without `pages` is eligible with every table), and only pairs with TEDS > 0 count as matched. An unmatched gold table scores 0 on every measure; unmatched predicted tables are counted and reported, not averaged in.
- **Pooling (`pool`).** The point value of each measure is the mean over all gold tables in the set (exact match as a fraction). The 95% interval is a percentile bootstrap that resamples **documents** with replacement: 2,000 draws by default, `random.Random(seed)`, documents in sorted-id order, table-weighted mean per draw, lower 2.5% and upper 97.5% quantiles. Seed and draws are fixed in the registration.
- **Comparator adapters.** A reader that writes tables in visual left-to-right order may get a per-lane adapter (mirror the columns) fixed on DEV and pinned in the registration; the scorer itself never mirrors.

## Not here yet

These come with the 0032 preregistration:
- cross-page stitching accuracy;
- form measures (field/value F1, section-tree similarity).
