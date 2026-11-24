# Table-structure metrics

Module: `heocr_bench.tables`. Status: scorer only. No set is registered yet; the tables and forms benchmark (decision 0032) will pin this scorer by commit before any held-out pass.

## Table format: `heocr_table.v1`

```json
{"rows": 3, "cols": 3, "header_rows": 1, "direction": "rtl",
 "cells": [{"r": 0, "c": 0, "rowspan": 1, "colspan": 2, "text": "..."}]}
```

- **Logical column order.** `c = 0` is the first column in reading order. In a right-to-left table that is the rightmost column on the page. The scorer never mirrors a table, so a prediction written in visual left-to-right order is scored as written and loses points. `direction` is informational.
- **Coverage.** Every grid position belongs to exactly one cell. Overlapping or out-of-grid cells are an error. Positions that no cell covers are filled with empty 1×1 cells.
- **Header rows.** `header_rows` counts the leading rows that form the column header (the `thead`).
- **Cell text normalisation.** The same as the page scorer (`metrics.normalize` with the registered `FOLDS`):
  - NFC;
  - bidi control characters removed;
  - maqaf → hyphen, geresh/gershayim and curly quotes → ASCII, em dash → en dash;
  - runs of whitespace (line breaks included) → one space, trimmed.

  Hebrew pointing is kept.

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

## Not here yet

These measures come with the 0032 preregistration:
- matching predicted tables to gold tables within a document;
- cross-page stitching accuracy;
- form measures (field/value F1, section-tree similarity).
