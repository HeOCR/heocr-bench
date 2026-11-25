# HeOCR Bench

HeOCR Bench scores OCR submissions against versioned reference sets. The
first registry entry is P-mod-open v0, a real held-out benchmark of modern
unpointed Hebrew print.

## Install and score

```sh
pip install git+https://github.com/HeOCR/heocr-bench.git
heocr-bench score --set p-mod-open-v0 --preds ./submission
```

Use Python 3.11 or later. The default registry fetches the public bundle's
manifest, references and masks from Hugging Face. Each file is checked
against a SHA-256 pin. For an offline copy of the same bytes, pass
`--data /path/to/p-mod-open-v0`. The CLI can work offline before the public
dataset is available if you already have the pinned bundle.

A submission directory contains exactly one UTF-8 `<carrier-id>.txt` per
carrier listed in the manifest, plus `metadata.json`. Use an empty file
for a blank OCR result. Write text in logical reading order, preserving
line breaks. For HeOCR outputs that use the literal ` | ` as a table cell
delimiter, set `text_format` to `heocr-cell-delimited`. The default is
`plain`.

```json
{
  "reader": "My OCR reader",
  "version": "1.0.0",
  "license": "Apache-2.0",
  "weights": "open",
  "params": {"lang": "he"},
  "text_format": "plain"
}
```

## Metrics

The headline is pooled **micro page CER** on the common filtered view.
Page CER uses character edits divided by reference characters, pooled over
carriers. The report also includes the all-reference view, word error rate,
character insertions, deletions and substitutions, and missed numeral
counts. Strata include archive, issuer render, office scanner, Phone A
(iPhone) and Phone B (Pixel). The registered issuer-scan stratum has zero
support. An unsupported stratum returns a null rate.

The registered `line_fold_v1` folds, NFC and bidi-control removal apply
before alignment. The filtered view uses the pinned common reference mask
and the registered full-page alignment and insertion attribution. Both
views keep Hebrew pointing and distinguish en dash from hyphen. A
supplementary all-reference line error rate treats each supplied line as
one sequence element and is sensitive to line breaks. Line CER needs
verified line joins, which v0 lacks. Filtered line error rate is unavailable
because the mask is projected at the character level.

## External Hebrew OCR

The [external benchmark harness](docs/external/README.md) reads native Pinkas
PAGE XML, CHURRO Hebrew Parquet and BiblIA ALTO. It runs local readers and emits
scores separately from the registered HeOCR benchmarks.

## Table metrics

`heocr_bench.tables` scores table structure: TEDS, TEDS-struct, GriTS-Con,
GriTS-Top and exact match, on the `heocr_table.v1` format (logical column
order, so right-to-left tables are never mirrored). See
[docs/tables.md](docs/tables.md) for the exact definitions. Form structure
(field/value F1 and section-tree similarity) is in `heocr_bench.forms`; see
[docs/forms.md](docs/forms.md). No table or form set is registered yet.

## Efficiency

The [v0 efficiency benchmark](efficiency/README.md) measures pages per minute,
peak RSS and cost on the public image panel separately from accuracy.

## Baseline replay

Place the released baseline submission directories directly under
`baseline/`, keeping their reader IDs as directory names. Then run:

```sh
heocr-bench replay --set p-mod-open-v0 --preds ./baseline --data ./p-mod-open-v0
```

The replay recomputes the frozen ten-reader table, including the paired
issue-cluster comparisons. The accompanying predictions record the scored
unmasked originals. Any release-only image masking is disclosed in the
dataset card. Code is Apache-2.0. Reader weights and dataset assets retain
their own licences.

The [set registry](src/heocr_bench/registry.json) pins the public split and
scoring protocol. The [hidden split plan](HIDDEN_SPLIT.md) describes a
possible v1 server-side split.
