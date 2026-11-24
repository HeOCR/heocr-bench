# Form-structure metrics

Module: `heocr_bench.forms`. Status: scorer only. No set is registered yet; decision 0032's preregistration will pin this scorer by commit.

## Format: `hdoc_form.v1`

The gold schema heocr-eval approved on 2026-11-25.

- **form:** `form → section* → (field | table | group | section)`.
- **field:** `{id, label, value, kind, slot, pages}`.
  - `id` is stable in gold; predictions don't need it.
  - `kind` is one of text, number, date, checkbox, choice or signature.
  - Checkbox values are `checked` or `unchecked`. A choice value is the printed option text.
  - An unfilled field has the value `""`. A `null` value is an error.
- **group:** `{columns, children}`, layout only (side-by-side sections or fields). It is flattened into its section.
- **section:** `{title?, children, pages}`. A section may nest inside another section, but doesn't have to.
- **Tables inside a form** are scored by the table metrics (`docs/tables.md`), not here.

## Normalisation

- **Value:** the cell normaliser (`tables.cell_text`, which equals `line_fold_v1`). A multi-line value becomes one line joined by spaces. Numbers and dates are compared as strings; nothing is parsed in v1.
- **Label:** the same, and then one trailing `:`, `-` or `–` is removed, together with the space before it.

## Field/value F1

- **Matching** is one-to-one within each form, by a Hungarian assignment on a lexicographic key: first the number of exact pairs (normalised label and value both exact), then the summed label similarity (`2·LCS / (|a| + |b|)`). The weight of a pair is `W·[exact] + similarity`, with `W = (gold fields in the form) + 1`. Because `W` exceeds any possible similarity total, no set of similar-label pairs can outweigh an exact pair, and the matched count is the maximum number of exact pairs.
- **A match counts** only when the normalised label and value are both exact.
- **Precision, recall and F1** use matched, gold and predicted counts summed over all documents:
  - precision = matched / predicted;
  - recall = matched / gold;
  - F1 = 2·matched / (gold + predicted).
- **An unmatched gold form** is scored against an empty prediction.

## Section-tree similarity

- **Tree:** `form → section → (section | field)`.
  - A section node carries its normalised title. A field node carries its normalised label; values are excluded.
  - Groups are flattened, and tables are left out.
- **Score:** TEDS, `1 − TED / max(|T_gold|, |T_pred|)`. The TED is the same exact Zhang–Shasha implementation as for tables.
  - Insertion and deletion cost 1.
  - Substitution costs 1 between node types, and the normalised Levenshtein distance between same-type texts (0 if both are empty).
- **Pooling:** the mean over gold forms.

## Pooling and intervals

`pool_forms` reports the point values above and a 95% percentile bootstrap that resamples documents: 2,000 draws by default, `random.Random(seed)`, documents in sorted-id order. The interval is descriptive. The launch test is the paired one-sided bound that the registration defines (0029/0032).

Generated forms (hdocsyngen) are for DEV and training only and are never scored as results.
