# CHURRO-DS Hebrew

Use a local Parquet file already restricted to the authorized upstream test
split. Native rows have no split column. The adapter keeps source order,
filters `main_language == 'Hebrew'` and `main_script == 'Hebrew'`,
and joins predictions by `dataset_id/example_id`. It reads the image
struct's encoded bytes and uses `cleaned_transcription` as the reference.

```sh
heocr-bench external predict --set churro --data /local/churro-hebrew-test.parquet \
  --reader rivok-full --reader-spec efficiency/readers.json --assets /local/reader-assets \
  > /private-run/churro-rivok-full.jsonl
heocr-bench external score --set churro --data /local/churro-hebrew-test.parquet \
  --preds /private-run/churro-rivok-full.jsonl
```

The publisher's normalized Levenshtein similarity first applies the published
Hebrew normalization. Each page contributes `1 − distance / max(lengths)`.
The harness reports the mean per native document type and across the admitted
Hebrew pages, alongside pooled `line_fold_v1` CER. The normalization code
is adapted from the [Apache-2.0 publisher scorer](https://github.com/stanford-oval/Churro/blob/23c3ac65781443b18d87c8cdac5f77445c3319e6/tooling/evaluation/normalization.py).
The Hebrew selection overlaps Pinkas and BiblIA source records. Treat the
results as separate sets.

The dataset is for research use under its publisher terms and retains each
source's conditions. [Dataset card](https://huggingface.co/datasets/stanford-oval/churro-dataset/blob/cbeaba69179d5de122fbbc9e97b2a7d97942f688/README.md) · [Paper, section 4.2](https://arxiv.org/html/2509.19768#S4.SS2)
