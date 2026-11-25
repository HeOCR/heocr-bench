# Pinkas v1

Use the publisher's Hebrew manuscript JPG/PAGE XML pairs. Extract the
authorized archive locally and point `--data` at its root. The adapter
uses only the entries in the publisher's `test_set.txt`, in listed order.
It pairs each listed stem to its PAGE annotation and carrier. Transcription
uses PAGE region reading order when present and XML line/word order,
preferring line text to word alternatives.

```sh
heocr-bench external predict --set pinkas --data /local/pinkas \
  --reader heocr > /private-run/pinkas-heocr.jsonl
heocr-bench external score --set pinkas --data /local/pinkas \
  --preds /private-run/pinkas-heocr.jsonl --text-format heocr-cell-delimited
```

The page transcript metric is pooled `line_fold_v1` CER. The authors'
published mean average precision (mAP) is for query-by-example word
spotting. The harness marks it inapplicable to these transcription lanes
because they do not emit ranked word-spotting hits.

Attribute Berat Kurar, Irina Rabaev and Jihad El-Sana. [Publisher deposit
(CC BY 4.0)](https://zenodo.org/records/3569694) · [Paper, sections V–VI](https://beratkurar.github.io/papers/icdar2019_pinkas.pdf)
