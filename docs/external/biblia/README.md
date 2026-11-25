# BiblIA v1.0

Extract the publisher's ALTO 4.2 annotations and authorized local images.
Pass `--pages` with one admitted XML path per line, relative to the
extracted root. The adapter resolves ALTO `fileName` locally and rejects
link-only pages. It walks transcribed `TextLine` entries in XML order,
concatenates `String CONTENT` and `SP` in logical order, and identifies
each line by its XML path and ALTO ID. It crops to the ALTO polygon using
white outside the shape, with a line rectangle when no polygon is supplied.
Every reader gets the same line pixels.

```sh
heocr-bench external predict --set biblia --data /local/biblia \
  --pages /private-run/admitted-alto.txt --reader tessdata-best \
  --reader-spec efficiency/readers.json --assets /local/reader-assets \
  > /private-run/biblia-best.jsonl
heocr-bench external score --set biblia --data /local/biblia \
  --pages /private-run/admitted-alto.txt --preds /private-run/biblia-best.jsonl
```

The authors report character accuracy, CER and WER for recognition. The
harness pools per-line edits on whitespace-normalized transcriptions for
these metrics, and reports pooled `line_fold_v1` CER separately. Scope
and crop policy are set by this track. The paper's results used the authors'
own split and Kraken recognition setup. See the
[Kraken 3.0.5 scoring implementation](https://github.com/mittagessen/kraken/blob/3.0.5/kraken/ketos.py#L528-L635).

The authors' deposit uses CC BY-NC-SA 4.0. [Publisher dataset and
attribution](https://zenodo.org/records/5167263) · [Paper, section 6](https://hal.science/hal-04937186/document)
