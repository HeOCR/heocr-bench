# External Hebrew OCR harness

The adapters load caller-supplied local inputs. Dataset assets, references
and predictions retain their original terms. Record the authorized scope,
input hashes, exact runtime and reader settings before a benchmark run.
This harness does not fetch datasets, contact OCR APIs or publish scores.

- [Pinkas](pinkas/README.md) uses publisher test pages and PAGE XML.
- [CHURRO-DS](churro/README.md) uses Hebrew page rows in native Parquet.
- [BiblIA](biblia/README.md) uses ALTO line crops from a supplied page list.

Install with Python 3.11 or later from a checkout.

```sh
python -m pip install '.[external]'
heocr-bench external predict --help
heocr-bench external score --help
```

## Local readers

Select one reader per invocation. `heocr` uses the HeOCR package with its
default model. `heocr-router-v2` opts into the capture router. Install the
registered HeOCR build separately and point `HEOCR_MODELS` at its local
weights. `--heocr-options` accepts a JSON object to pin settings such as
`ocr_model` (including a local traineddata path), `ocr_capture_type`,
`ocr_line_v6`, `ocr_layout`, `ocr_old_print_model` or a `crnn_model`
and `crnn_sha256`. The reader rejects hosted options. Use
`--tesseract /path/to/tesseract` for the registered binary.

The comparator IDs are `rivok-full`, `rivok-lite`, `paddle`,
`tessdata-best` and `tessdata-fast`. Pass
`--reader-spec efficiency/readers.json --assets /local/asset-root`.
Place the assets at the paths in the manifest under that root. The run
verifies every selected SHA-256 before inference. Install the pinned Paddle
wrapper and CPU dependencies from
[efficiency/requirements-paddle.txt](../../efficiency/requirements-paddle.txt).
Paddle uses the flagship word detector and cascade recognizer. Tesseract
uses OEM 1 with PSM 3 on pages and PSM 7 on BiblIA lines. The HeOCR lanes
receive the corresponding PSM through the package API.

## Prediction and scoring

Capture predictions as UTF-8 JSONL of `{"id":"…","text":"…"}`. A blank
read is an empty string. The reader receives an image only. An engine
failure aborts the run, and scoring requires exact prediction coverage.
Keep predictions in the run's private output location. The scorer emits
aggregate counts and rates only, with rates as fractions. It uses the
shared `line_fold_v1` with logical order and pools edit counts and
reference characters per set. Page and line units are reported separately.
For HeOCR text output, pass `--text-format heocr-cell-delimited` to the
scorer so its structural table delimiter is treated as a space.

Temporary raster files use the process's standard temporary directory
(`TMPDIR` on macOS and Linux). Point it to the run's private scratch
when processing private inputs. Scope, overlap handling, model exposure and
any publication of results remain with the evaluator.
