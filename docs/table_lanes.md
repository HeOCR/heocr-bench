# Table comparator lanes (0032)

`heocr_bench.table_lanes` runs each third-party table extractor in decision 0032 on PDF pages and
writes raw `heocr_table.v1` predictions plus a run receipt. Score the output with
`heocr_bench.tables` after applying the lane's registered adapter (`heocr_bench.table_adapters`).

```bash
python -m heocr_bench.table_lanes lanes
python -m heocr_bench.table_lanes run --lane tatr --manifest pages.jsonl --out preds/
```

`pages.jsonl` has one page per line: `{"doc_id": "...", "pdf": "path/to/file.pdf", "page": 3}`. A
relative path is read against the manifest's folder.

The run writes `preds/<lane>/<doc_id>.json`: a list of tables, or `{"error": "..."}` when the tool
failed on that page. A failure is recorded and the run continues.

## Lanes

Every lane runs the tool with its documented defaults. Lanes write columns in the tool's own order
and text as the tool returns it. Column and text order are fixed at scoring time, by the one adapter
registered for the lane.

| Lane | Tool, pinned | Input | Notes |
|---|---|---|---|
| `docling` | docling 2.152.0, docling-ibm-models 4.0.4 | one-page PDF | TableFormer accurate, PDF text layer, no OCR |
| `docling_ocr` | as above + Tesseract 5.5.3 `heb` | one-page PDF | full-page OCR forced (context lane) |
| `tatr` | transformers 5.24.1, timm 1.0.31 | 200 dpi image | see below |
| `ppstructure` | paddleocr 3.7.0, paddlepaddle 3.3.1 (paddlex 3.7.2) | 200 dpi image | see below |
| `marker` | marker-pdf 2.2.0 | one-page PDF | JSON renderer; tables are `Table` blocks' HTML |
| `mineru` | mineru 4.2.3 | one-page PDF | see below |
| `camelot_lattice`, `camelot_stream` | camelot-py 2.2.0 | PDF page | one flavour is registered per comparator |
| `pdfplumber` | pdfplumber 0.11.11 | PDF page | `extract_tables()` defaults (lines) |
| `img2table` | img2table 2.0.0 + Tesseract `heb` | 200 dpi image | borderless tables on (context lane) |

Lane details:

- **`tatr`** runs `microsoft/table-transformer-detection`, then
  `microsoft/table-transformer-structure-recognition-v1.1-all` on each crop. It uses the
  thresholds, padding and resize sizes of Microsoft's `src/inference.py`: 0.5, 10 px, and a longest
  side of 800 / 1000. TATR predicts structure only. A cell's text is the PDF text-layer words inside
  it, or Tesseract `heb` on the cell crop when the page has no text layer.
- **`ppstructure`** runs `PPStructureV3` with orientation and unwarping off, and reads each table's
  `pred_html`. PaddleOCR 3.7 ships no Hebrew recognition model, so cell text comes from the release's
  default OCR.
- **`mineru`** runs `mineru-kit parse -f middle_json` locally, never with `--remote`. It uses the
  release's default tier and OCR mode, and reads the table HTML in the middle JSON. For 4.2.3
  that default is the `hybrid/VLM` backend with `MinerU2.5-Pro-2605-1.2B`; the receipt records both by name.

The harness sets `HF_HUB_DISABLE_TELEMETRY`, `MINERU_TELEMETRY_DISABLED` and `DO_NOT_TRACK`.

## Environments

Each tool gets its own venv. The exact wheels used for the DEV run, with their PyPI SHA-256, are in
[`envs/tables/`](../envs/tables). They were built on macOS arm64 with CPython 3.11.15.

```bash
uv venv -p 3.11 ENV && VIRTUAL_ENV=ENV uv pip install --require-hashes --no-deps -r envs/tables/tatr.txt
VIRTUAL_ENV=ENV uv pip install --no-deps -e .   # this harness
```

`docling.txt` covers the `docling`, `docling_ocr`, `pdfplumber`, `camelot_*` and `img2table` lanes.

Model caches: the Hugging Face hub cache holds the TATR, Docling, Marker (`datalab-to/indra-ocr`) and
PaddleX downloads. Set `PADDLE_PDX_CACHE_HOME` for PaddleX's model folder and `MINERU_HOME` for
MinerU's. The receipt hashes whatever was loaded.

## Receipt (`run.json`)

- the manifest's SHA-256;
- the Python version and platform;
- every installed distribution with the SHA-256 of its `RECORD` file, plus one `env_sha256` over that
  list;
- tool versions, settings, and model revisions with the SHA-256 of every weight file;
- load time and per-page time;
- per-page errors.
