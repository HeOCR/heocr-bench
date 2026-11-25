"""Run third-party table extractors on PDF pages and write raw heocr_table.v1 predictions.

One lane per comparator in decision 0032. Each lane runs the tool as shipped, in its own
environment (see ``envs/tables/``), and writes the tables exactly as the tool emits them:
columns in the tool's order, cell text as the tool returns it. Column-order and text-order
repairs are applied at scoring time only, by the adapter pinned for that lane
(``heocr_bench.table_adapters``).

    python -m heocr_bench.table_lanes run --lane LANE --manifest PAGES.jsonl --out DIR
    python -m heocr_bench.table_lanes lanes

``PAGES.jsonl`` has one page per line: ``{"doc_id": ..., "pdf": PATH, "page": N}`` (page from 1;
a relative PATH is read against the manifest's folder). The lane writes ``DIR/LANE/<doc_id>.json``
(a list of tables, or ``{"error": ...}`` when the tool failed on that page) and
``DIR/LANE/run.json`` (tool versions, model revisions, an environment hash, timings).

Importing this module needs only the standard library. Each lane imports its tool lazily.
"""

import argparse
from hashlib import file_digest, sha256
from html.parser import HTMLParser
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

DPI = 200


# ---------------------------------------------------------------- helpers shared by lanes

def table(rows, cols, cells, page, header_rows=0):
    return {"schema": "heocr_table.v1", "rows": rows, "cols": cols, "header_rows": header_rows,
            "pages": [page, page], "cells": cells}


def from_grid(rows, page):
    """A plain list of rows (no spans) as one table; ``None`` cells become ``""``."""
    cols = max((len(r) for r in rows), default=0)
    cells = [{"r": i, "c": j, "text": "" if v is None else str(v)} for i, r in enumerate(rows) for j, v in enumerate(r)]
    return table(len(rows), cols, cells, page)


class _TableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self._rows, self._cell, self._depth = [], None, None, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._rows = []
        elif self._depth == 1 and tag == "tr":
            self._rows.append([])
        elif self._depth == 1 and tag in ("td", "th"):
            if not self._rows:
                self._rows.append([])

            def span(k):
                try:
                    return max(1, int(str(a.get(k) or "1").strip()))
                except ValueError:
                    return 1
            self._cell = {"rowspan": span("rowspan"), "colspan": span("colspan"), "th": tag == "th", "text": []}
        elif self._cell is not None and tag == "br":
            self._cell["text"].append("\n")

    def handle_endtag(self, tag):
        if tag == "table":
            if self._depth == 1 and self._rows is not None:
                self.tables.append(self._rows)
                self._rows = None
            self._depth = max(0, self._depth - 1)
        elif self._depth == 1 and tag in ("td", "th") and self._cell is not None:
            self._cell["text"] = " ".join("".join(self._cell["text"]).split(" ")).strip()
            self._rows[-1].append(self._cell)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"].append(data)


def from_html(html, page):
    """Every top-level ``<table>`` in ``html`` as heocr_table.v1, spans placed on an occupancy grid
    in the HTML's own column order (left to right)."""
    p = _TableHTML()
    p.feed(html or "")
    p.close()
    out = []
    for rows in p.tables:
        taken, cells, ncols = set(), [], 0
        for r, row in enumerate(rows):
            c = 0
            for cell in row:
                while (r, c) in taken:
                    c += 1
                rs, cs = cell["rowspan"], cell["colspan"]
                for dr in range(rs):
                    for dc in range(cs):
                        taken.add((r + dr, c + dc))
                cells.append({"r": r, "c": c, "rowspan": rs, "colspan": cs, "text": cell["text"]})
                c += cs
            ncols = max([ncols] + [cc + 1 for rr, cc in taken if rr == r])
        nrows = len(rows)
        # Clip spans the HTML declared past the table's last row.
        for cell in cells:
            cell["rowspan"] = min(cell["rowspan"], nrows - cell["r"])
        header = 0
        for r, row in enumerate(rows):
            if row and all(x["th"] for x in row):
                header = r + 1
            else:
                break
        if cells:
            out.append(table(nrows, ncols, cells, page, header))
    return out


def one_page_pdf(pdf, page, folder):
    import pypdfium2 as pdfium
    src = pdfium.PdfDocument(str(pdf))
    dst = pdfium.PdfDocument.new()
    dst.import_pages(src, [page - 1])
    out = Path(folder) / f"{Path(pdf).stem}_p{page}.pdf"
    dst.save(str(out))
    return out


def render(pdf, page, dpi=DPI):
    import pypdfium2 as pdfium
    return pdfium.PdfDocument(str(pdf))[page - 1].render(scale=dpi / 72).to_pil().convert("RGB")


def hf_revisions(*repos):
    """For each Hugging Face repo a lane loaded: the snapshot revision in the local hub cache and the
    SHA-256 of every file in it."""
    from huggingface_hub.constants import HF_HUB_CACHE
    out = {}
    for repo in repos:
        root = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
        ref = root / "refs" / "main"
        snaps = sorted(p.name for p in (root / "snapshots").glob("*")) if root.exists() else []
        rev = ref.read_text().strip() if ref.exists() else (snaps[-1] if len(snaps) == 1 else None)
        out[repo] = {"revision": rev, "snapshots": snaps,
                     "files": hash_tree(root / "snapshots" / rev) if rev else {}}
    return out


def hash_tree(root):
    """SHA-256 of every file under ``root`` (symlinks followed), keyed by relative path."""
    out = {}
    root = Path(root)
    for f in sorted(root.rglob("*")) if root.exists() else []:
        if f.is_file() and ".locks" not in f.parts:
            with f.open("rb") as s:
                out[str(f.relative_to(root))] = file_digest(s, "sha256").hexdigest()
    return out


def environment():
    """The interpreter and every installed distribution, each with the SHA-256 of its RECORD file
    (RECORD lists the hash of every file the wheel installed)."""
    dists = {}
    for d in metadata.distributions():
        name = d.metadata["Name"]
        if not name:
            continue
        record = d.read_text("RECORD")
        dists[f"{name.lower()}=={d.version}"] = sha256(record.encode()).hexdigest() if record else None
    listing = "\n".join(f"{k} {v}" for k, v in sorted(dists.items()))
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "env_sha256": sha256(listing.encode()).hexdigest(), "distributions": dict(sorted(dists.items()))}


def versions(*names):
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


# ---------------------------------------------------------------- lanes
# Each lane factory returns (run(pdf, page, folder) -> list[table], info() -> dict).

def lane_docling(ocr=False):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    if ocr:
        opts.ocr_options = TesseractCliOcrOptions(lang=["heb"], force_full_page_ocr=True)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    def run(pdf, page, folder):
        doc = conv.convert(str(one_page_pdf(pdf, page, folder))).document
        out = []
        for t in doc.tables:
            d, cells, seen = t.data, [], set()
            for c in d.table_cells:
                key = (c.start_row_offset_idx, c.start_col_offset_idx)
                if key in seen:
                    continue
                seen.add(key)
                cells.append({"r": c.start_row_offset_idx, "c": c.start_col_offset_idx,
                              "rowspan": max(1, c.end_row_offset_idx - c.start_row_offset_idx),
                              "colspan": max(1, c.end_col_offset_idx - c.start_col_offset_idx), "text": c.text})
            hdr = 0
            for r in range(d.num_rows):
                row = [c for c in d.table_cells if c.start_row_offset_idx == r]
                if row and all(c.column_header for c in row):
                    hdr = r + 1
                else:
                    break
            out.append(table(d.num_rows, d.num_cols, cells, page, hdr))
        return out

    def info():
        i = {"versions": versions("docling", "docling-ibm-models", "docling-parse"),
             "models": hf_revisions("docling-project/docling-models", "docling-project/docling-layout-heron"),
             "options": {"tableformer_mode": "accurate", "ocr": "tesseract-cli heb, full page" if ocr else None}}
        if ocr:
            i["tesseract"] = _tesseract_version()
        return i
    return run, info


def lane_pdfplumber():
    import pdfplumber

    def run(pdf, page, folder):
        with pdfplumber.open(str(pdf)) as doc:
            return [from_grid(rows, page) for rows in doc.pages[page - 1].extract_tables() if rows]
    return run, lambda: {"versions": versions("pdfplumber", "pdfminer.six"), "options": {"strategy": "lines"}}


def lane_camelot(flavor):
    import camelot

    def run(pdf, page, folder):
        return [from_grid(t.df.values.tolist(), page) for t in camelot.read_pdf(str(pdf), pages=str(page), flavor=flavor)]
    return run, lambda: {"versions": versions("camelot-py", "pdfminer.six", "pypdfium2"), "options": {"flavor": flavor}}


def lane_img2table():
    from img2table.document import Image
    from img2table.ocr import TesseractOCR
    ocr = TesseractOCR(lang="heb", n_threads=2)

    def run(pdf, page, folder):
        png = Path(folder) / "page.png"
        render(pdf, page).save(png)
        out = []
        for t in Image(str(png)).extract_tables(ocr=ocr, borderless_tables=True, implicit_rows=True):
            rows = list(t.content.values())
            seen, cells = set(), []
            for i, r in enumerate(rows):  # img2table repeats a spanning cell object at every covered position
                for j, cell in enumerate(r):
                    if cell is None or id(cell) in seen:
                        continue
                    seen.add(id(cell))
                    rs = sum(1 for ii in range(i, len(rows)) if any(x is cell for x in rows[ii]))
                    cs = sum(1 for x in r if x is cell)
                    cells.append({"r": i, "c": j, "rowspan": rs, "colspan": cs, "text": cell.value or ""})
            out.append(table(len(rows), max((len(r) for r in rows), default=0), cells, page))
        return out
    return run, lambda: {"versions": versions("img2table", "opencv-python", "opencv-contrib-python"),
                         "tesseract": _tesseract_version(),
                         "options": {"dpi": DPI, "ocr": "tesseract heb", "borderless_tables": True}}


TATR_DETECT = "microsoft/table-transformer-detection"
TATR_STRUCT = "microsoft/table-transformer-structure-recognition-v1.1-all"
TATR_EDGE = {False: 800, True: 1000}


def lane_tatr():
    """Table Transformer: detection, then structure recognition on each crop. TATR predicts structure
    only; each cell's text is the PDF text layer inside it (pdfplumber words), or Tesseract heb on the
    cell crop when the page has no text layer."""
    import pdfplumber
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    det_p = AutoImageProcessor.from_pretrained(TATR_DETECT)
    det = TableTransformerForObjectDetection.from_pretrained(TATR_DETECT).eval()
    st_p = AutoImageProcessor.from_pretrained(TATR_STRUCT)
    st = TableTransformerForObjectDetection.from_pretrained(TATR_STRUCT).eval()

    def boxes(model, proc, img, threshold):
        edge = TATR_EDGE[model is st]  # the TATR inference sizes: longest side 800 (detection), 1000 (structure)
        with torch.no_grad():
            out = model(**proc(images=img, size={"shortest_edge": edge, "longest_edge": edge}, return_tensors="pt"))
        res = proc.post_process_object_detection(out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
        labels = model.config.id2label
        return [(labels[int(l)], [float(v) for v in b], float(s))
                for l, b, s in zip(res["labels"], res["boxes"], res["scores"])]

    def run(pdf, page, folder):
        img = render(pdf, page)
        scale = DPI / 72
        with pdfplumber.open(str(pdf)) as doc:
            pg = doc.pages[page - 1]
            words = pg.extract_words(use_text_flow=False, keep_blank_chars=False)
        out = []
        for label, (x0, y0, x1, y1), _ in sorted(boxes(det, det_p, img, 0.5), key=lambda b: (b[1][1], b[1][0])):
            if label != "table":
                continue
            pad = 10
            ox, oy = max(0, x0 - pad), max(0, y0 - pad)
            crop = img.crop((ox, oy, min(img.width, x1 + pad), min(img.height, y1 + pad)))
            found = boxes(st, st_p, crop, 0.5)
            rows = sorted([b for l, b, _ in found if l == "table row"], key=lambda b: b[1])
            cols = sorted([b for l, b, _ in found if l == "table column"], key=lambda b: b[0])
            if not rows or not cols:
                continue
            spans = [b for l, b, _ in found if l == "table spanning cell"]
            heads = [b for l, b, _ in found if l == "table column header"]

            def to_pdf(b):  # crop pixels -> PDF points
                return ((b[0] + ox) / scale, (b[1] + oy) / scale, (b[2] + ox) / scale, (b[3] + oy) / scale)

            def text_in(b):
                bx0, by0, bx1, by1 = to_pdf(b)
                ws = [w for w in words if bx0 <= (w["x0"] + w["x1"]) / 2 <= bx1 and by0 <= (w["top"] + w["bottom"]) / 2 <= by1]
                if words:
                    ws.sort(key=lambda w: (round(w["top"]), w["x0"]))
                    return " ".join(w["text"] for w in ws)
                return _tesseract_crop(crop.crop(tuple(int(v) for v in b)), folder)

            def idx(lo, hi, bands, a, z):
                return [k for k, band in enumerate(bands) if min(hi, band[z]) - max(lo, band[a]) > 0.5 * (band[z] - band[a])]

            taken, cells = set(), []
            for s in spans:
                rr = idx(s[1], s[3], rows, 1, 3)
                cc = idx(s[0], s[2], cols, 0, 2)
                if not rr or not cc or any((r, c) in taken for r in rr for c in cc):
                    continue
                for r in rr:
                    for c in cc:
                        taken.add((r, c))
                cells.append({"r": rr[0], "c": cc[0], "rowspan": len(rr), "colspan": len(cc),
                              "text": text_in((cols[cc[0]][0], rows[rr[0]][1], cols[cc[-1]][2], rows[rr[-1]][3]))})
            for r, rb in enumerate(rows):
                for c, cb in enumerate(cols):
                    if (r, c) not in taken:
                        cells.append({"r": r, "c": c, "rowspan": 1, "colspan": 1,
                                      "text": text_in((cb[0], rb[1], cb[2], rb[3]))})
            header = sum(1 for rb in rows if any(min(rb[3], h[3]) - max(rb[1], h[1]) > 0.5 * (rb[3] - rb[1]) for h in heads))
            cells.sort(key=lambda x: (x["r"], x["c"]))
            out.append(table(len(rows), len(cols), cells, page, header))
        return out
    return run, lambda: {"versions": versions("transformers", "torch", "timm", "pdfplumber"),
                         "models": hf_revisions(TATR_DETECT, TATR_STRUCT),
                         "options": {"dpi": DPI, "longest_edge": {"detection": 800, "structure": 1000}, "detect_threshold": 0.5, "structure_threshold": 0.5, "crop_pad_px": 10, "defaults": "microsoft/table-transformer src/inference.py class thresholds and crop padding",
                                     "cell_text": "PDF text layer words; Tesseract heb when the page has none"},
                         "tesseract": _tesseract_version()}


def lane_ppstructure():
    """PP-StructureV3 on the page image, as shipped. PaddleOCR 3.7 ships no Hebrew recognition model, so
    its OCR is the release default; cell text comes from that OCR."""
    from paddleocr import PPStructureV3
    pipe = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)

    def run(pdf, page, folder):
        png = Path(folder) / "page.png"
        render(pdf, page).save(png)
        out = []
        for res in pipe.predict(str(png)):
            for t in res.json["res"].get("table_res_list", []):
                out.extend(from_html(t.get("pred_html", ""), page))
        return out
    return run, lambda: {"versions": versions("paddleocr", "paddlepaddle", "paddlex"), "models": _paddlex_models(),
                         "options": {"dpi": DPI, "pipeline": "PPStructureV3", "lang": "default (no Hebrew model shipped)",
                                     "use_doc_orientation_classify": False, "use_doc_unwarping": False}}


def lane_marker():
    """Marker (surya models) on the one-page PDF; tables from its JSON block tree."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.renderers.json import JSONRenderer  # noqa: F401 - pins the renderer import path
    conv = PdfConverter(artifact_dict=create_model_dict(), renderer="marker.renderers.json.JSONRenderer")

    def walk(block):
        if block.block_type == "Table":
            yield block.html
        for child in block.children or []:
            yield from walk(child)

    def run(pdf, page, folder):
        rendered = conv(str(one_page_pdf(pdf, page, folder)))
        out = []
        for child in rendered.children:
            for html in walk(child):
                out.extend(from_html(html, page))
        return out
    return run, lambda: {"versions": versions("marker-pdf", "surya-ocr", "torch"),
                         "models": hf_revisions("datalab-to/indra-ocr"), "options": {"renderer": "json", "defaults": True}}


def lane_mineru():
    """MinerU, local ``mineru-kit parse`` with the release defaults (tier and OCR mode chosen by MinerU);
    tables from the middle JSON's table blocks (their HTML)."""
    exe = Path(sys.executable).with_name("mineru-kit")

    def run(pdf, page, folder):
        one = one_page_pdf(pdf, page, folder)
        out_dir = Path(folder) / "mineru"
        r = subprocess.run([str(exe), "parse", str(one), "-o", str(out_dir), "-f", "middle_json"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        if r.returncode:
            raise RuntimeError(r.stdout.decode("utf-8", "replace")[-400:])
        out = []
        for f in sorted(out_dir.rglob("*.json")):
            out.extend(from_html("".join(_html_values(json.loads(f.read_text(encoding="utf-8")))), page))
        return out
    return run, lambda: {"versions": versions("mineru", "torch", "transformers"), "models": _mineru_models(),
                         "options": {"command": "mineru-kit parse -f middle_json", "tier": "release default",
                                     "ocr_mode": "auto (default)", "remote": False}}


def _html_values(node):
    """Every string in a MinerU JSON tree that holds a <table>, in document order."""
    if isinstance(node, dict):
        for v in node.values():
            yield from _html_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from _html_values(v)
    elif isinstance(node, str) and "<table" in node:
        yield node


LANES = {
    "docling": lambda: lane_docling(False),
    "docling_ocr": lambda: lane_docling(True),
    "pdfplumber": lane_pdfplumber,
    "camelot_lattice": lambda: lane_camelot("lattice"),
    "camelot_stream": lambda: lane_camelot("stream"),
    "img2table": lane_img2table,
    "tatr": lane_tatr,
    "ppstructure": lane_ppstructure,
    "marker": lane_marker,
    "mineru": lane_mineru,
}


def _tesseract_version():
    try:
        r = subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        return r.stdout.decode().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def _tesseract_crop(img, folder):
    p = Path(folder) / "cell.png"
    img.save(p)
    r = subprocess.run(["tesseract", str(p), "stdout", "-l", "heb", "--psm", "6"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    return " ".join(r.stdout.decode("utf-8", "replace").split())


def _paddlex_models():
    root = Path(os.environ.get("PADDLE_PDX_CACHE_HOME") or Path.home() / ".paddlex") / "official_models"
    return {"cache": str(root), "files": hash_tree(root)}


def _mineru_models():
    root = Path(os.environ.get("MINERU_HOME") or Path.home() / ".mineru") / "models"
    return {"cache": str(root), "files": hash_tree(root)}


# ---------------------------------------------------------------- runner

NO_TELEMETRY = {"HF_HUB_DISABLE_TELEMETRY": "1", "MINERU_TELEMETRY_DISABLED": "1", "DO_NOT_TRACK": "1",
                "SCARF_NO_ANALYTICS": "true", "DISABLE_TELEMETRY": "1"}


def run_lane(lane, manifest, out):
    os.environ.update(NO_TELEMETRY)
    pages = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    base = Path(manifest).resolve().parent
    out = Path(out) / lane
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fn, info = LANES[lane]()
    load_s = time.time() - started
    timings, errors = {}, {}
    for p in pages:
        pdf = Path(p["pdf"])
        pdf = pdf if pdf.is_absolute() else base / pdf
        t0 = time.time()
        with tempfile.TemporaryDirectory() as folder:
            try:
                preds = fn(pdf, int(p["page"]), folder)
            except Exception as exc:  # noqa: BLE001 - a tool failure on one page is a recorded result
                preds = {"error": repr(exc)[:500]}
                errors[p["doc_id"]] = preds["error"]
        timings[p["doc_id"]] = round(time.time() - t0, 2)
        (out / f"{p['doc_id']}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
        print(lane, p["doc_id"], len(preds) if isinstance(preds, list) else "error", flush=True)
    record = {"lane": lane, "manifest_sha256": sha256(Path(manifest).read_bytes()).hexdigest(), "pages": len(pages),
              "environment": environment(), "tool": info(), "load_seconds": round(load_s, 2),
              "page_seconds": timings, "errors": errors,
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
    (out / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m heocr_bench.table_lanes", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Run one lane on every page of a manifest")
    r.add_argument("--lane", choices=sorted(LANES), required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    sub.add_parser("lanes", help="List the lanes")
    args = parser.parse_args(argv)
    if args.command == "lanes":
        print("\n".join(sorted(LANES)))
        return 0
    rec = run_lane(args.lane, args.manifest, args.out)
    return 1 if len(rec["errors"]) == rec["pages"] else 0


if __name__ == "__main__":
    sys.exit(main())
