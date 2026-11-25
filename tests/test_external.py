"""Invented inputs covering the publisher formats, reader contracts and metrics."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pyarrow as pa
from pyarrow import parquet
import pytest

from heocr_bench.cli import main
from heocr_bench.external import load_predictions, normalized_levenshtein, score
from heocr_bench.external_data import load_units
from heocr_bench.external_readers import READERS, make_reader, predict


@pytest.fixture
def pinkas(tmp_path):
    root = tmp_path / "pinkas"
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    (root / "test_set.txt").write_text("pairs/test.jpg\n", encoding="utf-8-sig")
    (root / "train_set.txt").write_text("train.jpg\n", encoding="utf-8")
    (pairs / "train.xml").write_text("intentionally unreadable", encoding="utf-8")
    Image.new("RGB", (100, 80), "white").save(pairs / "test.jpg")
    (pairs / "test.xml").write_text("""<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageFilename="test.jpg">
        <ReadingOrder><OrderedGroup id="order">
          <RegionRefIndexed index="1" regionRef="left"/>
          <RegionRefIndexed index="0" regionRef="right"/>
        </OrderedGroup></ReadingOrder>
        <TextRegion id="left"><TextLine id="l1">
          <Word id="w1"><TextEquiv><Unicode>עץ</Unicode></TextEquiv></Word>
          <Word id="w2"><TextEquiv><Unicode>סגול</Unicode></TextEquiv></Word>
        </TextLine></TextRegion>
        <TextRegion id="right"><TextLine id="r1">
          <TextEquiv index="2"><Unicode>ignored alternative</Unicode></TextEquiv>
          <TextEquiv index="0"><Unicode>חלון־כחול</Unicode></TextEquiv>
        </TextLine></TextRegion>
      </Page>
    </PcGts>""", encoding="utf-8")
    return root


@pytest.fixture
def churro(tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (96, 80), "white").save(buffer, format="PNG")
    row = dict(image={"bytes": buffer.getvalue(), "path": "sample.png"},
               cleaned_transcription="אֶבֶן A", original_transcription="ignored original",
               dataset_id="fixture", main_language="Hebrew", main_script="Hebrew")
    rows = [
        dict(row, example_id="z", document_type="handwritten"),
        dict(row, example_id="a", document_type="printed", cleaned_transcription="עץ"),
        dict(row, example_id="script-mismatch", document_type="printed", main_script="Latin"),
        dict(row, example_id="language-mismatch", document_type="printed", main_language="Other"),
    ]
    path = tmp_path / "fixture.parquet"
    parquet.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def biblia(tmp_path):
    root = tmp_path / "biblia"
    root.mkdir()
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((5, 5, 85, 70), fill="black")
    image.save(root / "test.png")
    (root / "test.xml").write_text("""<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
      <Description><MeasurementUnit>pixel</MeasurementUnit>
        <sourceImageInformation><fileName>test.png</fileName></sourceImageInformation>
      </Description><Layout><Page WIDTH="100" HEIGHT="80"><PrintSpace><TextBlock>
        <TextLine ID="poly"><Shape><Polygon POINTS="10,10 70,15 70,25 10,29"/></Shape>
          <String CONTENT="אַב"/><SP/><String CONTENT="ג"/>
        </TextLine>
        <TextLine ID="rect" HPOS="10" VPOS="40" WIDTH="60" HEIGHT="25">
          <String CONTENT="ד ה"/>
        </TextLine>
        <TextLine ID="untranscribed"/>
      </TextBlock></PrintSpace></Page></Layout>
    </alto>""", encoding="utf-8")
    pages = tmp_path / "pages.txt"
    pages.write_text("test.xml\n", encoding="utf-8")
    return root, pages


def test_pinkas_split_order_and_fold(pinkas):
    units = list(load_units("pinkas", pinkas))
    assert [(x.id, x.reference) for x in units] == [("test", "חלון־כחול\nעץ סגול")]
    result = score(units, {"test": "חלון-כחול   עץ סגול"}, "pinkas")
    assert result["unit"] == "page"
    assert result["line_fold_v1"]["cer"] == 0
    assert result["publisher_metric"] == {"status": "not_applicable", "metric": "word_spotting_map"}
    with pytest.raises(ValueError, match="missing prediction"):
        score(units, {}, "pinkas")
    with pytest.raises(ValueError, match="unexpected prediction"):
        score(units, {"test": "", "train": "extra"}, "pinkas")


def test_churro_page_filter_and_publisher_normalization(churro):
    units = list(load_units("churro", churro))
    assert [x.id for x in units] == ["fixture/z", "fixture/a"]
    assert all(Image.open(BytesIO(x.image)).size == (96, 80) for x in units)
    scoring_only = list(load_units("churro", churro, with_images=False))
    assert all(x.image is None for x in scoring_only)
    scores = score(scoring_only, {"fixture/z": "אֶבֶן a", "fixture/a": "עץx"}, "churro")
    official = scores["publisher_metric"]
    assert official["by_document_type"]["handwritten"] == {"units": 1, "value": 1}
    assert official["by_document_type"]["printed"]["value"] == pytest.approx(2 / 3)
    assert official["value"] == pytest.approx(5 / 6)
    assert scores["line_fold_v1"]["edits"] == 2
    wrapped = "<output><HistoricalDocument><Page><Body>—אב <Gap>ה</Gap>ג</Body></Page></HistoricalDocument></output>"
    assert normalized_levenshtein(wrapped, "אב ג") == 1
    assert normalized_levenshtein("", "") == 1
    assert normalized_levenshtein("א", "") == 0


def test_biblia_line_rasters_and_metrics(biblia):
    root, pages = biblia
    units = list(load_units("biblia", root, pages))
    assert [(x.id, x.reference) for x in units] == [("test.xml#poly", "אַב ג"), ("test.xml#rect", "ד ה")]
    rasters = []

    def read(path):
        with Image.open(path) as image:
            rasters.append(image.copy())
        return "OCR only"

    assert list(predict(units, read)) == [{"id": unit.id, "text": "OCR only"} for unit in units]
    assert [raster.size for raster in rasters] == [(60, 19), (60, 25)]
    assert rasters[0].getpixel((59, 0)) == (255, 255, 255)
    assert rasters[0].getpixel((30, 9)) == rasters[1].getpixel((30, 9)) == (0, 0, 0)
    result = score(units, {"test.xml#poly": "אב ג", "test.xml#rect": "ד  ה"}, "biblia")
    assert result["unit"] == "line"
    assert result["publisher_metric"]["value"] == pytest.approx(7 / 8)
    assert result["publisher_metric"]["wer"] == pytest.approx(1 / 4)


def test_biblia_fails_closed_on_missing_scope_and_remote_carrier(biblia):
    root, pages = biblia
    with pytest.raises(ValueError, match="requires --pages"):
        list(load_units("biblia", root))
    annotation = root / "test.xml"
    annotation.write_text(annotation.read_text().replace("test.png", "https://example.invalid/test.png"))
    with pytest.raises(ValueError, match="no local carrier"):
        list(load_units("biblia", root, pages))


def write_local_settings(tmp_path, lane):
    assets = tmp_path / "assets"
    assets.mkdir()
    model = b"an invented mock model"
    (assets / "stub.traineddata").write_bytes(model)
    settings = {"assets": [{"path": "stub.traineddata", "sha256": sha256(model).hexdigest()}]}
    if lane == "paddle":
        settings["options"] = dict(provider="CPUExecutionProvider", pipeline="word",
                                   order="RTL", rec_mode="cascade", det_thresh=0.4)
    else:
        settings["argv"] = ["-l", "stub", "--oem", "1", "--psm", "3"]
    spec = tmp_path / "settings.json"
    spec.write_text(json.dumps({lane: settings}))
    return assets, spec


@pytest.mark.parametrize("lane", READERS)
def test_reader_lanes_are_local_and_use_line_mode(tmp_path, monkeypatch, lane):
    assets, spec = write_local_settings(tmp_path, lane)
    path = tmp_path / "line.png"
    stub = "טקסט"
    if lane.startswith("heocr"):
        def extract(source, **options):
            assert source == path
            assert options == {
                "ocr_psm": 7, "convert": False, "tesseract": "/local/tesseract",
                **({"ocr_router": "v2"} if lane.endswith("v2") else {}),
            }
            return SimpleNamespace(text=stub, flags=[], pages=[SimpleNamespace(route="ocr", flags=[])])
        monkeypatch.setitem(sys.modules, "heocr", SimpleNamespace(extract=extract))
    elif lane == "paddle":
        def word(model_path, **options):
            assert model_path == assets / "paddle"
            assert options == {"providers": ["CPUExecutionProvider"], "det_thresh": 0.4}

            def read(raster, *, rec_mode):
                assert raster == "bgr" and rec_mode == "cascade"
                return {"lines": [{"text": stub}]}
            return SimpleNamespace(read_array=read)
        monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
            imread=lambda filename, mode: "bgr", IMREAD_COLOR=1))
        monkeypatch.setitem(sys.modules, "paddleocr_hebrew",
                            SimpleNamespace(HebrewOCR=SimpleNamespace(word=word)))
    else:
        def run(argv, **kwargs):
            assert argv == ["/local/tesseract", str(path), "stdout", "--tessdata-dir",
                            str(assets), "-l", "stub", "--oem", "1", "--psm", "7"]
            assert kwargs["timeout"] == 180 and kwargs["check"]
            return subprocess.CompletedProcess(argv, 0, stdout=stub.encode("utf-8"))
        monkeypatch.setattr(subprocess, "run", run)
    args = SimpleNamespace(reader=lane, set_id="biblia", assets=assets,
                           reader_spec=spec, heocr_options=None, tesseract="/local/tesseract")
    assert make_reader(args)(path) == stub
    if not lane.startswith("heocr"):
        (assets / "stub.traineddata").write_text("model changed")
        with pytest.raises(ValueError, match="asset SHA-256 mismatch"):
            make_reader(args)


def test_cli_prediction_and_score_are_separate(tmp_path, pinkas, monkeypatch, capsys):
    import heocr_bench.external

    def reader(path):
        assert isinstance(path, Path) and path.suffix == ".jpg"
        return "חלון-כחול עץ סגול"

    monkeypatch.setattr(heocr_bench.external, "make_reader", lambda args: reader)
    common = ["--set", "pinkas", "--data", str(pinkas)]
    main(["external", "predict", *common, "--reader", "heocr"])
    raw = capsys.readouterr().out
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(raw, encoding="utf-8")
    main(["external", "score", *common, "--preds", str(predictions)])
    assert json.loads(capsys.readouterr().out)["line_fold_v1"]["cer"] == 0
    predictions.write_text(raw * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prediction"):
        load_predictions(predictions)


def test_heocr_options_exclude_hosted_routes(tmp_path, monkeypatch):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"ocr_hybrid": "haiku"}))
    monkeypatch.setitem(sys.modules, "heocr", SimpleNamespace())
    args = SimpleNamespace(reader="heocr", set_id="pinkas", heocr_options=options,
                           tesseract="/local/tesseract")
    with pytest.raises(ValueError, match="only local reader options"):
        make_reader(args)
