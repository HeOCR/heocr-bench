from argparse import Namespace
from decimal import Decimal
from math import exp, log
from pathlib import Path
from statistics import mean, stdev
import json
import sys

import pytest

from test_registered_efficiency import EFFICIENCY, make_plan

from c1_protocol import PROTOCOL, cost_fields, paired_bounds, schedule, student_t3_critical
from panel import sha256
from registered_run import load_plan, measure, run


def test_c1_order_and_t_bounds():
    assert [[r["id"] for r in order] for order in schedule([{"id": i} for i in ("z", "b", "A")])] == [
        ["A", "b", "z"], ["z", "b", "A"], ["b", "z", "A"], ["A", "z", "b"],
    ]
    # Independent SciPy Student-t(3) reference for two eligible comparators.
    critical = 5.391949071934048
    assert student_t3_critical(.025 / 4) == pytest.approx(critical)
    speeds = [2.6, 2.8, 3.0, 3.1]
    expenses = [.25, .26, .29, .22]

    def arm(rates, costs, cash):
        return {"trials": [
            {"turn": i, "complete": True, "pages_per_minute": speed, "fully_loaded_usd_per_1000": price,
             "marginal_usd_per_1000": cash, "mandatory_usd_per_1000": 0}
            for i, (speed, price) in enumerate(zip(rates, costs, strict=True), 1)
        ]}

    results = {"candidate": arm(speeds, expenses, 0), "hosted": arm([1]*4, [1]*4, .9),
               "local": arm([1]*4, [1]*4, 0)}
    stats = paired_bounds("candidate", ["hosted", "local"], results)
    logs = list(map(log, speeds))
    assert stats["hosted"]["throughput_ratio_lower"] == pytest.approx(exp(mean(logs) - critical * stdev(logs) / 2))
    assert stats["hosted"]["cost_ratio_upper"] == pytest.approx(mean(expenses) + critical * stdev(expenses) / 2)
    assert stats["local"]["cost_ratio_upper"] is None
    assert stats["local"]["throughput_ratio_lower"] == stats["hosted"]["throughput_ratio_lower"]
    results["hosted"]["trials"][0]["complete"] = False
    assert all(value is None for value in paired_bounds("candidate", ["hosted"], results)["hosted"].values())
    results["hosted"] = arm([1]*4, [0]*4, 0)
    results["candidate"] = arm([2]*4, [1]*4, .5)
    stats = paired_bounds("candidate", ["hosted"], results)["hosted"]
    assert stats["throughput_ratio_lower"] == 2
    assert stats["cost_ratio_upper"] is None


def test_cost_separates_amortization_power_and_cost_unavailable():
    row = {"seconds": 360, "completed": ["p"] * 100, "api_usd": "0.02", "complete": True}
    rates = {"amortization_usd_hour": ".8", "power_usd_hour": ".1"}
    cost_fields([row], rates, Decimal(".30"))
    assert row["amortization_usd_per_1000"] == .8
    assert row["power_usd_per_1000"] == .1
    assert row["runner_usd_per_1000"] == .9
    assert row["marginal_usd_per_1000"] == .2
    assert row["fully_loaded_usd_per_1000"] == 1.4
    cost_fields([row], None, Decimal(".30"))
    assert row["runner_usd_per_1000"] is None and row["fully_loaded_usd_per_1000"] is None
    assert row["marginal_usd_per_1000"] == .2
    cost_fields([row], rates, None)
    assert row["runner_usd_per_1000"] == .9 and row["fully_loaded_usd_per_1000"] is None


def c1_plan(root):
    plan = make_plan(root, "m4-cpu")
    plan.pop("warmup")
    plan["protocol"] = PROTOCOL
    for reader in plan["readers"]:
        if reader["kind"] == "local":
            reader["concurrency"] = 2
    evidence = root / "fixture-cost-evidence.json"
    evidence.write_text('{"fixture":true}')
    pin = {"path": evidence.name, "sha256": sha256(evidence)}
    plan["m4_cost"] = {"acquisition_usd": "5760", "mean_watts": "80", "electricity_usd_per_kwh": ".20",
                       "evidence": {k: pin for k in ("acquisition", "power", "electricity")}}
    return plan


def test_c1_mode_telemetry_and_missing_lanes(tmp_path, monkeypatch):
    plan = c1_plan(tmp_path)
    incomplete = dict(plan["readers"][0], id="partial")
    script = (
        f"import sys; sys.path.insert(0, {str(EFFICIENCY)!r}); from telemetry import Telemetry; "
        "t=Telemetry(); t.complete(t.pages[0]['id']); sys.exit(1)"
    )
    incomplete["argv"] = [sys.executable, "-c", script]
    plan["readers"].append(incomplete)
    plan["comparators"].append("partial")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    out = tmp_path / "receipt"
    monkeypatch.setattr("registered_run.hardware", lambda _: {"cpu": "mock M4"})
    run(Namespace(plan=path, plan_sha256=sha256(path), output=out, execute=False, allow_api=False, timeout_seconds=20))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["protocol"] == PROTOCOL and receipt["accuracy_repeat"] == 1
    assert receipt["orders"] == [["api", "local", "partial"], ["partial", "local", "api"],
                                 ["local", "partial", "api"], ["api", "partial", "local"]]
    assert receipt["paired_design"]["alpha_per_bound"] == .025 / 4
    assert Decimal(receipt["runner_usd_hour"]) == Decimal("1.016")
    assert receipt["m4_client_cost"]["evidence_sha256"]["power"]
    assert len(list(out.glob("*.jsonl"))) == 12
    assert not list(out.glob("*.first"))
    assert all(row["warmup"] is None for row in receipt["readers"].values())
    for t in receipt["readers"]["api"]["trials"]:
        assert t["complete"] and t["successful_pages"] == 2
        assert all(item["usage"]["input_tokens"] == 100 for item in t["requests"])
    for t in receipt["readers"]["partial"]["trials"]:
        assert not t["complete"] and t["successful_pages"] == t["failed_pages"] == 1
    assert receipt["table"][0]["pages_per_minute_ci95"] is None
    assert receipt["table"][2]["pages_per_minute"] is None
    assert receipt["paired_efficiency_bounds"]["partial"]["throughput_ratio_lower"] is None


def test_layout_pair_requires_same_build_and_settings(tmp_path):
    plan = c1_plan(tmp_path)
    path = tmp_path / "plan.json"
    wheel = tmp_path / "heocr.whl"
    wheel.write_bytes(b"fixture wheel")
    off = dict(plan["readers"][0], mode="router-v2", ocr_layout="off",
               heocr_wheel={"path": wheel.name, "sha256": sha256(wheel)})
    on = dict(off, id="layout", ocr_layout="v1")
    plan["readers"] = [off, on, plan["readers"][1]]
    path.write_text(json.dumps(plan))
    loaded, _ = load_plan(path, sha256(path), False)
    assert loaded["m4_rates"]["amortization_usd_hour"] == "1"
    assert [r.get("ocr_layout") for r in loaded["readers"]] == ["off", "v1", None]
    on["argv"] = ["another-runtime"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="identical settings and wheel"):
        load_plan(path, sha256(path), False)


def test_only_first_authorized_repeat_persists_outputs(tmp_path):
    # Invented input passed directly to the measurement contract. No API or benchmark plan.
    page = {"id": "invented", "carrier": "scanner", "path": "unused"}
    script = (
        f"import sys; sys.path.insert(0, {str(EFFICIENCY)!r}); from telemetry import Telemetry; "
        "t=Telemetry(); t.save_first_output('invented', b''); t.complete('invented')"
    )
    reader = {"id": "local", "kind": "local", "concurrency": 2, "argv": [sys.executable, "-c", script]}
    args = (reader, [page], "trial")
    first = measure(*args, 1, tmp_path, tmp_path, None, False, 3, c1=True)
    second = measure(*args, 2, tmp_path, tmp_path, None, False, 3, c1=True)
    assert first["first_output_sha256"] == {"invented": sha256(tmp_path/"local.first"/"invented.ocr")}
    assert first["complete"] and second["complete"] and "first_output_sha256" not in second


def test_layout_adapter_applies_effective_options_in_manifest_order(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from registered_adapter import ordered_cpu
    from run import CAPTURES, MODEL

    pages = [{"id": carrier, "carrier": carrier, "path": str(tmp_path/carrier)}
             for carrier in ("issuer-render", "scanner", "phone-a", "archive")]
    settings = {"runtime": "heocr", "models": dict(heb_rivok="a", heocr_modern_v1="b", v6="c")}
    received = {}

    def extract(path, **options):
        capture = options["ocr_capture_type"]
        model = MODEL[capture]
        components = {f"tesseract:{model}.traineddata": "sha256:" + settings["models"][model]}
        if options["ocr_line_v6"] is not None:
            components["tesseract:line_v6.traineddata"] = "sha256:c"
        received[path.name] = options
        return SimpleNamespace(pages=[SimpleNamespace(route="ocr")], flags=[], components=components,
                               to_json=lambda **_: path.name)

    monkeypatch.setenv("HEOCR_MODELS", "fixture")
    monkeypatch.setitem(sys.modules, "heocr", SimpleNamespace(extract=extract))
    for layout in ("off", "v1"):
        completed, written = [], []
        telemetry = SimpleNamespace(job={"ocr_layout": layout, "concurrency": 2}, pages=pages,
                                    complete=completed.append,
                                    save_first_output=lambda key, text: written.append((key, text)))
        ordered_cpu(telemetry, tmp_path, settings)
        assert completed == [p["id"] for p in pages]
        assert written == [(p["id"], p["id"].encode()) for p in pages]
        assert all(received[p["id"]]["ocr_capture_type"] == CAPTURES[p["carrier"]] for p in pages)
        assert all(received[p["id"]].get("ocr_layout") == (None if layout == "off" else "v1") for p in pages)


def test_imported_package_must_match_layout_wheel(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from zipfile import ZipFile
    from registered_adapter import verify_heocr_wheel

    module = tmp_path / "heocr" / "__init__.py"
    module.parent.mkdir()
    module.write_bytes(b"# pinned")
    wheel = tmp_path / "fixture.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("heocr/__init__.py", b"# pinned")
    monkeypatch.setitem(sys.modules, "heocr", SimpleNamespace(__file__=str(module)))
    verify_heocr_wheel(wheel)
    module.write_bytes(b"# changed")
    with pytest.raises(ValueError, match="pinned wheel"):
        verify_heocr_wheel(wheel)


def test_unfinished_mock_attempt_keeps_unknown_cost(tmp_path):
    from registered_run import read_events
    from registered_metrics import counts_cost

    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"event": "request_start", "page": "a", "request": 1, "start_ns": 10})+"\n")
    completed, fallbacks, requests = read_events(events, ["a"], True, 2, 1, partial=True)
    assert completed == fallbacks == []
    assert requests == [{"request": 1, "page": "a", "seconds": None, "ok": False, "usage": None}]
    assert counts_cost(requests, {"denominator": 1, "rates": {"units": "1"}}) is None
