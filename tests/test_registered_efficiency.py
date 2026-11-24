import csv
from decimal import Decimal
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

import pytest

EFFICIENCY = Path(__file__).resolve().parents[1] / "efficiency"
sys.path.insert(0, str(EFFICIENCY))
from panel import sha256  # noqa: E402
from registered_metrics import cost_fields, counts_cost, median_ci, paired_bounds  # noqa: E402
from registered_run import PRICES, REPEATS, load_plan, read_events  # noqa: E402
from report import export_registered  # noqa: E402


def make_plan(root, profile="github-ubuntu-cpu"):
    def image(name):
        input_path = root / name
        input_path.write_bytes(name.encode())
        return dict(id=name, carrier="scanner", path=name, sha256=sha256(input_path), egress=False)

    adapter = (
        "from time import sleep; from telemetry import Telemetry; t=Telemetry(); "
        "sleep(0.03); [t.complete(p['id']) for p in t.pages]"
    )
    return {
        "profile": profile, "purpose": "dev", "price_sha256": sha256(PRICES),
        "warmup": {**image("warmup"), "purpose": "dev"}, "pages": [image("page-1"), image("page-2")],
        "readers": [
            {"id": "local", "kind": "local", "mode": "fixture", "concurrency": 1,
             "argv": [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(EFFICIENCY)!r}); {adapter}"]},
            {"id": "api", "kind": "api", "mode": "sync", "concurrency": 1,
             "argv": [sys.executable, "-c", "raise SystemExit('real API adapter must not run in DEV')"],
             "service_region": "global", "price_key": "haiku-5-5-standard"},
        ],
        "candidate": "local", "comparators": ["api"],
    }


def test_exact_intervals_and_simultaneous_paired_ratios():
    assert median_ci(range(1, 10)) == {"median": 5, "ci95": [2, 8]}
    def arm(speed, cost):
        return {"trials": [{"pages_per_minute": speed * n, "fully_loaded_usd_per_1000": cost * n}
                           for n in range(1, 10)]}
    stats = paired_bounds("local", ["api"], {"local": arm(3, 1), "api": arm(1, 3)})["api"]
    assert stats == {"throughput_ratio_median": 3, "throughput_ratio_lower": 3,
                     "cost_ratio_median": 1 / 3, "cost_ratio_upper": 1 / 3}


def test_price_applies_per_attempt_and_unknown_spend_is_unavailable():
    price = json.loads(PRICES.read_text())["api"]["haiku-5-5-standard"]
    items = [{"usage": {"input_tokens": 100_000, "output_tokens": 2}, "ok": False},
             {"usage": {"input_tokens": 100_001, "output_tokens": 3}, "ok": True}]
    assert counts_cost(items, price) == Decimal("0.060009")
    assert counts_cost([*items, {"usage": None}], price) is None
    with pytest.raises(ValueError, match="billed category"):
        counts_cost([{"usage": {"input_tokens": 5}}], price)


def test_cost_includes_warmup_runner_rounding_and_fees():
    trial = {"seconds": 30, "completed": ["x"] * 100, "api_usd": "0.10"}
    warmup = {"seconds": 1, "api_usd": "0.01"}
    rate = json.loads(PRICES.read_text())["runners"]["github-ubuntu-cpu"]
    cost_fields([trial], warmup, rate, 1000, Decimal("0.20"))
    assert trial["runner_usd_per_1000"] == 0.036
    assert trial["marginal_usd_per_1000"] == 1.01
    assert trial["fully_loaded_usd_per_1000"] == 1.246
    cost_fields([trial], {**warmup, "api_usd": None}, rate, 1000)
    assert trial["fully_loaded_usd_per_1000"] is None


def test_prereg_gate_refuses_private_pages_and_mismatched_plan_hash(tmp_path):
    plan = make_plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="SHA-256"):
        load_plan(path, "a" * 64, False)
    with pytest.raises(ValueError, match="registered-open"):
        load_plan(path, sha256(path), True, True)
    plan.update(purpose="registered-open", registration="c1-v1")
    for reader in plan["readers"]:
        reader["pins"] = [plan["warmup"]]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="egress clearance"):
        load_plan(path, sha256(path), True, True)


def test_coverage_and_api_events_fail_closed(tmp_path):
    events = tmp_path / "events.jsonl"
    expected = ["a"]
    base = [
        {"event": "request_start", "request": 1, "page": "a", "start_ns": 10},
        {"event": "request_end", "request": 1, "end_ns": 20, "ok": True, "usage": {"units": 1}},
        {"event": "page", "page": "a"},
    ]
    for bad, reason in [
        (base[:-1], "page output"),
        ([base[0], base[-1]], "unfinished"),
        ([base[-1]], "hosted request"),
        ([*base, base[-1]], "repeated page"),
        ([base[0], {**base[0], "request": 2}, base[1], {**base[1], "request": 2}, base[-1]], "concurrency"),
    ]:
        events.write_text("\n".join(json.dumps(item) for item in bad))
        with pytest.raises(ValueError, match=reason):
            read_events(events, expected, api=True, concurrency=1, elapsed=1)

    events.write_text(json.dumps({"event": "page", "page": "a", "local_fallback": True}))
    with pytest.raises(ValueError, match="outside the registered policy"):
        read_events(events, expected, api=True, concurrency=1, elapsed=1)
    assert read_events(events, expected, api=True, concurrency=1, elapsed=1,
                       allow_local_fallback=True) == (["a"], ["a"], [])


def test_dev_runs_isolated_rotations_and_exports_only_with_opt_in(tmp_path, monkeypatch):
    from argparse import Namespace
    from registered_run import run

    plan = make_plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    output = tmp_path / "receipt"
    monkeypatch.setattr("registered_run.hardware", lambda _: {"runner_image": "fixture"})
    run(Namespace(plan=path, plan_sha256=sha256(path), output=output,
                  execute=False, allow_api=False, timeout_seconds=30))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["dry_run"] is True
    assert len(receipt["orders"]) == REPEATS
    assert receipt["orders"][:3] == [["local", "api"], ["api", "local"], ["local", "api"]]
    assert {item["page"] for result in receipt["readers"]["api"]["trials"] for item in result["requests"]} == {
        "page-1", "page-2",
    }
    measured = receipt["readers"]["api"]["trials"][0]
    assert measured["events_sha256"] == sha256(output / "api.trial.01.jsonl")
    assert all(item["seconds"] > 0 and item["usage"]["input_tokens"] == 100 for item in measured["requests"])
    assert receipt["table"][1]["api_pages_per_minute"] > 0
    assert receipt["table"][1]["api_pages_per_minute_ci95"] is not None
    assert "plan.json" not in (output / "receipt.json").read_text()
    assert not list(output.glob(".*.job.json"))
    with pytest.raises(ValueError, match="DEV is excluded"):
        export_registered(output / "receipt.json", StringIO())
    csv_output = StringIO()
    export_registered(output / "receipt.json", csv_output, allow_dry_run=True)
    rows = list(csv.DictReader(StringIO(csv_output.getvalue())))
    assert rows[0]["status"] == rows[1]["status"] == "dry-run"
    assert json.loads(rows[1]["peak_rss_mib_ci95"])[0] > 0


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    root.mkdir()
    path = root / "plan.json"
    path.write_text(json.dumps(make_plan(root)))
    subprocess.run([sys.executable, str(EFFICIENCY / "registered_run.py"), "--plan", str(path),
                    "--plan-sha256", sha256(path), "--output", str(root / "receipt")], check=True)
    receipt = json.loads((root / "receipt/receipt.json").read_text())
    assert receipt["runtime"]["runner_image"] == "ubuntu24"
    assert len(receipt["runtime"]["cpu_affinity"]) == 2
    assert receipt["dry_run"] and all(row["repetitions"] == REPEATS for row in receipt["table"])
