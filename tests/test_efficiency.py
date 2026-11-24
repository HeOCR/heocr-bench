from collections import Counter
from decimal import Decimal
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "efficiency"))
from panel import HERE, panel  # noqa: E402
from registered import API_TIMING_LABEL, read_registered, registered_fields  # noqa: E402
from report import PLATFORMS, rows  # noqa: E402
from run import measure, metrics  # noqa: E402


def test_public_panel_is_image_only_and_covers_all_carriers():
    assert Counter(kind for _, _, kind in panel()) == {
        "archive": 5, "issuer-render": 9, "scanner": 17, "phone-a": 17, "phone-b": 17
    }
    assert len({digest for _, digest, _ in panel()}) == 65


def test_metrics_include_median_time_and_maximum_sampled_memory():
    runs = [{"seconds": seconds, "peak_rss_bytes": mib * 1024**2,
             "completed_by_carrier": {"scanner": 65}}
            for seconds, mib in [(15, 10), (30, 21), (45, 14)]]
    assert metrics(runs, 0.36) == {
        "pages": 65, "pages_per_minute": 130.0, "peak_rss_mib": 21,
        "marginal_usd_per_1000": 0, "runner_hours_per_1000": 0.128205,
        "runner_usd_per_1000": 0.046154,
    }


def test_native_worker_coverage_remains_nested_in_the_measurement():
    with patch("run.sample", return_value={
        "worker_result": {"scanner": 65}, "seconds": 30, "peak_rss_bytes": 1024,
    }):
        assert measure(Path("images"), "tessdata-fast", None) == {
            "completed_by_carrier": {"scanner": 65}, "seconds": 30, "peak_rss_bytes": 1024,
        }


def test_external_export_joins_single_registered_rows_and_omits_router():
    api = read_registered(HERE / "receipts/registered-api.json")
    external = list(rows({host: {} for host in PLATFORMS}, api, external_only=True))
    assert len(external) == 15
    assert all(row["reader_id"] != "router-v2" for row in external)
    registered = {row["reader_id"]: row for row in external if row["platform"] == "registered-api"}
    assert len(registered) == 5
    assert all(row["api_timing_label"] == API_TIMING_LABEL for row in registered.values())
    assert all("pages_per_minute" not in row and "runner_hours_per_1000" not in row
               for row in registered.values())
    assert registered["gemini-vertex-v0"]["api_input_usd_per_million"] == Decimal("0.75")
    assert registered["claude-haiku-4-5-v0"]["marginal_usd_per_1000"] == Decimal("4.629431")
    assert registered["claude-haiku-5-5-v0"]["marginal_usd_per_1000"] == Decimal("0.458222")
    assert registered["google-vision-document-text-v0"]["marginal_usd_per_1000"] == Decimal("1.50")
    assert registered["google-document-ai-ocr-v21-v0"]["marginal_usd_per_1000"] == Decimal("1.50")


def test_request_latency_is_not_derived_from_batch_wall_and_usage_includes_reasoning():
    price = {"denominator": 1_000_000, "rates": {
        "input_tokens": Decimal("0.75"),
        "output_tokens": Decimal("3.75"),
        "reasoning_tokens": Decimal("3.75"),
    }}
    row = {"registration": "#1", "execution": "batch", "pages": 100,
           "usage": {"input_tokens": 100_000, "output_tokens": 10_000, "reasoning_tokens": 5_000},
           "timing": {"scope": "provider batch wall", "pages": 50, "wall_seconds": 120}}
    result = registered_fields(row, price)
    assert result["marginal_usd_per_1000"] == Decimal("1.3125")
    assert result["api_pages_per_minute"] == Decimal("25")
    assert "api_mean_request_latency_seconds" not in result
    row["timing"] = None
    assert "api_pages_per_minute" not in registered_fields(row, price)
    row["usage"].pop("reasoning_tokens")
    with pytest.raises(ValueError, match="billed category"):
        registered_fields(row, price)
