from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "efficiency"))
from panel import HERE, panel  # noqa: E402
from quote import quote  # noqa: E402
from report import PLATFORMS, rows  # noqa: E402
from run import metrics  # noqa: E402
from vertex import response_usage  # noqa: E402


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


def test_external_export_excludes_router_and_keeps_api_price_only_status():
    external = list(rows({host: {} for host in PLATFORMS}, external_only=True))
    assert len(external) == 12
    assert all(row["reader_id"] != "router-v2" for row in external)
    gemini = [row for row in external if row["reader_id"] == "gemini-vertex-v0"]
    assert len(gemini) == 2 and all(row["status"] == "list-price-only" for row in gemini)


def test_token_price_includes_reasoning():
    prices = json.loads((HERE / "api-pricing.json").read_text(), parse_float=Decimal)
    assert quote(100, 100_000, 10_000, 5_000, prices) == Decimal("1.3125")
    with pytest.raises(ValueError):
        quote(0, 0, 0, 0, prices)


def test_vertex_counts_only_billable_uncached_ocr_usage():
    response = {
        "modelVersion": "pinned",
        "candidates": [{"finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 1000, "candidatesTokenCount": 150,
            "thoughtsTokenCount": 10, "totalTokenCount": 1160,
        },
    }
    assert response_usage(response, "pinned") == {
        "promptTokenCount": 1000, "candidatesTokenCount": 150, "thoughtsTokenCount": 10,
    }
    with pytest.raises(ValueError, match="model version"):
        response_usage(response, "other")
    response["usageMetadata"]["cachedContentTokenCount"] = 1
    with pytest.raises(ValueError, match="caching"):
        response_usage(response, "pinned")
