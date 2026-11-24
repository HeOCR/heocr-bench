"""Project aggregate billing and timing receipts from registered API runs."""

from decimal import Decimal
import json
from pathlib import Path

from panel import HERE

API_TIMING_LABEL = "from registered run; network-bound, not comparable to local CPU"
PRICING = HERE / "api-pricing.json"


def read_registered(path: Path):
    receipt = json.loads(path.read_text(), parse_float=Decimal)
    if receipt["schema"] != "heocr.bench.registered-api.v1":
        raise ValueError("unrecognized registered API receipt")
    prices = json.loads(PRICING.read_text(), parse_float=Decimal)["readers"]
    if receipt["readers"].keys() != prices.keys():
        raise ValueError("registered API receipts and pricing cover different readers")
    return {name: registered_fields(row, prices[name]) for name, row in receipt["readers"].items()}


def list_cost_per_1000(pages, usage, price):
    if usage.keys() != price["rates"].keys():
        raise ValueError("recorded usage must cover each billed category")
    if any(type(value) is not int or value < 0 for value in usage.values()):
        raise ValueError("usage counts must be nonnegative integers")
    dollars = sum(Decimal(count) * price["rates"][kind] for kind, count in usage.items())
    return round(dollars * 1000 / (pages * price["denominator"]), 6)


def registered_fields(row, price):
    pages = row["pages"]
    if type(pages) is not int or pages <= 0:
        raise ValueError("registered page count must be a positive integer")
    rates = price["rates"]
    usage, timing = row["usage"], row["timing"]
    fields = {
        "platform": "registered-api",
        "execution": row["execution"],
        "status": {
            (True, True): "registered",
            (True, False): "registered-cost-only",
            (False, True): "registered-timing-only",
            (False, False): "list-price-only",
        }[(usage is not None, timing is not None)],
        "pages": pages,
        "repetitions": 1,
        "api_timing_label": API_TIMING_LABEL,
        "api_input_usd_per_million": rates.get("input_tokens"),
        "api_output_usd_per_million": rates.get("output_tokens"),
        "api_usd_per_1000_units": rates.get("units"),
        "api_registration": row["registration"],
    }
    if usage is not None:
        fields["marginal_usd_per_1000"] = list_cost_per_1000(pages, usage, price)
    if timing is not None:
        fields["api_timing_scope"] = timing["scope"]
        timed_pages = timing["pages"]
        if type(timed_pages) is not int or not 0 < timed_pages <= pages:
            raise ValueError("recorded timing coverage must fit within the registered run")
        fields["api_timed_pages"] = timed_pages
        for key in ("wall_seconds", "mean_request_seconds"):
            value = timing.get(key)
            if value is not None and (not Decimal(value).is_finite() or value <= 0):
                raise ValueError("recorded timing must be finite and positive")
        if timing.get("wall_seconds") is not None:
            fields["api_pages_per_minute"] = round(Decimal(60 * timed_pages) / timing["wall_seconds"], 4)
        if timing.get("mean_request_seconds") is not None:
            fields["api_mean_request_latency_seconds"] = timing["mean_request_seconds"]
    return fields
