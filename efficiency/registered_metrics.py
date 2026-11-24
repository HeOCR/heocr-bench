"""Prespecified decision 0030 statistics and usage costing."""

from decimal import Decimal, ROUND_CEILING
from math import comb
from statistics import median

def median_ci(values):
    ordered = sorted(values)
    n = len(ordered)
    if n < 6:
        raise ValueError("at least six trials needed for the exact two-sided median interval")
    k = max(i for i in range(1, (n + 1) // 2 + 1)
            if 40 * sum(comb(n, j) for j in range(i)) <= 2**n)
    return {"median": median(ordered), "ci95": [ordered[k - 1], ordered[-k]]}


def request_cost(usage, price):
    if usage is None:
        return None
    rates = price["rates"]
    if usage.keys() != rates.keys() or any(type(count) is not int or count < 0 for count in usage.values()):
        raise ValueError("request usage must specify each billed category as a nonnegative integer")
    if "long_prompt_over" in price and usage["input_tokens"] > price["long_prompt_over"]:
        rates = price["long_rates"]
    return sum(Decimal(rates[key]) * count for key, count in usage.items()) / price["denominator"]


def counts_cost(requests, price):
    costs = [request_cost(item["usage"], price) for item in requests]
    return None if None in costs else sum(costs, Decimal(0))


def cost_fields(trials, warmup, rate, workload, mandatory_fees_usd=Decimal(0)):
    hourly = Decimal(rate["usd"]) / Decimal(rate["hours"])
    setup_seconds = Decimal(str(warmup["seconds"]))
    warmup_api = warmup["api_usd"]
    for trial in trials:
        n = len(trial["completed"])
        run_seconds = Decimal(str(trial["seconds"]))
        billed_minutes = (setup_seconds + run_seconds * workload / n) / 60
        if rate["minute_rounding"]:
            billed_minutes = billed_minutes.to_integral_value(rounding=ROUND_CEILING)
        trial["runner_usd_per_1000"] = float(billed_minutes / 60 * hourly * 1000 / workload)
        if warmup_api is None or trial["api_usd"] is None:
            trial["marginal_usd_per_1000"] = None
            trial["fully_loaded_usd_per_1000"] = None
            continue
        marginal = (Decimal(trial["api_usd"]) * workload / n + Decimal(warmup_api)) * 1000 / workload
        trial["marginal_usd_per_1000"] = float(marginal)
        trial["fully_loaded_usd_per_1000"] = (
            float(billed_minutes / 60 * hourly * 1000 / workload + marginal + mandatory_fees_usd)
        )


def row(reader, platform, result, dry_run, *, four_repeats=False):
    trials = result["trials"]
    api = reader["kind"] == "api"
    requests = [item for trial in trials for item in trial["requests"]]
    complete = not four_repeats or all(item["complete"] for item in trials)
    pages = min(len(item["completed"]) for item in trials)
    rss_values = [item["peak_rss_bytes"] / 1024**2 for item in trials if item["peak_rss_bytes"] > 0]
    speed_values = [item["pages_per_minute"] for item in trials] if complete else []
    if four_repeats:
        timing = {"median": median(speed_values) if speed_values else None, "ci95": None}
        memory = {"median": median(rss_values) if rss_values else None, "ci95": None}
    else:
        timing, memory = median_ci(speed_values), median_ci(rss_values)
    runner_costs = [item["runner_usd_per_1000"] for item in trials]
    data = {
        "reader_id": reader["id"], "platform": platform,
        "execution": reader["mode"], "status": "dry-run" if dry_run else "registered",
        "pages": pages, "repetitions": len(trials),
        "pages_per_minute": None if api else timing["median"],
        "pages_per_minute_ci95": None if api else timing["ci95"],
        "peak_rss_mib": memory["median"], "peak_rss_mib_ci95": memory["ci95"],
        "runner_hours_per_1000": (median(item["seconds"] / len(item["completed"]) * 1000 / 3600
                                         for item in trials) if complete and pages else None),
        "runner_usd_per_1000": None if None in runner_costs else median(runner_costs),
    }
    for key in ("marginal_usd_per_1000", "fully_loaded_usd_per_1000"):
        values = [item[key] for item in trials]
        data[key] = None if None in values else median(values)
    if api:
        durations = [item["seconds"] for item in requests if item["seconds"] is not None]
        data.update({
            "api_price_key": reader["price_key"],
            "api_service_region": reader["service_region"],
            "api_pages_per_minute": timing["median"], "api_pages_per_minute_ci95": timing["ci95"],
            "api_mean_request_latency_seconds": sum(durations) / len(durations) if durations else None,
            "api_timed_pages": len({item["page"] for item in requests if item["seconds"] is not None}),
            "api_timing_scope": "end-to-end synchronous client, all attempts included; RSS is client-only",
            "api_timing_label": "from registered run; network-bound; RSS measures the client",
        })
    return data


def paired_bounds(candidate, comparators, results):
    """Bonferroni exact one-sided sign intervals on paired repeat ratios."""
    if not comparators:
        return {}
    alpha = Decimal("0.05") / (2 * len(comparators))
    trials = results[candidate]["trials"]
    n = len(trials)
    rank = max((i for i in range(1, (n + 1) // 2 + 1)
                if Decimal(sum(comb(n, j) for j in range(i))) / 2**n <= alpha), default=0)
    bounds = {}
    for name in comparators:
        pairs = list(zip(trials, results[name]["trials"], strict=True))
        throughput = sorted(a["pages_per_minute"] / b["pages_per_minute"] for a, b in pairs)
        key = "fully_loaded_usd_per_1000"
        comparable_cost = all(a[key] is not None and b[key] for a, b in pairs)
        costs = sorted(a[key] / b[key] for a, b in pairs) if comparable_cost else []
        bounds[name] = {
            "throughput_ratio_median": median(throughput),
            "throughput_ratio_lower": throughput[rank - 1] if rank else 0,
            "cost_ratio_median": median(costs) if costs else None,
            "cost_ratio_upper": costs[-rank] if costs and rank else None,
        }
    return bounds
