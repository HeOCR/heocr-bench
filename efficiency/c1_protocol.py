"""The four-repeat C1 efficiency protocol, separate from the legacy sign-bound run."""

from decimal import Decimal, InvalidOperation
from math import atan, exp, isfinite, log, pi, sqrt
from statistics import fmean, median, stdev

PROTOCOL = "c1-740"
REPEATS = 4
HOURS = 36 * 160


def schedule(readers):
    first = sorted(readers, key=lambda reader: reader["id"].encode("ascii"))
    pivot = len(first) // 2
    rotated = first[pivot:] + first[:pivot]
    return [first, first[::-1], rotated, rotated[::-1]]


def m4_rates(config, verify_evidence):
    """Validate the pinned USD acquisition, wattage and electricity evidence."""
    if config is None:
        return None
    values = {}
    for key in ("acquisition_usd", "mean_watts", "electricity_usd_per_kwh"):
        try:
            value = Decimal(str(config[key]))
        except (KeyError, InvalidOperation) as exc:
            raise ValueError(f"invalid M4 cost: {key}") from exc
        if not value.is_finite() or value < 0 or (key == "acquisition_usd" and not value):
            raise ValueError(f"invalid M4 cost: {key}")
        values[key] = str(value)
    evidence = config.get("evidence", {})
    required = {"acquisition", "power", "electricity"}
    if not required <= evidence.keys() <= required | {"exchange_rate"}:
        raise ValueError("M4 cost needs acquisition, power and electricity evidence")
    for item in evidence.values():
        verify_evidence(item)
    amortization = Decimal(values["acquisition_usd"]) / HOURS
    power = Decimal(values["mean_watts"]) * Decimal(values["electricity_usd_per_kwh"]) / 1000
    return {**values, "amortization_hours": HOURS, "amortization_usd_hour": str(amortization),
            "power_usd_hour": str(power), "evidence_sha256": {key: item["sha256"] for key, item in evidence.items()}}


def cost_fields(trials, rates, fixed_fee):
    """Count the client wall clock once. Keep setup and request charges distinct."""
    for trial in trials:
        n = len(trial["completed"])
        eligible = trial["complete"] and n > 0
        hours = Decimal(str(trial["seconds"])) * 1000 / (n * 3600) if eligible else None
        cash = Decimal(trial["api_usd"]) * 1000 / n if eligible and trial["api_usd"] is not None else None
        capital = power = None
        if rates is not None and hours is not None:
            capital = hours * Decimal(rates["amortization_usd_hour"])
            power = hours * Decimal(rates["power_usd_hour"])
        runner = capital + power if capital is not None else None
        total = runner + cash + fixed_fee if None not in (runner, cash, fixed_fee) else None
        fields = {"amortization_usd_per_1000": capital, "power_usd_per_1000": power,
                  "runner_usd_per_1000": runner, "marginal_usd_per_1000": cash,
                  "mandatory_usd_per_1000": fixed_fee, "fully_loaded_usd_per_1000": total}
        trial.update({key: float(value) if value is not None else None for key, value in fields.items()})


def student_t3_critical(alpha):
    """Invert the closed-form Student-t survival function for three degrees of freedom."""
    def tail(x):
        z = sqrt(3) / x
        return (atan(z) - z / (1 + z * z)) / pi

    low, high = 0.0, 1.0
    while tail(high) > alpha:
        high *= 2
    for _ in range(60):
        middle = (low + high) / 2
        if tail(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _paired_ratios(pairs, field, *, positive):
    ratios = []
    for a, b in pairs:
        base, control = a.get(field), b.get(field)
        if (base is None or control is None or not isfinite(base) or not isfinite(control)
                or control <= 0 or base < 0 or (positive and not base)):
            return []
        ratio = base / control
        if not isfinite(ratio) or (positive and not ratio):
            return []
        ratios.append(ratio)
    return ratios


def paired_bounds(candidate, comparators, results):
    bounds = {}
    if not comparators:
        return bounds
    critical = student_t3_critical(.025 / (2 * len(comparators)))
    first = results[candidate]["trials"]
    for comparator in comparators:
        second = results[comparator]["trials"]
        pairs = list(zip(first, second)) if len(first) == len(second) == REPEATS else []
        if any(a.get("turn") != n or b.get("turn") != n or not (a["complete"] and b["complete"])
               for n, (a, b) in enumerate(pairs, 1)):
            pairs = []
        speeds = _paired_ratios(pairs, "pages_per_minute", positive=True)
        costs = _paired_ratios(pairs, "fully_loaded_usd_per_1000", positive=False)
        # Both zero marginal costs use the speed leg even if M4 client costs are present.
        if pairs and all(trial.get("marginal_usd_per_1000") == 0 and trial.get("mandatory_usd_per_1000") == 0
                         for pair in pairs for trial in pair):
            costs = []
        logs = list(map(log, speeds))
        lower = exp(fmean(logs) - critical * stdev(logs) / 2) if logs else None
        try:
            upper = fmean(costs) + critical * stdev(costs) / 2 if costs else None
        except OverflowError:
            upper = None
        bounds[comparator] = {
            "throughput_ratio_median": median(speeds) if speeds else None,
            "throughput_ratio_lower": lower,
            "cost_ratio_median": median(costs) if costs else None,
            "cost_ratio_upper": upper if upper is not None and isfinite(upper) else None,
        }
    return bounds
