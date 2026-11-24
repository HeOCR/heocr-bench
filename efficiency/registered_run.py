"""Run a pin-checked decision 0030 efficiency plan without scoring OCR text."""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
from time import monotonic, sleep

from panel import HERE, sha256
from registered_metrics import cost_fields, counts_cost, paired_bounds, row
from run import TESSERACT_ENV, tree_rss

SCHEMA = "heocr.bench.registered-efficiency.v2"
PRICES = HERE / "registered-prices-2026-11-24.json"
PROFILES = ("m4-cpu", "github-ubuntu-cpu")
REPEATS = 9
SAMPLE_SECONDS = 0.02
LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}")
SHA = re.compile(r"[0-9a-f]{64}")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin_file(root, entry):
    path = (root / entry["path"]).resolve()
    match = SHA.fullmatch(entry["sha256"])
    need(match and path.is_file() and sha256(path) == entry["sha256"], "missing or changed pinned file")
    return str(path)


def load_plan(path, digest, execute, allow_api=False):
    path = path.resolve()
    need(SHA.fullmatch(digest) and sha256(path) == digest, "plan SHA-256 differs")
    plan = json.loads(path.read_text())
    need(plan["profile"] in PROFILES, "unknown hardware profile")
    purpose = "registered-open" if execute else "dev"
    need(plan["purpose"] == purpose, f"this mode needs a {purpose} plan")
    prices = json.loads(PRICES.read_text())
    need(plan["price_sha256"] == sha256(PRICES), "pricing SHA-256 differs")
    panel = plan["pages"]
    warmup = plan["warmup"]
    entries = [warmup, *panel]
    ids = [item["id"] for item in entries]
    need(panel and len(set(ids)) == len(ids) and all(LABEL.fullmatch(key) for key in ids),
         "page IDs must be unique, including warm-up")
    need(all(LABEL.fullmatch(item["carrier"]) for item in entries), "invalid carrier ID")
    need(warmup.get("purpose") == "dev" and warmup["sha256"] not in {item["sha256"] for item in panel},
         "warm-up needs an independent DEV page")
    need(len({item["sha256"] for item in panel}) == len(panel), "duplicate image bytes in panel")
    readers = plan["readers"]
    names = [item["id"] for item in readers]
    need(readers and len(set(names)) == len(names) and all(LABEL.fullmatch(key) for key in names),
         "reader IDs must be unique")
    comparisons = plan.get("comparators", [])
    candidate = plan.get("candidate")
    need(not comparisons or (candidate in names and set(comparisons) <= set(names) - {candidate}
                             and len(set(comparisons)) == len(comparisons)), "invalid comparison roster")
    for reader in readers:
        api = reader["kind"] == "api"
        need(reader["kind"] in ("local", "api") and LABEL.fullmatch(reader["mode"]), "invalid reader mode")
        valid_argv = (isinstance(reader["argv"], list) and reader["argv"]
                      and all(isinstance(arg, str) and arg for arg in reader["argv"]))
        need(valid_argv, "reader argv must be a nonempty argument vector")
        need(type(reader["concurrency"]) is int and 1 <= reader["concurrency"] <= (64 if api else 2),
             "invalid reader concurrency")
        if "allow_local_fallback" in reader:
            need(api and type(reader["allow_local_fallback"]) is bool, "invalid API fallback policy")
        try:
            fee = Decimal(str(reader.get("mandatory_usd_per_1000", 0)))
        except InvalidOperation:
            raise ValueError("invalid mandatory fee") from None
        need(fee.is_finite() and fee >= 0, "invalid mandatory fee")
        if api:
            need(LABEL.fullmatch(reader["service_region"]), "API service region must be explicit")
            price = prices["api"][reader["price_key"]]
            need("service_region" not in price or reader["service_region"] == price["service_region"],
                 "reader region differs from the pricing SKU")
            if execute:
                need(allow_api and all(item.get("egress") is True for item in entries),
                     "API execution needs explicit opt-in and egress clearance for every page")
                need("expires" not in price or date.today() <= date.fromisoformat(price["expires"]),
                     "registered API price has expired")
        if execute:
            need(reader.get("pins"), "registered reader needs pinned adapter/assets")
        for asset in reader.get("pins", []):
            pin_file(path.parent, asset)
    if execute:
        need(LABEL.fullmatch(plan["registration"]), "registered run needs a registration ID")
    for item in entries:
        item["path"] = pin_file(path.parent, item)
    return plan, prices


def hardware(profile):
    """Constrain the Ubuntu worker tree to two vCPUs; verify the M4 Max host."""
    import psutil

    info = {"system": platform.system(), "machine": platform.machine(),
            "os": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "runner_image": os.getenv("ImageOS"),
            "runner_image_version": os.getenv("ImageVersion")}
    if profile == "m4-cpu":
        need((info["system"], info["machine"], info["logical_cpus"]) == ("Darwin", "arm64", 16),
             "m4-cpu requires a native 16-core Apple M4 Max")
        cpu, memory = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True
        ).splitlines()
        need(cpu == "Apple M4 Max" and int(memory) == 128 * 1024**3, "m4-cpu requires 128 GiB of RAM")
        return {**info, "cpu": cpu, "memory_bytes": int(memory)}
    need((info["system"], info["machine"], info["runner_image"]) == ("Linux", "x86_64", "ubuntu24")
         and os.getenv("GITHUB_ACTIONS") == "true", "github-ubuntu-cpu requires a GitHub Ubuntu 24.04 x64 runner")
    affinity = sorted(os.sched_getaffinity(0))[:2]
    need(len(affinity) == 2, "runner needs two available vCPUs")
    os.sched_setaffinity(0, affinity)
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    return {**info, "cpu": cpu, "cpu_affinity": affinity, "memory_bytes": psutil.virtual_memory().total}


def read_events(path, expected, api, concurrency, elapsed, allow_local_fallback=False):
    completed, fallbacks, starts, ends = [], [], {}, {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        event = item["event"]
        if event == "page":
            completed.append(item["page"])
            if item.get("local_fallback", False):
                need(api and allow_local_fallback, "local fallback is outside the registered policy")
                fallbacks.append(item["page"])
        elif event == "request_start":
            number = item["request"]
            need(api and number not in starts and item["page"] in expected, "invalid request start")
            starts[number] = item
        elif event == "request_end":
            number = item["request"]
            need(api and number in starts and number not in ends, "invalid request end")
            ends[number] = item
        else:
            raise ValueError("unknown telemetry event")
    need(Counter(completed) == Counter(expected), "incomplete or repeated page output")
    need(starts.keys() == ends.keys(), "unfinished API request")
    requests, timeline = [], []
    for number, start in starts.items():
        end = ends[number]
        seconds = (end["end_ns"] - start["start_ns"]) / 1e9
        need(0 < seconds <= elapsed and type(end["ok"]) is bool, "invalid API request duration/status")
        timeline.extend(((start["start_ns"], 1), (end["end_ns"], -1)))
        requests.append({"request": number, "page": start["page"], "seconds": seconds,
                         "ok": end["ok"], "usage": end["usage"]})
    active = 0
    for _, delta in sorted(timeline):
        active += delta
        need(active <= concurrency, "API concurrency exceeds registration")
    if api:
        served = {item["page"] for item in requests if item["ok"]}
        need(served | set(fallbacks) == set(expected),
             "API reader did not complete a hosted request for every page")
    return completed, fallbacks, requests


def sample(command, root, job_path, events, timeout):
    import psutil

    env = {**os.environ, **TESSERACT_ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": "",
           "HEOCR_EFFICIENCY_JOB": str(job_path), "HEOCR_EFFICIENCY_EVENTS": str(events)}
    start = monotonic()
    with subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True) as worker:
        peak = 0
        try:
            process = psutil.Process(worker.pid)
            while worker.poll() is None:
                peak = max(peak, tree_rss(process))
                if monotonic() - start >= timeout:
                    raise TimeoutError("reader deadline exceeded")
                sleep(SAMPLE_SECONDS)
        finally:
            # A failed adapter can leave descendants holding the original process group.
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elapsed = monotonic() - start
        need(worker.returncode == 0 and peak > 0, "reader failed or RSS sampler saw no process")
    return elapsed, peak


def measure(reader, pages, phase, turn, root, output, price, dry_run, timeout):
    api = reader["kind"] == "api"
    stem = f"{reader['id']}.{phase}.{turn:02d}"
    job = output / f".{stem}.job.json"
    events = output / f"{stem}.jsonl"
    # The receipt omits original paths. Only the isolated worker needs the raster paths.
    job.write_text(json.dumps({"kind": reader["kind"], "phase": phase, "turn": turn,
                              "concurrency": reader["concurrency"], "pages": pages,
                              "billing_categories": list(price["rates"]) if api else []}))
    events.touch(exist_ok=False)
    command = [sys.executable, str(HERE / "registered_adapter.py"), "--mock-api"] if api and dry_run else reader["argv"]
    try:
        seconds, peak = sample(command, root, job, events, timeout)
    finally:
        job.unlink()
    completed, fallbacks, requests = read_events(events, [item["id"] for item in pages], api, reader["concurrency"],
                                                 seconds, reader.get("allow_local_fallback", False))
    cost = counts_cost(requests, price) if api else 0
    return {"turn": turn, "seconds": seconds, "peak_rss_bytes": peak, "completed": completed,
            "completed_by_carrier": dict(Counter(item["carrier"] for item in pages)),
            "requests": requests, "local_fallbacks": fallbacks, "events_sha256": sha256(events),
            "api_usd": None if cost is None else str(cost), "pages_per_minute": 60 * len(completed) / seconds}


def run(args):
    begin = monotonic()
    plan, prices = load_plan(args.plan, args.plan_sha256, args.execute, args.allow_api)
    runtime = hardware(plan["profile"])
    setup = monotonic() - begin
    output = args.output.resolve()
    output.mkdir(mode=0o700)
    root = args.plan.resolve().parent
    results = {}
    readers = plan["readers"]
    runner_rate = prices["runners"][plan["profile"]]

    def trial(reader, pages, phase, turn):
        try:
            outcome = measure(reader, pages, phase, turn, root, output, prices["api"].get(reader.get("price_key")),
                              not args.execute, args.timeout_seconds)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"{reader['id']} {phase} {turn}: {exc}") from None
        print(f"{reader['id']} {phase} {turn} complete", file=sys.stderr, flush=True)
        return outcome

    for reader in readers:
        results[reader["id"]] = {"warmup": trial(reader, [plan["warmup"]], "warmup", 0), "trials": []}
    orders = []
    for turn in range(REPEATS):
        pivot = turn % len(readers)
        schedule = readers[pivot:] + readers[:pivot]
        orders.append([reader["id"] for reader in schedule])
        for reader in schedule:
            results[reader["id"]]["trials"].append(trial(reader, plan["pages"], "trial", turn + 1))
    table = []
    for reader in readers:
        result = results[reader["id"]]
        mandatory_fee = Decimal(str(reader.get("mandatory_usd_per_1000", 0)))
        cost_fields(result["trials"], result["warmup"], runner_rate, prices["workload_pages"], mandatory_fee)
        fields = row(reader, plan["profile"], result, not args.execute)
        if args.execute and fields["fully_loaded_usd_per_1000"] is None:
            fields["status"] = "registered-cost-pending"
        table.append(fields)
        settings = ("id", "kind", "mode", "concurrency", "service_region", "price_key", "mandatory_usd_per_1000",
                    "allow_local_fallback")
        result["settings"] = {key: reader[key] for key in settings if key in reader}
    receipt = {
        "schema": SCHEMA, "dry_run": not args.execute, "utc_generated": datetime.now(timezone.utc).isoformat(),
        "registration": plan.get("registration"), "plan_sha256": args.plan_sha256,
        "price_sha256": plan["price_sha256"], "price_date": prices["checked"], "pricing": prices,
        "runtime": runtime, "harness_setup_seconds": setup, "orders": orders,
        "rss_sample_seconds": SAMPLE_SECONDS, "worker_timeout_seconds": args.timeout_seconds,
        "runner_usd_hour": str(Decimal(runner_rate["usd"]) / Decimal(runner_rate["hours"])),
        "confidence_interval": "exact two-sided 95% median sign interval; independent ordered repeat blocks",
        "readers": results, "table": table, "paired_efficiency_bounds": paired_bounds(
            plan.get("candidate"), plan.get("comparators", []), results),
    }
    temporary = output / ".receipt.json"
    temporary.write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    temporary.replace(output / "receipt.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private directory for event logs and receipt")
    parser.add_argument("--execute", action="store_true", help="run a preregistered OPEN plan")
    parser.add_argument("--allow-api", action="store_true", help="enable live API adapters for a cleared OPEN plan")
    parser.add_argument("--timeout-seconds", type=int, default=3600, help="hard time limit per isolated worker")
    args = parser.parse_args()
    if args.timeout_seconds < 1 or (args.allow_api and not args.execute):
        parser.error("positive timeout required; --allow-api needs --execute")
    try:
        run(args)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"registered efficiency: {exc}\n")


if __name__ == "__main__":
    main()
