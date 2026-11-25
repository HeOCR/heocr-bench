"""Run a pin-checked decision 0030 efficiency plan without scoring OCR text."""

import argparse
import c1_protocol
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
    protocol = plan.get("protocol")
    need(protocol in (None, c1_protocol.PROTOCOL), "unknown timing protocol")
    c1 = protocol == c1_protocol.PROTOCOL
    need(plan["profile"] in PROFILES, "unknown hardware profile")
    if c1:
        need(plan["profile"] == "m4-cpu", "C1 #740 needs the M4 CPU profile")
        need("warmup" not in plan, "C1 #740 has four timed repeats and no runner warm-up")
    purpose = "registered-open" if execute else "dev"
    need(plan["purpose"] == purpose, f"this mode needs a {purpose} plan")
    prices = json.loads(PRICES.read_text())
    need(plan["price_sha256"] == sha256(PRICES), "pricing SHA-256 differs")
    panel = plan["pages"]
    warmup = None if c1 else plan["warmup"]
    entries = panel if c1 else [warmup, *panel]
    ids = [item["id"] for item in entries]
    need(panel and len(set(ids)) == len(ids) and all(LABEL.fullmatch(key) for key in ids),
         "page IDs must be unique, including warm-up")
    need(all(LABEL.fullmatch(item["carrier"]) for item in entries), "invalid carrier ID")
    if warmup is not None:
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
    layout_arms = []
    for reader in readers:
        api = reader["kind"] == "api"
        need(reader["kind"] in ("local", "api") and LABEL.fullmatch(reader["mode"]), "invalid reader mode")
        valid_argv = (isinstance(reader["argv"], list) and reader["argv"]
                      and all(isinstance(arg, str) and arg for arg in reader["argv"]))
        need(valid_argv, "reader argv must be a nonempty argument vector")
        need(type(reader["concurrency"]) is int and 1 <= reader["concurrency"] <= (64 if api else 2),
             "invalid reader concurrency")
        if c1:
            need(reader["concurrency"] <= 2 if api else reader["concurrency"] == 2,
                 "C1 #740 needs two local slots and at most two hosted requests")
        if "ocr_layout" in reader or "heocr_wheel" in reader:
            need(c1 and not api and reader["mode"] == "router-v2" and reader.get("ocr_layout") in ("off", "v1")
                 and "heocr_wheel" in reader, "layout diagnostics need router-v2, off/v1, and a wheel pin")
            pin_file(path.parent, reader["heocr_wheel"])
            layout_arms.append(reader)
        if "allow_local_fallback" in reader:
            need(api and type(reader["allow_local_fallback"]) is bool, "invalid API fallback policy")
        try:
            fee_value = reader.get("mandatory_usd_per_1000", 0)
            fee = None if c1 and fee_value is None else Decimal(str(fee_value))
        except InvalidOperation:
            raise ValueError("invalid mandatory fee") from None
        need(fee is None or (fee.is_finite() and fee >= 0), "invalid mandatory fee")
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
    if layout_arms:
        comparable = [{k: v for k, v in r.items() if k not in ("id", "ocr_layout")} for r in layout_arms]
        need(len(layout_arms) == 2 and {r["ocr_layout"] for r in layout_arms} == {"off", "v1"}
             and comparable[0] == comparable[1], "layout comparison requires identical settings and wheel except off/v1")
        layout_on = next(r["id"] for r in layout_arms if r["ocr_layout"] == "v1")
        layout_off = next(r["id"] for r in layout_arms if r["ocr_layout"] == "off")
        need(candidate == layout_on and layout_off not in comparisons,
             "layout v1 must be the launch candidate and layout off cannot be a launch comparator")
    if c1:
        need(not execute or bool(comparisons), "registered #740 needs its eligible comparator roster")
        plan["m4_rates"] = c1_protocol.m4_rates(plan.get("m4_cost"), lambda item: pin_file(path.parent, item))
        need(prices["workload_pages"] == 1000, "C1 #740 needs a 1,000-page monthly workload")
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
        build = subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()
        return {**info, "cpu": cpu, "memory_bytes": int(memory), "os_build": build}
    need((info["system"], info["machine"], info["runner_image"]) == ("Linux", "x86_64", "ubuntu24")
         and os.getenv("GITHUB_ACTIONS") == "true", "github-ubuntu-cpu requires a GitHub Ubuntu 24.04 x64 runner")
    affinity = sorted(os.sched_getaffinity(0))[:2]
    need(len(affinity) == 2, "runner needs two available vCPUs")
    os.sched_setaffinity(0, affinity)
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    return {**info, "cpu": cpu, "cpu_affinity": affinity, "memory_bytes": psutil.virtual_memory().total}


def read_events(path, expected, api, concurrency, elapsed, allow_local_fallback=False, *, partial=False):
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
    need(Counter(completed) <= Counter(expected) if partial else Counter(completed) == Counter(expected),
         "incomplete or repeated page output")
    need(partial or starts.keys() == ends.keys(), "unfinished API request")
    requests, timeline = [], []
    for number, start in starts.items():
        end = ends.get(number)
        if end is None:
            timeline.append((start["start_ns"], 1))
            requests.append({"request": number, "page": start["page"], "seconds": None, "ok": False, "usage": None})
            continue
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
        need(set(completed) <= served | set(fallbacks) if partial else served | set(fallbacks) == set(expected),
             "API reader did not complete a hosted request for every completed page")
    return completed, fallbacks, requests


def sample(command, root, job_path, events, timeout, *, partial=False):
    import psutil

    env = {**os.environ, **TESSERACT_ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": "",
           "HEOCR_EFFICIENCY_JOB": str(job_path), "HEOCR_EFFICIENCY_EVENTS": str(events)}
    start = monotonic()
    with subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True) as worker:
        peak = 0
        timed_out = False
        try:
            process = psutil.Process(worker.pid)
            while worker.poll() is None:
                peak = max(peak, tree_rss(process))
                if monotonic() - start >= timeout:
                    if not partial:
                        raise TimeoutError("reader deadline exceeded")
                    timed_out = True
                    break
                sleep(SAMPLE_SECONDS)
        finally:
            # A failed adapter can leave descendants holding the original process group.
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elapsed = monotonic() - start
        if partial:
            return elapsed, peak, "timeout" if timed_out else (
                "worker_failed" if worker.poll() != 0 or peak <= 0 else None)
        need(worker.returncode == 0 and peak > 0, "reader failed or RSS sampler saw no process")
    return elapsed, peak


def measure(reader, pages, phase, turn, root, output, price, dry_run, timeout, *, c1=False):
    api = reader["kind"] == "api"
    stem = f"{reader['id']}.{phase}.{turn:02d}"
    job = output / f".{stem}.job.json"
    events = output / f"{stem}.jsonl"
    expected = [item["id"] for item in pages]
    # The receipt omits original paths. Only the isolated worker needs the raster paths.
    payload = {"kind": reader["kind"], "phase": phase, "turn": turn,
               "concurrency": reader["concurrency"], "pages": pages,
               "billing_categories": list(price["rates"]) if api else []}
    retained = None
    if c1:
        if turn == 1 and not dry_run:
            retained = output / f"{reader['id']}.first"
            retained.mkdir(mode=0o700)
        payload.update(protocol=c1_protocol.PROTOCOL, ocr_layout=reader.get("ocr_layout", "off"),
                       accuracy_output_dir=str(retained) if retained is not None else None)
    job.write_text(json.dumps(payload))
    events.touch(exist_ok=False)
    command = [sys.executable, str(HERE / "registered_adapter.py"), "--mock-api"] if api and dry_run else reader["argv"]
    failure = None
    try:
        if c1:
            seconds, peak, failure = sample(command, root, job, events, timeout, partial=True)
        else:
            seconds, peak = sample(command, root, job, events, timeout)
    finally:
        job.unlink()
    try:
        completed, fallbacks, requests = read_events(events, expected, api, reader["concurrency"], seconds,
                                                     reader.get("allow_local_fallback", False), partial=c1)
        cost = counts_cost(requests, price) if api else 0
    except (KeyError, ValueError):
        if not c1:
            raise
        completed, fallbacks, requests, cost = [], [], [], None
        failure = failure or "invalid_telemetry"
    detail = {}
    if c1:
        if retained is not None:
            paths = {p.stem: p for p in retained.iterdir() if p.suffix == ".ocr" and p.is_file() and not p.is_symlink()}
            if set(paths) != set(completed) or len(list(retained.iterdir())) != len(paths):
                failure = failure or "first_output_coverage"
            detail["first_output_sha256"] = {page: sha256(p) for page, p in paths.items()}
        if set(completed) != set(expected) or any(item["seconds"] is None for item in requests):
            failure = failure or "incomplete_coverage"
        detail.update(complete=failure is None, failure=failure, successful_pages=len(completed),
                      failed_pages=len(set(expected) - set(completed)))
    successful = set(completed)
    return {**detail, "turn": turn, "seconds": seconds, "peak_rss_bytes": peak, "completed": completed,
            "completed_by_carrier": dict(Counter(item["carrier"] for item in pages if item["id"] in successful)),
            "requests": requests, "local_fallbacks": fallbacks, "events_sha256": sha256(events),
            "api_usd": None if cost is None else str(cost), "pages_per_minute": 60 * len(completed) / seconds}


def run(args):
    begin = monotonic()
    plan, prices = load_plan(args.plan, args.plan_sha256, args.execute, args.allow_api)
    layout_reader = next((r for r in plan["readers"] if "heocr_wheel" in r), None)
    if layout_reader is not None:
        wheel = (args.plan.resolve().parent / layout_reader["heocr_wheel"]["path"]).resolve()
        check = subprocess.run([layout_reader["argv"][0], str(HERE / "registered_adapter.py"),
                                "--verify-heocr-wheel", str(wheel)],
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
        need(check.returncode == 0, "installed HeOCR differs from the pinned wheel or verifier failed")
    runtime = hardware(plan["profile"])
    setup = monotonic() - begin
    output = args.output.resolve()
    output.mkdir(mode=0o700)
    root = args.plan.resolve().parent
    results = {}
    readers = plan["readers"]
    c1 = plan.get("protocol") == c1_protocol.PROTOCOL
    runner_rate = prices["runners"][plan["profile"]]

    def trial(reader, pages, phase, turn):
        try:
            outcome = measure(reader, pages, phase, turn, root, output, prices["api"].get(reader.get("price_key")),
                              not args.execute, args.timeout_seconds, c1=c1)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"{reader['id']} {phase} {turn}: {exc}") from None
        status = outcome.get("failure") or "complete"
        print(f"{reader['id']} {phase} {turn} {status}", file=sys.stderr, flush=True)
        return outcome

    for reader in readers:
        warmup = None if c1 else trial(reader, [plan["warmup"]], "warmup", 0)
        results[reader["id"]] = {"warmup": warmup, "trials": []}
    orders = []
    schedules = c1_protocol.schedule(readers) if c1 else (
        readers[pivot:] + readers[:pivot] for pivot in (turn % len(readers) for turn in range(REPEATS))
    )
    for turn, schedule in enumerate(schedules, 1):
        orders.append([reader["id"] for reader in schedule])
        for reader in schedule:
            results[reader["id"]]["trials"].append(trial(reader, plan["pages"], "trial", turn))
    table = []
    for reader in readers:
        result = results[reader["id"]]
        fee = reader.get("mandatory_usd_per_1000", 0)
        mandatory_fee = None if fee is None else Decimal(str(fee))
        if c1:
            c1_protocol.cost_fields(result["trials"], plan["m4_rates"], mandatory_fee)
        else:
            cost_fields(result["trials"], result["warmup"], runner_rate, prices["workload_pages"], mandatory_fee)
        fields = row(reader, plan["profile"], result, not args.execute, four_repeats=c1)
        if args.execute and c1 and not all(item["complete"] for item in result["trials"]):
            fields["status"] = "registered-incomplete"
        elif args.execute and fields["fully_loaded_usd_per_1000"] is None:
            fields["status"] = "registered-cost-pending"
        table.append(fields)
        settings = ("id", "kind", "mode", "concurrency", "service_region", "price_key", "mandatory_usd_per_1000",
                    "allow_local_fallback", "ocr_layout")
        result["settings"] = {key: reader[key] for key in settings if key in reader}
        if "heocr_wheel" in reader:
            result["settings"]["heocr_wheel_sha256"] = reader["heocr_wheel"]["sha256"]
    comparators = plan.get("comparators", [])
    candidate = plan.get("candidate")
    compare = c1_protocol.paired_bounds if c1 else paired_bounds
    basis = plan.get("m4_rates") if c1 else None
    hourly = None
    if not c1:
        hourly = Decimal(runner_rate["usd"]) / Decimal(runner_rate["hours"])
    elif basis is not None:
        hourly = Decimal(basis["amortization_usd_hour"]) + Decimal(basis["power_usd_hour"])
    receipt = {
        "schema": SCHEMA, "dry_run": not args.execute, "utc_generated": datetime.now(timezone.utc).isoformat(),
        "registration": plan.get("registration"), "plan_sha256": args.plan_sha256,
        "price_sha256": plan["price_sha256"], "price_date": prices["checked"],
        "pricing": {key: value for key, value in prices.items() if key != "runners"} if c1 else prices,
        "runtime": runtime, "harness_setup_seconds": setup, "orders": orders,
        "rss_sample_seconds": SAMPLE_SECONDS, "worker_timeout_seconds": args.timeout_seconds,
        "runner_usd_hour": None if hourly is None else str(hourly),
        "confidence_interval": ("four-repeat paired Student-t bounds; no exact two-sided 95% median interval"
                                if c1 else "exact two-sided 95% median sign interval; independent ordered repeat blocks"),
        "readers": results, "table": table, "paired_efficiency_bounds": compare(candidate, comparators, results),
    }
    if c1:
        receipt.update(protocol=c1_protocol.PROTOCOL, accuracy_repeat=1, m4_client_cost=basis,
                       ordered_pages=[{k: r[k] for k in ("id", "carrier", "sha256")} for r in plan["pages"]],
                       paired_design={"candidate": candidate, "comparators": comparators, "repeats": 4,
                                      "alpha_per_bound": .025 / (2 * len(comparators)) if comparators else None},
                       worker_controls={**TESSERACT_ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                        "CUDA_VISIBLE_DEVICES": ""})
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
