# Registered efficiency runs

This harness implements the timing and cost portion of decision 0030. It
drives the registered readers on the same ordered image panel, records
performance separately from accuracy, and exports JSON and table rows. The
existing v0 receipts and exporter are still available.

## Freeze the plan

Use a private directory for the plan and the receipt. Paths are relative to
the plan file or absolute. Hash the image files, adapter, model weights and
other effective inputs. For a registered evaluation, write the plan's SHA-256
and this code revision into the preregistration. Freeze each reader's model,
operating mode, runtime, retry policy, concurrency, region and output
configuration there as well. The first measured repeat can write the OCR files
for the single accuracy scoring pass. Have the adapter save those files in
Eval's private output directory only when `phase == "trial"` and `turn == 1`.

The profiles are `m4-cpu` (native Apple M4 Max, 16 cores, 128 GiB) and
`github-ubuntu-cpu` (GitHub Actions `ubuntu-24.04` x64, pinned to two logical
CPUs for the whole child process tree). Keep all comparisons on the same
runner allocation. The receipt records OS, CPU, Python and runner image
version. Local adapters declare one or two workers with one thread each.
Pin CPU-only dependencies and request settings in the adapter. The generic
runner measures native workers and their descendants and validates observed
API concurrency.

The plan uses this shape. The array of reader entries is the exact frozen
roster, including comparators. Set `mode` to an identifier for the effective
execution settings. Every page and warm-up has the same shape.

```json
{
  "purpose": "dev",
  "profile": "m4-cpu",
  "price_sha256": "<hash of registered-prices-2026-11-24.json>",
  "warmup": {"purpose": "dev", "id": "dev-1", "carrier": "scanner",
             "path": "dev-1.png", "sha256": "<image hash>", "egress": false},
  "pages": [
    {"id": "dev-2", "carrier": "scanner", "path": "dev-2.png",
     "sha256": "<image hash>", "egress": false}
  ],
  "readers": [
    {"id": "candidate", "kind": "local", "mode": "cpu", "concurrency": 1,
     "argv": ["python", "/path/to/adapter.py"],
     "pins": [{"path": "/path/to/adapter.py", "sha256": "<adapter hash>"}]},
    {"id": "haiku", "kind": "api", "mode": "synchronous", "concurrency": 1,
     "argv": ["python", "/path/to/hosted_adapter.py"],
     "pins": [{"path": "/path/to/hosted_adapter.py", "sha256": "<adapter hash>"}],
     "service_region": "global", "price_key": "haiku-5-5-standard"}
  ],
  "candidate": "candidate",
  "comparators": ["haiku"]
}
```

Run this first on DEV. The dry-run substitutes an internal mock for every
hosted adapter and can exercise the real local adapter. It records its output
as DEV. The CI job uses an offline fixture on the actual Ubuntu profile.

```sh
python -m pip install psutil==7.2.2
python efficiency/registered_run.py --plan /private/dev/plan.json \
  --plan-sha256 "$DEV_PLAN_SHA256" --output /private/dev/efficiency
python efficiency/report.py --registered-run /private/dev/efficiency/receipt.json --allow-dry-run
```

For the registered OPEN run, set `purpose` to `registered-open`, supply the
preregistration ID in `registration`, and pin at least the executable adapter
and its effective assets under `pins`. Use the same command shape with
`--execute`. Add `--allow-api` only for approved outbound work. Every image,
including the separate DEV warm-up, needs a recorded egress approval in the
plan whenever a live hosted adapter is present. Retain the new output directory
and its JSONL logs with the registration. An interrupted run keeps its
partial logs and has no final receipt. Decide any rerun under the
preregistration's interruption policy.

## Instrument reader adapters

The runner passes a private job JSON in `HEOCR_EFFICIENCY_JOB` and a JSONL
destination in `HEOCR_EFFICIENCY_EVENTS`. Put `efficiency/` on the adapter's
Python path and use `telemetry.Telemetry`. Use exactly one recorder in the
adapter process; it is thread-safe. `pages` has each page's ID, carrier and
resolved raster path. `job` also has the phase, repeat number, registered
concurrency and expected billed categories.

```python
from telemetry import Telemetry

telemetry = Telemetry()
for page in telemetry.pages:
    # Place this at the transport boundary of the registered API reader.
    with telemetry.request(page["id"]) as record_usage:
        response = send_one_attempt(page["path"])
        record_usage(provider_usage(response))
    write_usable_output(page["id"], response)
    telemetry.complete(page["id"])
```

Keep existing provider payloads and result processing in the scoring adapter.
The timing hook surrounds one synchronous request through response arrival.
Disable hidden SDK retries and wrap each actual attempt. If a failed attempt
was billed, report its actual counts before letting the error propagate out
of the timing context. An explicitly known zero charge uses zero counts for
every billed category. A request without usage leaves cost pending. Usage
uses actual provider fields, with output and reasoning separated for Vertex,
output inclusive of thinking for Haiku, and completed operation units for
Vision and Document AI. The registered SKUs use uncached standard requests.
Batch API timing requires a separately registered transport and pricing
policy. The hook emits timing, status and usage, with OCR contents, response
bodies, paths and credentials kept in the adapter only. The built-in
`registered_adapter.py` reuses the packaged native Tesseract, Paddle CPU and
router v2 implementations. Pass `--reader ID --assets DIRECTORY` with the
layout used by `run.py`; for router v2, use the pinned model directory.

An explicitly registered hybrid adapter can set `allow_local_fallback: true`
and report `telemetry.complete(page_id, local_fallback=True)` on pages it
completed locally. The receipt counts those pages and charges all attempted
requests. A hosted-only comparator requires a successful hosted request for
every completed page.

## Timing, bounds and cost

One separate DEV page is run per reader before timing. The nine measured
repeats cycle reader order by one position per repeat. Each starts a fresh
process, runs the same panel completely and measures launch through usable
output, including image I/O, model initialization and postprocessing. The
receipt retains warm-up elapsed, page counts, usage and hash/plan validation
time separately. RSS is peak summed resident memory of the process tree,
sampled every 20 ms. Hosted RSS measures the client only. The sampler and
adapter logs stay on the client; provider latency and network time appear in
the end-to-end hosted figure.

The table reports the median and the exact distribution-free 95% interval
for pages/min and peak RSS from nine repeat measurements (second and eighth
ordered values, with 96.094% nominal coverage under independent repeat
blocks). The receipt has a one-sided throughput lower bound and cost upper
bound based on ratios paired by repeat. Bonferroni uses 5% divided across
both measures and the prespecified comparators. When nine repeats cannot
support a finite bound, the receipt leaves the upper bound blank and uses
zero for the lower bound. These are efficiency bounds. Eval performs the
paired source-cluster CER test and ties/eligibility decisions separately.

[Pricing snapshot](registered-prices-2026-11-24.json) covers USD list prices
checked on 2026-11-24. The Ubuntu profile uses $0.36 per runner-hour on the
paid 2-vCPU rate, with minute rounding at the modeled 1,000-page job size.
M4 uses $559 per 720 allocated hours at the published monthly rate for the
same CPU and RAM. That is a dedicated rental estimate with continuous use.
Each repeat projects to a 1,000-page deployment. Fully loaded cost includes
the once-per-deployment warm-up (runner and API expense), all actual request
attempts, timed compute and any declared mandatory fees per 1,000 pages.
Prices exclude credits and taxes. Asset installation and panel verification
are outside the timed deployment. Preregister any additional mandatory
provisioning or provider fees. Costs stay blank if usage is missing.
[GitHub paid runner rates](https://docs.github.com/en/billing/reference/actions-runner-pricing)
[M4 Max monthly rental rate](https://bookingmac.com/mac-studio/m4-max)

The output is `receipt.json`, schema `heocr.bench.registered-efficiency.v2`,
with per-reader warm-up and trials, request timing and usage, table columns,
and the ordered comparison bounds. Export a registered receipt with
`python efficiency/report.py --registered-run /private/run/receipt.json`.

## Four-repeat C1 mode

Set `"protocol": "c1-740"`, `"profile": "m4-cpu"`, and omit `warmup`. The
legacy plan shape above continues to select the nine-repeat protocol. The
C1 mode keeps the plan and input hashes, per-lane event logs, 20 ms RSS
sampling, and CSV exporter. It processes the same ordered page list in each
lane. It runs the ASCII-sorted arm order, its reverse, the first order
rotated left by half the arm count rounded down, and that order's reverse.
Every lane starts in a new process. Use two local page slots and at most two
hosted request slots. The built-in CPU adapters submit pages in manifest order
and serialize each result before marking it complete.

List the eligible arms under `comparators` and identify the registered
`candidate`. Each paired bound uses one-sided alpha `0.025 / (2K)`, where
`K` is the size of that list. The lower Student-t bound is computed on log
throughput ratios and exponentiated. The upper bound uses raw cost ratios.
Both use four paired observations and three degrees of freedom. The table
shows medians without a distribution-free 95% median interval. A failed
or missing page leaves the affected trial incomplete. The receipt keeps its
success and failure counts and raw telemetry, and the corresponding paired
bounds are blank. Zero comparator cost leaves the cost leg blank. Two lanes
with zero marginal cash costs are compared on speed.

### M4 cost evidence

Supply the acquisition and power tariffs in USD with exact local evidence
pins. The structure is as follows. The amounts below are invented examples.

```json
{
  "acquisition_usd": "5760",
  "mean_watts": "80",
  "electricity_usd_per_kwh": "0.20",
  "evidence": {
    "acquisition": {"path": "purchase.json", "sha256": "<digest>"},
    "power": {"path": "power.json", "sha256": "<digest>"},
    "electricity": {"path": "tariff.json", "sha256": "<digest>"}
  }
}
```

Place this object in `m4_cost`. Add an `exchange_rate` evidence pin when
an acquisition value is converted to USD. The receipt records the hashes,
purchase cost divided by `36 * 160` provisioned hours, and watts converted
to kW at the pinned electricity rate. Trial costs charge measured client
wall time once and scale to 1,000 successful pages. Client acquisition,
power and marginal request costs are separate in the receipt. Required
recurring fees at the registered workload use `mandatory_usd_per_1000`;
use `null` when the applicable fee is unresolved. Omitting `m4_cost` keeps
timing available and leaves complete cost blank. Keep setup evidence and
one-time setup costs separately. Adapters report the billed SKU units,
including applicable rounding and minimums, and send usage for billed
retries through the existing telemetry hook.

### Same-build layout comparison and output custody

Use two local readers with `mode: "router-v2"`, the same built-in adapter
invocation, asset pins and concurrency, and the same `heocr_wheel` file pin.
Give them distinct arm IDs and set `ocr_layout` to `"off"` and `"v1"`. The
runner verifies the imported package bytes against the wheel before timing.
Use an installed wheel from the #208 build for both arms. The adapter uses
the existing capture and model rules with the layout option forwarded
per page. These diagnostic arms stay outside the eligible comparison roster
in a registered C1 execution.

For an approved registered run, the job provides `accuracy_output_dir`
on repeat 1 for the accuracy lanes. After validating and serializing each
OCR result, call `telemetry.save_first_output(page_id, serialized_bytes)`
before `telemetry.complete(page_id)`. The adapter still validates and
serializes the same result on each repeat. The receipt pins the first
artifacts and retains timing, RSS and request usage for later repeats.
DEV mode uses the mock for hosted lanes and emits no OCR payloads. The
actual production deployment, adapters, pricing evidence, and input roster
belong to the independently reviewed execution freeze.
