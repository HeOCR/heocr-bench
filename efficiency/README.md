# v0 efficiency

This table measures end-to-end OCR resource use on the public v0 image bytes.
It is independent of the accuracy scorer. The public dataset card describes the
release-only mask on nine carriers. The accuracy replay used its registered
inputs. Read efficiency and accuracy with those input scopes in view.
[Public v0 dataset card](https://huggingface.co/datasets/HeOCR/p-mod-open-v0/blob/54d637d0f7f327597d67e4c474babec9586db33e/README.md)

- [Efficiency table](efficiency.csv) covers the pinned public leaderboard
  roster and the opt-in HeOCR router v2.
- [External-only table](external.csv) uses the same columns keyed by
  `reader_id` and `platform`. Use it for leaderboard joins.
- [Receipts](receipts/) record the hardware, runtimes, trials and complete
  counts by capture. [Image hashes](images.sha256) pin all 65 carriers from
  the public release. [Leaderboard roster](leaderboard.json)
  and [reader settings](readers.json) pin the reader IDs, model assets
  and options for this run.

The two-worker router sustained 20.15 pages/min on M4 and 12.59 on GitHub
Ubuntu. Standalone Rivok Full sustained 60.44 and 57.82 pages/min on the same
images. The M4 and router rows each use one pass. The Ubuntu external rows
use the median of three passes.

## Measurement and price

- A page is one carrier image. All available local readers process all 65
  images, including the five archive images. Timing covers worker startup,
  image loading, recognition and normal reader output generation. It excludes
  dependency installation, dataset and model fetches, checksum verification,
  and accuracy scoring. The worker does not retain OCR text.
- Pages/min is 60 times completed pages divided by median elapsed seconds.
  Peak RSS is the highest 20 ms sample sum for the worker and its descendants,
  in MiB. It includes shared resident memory in each process using it. Hosted
  readers can expose client RSS only. The CSV states the repeat count.
- Local readers have zero marginal API charges. Runner hours per 1,000 pages
  are measured wall time projected linearly. The M4 has no hourly rental
  price in this table. For Ubuntu the dollar estimate uses the standard
  2-core Linux rate of $0.36/h as a common proxy. Public standard jobs are
  free. The external Ubuntu job had four vCPUs and the router job had two.
  Use measured hours with your
  deployment's own rates. [GitHub runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing)
  [GitHub runner hardware](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- The Paddle CPU row times the pinned flagship wrapper and model with
  ONNX Runtime's CPU provider. The leaderboard's accuracy row used CUDA.
  CPU timing is a separate execution variant. The Tesseract rows use
  the same model hashes and recognition flags with native Tesseract
  builds. The receipts identify each executable.
- Router v2 declares phone A as iPhone and phone B as Pixel. It uses modern
  v1 on phones, Rivok Full with the pinned v6 line reader on scans and
  issuer renders, and stock Rivok Full on archive. The benchmark uses two
  independent image workers and serializes complete records. The exact
  HeOCR commit and model digests are recorded.
## API timing from registered runs

**from registered run; network-bound, not comparable to local CPU**

API readers have one `registered-api` row each. The API timing columns use
elapsed time or request latency recorded during the registered run. Batch
wall time, when available, includes the provider queue. It yields batch
throughput, with per-request latency reported separately when recorded. The
timed-page count and scope describe the measurements as recorded. A blank
timing or RSS field means the receipt lacks that measurement.

List cost per 1,000 pages is recorded token or unit usage times the matching
published rate, divided by the number of completed pages and scaled to 1,000.
The Google OCR rates use the first paid tier before free quotas and credits.
The Haiku rates follow the Message Batches SKU. The Gemini rate includes
reasoning tokens at the output rate. Unit and token rates are beside the
projected page cost. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
[Vision pricing](https://cloud.google.com/vision/pricing)
[Document AI pricing](https://cloud.google.com/document-ai/pricing)
[Vertex pricing](https://cloud.google.com/model-garden/pricing)

The [aggregate API receipts](receipts/registered-api.json) cite the
registered run and pin the source receipt. They contain usage, timing and
coverage only. The exporter reads these files offline.

## Reproduce

Use Python 3.12 in a clean environment with native Tesseract on the PATH.
The public Ubuntu workflow installs the readers and downloads assets from
their pinned sources. On your own CPU, for example

```sh
python -m pip install -r efficiency/requirements.txt -r efficiency/requirements-paddle.txt
python efficiency/run.py ./v0-images --download --repeat 3 > m4-cpu.json
```

For router v2, install HeOCR from the commit in the router receipt. Run with
`--reader router-v2 --models /path/to/local/model-directory --heocr-revision COMMIT`.
The directory needs `heb_rivok.traineddata`, `heocr_modern_v1.traineddata`
and `v6.traineddata` from authorized custody. The runner verifies each model
before timing. GPU use is disabled for both the Tesseract and Paddle CPU
routes. To generate both CSVs from verified receipts

```sh
python efficiency/report.py > efficiency/efficiency.csv
python efficiency/report.py --external-only > efficiency/external.csv
```

## Decision 0030 registered runs

Use the [registered efficiency harness](REGISTERED.md) for the v1 one-shot
efficiency receipt. The workflow, cost basis, adapter hooks and table
export are documented separately from this historical v0 table.
