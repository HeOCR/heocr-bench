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
  free. The external Ubuntu job and the private router job have different
  CPU allocations, which the receipts record. Use measured hours with your
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
- Gemini uses Vertex global standard pricing. The November 2026 list price
  is $0.75 per million input tokens and $3.75 per million output tokens
  including reasoning, through December 31. List price per 1,000 pages
  equals `(0.75 × input + 3.75 × (output + reasoning)) / (1,000 × pages)`
  for total measured tokens and completed pages. A `list-price-only` row
  has no matched usage, wall time or client RSS receipt. Empty fields denote
  unavailable or inapplicable measurements. [Vertex pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)

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

For Gemini, project the official list rate from a matched Vertex
`usageMetadata` receipt with `efficiency/quote.py`. Supply aggregate prompt,
candidate and thought tokens separately and exclude cached-token pricing
unless the request policy includes caching.
