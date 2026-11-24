# P-mod-open v0 · Tesseract legacy Hebrew registration

This additional comparison uses the original 65 scored rasters from 31 reference pages, 14 issue clusters and the existing pinned common mask. The scorer and frozen ten-reader replay remain version 0.1.0. Scored originals and published images can differ where the release masks emblems.

- Tesseract 5.5.3, Mach-O ARM64, executable SHA-256 d3d15b31535c2dd88cbbb53ec54a16b9a739f4146e1648b4b6ad57cf5ce79f1d.
- Combined tessdata release 4.1.0, Hebrew model SHA-256 7da6ea6b7a2620ec8e8b41de2967a13d429635a56657a0b30b622501a573d3e1. Force OEM 0 so the recognizer uses the legacy engine. PSM 3, one thread, local CPU and unmodified UTF-8 output. [Upstream model revision](https://github.com/tesseract-ocr/tessdata/tree/4767ea922bcc460e70b87b1d303ebdfed0897da8)
- Unchanged manifest SHA-256 f173417a49f960f8bdd2bf9950a9e676d2070aeb6d6a6a3b0066887af6a11190. The runner verifies the exact scored hash for every carrier before invocation. Force one attempt per carrier and score only complete submissions. The scorer reports the same filtered and all-reference CER denominators and carrier strata. Training membership of the upstream model is unknown.
- A separate real DEV fixture established Hebrew output and a 6.44-second CPU runtime. No DEV accuracy is used. Estimated full run about seven minutes and $0 incremental API or compute charges.

The reproducible invocation uses [the exact OCR-only runner](../../run_tesseract_legacy.py), the original roster manifest in Eval custody, and the published manifest. Both input and output custody are local. A fixture and this registration precede the additional held-out run. The legacy engine is an additional Tesseract-family diagnostic. The original best and fast comparisons retain their registered roles.
