# C1 v0 comparator results

These rows use the same 65 original rasters, 31 reference pages and 14 issue clusters as the [frozen public v0 release](https://huggingface.co/datasets/HeOCR/p-mod-open-v0/tree/54d637d0f7f327597d67e4c474babec9586db33e). The release manifest is `f173417a49f960f8bdd2bf9950a9e676d2070aeb6d6a6a3b0066887af6a11190`. Both views use heocr-bench 0.1.0 and the existing `line_fold_v1` profile. Each new JSON report includes the pooled and carrier-specific counts. In the published image set some state emblems are masked. These additional local scores use the exact scored originals. Upstream training membership remains unverified.

## Local comparators

| Comparator | v0 disposition |
| --- | --- |
| Tessdata Best Hebrew | Complete, 65 carriers, original replay. Filtered CER 15.95%. |
| Tessdata Fast Hebrew | Complete, 65 carriers, original replay. Filtered CER 15.74%. |
| Rivok Tesseract Full | Complete, 65 carriers, original replay. Filtered CER 14.80%. It participated in part of reference construction. |
| Rivok Tesseract Lite | Complete, 65 carriers, original replay. Filtered CER 15.44%. |
| Rivok Paddle flagship | Complete, 65 carriers on CUDA, original replay. Filtered CER 17.85%. It participated in part of reference construction. |
| Kraken 7.2 with PP-OCRv6 medium | Complete, 65 carriers, CPU. Filtered CER 79.06%, all-reference CER 78.64%. [Aggregate](kraken-7.2.json) · [Receipt](kraken-7.2-receipt.json). |
| Tesseract legacy Hebrew | Complete, 65 carriers, CPU diagnostic. Filtered CER 24.35%, all-reference CER 28.55%. [Registration](tesseract-legacy-registration.md) · [Aggregate](tesseract-legacy.json) · [Receipt](tesseract-legacy-receipt.json). |
| Qwen3-VL-2B-Instruct MLX 4-bit | Registered local vision-language diagnostic, execution pending aggregate. [Weights](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit/tree/9c4f5209e57b31f4b9dfba735de3fb983739c9cc). |

The table lists external readers. The release also includes five project lanes in the ten-reader replay and a separate HeOCR scanner-router subset. Existing external rows were reused. The additional local runs incurred $0 in incremental compute charges and needed no provider account.

## Eligibility gaps

- EasyOCR v1.7.2 lacks a registered Hebrew recognition model. Stock docTR/OnnxTR lacks a verified published checkpoint with Hebrew decoder classes. No score. [EasyOCR language registry](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/config.py) · [docTR stock PARSeq](https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/recognition/parseq/pytorch.py) · [Experimental OnnxTR decoder configuration](https://huggingface.co/Felix92/onnxtr-parseq-multilingual-v1/blob/7d7329948dbcff02637a3eb9c14ae0c4440c5ed0/config.json)
- Surya and ABBYY FineReader need written clearance for the pinned engine and public disclosure. Surya weight terms vary by release. No score. [Surya code and model terms](https://github.com/datalab-to/surya) · [ABBYY Hebrew specifications](https://support.abbyy.com/hc/en-us/articles/360003319260-FineReader-Engine-12-Specifications)
- Stock PaddleOCR's published PP-OCRv6 language roster does not list Hebrew. The fitted Rivok recognizer and the multilingual Kraken checkpoint have separate registered rows. PaddleOCR-VL and other multilingual OCR/VLM families require a pinned, rights-cleared Hebrew page adapter and qualification before they can be added. [PaddleOCR release documentation](https://www.paddleocr.ai/main/en/index.html) · [PaddleOCR-VL model](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- Google Cloud Vision and Enterprise Document OCR, Azure Document Intelligence Read and Amazon Textract DetectDocumentText are the conditional C1 vendor OCR roster. Full-roster v0 results remain pending. Azure has an account-access hold. [C1 comparator inventory](https://github.com/HeOCR/heocr-engine/blob/930748d4efe9ece7043664ef07d440b4ab361088/docs/comparators.md)
- Gemini 3.8 Flash has a separate sanitized diagnostic run on 56 original-equivalent images and 9 masked derivatives. The Gemini family participated in some v0 reference construction. Claude Haiku and GPT vision are additional diagnostic families and have no v0 result in this record. The vision-language diagnostic stratum does not determine the registered launch gate.

The public v0 replay has HeOCR Modern 1 at 14.97% and Rivok Full at 14.80% on the filtered pooled view. The two additional local readers do not change the leading external row. The registered C1 launch gate requires paired comparison with the eligible roster, and the vendor and permission gaps remain open. The historical ten-reader replay retains the same masks and denominators.
