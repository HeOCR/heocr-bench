"""Opt-in Vertex timing adapter. Save usage and timing aggregates only."""

from base64 import b64encode
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen

from panel import HERE

MODEL = "gemini-3.8-flash"
PRICE_FILE = HERE / "api-pricing.json"
USAGE = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")


def price():
    return json.loads(PRICE_FILE.read_text(), parse_float=Decimal)


def response_usage(response, model_version):
    if response.get("modelVersion") != model_version:
        raise ValueError("Vertex response model version differs")
    candidates = response.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") not in ("STOP", "MAX_TOKENS"):
        raise ValueError("Vertex did not return a complete response")
    usage = response["usageMetadata"]
    if usage.get("cachedContentTokenCount", 0) or usage.get("toolUsePromptTokenCount", 0):
        raise ValueError("unexpected caching or tool-use billing")
    counts = {key: usage.get(key, 0) for key in USAGE}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("invalid Vertex usage totals")
    if counts["promptTokenCount"] == 0 or sum(counts.values()) != usage.get("totalTokenCount"):
        raise ValueError("incomplete Vertex usage totals")
    return counts


def read(paths, project, prompt_file, model_version, budget):
    from quote import quote

    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project):
        raise ValueError("expected a Google Cloud project ID")
    token = os.environ["HEOCR_BENCH_VERTEX_ACCESS_TOKEN"]
    prompt = Path(prompt_file).read_text()
    if not prompt.strip():
        raise ValueError("expected an OCR instruction")
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{MODEL}:generateContent"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    prices = price()
    totals = Counter()
    completed = Counter()
    # Leave headroom for the next request. No retries are issued by this adapter.
    reserve = Decimal("0.06")
    for path, carrier in paths:
        accrued = quote(1, *(totals[key] for key in USAGE), prices) / 1000
        if accrued + reserve > budget:
            raise RuntimeError("Vertex budget stop before issuing another request")
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        payload = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": mime, "data": b64encode(path.read_bytes()).decode("ascii")}},
                {"text": prompt},
            ]}],
            "generationConfig": {
                "candidateCount": 1, "temperature": 0, "seed": 0, "maxOutputTokens": 4096,
                "thinkingConfig": {"thinkingLevel": "LOW"}, "responseMimeType": "text/plain",
            },
        }
        with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode()), timeout=90) as handle:
            response = json.load(handle)
        totals.update(response_usage(response, model_version))
        completed[carrier] += 1
    return {"completed_by_carrier": dict(completed), "vertex_usage": dict(totals)}
