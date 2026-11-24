"""Timing-only adapter interface for the registered efficiency runner."""

from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic_ns


class Telemetry:
    """Record completed pages and each billed transport attempt as it happens."""

    def __init__(self):
        self.job = json.loads(Path(os.environ["HEOCR_EFFICIENCY_JOB"]).read_text())
        self.pages = self.job["pages"]
        self._events = Path(os.environ["HEOCR_EFFICIENCY_EVENTS"])
        self._lock = Lock()
        self._next = itertools.count(1)

    def _emit(self, **entry):
        with self._lock, self._events.open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def save_first_output(self, page_id, serialized):
        """Persist the same serialized output only when repeat 1 requests an accuracy artifact."""
        folder = self.job.get("accuracy_output_dir")
        if folder is None:
            return
        if page_id not in {page["id"] for page in self.pages}:
            raise ValueError("output page outside trial")
        with (Path(folder) / f"{page_id}.ocr").open("xb") as stream:
            stream.write(serialized)

    def complete(self, page_id, *, local_fallback=False):
        """Call after the usable OCR output has been produced."""
        self._emit(event="page", page=page_id, local_fallback=local_fallback)

    @contextmanager
    def request(self, page_id):
        """Wrap one transport attempt. Supply usage even for a known billed error."""
        if self.job["kind"] != "api":
            raise ValueError("only an API adapter may record a request")
        if page_id not in {page["id"] for page in self.pages}:
            raise ValueError("request page outside trial")
        with self._lock:
            number = next(self._next)
        started = monotonic_ns()
        self._emit(event="request_start", request=number, page=page_id, start_ns=started)
        usage = None
        ok = False

        def record_usage(counts):
            nonlocal usage
            usage = dict(counts)

        try:
            yield record_usage
            ok = True
        finally:
            ended = monotonic_ns()
            self._emit(event="request_end", request=number, end_ns=ended, ok=ok, usage=usage)
