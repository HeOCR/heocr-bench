import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heocr_bench.data import Carrier, load_submission, read_asset, split_descriptor
from heocr_bench.metrics import FOLDS, normalize
from heocr_bench.scoring import score_submission


class ScoringTests(unittest.TestCase):
    def test_frozen_folds_keep_en_dash_distinct_from_hyphen(self):
        self.assertEqual(normalize("\u200fA\u05beB  C\u2014D \u05d0\u05b8\u05f3", FOLDS), "A-B C\u2013D \u05d0\u05b8'")

    def test_micro_cer_and_mask_preserve_registered_views(self):
        carrier = Carrier("p1", "issue-01", "scanner", ["abc", "xyz"], [True, False])
        result = score_submission({carrier.id: carrier}, ({}, {carrier.id: "abX\nxYz"}))
        self.assertAlmostEqual(result["views"]["all"]["pooled"]["page_cer"], 2 / 7)
        self.assertAlmostEqual(result["views"]["filtered"]["scanner"]["page_cer"], 1 / 3)
        self.assertEqual(result["views"]["all"]["scanner"]["line_edits"], 2)
        self.assertIsNone(result["views"]["filtered"]["scanner"]["line_error_rate"])
        self.assertIsNone(result["views"]["all"]["issuer-scan"]["page_cer"])

    def test_submission_rejects_missing_and_extra_entries(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            metadata = {"reader": "example", "version": "0", "license": "Apache-2.0", "weights": "open", "params": {}}
            (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_submission(folder, ["p1"])
            (folder / "p1.txt").write_bytes(b"")
            self.assertEqual(load_submission(folder, ["p1"]), (metadata, {"p1": ""}))
            (folder / "unexpected.txt").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Extra"):
                load_submission(folder, ["p1"])

    def test_registry_and_local_asset_require_exact_pin(self):
        descriptor = split_descriptor("p-mod-open-v0", "test")
        with TemporaryDirectory() as root:
            folder = Path(root).resolve()
            (folder / "manifest.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_asset(descriptor, folder, "manifest.json", descriptor["manifest_sha256"])
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                read_asset(descriptor, folder, "../manifest.json", descriptor["manifest_sha256"])
