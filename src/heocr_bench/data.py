"""Read pinned public references and validate a submission's exact carrier roster."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from urllib.request import urlopen

from .metrics import PROFILE_SHA, STRATA


ID = re.compile(r"[A-Za-z0-9_-]+")
PLAIN = "plain"
CELLS = "heocr-cell-delimited"


@dataclass(frozen=True)
class Carrier:
    id: str
    issue: str
    stratum: str
    lines: list[str]
    retained: list[bool]


def split_descriptor(set_id, split):
    registry = json.loads(files("heocr_bench").joinpath("registry.json").read_bytes())
    try:
        entry = registry["sets"][set_id]["splits"][split]
    except KeyError as exc:
        raise ValueError(f"Unknown set or split: {set_id}/{split}") from exc
    if entry["reference_access"] != "public":
        raise ValueError(f"{set_id}/{split} requires server-side scoring")
    if (entry["protocol"], entry["normalization"], entry["normalization_source_sha256"]) != (
        "page-cer-v1", "line_fold_v1", PROFILE_SHA
    ):
        raise ValueError("Unsupported registered scoring protocol")
    return entry


def read_asset(descriptor, root, name, expected_sha):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError("Unsafe dataset asset path")
    if root is None:
        dataset = descriptor["dataset"]
        revision = quote(descriptor["revision"], safe="")
        uri = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{quote(name)}"
        with urlopen(uri, timeout=30) as response:
            raw = response.read()
    else:
        local = (root / relative).resolve(strict=True)
        if not local.is_relative_to(root):
            raise ValueError("Dataset asset leaves the supplied directory")
        raw = local.read_bytes()
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"SHA-256 mismatch: {name}")
    return raw


def load_carriers(descriptor, data_dir=None):
    root = data_dir.resolve(strict=True) if data_dir is not None else None
    pinned = read_asset(descriptor, root, "manifest.json", descriptor["manifest_sha256"])
    manifest = json.loads(pinned)
    if (manifest["schema"] != "heocr.pmod-open-v0.release-manifest.v1"
            or len(manifest["pages"]) != descriptor["pages"]):
        raise ValueError("Unsupported or incomplete set manifest")
    carriers = {}
    for page in manifest["pages"]:
        reference = page["reference"]
        mask = page["mask"]
        lines = read_asset(descriptor, root, reference["path"], reference["sha256"]).decode("utf-8").splitlines()
        flags = json.loads(read_asset(descriptor, root, mask["path"], mask["sha256"]))
        retained = flags["retained_positions"]
        if (flags["id"] != page["id"] or len(lines) != reference["line_count"] or len(lines) != len(retained)
                or any(type(flag) is not bool for flag in retained)):
            raise ValueError(f"Reference and mask mismatch on {page['id']}")
        for member in page["carriers"]:
            key, stratum = member["id"], member["stratum"]
            if not ID.fullmatch(key) or key in carriers or stratum not in STRATA:
                raise ValueError("Invalid or duplicate manifest carrier")
            carriers[key] = Carrier(key, page["issue_cluster"], stratum, lines, retained)
    if len(carriers) != descriptor["carriers"]:
        raise ValueError("Incomplete carrier roster")
    return carriers


def load_submission(directory: Path, carriers):
    root = directory.resolve(strict=True)
    files_by_name = {path.name: path for path in root.iterdir()}
    expected = {f"{key}.txt" for key in carriers} | {"metadata.json"}
    missing = expected - files_by_name.keys()
    extra = files_by_name.keys() - expected
    if missing or extra:
        raise ValueError(f"Submission roster differs. Missing {sorted(missing)}. Extra {sorted(extra)}")
    if any(path.is_symlink() or not path.is_file() for path in files_by_name.values()):
        raise ValueError("Submission entries must be regular files")
    metadata = json.loads(files_by_name["metadata.json"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Submission metadata must be a JSON object")
    for key in ("reader", "version", "license"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Expected nonempty metadata field: {key}")
    if metadata.get("weights") not in ("open", "closed") or not isinstance(metadata.get("params"), dict):
        raise ValueError("Metadata must declare open or closed weights and a params object")
    if metadata.get("text_format", PLAIN) not in (PLAIN, CELLS):
        raise ValueError("Unrecognized prediction text format")
    predictions = {key: files_by_name[f"{key}.txt"].read_bytes().decode("utf-8") for key in carriers}
    return metadata, predictions
