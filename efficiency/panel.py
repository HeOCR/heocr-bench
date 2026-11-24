"""Fetch and validate the image-only public v0 snapshot and open reader assets."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import re
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
DATASET_REVISION = "54d637d0f7f327597d67e4c474babec9586db33e"
DATASET = f"https://huggingface.co/datasets/HeOCR/p-mod-open-v0/resolve/{DATASET_REVISION}"
READERS = json.loads((HERE / "readers.json").read_text())
LEADERBOARD_REVISION = json.loads((HERE / "leaderboard.json").read_text())["leaderboard_revision"]
IMAGE_PATH = re.compile(r"images/pmod-\d{3}-(archive|issuer-render|phone-a|phone-b|scanner)\.(png|jpg)")


def sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def panel():
    rows = []
    for line in (HERE / "images.sha256").read_text().splitlines():
        digest, name = line.split()
        match = IMAGE_PATH.fullmatch(name)
        if not match or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("unexpected entry in the public image manifest")
        rows.append((name, digest, match[1]))
    if len(rows) != 65 or len({name for name, _, _ in rows}) != 65:
        raise ValueError("expected 65 unique public carriers")
    return rows


def fetch(item):
    path, digest, url = item
    if path.is_file() and sha256(path) == digest:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as response, pending.open("wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
        if sha256(pending) != digest:
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def assets(root, readers):
    return {
        (root / "assets" / asset["path"], asset["sha256"], asset["url"])
        for name in readers
        for asset in READERS[name].get("assets", [])
    }


def prepare(root, readers, download):
    inputs = {(root / name, digest, f"{DATASET}/{name}") for name, digest, _ in panel()}
    needed = inputs | assets(root, readers)
    if download:
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(fetch, needed))
    for path, digest, _ in needed:
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"missing or changed file {path}, use --download for public inputs")
