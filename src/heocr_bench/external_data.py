"""Local native-format adapters for the external Hebrew OCR track."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


SETS = ("pinkas", "churro", "biblia")
RASTERS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Unit:
    id: str
    reference: str
    image: Path | bytes | None
    document_type: str = ""
    polygon: tuple[tuple[float, float], ...] = ()
    dimensions: tuple[int, int] | None = None


def local_name(element):
    return element.tag.rsplit("}", 1)[-1]


def name_from_path(name):
    return PurePosixPath(name.replace("\\", "/")).name


def listed(path):
    entries = path.read_text(encoding="utf-8-sig").splitlines()
    if not any(entry.strip() for entry in entries):
        raise ValueError("page list is empty")
    return (entry.strip() for entry in entries if entry.strip())


def local_files(root, suffixes, *, by_stem=False):
    if not root.is_dir():
        raise ValueError("input must be an extracted dataset directory")
    result = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in suffixes and path.is_file():
            key = path.stem if by_stem else path.name
            result.setdefault(key.casefold(), []).append(path)
    return result


def unique_file(index, name):
    matches = index.get(name.casefold(), [])
    if len(matches) != 1:
        raise ValueError(f"expected one local file for {name!r}, found {len(matches)}")
    return matches[0]


def text_equiv(element):
    equivalents = element.findall("{*}TextEquiv")
    for alternative in sorted(equivalents, key=lambda node: int(node.get("index", "0"))):
        for field in ("Unicode", "PlainText"):
            value = alternative.findtext(f"{{*}}{field}")
            if value and value.strip():
                return value
    return ""


def page_tokens(element, child_kind, ordered=None):
    children = element.findall(f"{{*}}{child_kind}")
    if ordered is not None:
        children.sort(key=lambda node: ordered.get(node.get("id"), len(ordered)))
    return children


def region_order(page):
    """Resolve nested PAGE ReadingOrder groups, with document order for unordered groups."""
    def refs(node):
        if node.get("regionRef"):
            yield node.attrib["regionRef"]
        children = list(node)
        if local_name(node).startswith("OrderedGroup"):
            children.sort(key=lambda child: int(child.get("index", "0")))
        for child in children:
            yield from refs(child)

    order = page.find("{*}ReadingOrder")
    return {} if order is None else {name: i for i, name in enumerate(refs(order))}


def page_text(page):
    def region_text(region):
        lines = []
        for line in page_tokens(region, "TextLine"):
            text = text_equiv(line) or " ".join(
                value for word in page_tokens(line, "Word") if (value := text_equiv(word))
            )
            if text:
                lines.append(text)
        if not lines and (text := text_equiv(region)):
            lines.append(text)
        for child in page_tokens(region, "TextRegion", order):
            lines.extend(region_text(child))
        return lines

    order = region_order(page)
    lines = []
    for region in page_tokens(page, "TextRegion", order):
        lines.extend(region_text(region))
    return "\n".join(lines) or text_equiv(page)


def pinkas(root):
    annotations = local_files(root, {".xml"}, by_stem=True)
    images = local_files(root, RASTERS)
    lists = local_files(root, {".txt"})
    for entry in listed(unique_file(lists, "test_set.txt")):
        name = name_from_path(entry)
        stem = Path(name).stem if Path(name).suffix.lower() in RASTERS | {".xml"} else name
        annotation = unique_file(annotations, stem)
        page = ET.parse(annotation).getroot().find("{*}Page")
        if page is None:
            raise ValueError(f"no PAGE Page in {annotation.name!r}")
        filename = name_from_path(page.attrib["imageFilename"])
        if Path(filename).stem.casefold() != stem.casefold():
            raise ValueError(f"PAGE carrier and test list differ for {stem!r}")
        yield Unit(stem, page_text(page), unique_file(images, filename))


def churro(path, *, with_images):
    from pyarrow.parquet import ParquetFile

    columns = ["dataset_id", "example_id", "main_language", "main_script",
               "document_type", "cleaned_transcription"]
    if with_images:
        columns.append("image")
    with ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=4, columns=columns):
            if set(batch.schema.names) != set(columns):
                raise ValueError("CHURRO Parquet is missing native columns")
            for row in batch.to_pylist():
                if (row["main_language"], row["main_script"]) != ("Hebrew", "Hebrew"):
                    continue
                image = row["image"]["bytes"] if with_images else None
                if with_images and (not isinstance(image, bytes) or not image):
                    raise ValueError("CHURRO image must contain embedded bytes")
                fields = [row[key] for key in ("dataset_id", "example_id", "document_type")]
                if any(not isinstance(value, str) or not value.strip() for value in fields):
                    raise ValueError("CHURRO row has invalid identifiers or document type")
                source, example, doc_type = fields
                yield Unit(f"{source}/{example}", row["cleaned_transcription"], image, doc_type)


def alto_polygon(line):
    shape = line.find("{*}Shape/{*}Polygon")
    if shape is not None:
        coordinates = [float(value) for value in shape.attrib["POINTS"].replace(",", " ").split()]
        points = tuple(zip(coordinates[::2], coordinates[1::2], strict=True))
    else:
        x, y, w, h = (float(line.attrib[key]) for key in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
        if w <= 0 or h <= 0:
            raise ValueError("invalid ALTO line rectangle")
        points = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    if len(set(points)) < 3 or not all(isfinite(value) for point in points for value in point):
        raise ValueError("invalid ALTO line polygon")
    return points


def biblia(root, pages):
    images = local_files(root, RASTERS)
    for entry in listed(pages):
        relative = PurePosixPath(entry)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".xml":
            raise ValueError("BiblIA page list requires relative ALTO XML paths")
        annotation = (root / relative).resolve()
        if not annotation.is_relative_to(root.resolve()):
            raise ValueError("BiblIA ALTO path escapes the extracted directory")
        alto = ET.parse(annotation).getroot()
        page = alto.find("{*}Layout/{*}Page")
        if local_name(alto) != "alto" or page is None:
            raise ValueError(f"no ALTO Page in {entry!r}")
        units = alto.findtext("{*}Description/{*}MeasurementUnit", default="pixel")
        if units.strip() != "pixel":
            raise ValueError("BiblIA ALTO coordinates must be in pixels")
        filename = alto.findtext(".//{*}sourceImageInformation/{*}fileName", default="").strip()
        if not filename or "://" in filename:
            raise ValueError(f"BiblIA page has no local carrier in {entry!r}")
        image = unique_file(images, name_from_path(filename))
        width, height = (float(page.attrib[key]) for key in ("WIDTH", "HEIGHT"))
        if not (width.is_integer() and height.is_integer() and width > 0 and height > 0):
            raise ValueError(f"invalid ALTO page dimensions in {entry!r}")
        dimensions = int(width), int(height)
        count = 0
        for line in page.findall(".//{*}TextLine"):
            reference = "".join(
                node.get("CONTENT") or " " for node in line.iter()
                if local_name(node) in {"String", "SP"}
            )
            if reference.strip():
                count += 1
                polygon = alto_polygon(line)
                yield Unit(f"{relative.as_posix()}#{line.attrib['ID']}",
                           reference, image, polygon=polygon, dimensions=dimensions)
        if not count:
            raise ValueError(f"no transcribed lines in {entry!r}")


def load_units(set_id, data, pages=None, *, with_images=True):
    if set_id == "biblia":
        if pages is None:
            raise ValueError("BiblIA requires --pages with the registered ALTO page paths")
        units = biblia(data, pages)
    elif pages is not None:
        raise ValueError("--pages is only for BiblIA")
    elif set_id == "pinkas":
        units = pinkas(data)
    elif set_id == "churro":
        units = churro(data, with_images=with_images)
    else:
        raise ValueError(f"unknown external set: {set_id}")
    seen = set()
    for unit in units:
        if unit.id in seen:
            raise ValueError(f"duplicate external unit ID: {unit.id!r}")
        if not isinstance(unit.reference, str) or not unit.reference.strip():
            raise ValueError(f"missing transcription for {unit.id!r}")
        seen.add(unit.id)
        yield unit
    if not seen:
        raise ValueError("no external units selected")
