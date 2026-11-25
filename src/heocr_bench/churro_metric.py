"""CHURRO Hebrew metric preprocessing, adapted from Apache-2.0 upstream.

Source: stanford-oval/Churro at 23c3ac65781443b18d87c8cdac5f77445c3319e6.
The Hebrew evaluation path omits Arabic rules, logging and BLEU dependencies.
"""

import html
import re
import unicodedata
from xml.etree import ElementTree as ET


DEFAULT_OCR_OUTPUT_TAG = "output"


def strip_ocr_output_tag(text: str, *, output_tag: str = DEFAULT_OCR_OUTPUT_TAG) -> str:
    """Remove outer OCR output tags and any stray tag tokens when present.

    :param text: Raw OCR response text.
    :param output_tag: Expected wrapper tag name.
    :returns: OCR text with the outer wrapper removed when present.
    """
    outer_wrapper_pattern = re.compile(
        rf"^\s*<{re.escape(output_tag)}>\s*(.*?)\s*</{re.escape(output_tag)}>\s*$",
        flags=re.DOTALL,
    )
    match = outer_wrapper_pattern.match(text)
    if match is not None:
        return match.group(1).strip()

    stray_tag_pattern = re.compile(rf"</?{re.escape(output_tag)}\b[^>]*>", flags=re.IGNORECASE)
    return stray_tag_pattern.sub("", text).strip()


def strip_rich_ocr_markup_to_plain_text(text: str) -> str:
    """Best-effort plain-text conversion for OCR markdown/HTML output."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"!\[[^\]]*]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)

    html_replacements = (
        (r"(?is)<\s*input\b[^>]*\bchecked(?:=(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?[^>]*>", "[x]"),
        (r"(?is)<\s*input\b[^>]*>", "[ ]"),
        (r"(?i)<\s*br\s*/?\s*>", "\n"),
        (r"(?i)<\s*hr\s*/?\s*>", "\n"),
        (r"(?i)</\s*(?:p|div|h[1-6]|ul|ol|table|tr|li|pre|caption)\s*>", "\n"),
        (r"(?i)</\s*(?:td|th)\s*>", " | "),
        (r"(?i)<\s*li\b[^>]*>", ""),
        (
            r"(?i)</?\s*(?:table|thead|tbody|tfoot|tr|td|th|p|div|span|h[1-6]|ul|ol|"
            r"strong|em|b|i|u|sup|sub|code|pre|a|math|chem|caption|small|big)\b[^>]*>",
            "",
        ),
        (r"(?i)<\s*img\b[^>]*>", ""),
    )
    for pattern, replacement in html_replacements:
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"(?is)</?[a-z][^>]*>", "", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-+*]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s?", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    normalized_lines: list[str] = []
    saw_content = False
    for raw_line in cleaned.splitlines():
        line = re.sub(r"\s*\|\s*$", "", raw_line.strip())
        line = re.sub(r"^\|\s*", "", line)
        line = re.sub(r"\s*\|\s*", " | ", line)
        if line:
            normalized_lines.append(line)
            saw_content = True
            continue
        if saw_content and normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")
    return "\n".join(normalized_lines).strip()


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _remove_tag(xml_content: str, tag_name: str) -> str:
    if f"<{tag_name}" not in xml_content:
        return xml_content
    return re.sub(
        rf"<{tag_name}\b[^>]*/>",
        "",
        re.sub(rf"<{tag_name}\b[^>]*>.*?</{tag_name}>", "", xml_content, flags=re.DOTALL),
    )


def extract_actual_text_from_xml(xml_content: str) -> str:
    """Extract text from HistoricalDocument XML, or return the raw input when not XML."""
    if "HistoricalDocument" not in xml_content:
        return xml_content

    for tag_name in ("Description", "Deletion", "Illegible", "Gap"):
        xml_content = _remove_tag(xml_content, tag_name)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return ""

    page_texts: list[str] = []
    for page in root.iter():
        if _local_name(page.tag) != "Page":
            continue
        section_texts: list[str] = []
        for child in page.iter():
            if _local_name(child.tag) not in {"Header", "Body", "Footer"}:
                continue
            lines = [line.strip() for line in child.itertext() if line.strip()]
            if lines:
                section_texts.append("\n".join(lines))
        if section_texts:
            page_texts.append("\n".join(section_texts))

    return "\n\n".join(page_texts).strip()


SUBSTITUTIONS = {
    "\ueada": "st",
    "\ueec5": "ct",
    "\ueba6": "ss",
    "\ueba2": "si",
    "\ueba7": "ssi",
    "\ueba3": "sl",
    "’": "'",
    "¬": "-",
}
SUBSTITUTION_PATTERN = re.compile("|".join(map(re.escape, SUBSTITUTIONS.keys())))


def normalize_characters(text: str, *, keep_long_s: bool = True) -> str:
    """Replace document-specific glyphs with normalized equivalents."""
    text = re.sub(r"(?<=\d)(?=[↉½⅓¼⅕⅙⅐⅛⅑⅒⅔⅖¾⅗⅜⅘⅚⅞])", " ", text)

    placeholder = "\ue000"
    if keep_long_s:
        text = text.replace("ſ", placeholder)

    text = unicodedata.normalize("NFKC", text)

    if keep_long_s:
        text = text.replace(placeholder, "ſ")

    return re.sub(
        r"(^|\s)~(?=\w)",
        r"\1",
        SUBSTITUTION_PATTERN.sub(lambda match: SUBSTITUTIONS[match.group(0)], text),
    )


def normalize_text_for_evaluation(text: str) -> str:
    """Normalize raw OCR text before metric computation."""
    text = strip_rich_ocr_markup_to_plain_text(text)

    text = text.lower()
    text = re.sub(r"[*_`~#]", "", text)
    text = re.sub(r"[–—−‑‒―‐]", "-", text)
    text = re.sub(r"\[figure\s+\d+\]", "", text)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = text.strip("-")
    return re.sub(r"\s+", " ", normalize_characters(text, keep_long_s=False)).strip()


def normalize_churro(text):
    """Apply the published Hebrew scorer preprocessing to plain or marked-up text."""
    return normalize_text_for_evaluation(extract_actual_text_from_xml(strip_ocr_output_tag(text)))
