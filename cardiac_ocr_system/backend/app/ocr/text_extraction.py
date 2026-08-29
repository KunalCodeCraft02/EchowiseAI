"""
STAGE 1 -- GEOMETRIC TABLE RECONSTRUCTION
=========================================
Turns PaddleOCR's raw per-line output (4-point bounding polygon + text + confidence) into a
correctly-ordered, confidence-tagged representation BEFORE any semantic matching happens.

WHY THIS EXISTS
---------------
The previous pipeline threw the geometry away and flattened OCR into a flat list of text lines.
On a dense multi-column echo report that destroys the label->value association: "IVSd" and
"1.0 cm" end up as unrelated entries, and a value from the row BELOW can be paired with the
label above it. No amount of prompting fixes that downstream, because by then the information
is genuinely gone. So structure is recovered here, from pixel coordinates, and the semantic
layer (Stage 2) is only ever shown text it cannot mis-associate.

PIPELINE
--------
  raw paddle output
    -> normalize_paddle_output()   polygon -> left/top/width/height, confidence -> 0-100
    -> raw_text                    EVERY line, unfiltered, in reading order (preview only)
    -> _filter_lines()             drop conf < OCR_CONFIDENCE_REJECT_THRESHOLD, drop
                                   punctuation-only noise
    -> cluster_rows()              explicit vertical-overlap clustering (NOT Paddle's own
                                   reading order, which is exactly what breaks on 3 columns)
    -> _split_table_and_narrative()
    -> cluster_columns()           left-edge gap clustering (> OCR_COLUMN_GAP_PX starts a column)
    -> build_grid()                (row, column) cells, text joined left-to-right, conf averaged

CONFIDENCE SCALE
----------------
PaddleOCR returns 0-1. It is converted to 0-100 here, at the ONE place raw Paddle output enters
the system, so every threshold in config.py and every confidence downstream is on one scale.
"""
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

from app.config import (
    OCR_COLUMN_GAP_PX,
    OCR_COLUMN_GAP_PX_EXPLICIT,
    OCR_COLUMN_GAP_RATIO,
    OCR_CONFIDENCE_FLAG_THRESHOLD,
    OCR_DET_LIMIT_SIDE_LEN,
    OCR_DET_UNCLIP_RATIO,
    OCR_DROP_SCORE,
    OCR_MIN_LONG_SIDE,
    OCR_CONFIDENCE_REJECT_THRESHOLD,
    OCR_ROW_OVERLAP_RATIO,
)

# A line consisting only of punctuation/brackets is never a valid label or value -- it is table
# rule-line noise or a stray glyph from a blank cell. Dropping it here prevents it from ever
# becoming a "value" for the row it happens to sit in.
_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

# A row whose cells are long running text is prose that happens to be split into two boxes, not
# a two-column table row. Used to keep Conclusion/Impression paragraphs out of the grid.
_NARRATIVE_WORDS_PER_CELL = 8


class OCRLine(dict):
    """A single OCR line with its geometry. A plain dict subclass so it stays JSON-serializable
    (it ends up in quality_info / debug payloads) while allowing attribute-free dict access:
      {"text", "confidence" (0-100), "left", "top", "width", "height", "polygon"}
    """

    @property
    def bottom(self) -> float:
        return self["top"] + self["height"]

    @property
    def right(self) -> float:
        return self["left"] + self["width"]


def _poly_bounds(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """left/top/width/height from a 4-point polygon's min/max x and y.

    Uses min/max rather than assuming point order: PaddleOCR emits points clockwise from the
    top-left, but a rotated/deskewed box has no axis-aligned "first" corner, and an angle-
    classified line can come back in a different order entirely.
    """
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    left, top = min(xs), min(ys)
    return left, top, max(xs) - left, max(ys) - top


def _to_percent(conf: float, scale: str = "auto") -> float:
    """Normalize a confidence onto the project-wide 0-100 scale.

    scale="auto"  -- values <= 1.0 are assumed to be on PaddleOCR's native 0-1 scale.
    scale="0-100" -- passed through unchanged. Required for Tesseract, whose conf IS 0-100:
                     under "auto" a genuine 1% confidence would be read as 100%, promoting the
                     single worst-read token on the page to fully trusted.
    """
    conf = float(conf)
    if scale == "0-100":
        return conf
    return conf * 100.0 if conf <= 1.0 else conf


def _looks_like_page(item) -> bool:
    """True when `item` is a list of detections (so its container is a list of pages).

    Both detection encodings must be recognised: the [polygon, (text, conf)] pair PaddleOCR
    emits, and the pre-normalized dict form used by tesseract_detections() and the test
    fixtures. Getting this wrong silently shifts a nesting level and every detection then fails
    to parse, so it is worth stating explicitly rather than inferring from list-ness alone.
    """
    if not isinstance(item, (list, tuple)):
        return False
    if not item:
        return True                       # an empty page is still a page
    head = item[0]
    if isinstance(head, dict):
        return True
    return (isinstance(head, (list, tuple)) and len(head) == 2
            and isinstance(head[0], (list, tuple)))


def normalize_paddle_output(raw, conf_scale: str = "auto") -> List[OCRLine]:
    """Accept any of the shapes PaddleOCR (and our test fixtures) produce and return OCRLines.

    Supported:
      * PaddleOCR 2.x:  [[ [poly, (text, conf)], ... ]]   (list of pages)
      * a single page:  [ [poly, (text, conf)], ... ]
      * pre-normalized: [ {"text":..., "confidence":..., "polygon": [...]}, ... ]
                        (or with explicit left/top/width/height instead of a polygon)
    """
    lines: List[OCRLine] = []
    if not raw:
        return lines

    # `raw` is a list of PAGES if its first element is itself a list of detections; otherwise it
    # is a single page's worth of detections.
    pages: Iterable = raw if _looks_like_page(raw[0]) else [raw]

    for page in pages:
        if not page:  # PaddleOCR returns [None] for a page where nothing was detected
            continue
        for det in page:
            line = _normalize_detection(det, conf_scale)
            if line is not None:
                lines.append(line)
    return lines


def _normalize_detection(det, conf_scale: str = "auto") -> Optional[OCRLine]:
    if isinstance(det, dict):
        text = str(det.get("text") or "")
        conf = _to_percent(det.get("confidence", det.get("conf", 0.0)), conf_scale)
        if "polygon" in det and det["polygon"]:
            left, top, width, height = _poly_bounds(det["polygon"])
            polygon = [[float(p[0]), float(p[1])] for p in det["polygon"]]
        else:
            left = float(det.get("left", 0.0))
            top = float(det.get("top", 0.0))
            width = float(det.get("width", 0.0))
            height = float(det.get("height", 0.0))
            polygon = [[left, top], [left + width, top],
                       [left + width, top + height], [left, top + height]]
    else:
        try:
            polygon_raw, payload = det[0], det[1]
            text = str(payload[0])
            conf = _to_percent(payload[1], conf_scale)
        except (IndexError, TypeError, ValueError):
            return None
        left, top, width, height = _poly_bounds(polygon_raw)
        polygon = [[float(p[0]), float(p[1])] for p in polygon_raw]

    return OCRLine(text=text, confidence=round(conf, 2), left=left, top=top,
                   width=width, height=height, polygon=polygon)


# ---------------------------------------------------------------------------
# Raw preview text -- NEVER filtered
# ---------------------------------------------------------------------------
def build_raw_text(lines: Sequence[OCRLine]) -> str:
    """Every detected line, in top-to-bottom / left-to-right reading order, with NO confidence
    filtering and no punctuation filtering at all.

    This is the user-facing verification preview: its whole purpose is to show what the OCR
    engine actually saw, including the garbage. Filtering it would defeat that -- a doctor
    checking why a field came out blank needs to see the unreadable smudge that caused it.
    Kept strictly separate from the reconstruction below, which IS filtered.
    """
    ordered = sorted(lines, key=lambda ln: (round(ln["top"], 1), ln["left"]))
    return "\n".join(ln["text"] for ln in ordered)


# ---------------------------------------------------------------------------
# Filtering (applies to the reconstruction only)
# ---------------------------------------------------------------------------
def _is_punctuation_only(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return bool(_PUNCT_ONLY_RE.match(stripped))


def _filter_lines(lines: Sequence[OCRLine]) -> Tuple[List[OCRLine], Dict[str, int]]:
    kept: List[OCRLine] = []
    stats = {"dropped_low_confidence": 0, "dropped_punctuation_only": 0, "dropped_empty": 0}
    for ln in lines:
        if not (ln["text"] or "").strip():
            stats["dropped_empty"] += 1
            continue
        # Confidence gate runs BEFORE clustering, so a low-confidence smudge can never be
        # clustered into a real row and become that row's value.
        if ln["confidence"] < OCR_CONFIDENCE_REJECT_THRESHOLD:
            stats["dropped_low_confidence"] += 1
            continue
        if _is_punctuation_only(ln["text"]):
            stats["dropped_punctuation_only"] += 1
            continue
        kept.append(ln)
    return kept, stats


# ---------------------------------------------------------------------------
# Row clustering
# ---------------------------------------------------------------------------
def _vertical_overlap_ratio(a: OCRLine, b: OCRLine) -> float:
    """Overlap of the two [top, top+height] spans, as a fraction of the SHORTER box's height.

    Using the shorter box is what makes a small-font value cell still cluster with the taller
    label cell printed beside it -- with the taller box as the denominator the ratio would be
    dragged below the threshold and the row would split in two.
    """
    overlap = min(a.bottom, b.bottom) - max(a["top"], b["top"])
    if overlap <= 0:
        return 0.0
    shorter = min(a["height"], b["height"])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def cluster_rows(lines: Sequence[OCRLine]) -> List[List[OCRLine]]:
    """Group boxes into rows by explicit vertical-overlap clustering.

    Deliberately does NOT use PaddleOCR's own output ordering: Paddle emits detections in its
    own reading order, which on a multi-column report interleaves columns and is precisely the
    behaviour that produced the row/column confusion this module exists to fix.
    """
    ordered = sorted(lines, key=lambda ln: (ln["top"], ln["left"]))
    rows: List[List[OCRLine]] = []

    for ln in ordered:
        placed = False
        for row in rows:
            # Compare against every member: a long row drifts vertically across the page, so
            # the nearest neighbour is a better test than the row's first box alone.
            if any(_vertical_overlap_ratio(ln, other) > OCR_ROW_OVERLAP_RATIO for other in row):
                row.append(ln)
                placed = True
                break
        if not placed:
            rows.append([ln])

    for row in rows:
        row.sort(key=lambda ln: ln["left"])
    rows.sort(key=lambda row: min(ln["top"] for ln in row))
    return rows


# ---------------------------------------------------------------------------
# Table region vs. narrative region
# ---------------------------------------------------------------------------
def _looks_like_prose(row: Sequence[OCRLine]) -> bool:
    """A row of long running text is a wrapped paragraph, not a table row -- even if the OCR
    engine happened to split it into two boxes."""
    if not row:
        return True
    avg_words = sum(len(ln["text"].split()) for ln in row) / len(row)
    return avg_words > _NARRATIVE_WORDS_PER_CELL


def _row_gap(a: Sequence[OCRLine], b: Sequence[OCRLine]) -> float:
    """Vertical white space between two rows (0 if they overlap)."""
    a_top, a_bottom = min(ln["top"] for ln in a), max(ln.bottom for ln in a)
    b_top, b_bottom = min(ln["top"] for ln in b), max(ln.bottom for ln in b)
    return max(0.0, max(a_top, b_top) - min(a_bottom, b_bottom))


def _split_table_and_narrative(
    rows: Sequence[Sequence[OCRLine]]
) -> Tuple[List[List[OCRLine]], List[List[OCRLine]]]:
    """A row with 2+ boxes that is not running prose is a table row; everything else (headings,
    sentences, wrapped paragraph lines) is narrative.

    Narrative rows are NEVER forced into the grid -- a Conclusion paragraph squeezed into
    (row, column) cells would produce nonsense labels and nonsense values.

    ONE EXCEPTION, and it matters clinically: a row left with a single box because its value
    cell was dropped by the confidence filter (or was blank on the page) is still a table row.
    Such a row is promoted back into the table when it sits immediately above or below a real
    table row -- close enough to be part of the same printed block. Without this, a label whose
    value was unreadable would fall through to the prose path and the field would render as
    "Not detected" with no flag, which reads as "the report doesn't contain this measurement"
    rather than "we couldn't read it".
    """
    is_table = [len(row) >= 2 and not _looks_like_prose(row) for row in rows]

    heights = sorted(ln["height"] for row in rows for ln in row if ln["height"] > 0)
    median_h = heights[len(heights) // 2] if heights else 0.0
    # Same "one blank line of white space" notion used for paragraph breaks below.
    max_gap = median_h * 2.5

    # Snapshot so a promoted row cannot itself promote its neighbour -- one step only, otherwise
    # a table at the top of the page would cascade all the way through the narrative below it.
    originally_table = list(is_table)
    if median_h:
        for i, row in enumerate(rows):
            if originally_table[i] or _looks_like_prose(row):
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(rows) and originally_table[j] and _row_gap(row, rows[j]) <= max_gap:
                    is_table[i] = True
                    break

    table_rows = [list(row) for row, flag in zip(rows, is_table) if flag]
    narrative_rows = [list(row) for row, flag in zip(rows, is_table) if not flag]
    return table_rows, narrative_rows


# ---------------------------------------------------------------------------
# Column clustering
# ---------------------------------------------------------------------------
def column_gap_for(lines: Iterable[OCRLine]) -> float:
    """The column-splitting gap for THIS page, in pixels.

    Derived from the page's own median line height (config.OCR_COLUMN_GAP_RATIO) so the same
    threshold works at 300 DPI and on a 630px phone photo -- see the note in config.py for why a
    fixed pixel value silently merged columns on low-resolution uploads. An explicitly-set
    OCR_COLUMN_GAP_PX still wins, and is also the fallback when no line heights are available.
    """
    if OCR_COLUMN_GAP_PX_EXPLICIT:
        return OCR_COLUMN_GAP_PX
    heights = sorted(ln["height"] for ln in lines if ln["height"] > 0)
    if not heights:
        return OCR_COLUMN_GAP_PX
    return heights[len(heights) // 2] * OCR_COLUMN_GAP_RATIO


def cluster_columns(table_rows: Sequence[Sequence[OCRLine]],
                    gap_px: Optional[float] = None) -> List[float]:
    """Return the sorted left-edge boundary of each detected column.

    Collect the `left` x-coordinate of every box across the table region, sort the unique
    values, and start a new column boundary wherever the gap to the next value exceeds gap_px.

    NOTE ON SCOPE: the spec phrased this as "the left edge of the FIRST box in every row". That
    literal reading can only ever yield column 0's boundary, so a 4-cell row would collapse into
    a single bin and the multi-column case this module exists for would still fail. Every box's
    left edge is used instead -- a strict superset that includes all the first-box edges.

    gap_px defaults to column_gap_for() -- scaled to the page's own text size rather than fixed
    in pixels. Pass an explicit value to override (config.OCR_COLUMN_GAP_PX does this globally).
    """
    if gap_px is None:
        gap_px = column_gap_for(ln for row in table_rows for ln in row)
    lefts = sorted({round(ln["left"], 1) for row in table_rows for ln in row})
    if not lefts:
        return []

    boundaries = [lefts[0]]
    for prev, cur in zip(lefts, lefts[1:]):
        if cur - prev > gap_px:
            boundaries.append(cur)
    return boundaries


def _nearest_column(left: float, boundaries: Sequence[float]) -> int:
    return min(range(len(boundaries)), key=lambda i: abs(left - boundaries[i]))


def build_grid(table_rows: Sequence[Sequence[OCRLine]],
               boundaries: Sequence[float]) -> List[List[Dict]]:
    """Assign every box to its nearest column bin, join text within a (row, column) cell in
    left-to-right order, and average the constituent boxes' confidence for that cell."""
    grid: List[List[Dict]] = []
    if not boundaries:
        return grid

    for row in table_rows:
        cells: List[List[OCRLine]] = [[] for _ in boundaries]
        for ln in row:
            cells[_nearest_column(ln["left"], boundaries)].append(ln)

        built = []
        for bucket in cells:
            if not bucket:
                # An empty cell is a real, meaningful observation (a blank measurement) -- it is
                # kept as an explicit empty cell so column indices stay aligned across rows.
                built.append({"text": "", "confidence": 0.0, "left": None, "top": None})
                continue
            bucket.sort(key=lambda ln: ln["left"])
            built.append({
                "text": " ".join(ln["text"].strip() for ln in bucket).strip(),
                "confidence": round(sum(ln["confidence"] for ln in bucket) / len(bucket), 2),
                "left": bucket[0]["left"],
                "top": min(ln["top"] for ln in bucket),
            })
        grid.append(built)
    return grid


# ---------------------------------------------------------------------------
# Narrative blocks
# ---------------------------------------------------------------------------
def build_narrative_blocks(narrative_rows: Sequence[Sequence[OCRLine]]) -> List[str]:
    """Group consecutive narrative rows into paragraph-like blocks.

    A block break is a vertical gap larger than ~1.8x the median line height -- i.e. visible
    white space between paragraphs/sections. Each block keeps its own line breaks so the Stage 2
    narrative reader still sees "Conclusion:" attached to the sentences underneath it.
    """
    if not narrative_rows:
        return []

    heights = sorted(ln["height"] for row in narrative_rows for ln in row if ln["height"] > 0)
    median_h = heights[len(heights) // 2] if heights else 0.0
    gap_limit = median_h * 1.8 if median_h else float("inf")

    blocks: List[str] = []
    current: List[str] = []
    prev_bottom: Optional[float] = None

    for row in sorted(narrative_rows, key=lambda r: min(ln["top"] for ln in r)):
        text = " ".join(ln["text"].strip() for ln in sorted(row, key=lambda ln: ln["left"])).strip()
        top = min(ln["top"] for ln in row)
        bottom = max(ln.bottom for ln in row)
        if prev_bottom is not None and (top - prev_bottom) > gap_limit and current:
            blocks.append("\n".join(current))
            current = []
        if text:
            current.append(text)
        prev_bottom = bottom

    if current:
        blocks.append("\n".join(current))
    return blocks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def extract_text(raw_ocr_output, conf_scale: str = "auto") -> Dict:
    """Reconstruct one document from raw OCR output.

    Returns:
      {
        "raw_text":         str   -- EVERY detected line, unfiltered, for the preview page
        "table_grid":       [[{"text": str, "confidence": float}, ...cols], ...rows]
        "narrative_blocks": [str, ...]  -- paragraph regions outside the table grid
        "quality_info":     {...} -- what was dropped and why, for debugging/QA
        "lines":            [(text, confidence), ...] -- filtered rows, row-joined, for
                            extractor.py's last-resort flat-line fallback only
      }
    """
    lines = normalize_paddle_output(raw_ocr_output, conf_scale)
    raw_text = build_raw_text(lines)

    kept, drop_stats = _filter_lines(lines)
    rows = cluster_rows(kept)
    table_rows, narrative_rows = _split_table_and_narrative(rows)
    boundaries = cluster_columns(table_rows)
    table_grid = build_grid(table_rows, boundaries)
    narrative_blocks = build_narrative_blocks(narrative_rows)

    confidences = [ln["confidence"] for ln in kept]
    quality_info = {
        "total_lines_detected": len(lines),
        "lines_kept": len(kept),
        **drop_stats,
        "rows_detected": len(rows),
        "table_rows": len(table_rows),
        "columns_detected": len(boundaries),
        "column_boundaries": [round(b, 1) for b in boundaries],
        "narrative_blocks": len(narrative_blocks),
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
        "min_confidence": round(min(confidences), 2) if confidences else 0.0,
        "reject_threshold": OCR_CONFIDENCE_REJECT_THRESHOLD,
        "flag_threshold": OCR_CONFIDENCE_FLAG_THRESHOLD,
        "column_gap_px": round(column_gap_for(kept), 2),
        "column_gap_ratio": OCR_COLUMN_GAP_RATIO,
        "row_overlap_ratio": OCR_ROW_OVERLAP_RATIO,
    }

    # Row-joined text for extractor.py's flat-line fallback. Confidence is the MINIMUM of the
    # row's boxes, not the mean: a flat line is only as trustworthy as its worst-read token, and
    # this path has no cell structure to isolate a bad one.
    lines = [
        (" ".join(ln["text"].strip() for ln in row).strip(),
         round(min(ln["confidence"] for ln in row), 2))
        for row in rows
    ]

    return {
        "raw_text": raw_text,
        "table_grid": table_grid,
        "narrative_blocks": narrative_blocks,
        "quality_info": quality_info,
        "lines": [(t, c) for t, c in lines if t],
    }


def upscale_for_ocr(image, min_long_side: int = OCR_MIN_LONG_SIDE):
    """Enlarge a small page so the recogniser sees text at a workable size.

    Real uploads are phone photos and low-DPI scans -- the ones in tests/fixtures run from
    400x452 to 1085px on the long side. At that scale PaddleOCR mis-reads whole labels ("IVSd"
    came back as "PSAI"), and the detector fuses adjacent columns into one box because the gap
    between them is only a few pixels.

    Only ever upscales: a page already at or above the target is returned untouched, so a proper
    300-DPI scan is not resampled for nothing. INTER_CUBIC because the recogniser is trained on
    anti-aliased text and nearest/linear leave stair-stepped glyph edges.

    Coordinates come back in the SCALED space, which is fine -- every downstream threshold is
    relative (rows cluster by proportional overlap, columns by a multiple of median line height).
    """
    if image is None:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= 0 or long_side >= min_long_side:
        return image
    scale = min_long_side / float(long_side)
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))),
                      interpolation=cv2.INTER_CUBIC)


def run_paddle_with_geometry(image) -> Dict:
    """Run PaddleOCR on one page image and reconstruct it. Kept separate from extract_text() so
    the reconstruction logic stays testable without the (heavy, optional) paddle dependency."""
    from paddleocr import PaddleOCR  # imported lazily; optional dependency

    global _PADDLE_INSTANCE
    try:
        _PADDLE_INSTANCE
    except NameError:
        _PADDLE_INSTANCE = PaddleOCR(
            use_angle_cls=True, lang="en", show_log=False,
            # See config.py: the limit MUST be at least OCR_MIN_LONG_SIDE, or PaddleOCR resizes
            # the upscaled page straight back down and the upscale accomplishes nothing.
            det_limit_side_len=OCR_DET_LIMIT_SIDE_LEN,
            det_limit_type="max",
            det_db_unclip_ratio=OCR_DET_UNCLIP_RATIO,
            drop_score=OCR_DROP_SCORE,
        )

    return extract_text(_PADDLE_INSTANCE.ocr(upscale_for_ocr(image), cls=True))


def tesseract_detections(image) -> List[dict]:
    """Tesseract word boxes regrouped into line-level detections with geometry.

    WHY THIS EXISTS: PaddleOCR is not always importable (its bundled protobuf stubs conflict with
    newer protobuf runtimes, among other things). Without geometry the pipeline loses table
    reconstruction entirely and a dense multi-column report extracts almost nothing -- so the
    fallback reader must supply boxes too, not just text. pytesseract's image_to_data already
    returns left/top/width/height and a 0-100 confidence per word; grouping those by Tesseract's
    own (block, paragraph, line) numbering reproduces the same line-level shape PaddleOCR emits,
    which is all Stage 1 needs.
    """
    import pytesseract

    from app.config import TESSERACT_CMD
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    grouped: Dict[Tuple[int, int, int], List[dict]] = {}
    for i in range(len(data["text"])):
        word = (data["text"][i] or "").strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:   # conf == -1 marks a non-text box
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append({
            "text": word, "conf": conf,
            "left": float(data["left"][i]), "top": float(data["top"][i]),
            "width": float(data["width"][i]), "height": float(data["height"][i]),
        })

    detections: List[dict] = []
    for key in sorted(grouped):
        words = sorted(grouped[key], key=lambda w: w["left"])
        left = min(w["left"] for w in words)
        top = min(w["top"] for w in words)
        right = max(w["left"] + w["width"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)
        detections.append({
            "text": " ".join(w["text"] for w in words),
            # Line confidence is the MINIMUM of its words: a line is only as trustworthy as its
            # worst-read token, and averaging would let one clear word mask an unreadable value.
            "confidence": min(w["conf"] for w in words),
            "left": left, "top": top, "width": right - left, "height": bottom - top,
        })
    return detections


def run_tesseract_with_geometry(image) -> Dict:
    """Geometry-preserving Tesseract read, reconstructed exactly like the PaddleOCR path."""
    return extract_text([tesseract_detections(image)], conf_scale="0-100")


def merge_documents(docs: Sequence[Dict]) -> Dict:
    """Combine per-page extract_text() results into one document-level result."""
    merged = {"raw_text": "", "table_grid": [], "narrative_blocks": [], "lines": [],
              "quality_info": {"pages": []}}
    raw_parts = []
    for doc in docs:
        raw_parts.append(doc.get("raw_text", ""))
        merged["table_grid"].extend(doc.get("table_grid") or [])
        merged["narrative_blocks"].extend(doc.get("narrative_blocks") or [])
        merged["lines"].extend(doc.get("lines") or [])
        merged["quality_info"]["pages"].append(doc.get("quality_info") or {})
    merged["raw_text"] = "\n".join(p for p in raw_parts if p)
    return merged
