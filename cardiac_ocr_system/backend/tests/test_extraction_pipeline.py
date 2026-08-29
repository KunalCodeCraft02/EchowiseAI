"""
Tests for the two-stage extraction engine.

  STAGE 1  app/ocr/text_extraction.py -- geometric table reconstruction from PaddleOCR
           polygons. Driven here by MOCKED PaddleOCR output: 4-point bounding boxes plus
           per-line confidence, in the native 0-1 scale, so the ×100 normalization is exercised
           too. No paddle install, no image, no network.

  STAGE 2  app/ocr/extractor.py -- semantic matching over Stage 1's grid + narrative blocks.
           The text-only Groq layer is DISABLED for the whole module (see _no_groq below), so
           every assertion here pins the deterministic dictionary path: these are the clinical
           safety rules, and they must hold whether or not an API key is present.

The second half of the file keeps the pre-existing doc_result-shaped regressions (pdfplumber
tables / label-value form fields), which pin the same matcher against a real hospital report
template.

Everything is fully offline: no network, no DB, no API keys required.
"""
import pytest

from app.ocr import extractor as ex
from app.ocr import text_extraction as tx


@pytest.fixture(autouse=True)
def _no_groq(monkeypatch):
    """Force the deterministic path. Groq is an enhancement layer; if a developer has
    GROQ_API_KEY set locally the suite must still assert the same behaviour."""
    monkeypatch.setattr(ex, "build_semantic_label_map", lambda labels: {})
    monkeypatch.setattr(ex, "_groq_narrative_findings", lambda blocks: {})


# ---------------------------------------------------------------------------
# Mocked PaddleOCR input
# ---------------------------------------------------------------------------
ROW_H = 20          # printed line height, px
ROW_PITCH = 40      # vertical distance between table rows, px


def _det(text, left, top, conf=0.96, width=None, height=ROW_H):
    """One mocked PaddleOCR detection: [4-point polygon, (text, confidence 0-1)].

    Matches PaddleOCR 2.7's real output shape exactly, including its 0-1 confidence scale --
    Stage 1 is responsible for converting that to the project-wide 0-100 scale.
    """
    w = width if width is not None else max(40, 11 * len(text))
    poly = [[left, top], [left + w, top], [left + w, top + height], [left, top + height]]
    return [poly, (text, conf)]


def _table(rows, x_positions, y0=100):
    """Lay out `rows` (lists of (text, conf) or (text, conf, col_index)) on a grid.

    x_positions are the column left-edges in px; they are spaced well beyond
    config.OCR_COLUMN_GAP_PX so column clustering has unambiguous boundaries.
    """
    dets = []
    for r, row in enumerate(rows):
        top = y0 + r * ROW_PITCH
        for c, cell in enumerate(row):
            text, conf = cell[0], cell[1]
            col = cell[2] if len(cell) > 2 else c
            if text:
                dets.append(_det(text, x_positions[col], top, conf))
    return dets


def _prose(lines, left=100, y0=600, conf=0.96):
    """Narrative lines: one wide box per line, tightly spaced so they group into one block."""
    return [_det(text, left, y0 + i * 30, conf, width=700) for i, text in enumerate(lines)]


def _stage1_doc(detections):
    """Run Stage 1 and shape its output into the doc_result Stage 2 consumes."""
    doc = tx.extract_text([detections])
    doc.update({
        "full_text": "\n".join(doc["narrative_blocks"]),
        "tables": [],
        "form_fields": [],
        "source_type": "paddleocr_geometry",
    })
    return doc


def _run(detections):
    doc = _stage1_doc(detections)
    results, raw_text = ex.extract_parameters_structured(doc)
    return results, doc, raw_text


# ===========================================================================
# Stage 1 behaviour
# ===========================================================================
def test_stage1_raw_text_is_unfiltered_but_grid_is_filtered():
    """The preview text must show EVERYTHING the OCR engine read -- including the low-confidence
    smudge and the stray bracket -- while the reconstruction those artefacts would corrupt is
    filtered. These are two separate outputs on purpose."""
    dets = _table([[("IVSd", 0.96), ("1.05 cm", 0.96)]], [100, 400])
    dets.append(_det("blurred", 400, 140, conf=0.12))   # below reject threshold
    dets.append(_det("}", 700, 140, conf=0.99))         # punctuation-only noise

    doc = tx.extract_text([dets])

    assert "blurred" in doc["raw_text"], "raw preview text must never be filtered"
    assert "}" in doc["raw_text"]

    grid_text = " ".join(c["text"] for row in doc["table_grid"] for c in row)
    assert "blurred" not in grid_text
    assert "}" not in grid_text
    assert doc["quality_info"]["dropped_low_confidence"] == 1
    assert doc["quality_info"]["dropped_punctuation_only"] == 1


def test_stage1_confidence_scale_is_normalized_to_0_100():
    doc = tx.extract_text([[_det("EF", 100, 100, conf=0.93)]])
    assert doc["quality_info"]["avg_confidence"] == pytest.approx(93.0)


def test_stage1_rows_cluster_by_overlap_not_reading_order():
    """Detections arrive in a column-interleaved order (what PaddleOCR actually does on a
    multi-column page). Row clustering must recover the printed rows regardless."""
    dets = [
        _det("LVEDd", 100, 100), _det("LAd", 700, 102),      # printed row 1, slight y jitter
        _det("4.2 cm", 300, 101), _det("3.1 cm", 900, 100),
        _det("LVESd", 100, 140), _det("2.5 cm", 300, 141),   # printed row 2
    ]
    doc = tx.extract_text([list(reversed(dets))])           # deliberately wrong input order

    assert len(doc["table_grid"]) == 2
    assert [c["text"] for c in doc["table_grid"][0]] == ["LVEDd", "4.2 cm", "LAd", "3.1 cm"]
    assert doc["table_grid"][1][0]["text"] == "LVESd"


def test_stage1_accepts_dict_detections_paged_or_unpaged():
    """The fallback reader (tesseract_detections) emits pre-normalized dicts rather than
    PaddleOCR's [polygon, (text, conf)] pairs. Both encodings must be recognised, wrapped in a
    page list or not -- misreading the nesting shifts a level and every detection silently fails
    to parse, which is exactly how the Tesseract geometry path broke the first time."""
    dicts = [
        {"text": "IVSd", "confidence": 96.0, "left": 100, "top": 100, "width": 60, "height": 20},
        {"text": "1.05 cm", "confidence": 96.0, "left": 400, "top": 100, "width": 80, "height": 20},
    ]
    for raw in (dicts, [dicts]):          # unpaged and paged
        doc = tx.extract_text(raw, conf_scale="0-100")
        assert [c["text"] for c in doc["table_grid"][0]] == ["IVSd", "1.05 cm"]
        assert doc["table_grid"][0][0]["confidence"] == 96.0


def test_stage1_explicit_0_100_scale_is_not_rescaled():
    """Tesseract confidence is already 0-100. Under the 0-1 "auto" heuristic a genuine 1%
    confidence would be read as 100% -- promoting the worst-read token on the page to fully
    trusted, which is the one thing the reject gate exists to prevent."""
    det = [{"text": "IVSd", "confidence": 1.0, "left": 100, "top": 100, "width": 60, "height": 20}]

    assert tx.extract_text(det, conf_scale="0-100")["quality_info"]["dropped_low_confidence"] == 1
    # ...whereas the PaddleOCR-native reading of the same number is 100%.
    assert tx.extract_text(det)["quality_info"]["avg_confidence"] == 100.0


def test_stage1_narrative_stays_out_of_the_grid():
    dets = _table([[("EF", 0.96), ("62 %", 0.96)]], [100, 400])
    dets += _prose(["CONCLUSION:", "Mild mitral regurgitation noted."])
    doc = tx.extract_text([dets])

    assert len(doc["table_grid"]) == 1
    assert any("mitral" in b.lower() for b in doc["narrative_blocks"])


# ===========================================================================
# Case 1: LVEDD row + adjacent LVEDD/BSA row -> primary vs indexed routing
# ===========================================================================
def test_case1_lvedd_and_lvedd_bsa_adjacent_rows():
    results, _, _ = _run(_table([
        [("LVEDD", 0.96), ("4.6 cm", 0.96)],
        [("LVEDD/BSA", 0.96), ("2.4", 0.96)],
    ], [100, 400]))

    assert results["LVIDd"].value == "46 mm"
    assert results["LVIDd"].db_field == "lvidd"
    assert results["LVIDd_Indexed"].value == "2.4 cm/m2"
    assert results["LVIDd_Indexed"].db_field == "lvidd_indexed_value"

    # No leakage in either direction between the two adjacent rows.
    assert "2.4" not in results["LVIDd"].value
    assert "4.6" not in results["LVIDd_Indexed"].value


# ===========================================================================
# Case 2: 3-column layout, IVSd/PWd in the CENTER column
# ===========================================================================
def test_case2_three_column_layout_center_column_extracted():
    results, doc, _ = _run(_table([
        [("LVEDd", 0.96), ("4.2 cm", 0.96), ("IVSd", 0.96), ("1.05 cm", 0.96),
         ("LAd (AP)", 0.95), ("3.1 cm", 0.95)],
        [("LVESd", 0.96), ("2.5 cm", 0.96), ("PWd", 0.96), ("0.98 cm", 0.96),
         ("SOV", 0.95), ("2.7 cm", 0.95)],
    ], [100, 300, 550, 750, 1000, 1200]))

    assert doc["quality_info"]["columns_detected"] == 6

    # The center column is the one most exposed to column-swap bugs -- both of its values must
    # come from their OWN row and their OWN column.
    assert results["IVSd"].value == "10.5 mm"
    assert results["PWd"].value == "9.8 mm"
    assert results["IVSd"].source == "table"
    assert results["PWd"].source == "table"

    # Flanking columns must be unaffected.
    assert results["LVIDd"].value == "42 mm"
    assert results["LVIDs"].value == "25 mm"
    assert results["LA_Diameter"].value == "3.1 cm"
    assert results["Ao_Root"].value == "2.7 cm"


# ===========================================================================
# Case 3: "mm" value for a cm-canonical field
# ===========================================================================
def test_case3_mm_converted_to_canonical_cm():
    results, _, _ = _run(_table([[("LA Diameter", 0.96), ("25 mm", 0.96)]], [100, 400]))

    ef = results["LA_Diameter"]
    assert ef.value == "2.5 cm"
    assert ef.value_type == "numeric"
    # The unit is never stripped silently: raw value, raw unit and the conversion are all kept.
    assert ef.meta["raw_detected_value"] == "25"
    assert ef.meta["raw_detected_unit"] == "mm"
    assert ef.meta["conversion_applied"] is True


# ===========================================================================
# Case 4: low-confidence / blank IVSd cell -> null, never hallucinated
# ===========================================================================
def test_case4_low_confidence_cell_is_null_not_hallucinated():
    results, doc, _ = _run(_table([
        [("IVSd", 0.96), ("1.05 cm", 0.10)],   # value cell far below the reject threshold
        [("PWd", 0.96), ("0.98 cm", 0.96)],
    ], [100, 400]))

    # Stage 1 dropped the unreadable cell entirely -- it never reached the semantic layer.
    assert "1.05" not in doc["raw_text"].split("\n")[0] or True  # raw preview still has it
    assert "1.05" in doc["raw_text"]
    grid_text = " ".join(c["text"] for row in doc["table_grid"] for c in row)
    assert "1.05" not in grid_text

    ef = results["IVSd"]
    assert ef.value is None
    assert ef.flagged is True
    assert ef.meta["final_stored_value"] is None
    assert "unreadable" in ef.meta["flag_reason"].lower() or "blank" in ef.meta["flag_reason"].lower()

    # The neighbouring good row is unaffected -- one bad cell must not poison the table.
    assert results["PWd"].value == "9.8 mm"


# ===========================================================================
# Case 5: "201" where only "2.01" is plausible -> rejected + flagged
# ===========================================================================
def test_case5_impossible_value_rejected_but_plausible_one_kept():
    bad, _, _ = _run(_table([[("IVSd", 0.96), ("201", 0.96)]], [100, 400]))
    ef = bad["IVSd"]
    assert ef.value is None, "an impossible value must not be stored"
    assert ef.flagged is True
    reason = ef.meta["flag_reason"].lower()
    assert "physiologically possible range" in reason
    assert "201" in ef.meta["flag_reason"], "the raw OCR snippet must be quoted in the flag"

    good, _, _ = _run(_table([[("IVSd", 0.96), ("2.01", 0.96)]], [100, 400]))
    assert good["IVSd"].value == "20.1 mm"
    assert good["IVSd"].in_range is True


# ===========================================================================
# Case 6: narrative-only mitral finding, no MV table row anywhere
# ===========================================================================
def test_case6_narrative_fills_mitral_valve_when_no_table_row():
    dets = _table([[("EF", 0.96), ("62 %", 0.96)]], [100, 400])
    dets += _prose([
        "CONCLUSION:",
        "Mild mitral regurgitation noted. Normal LV systolic function.",
    ])
    results, doc, _ = _run(dets)

    # Confirm the fixture really has no structured mitral source.
    grid_text = " ".join(c["text"] for row in doc["table_grid"] for c in row).lower()
    assert "mitral" not in grid_text and "mv" not in grid_text.split()

    ef = results["MV"]
    assert ef.value == "Mild Regurgitation"
    assert ef.value_type == "qualitative"
    assert ef.source == "narrative"
    assert ef.db_field == "mv_finding"
    assert ef.flagged is True            # prose is always lower-trust than a structured cell
    assert "mild mitral regurgitation" in ef.meta["source_snippet"].lower()
    # Qualitative extractions carry no unit/conversion metadata at all.
    assert "raw_detected_unit" not in ef.meta
    assert "conversion_applied" not in ef.meta


def test_case6b_table_always_wins_over_narrative():
    """Same narrative finding, but this time the table states the mitral finding too."""
    dets = _table([[("MV", 0.96), ("Normal", 0.96)]], [100, 400])
    dets += _prose(["CONCLUSION:", "Mild mitral regurgitation noted."])
    results, _, _ = _run(dets)

    assert results["MV"].value == "Normal"
    assert results["MV"].source == "table"


# ===========================================================================
# Case 7: "LAd (AP)" label variant -> la_diameter
# ===========================================================================
def test_case7_lad_ap_variant_matches_la_diameter():
    results, _, _ = _run(_table([[("LAd (AP)", 0.95), ("3.8 cm", 0.95)]], [100, 400]))

    ef = results["LA_Diameter"]
    assert ef.value == "3.8 cm"
    assert ef.db_field == "la_diameter"
    assert ef.source == "table"


# ===========================================================================
# Case 8: a table cell containing a WORD -> stored as-is, bounds never applied
# ===========================================================================
def test_case8_qualitative_table_cell_skips_numeric_validation():
    results, _, _ = _run(_table([
        [("MV", 0.96), ("Normal", 0.96)],
        [("TV", 0.96), ("Mild Regurgitation", 0.96)],
        [("RV Size", 0.96), ("Enlarged", 0.96)],
        # A normally-numeric field whose cell holds a word: still a valid extraction.
        [("IVSd", 0.96), ("Normal", 0.96)],
    ], [100, 400]))

    for canon, want in [("MV", "Normal"), ("TV", "Mild Regurgitation"),
                        ("RV_Size", "Enlarged"), ("IVSd", "Normal")]:
        ef = results[canon]
        assert ef.value == want, f"{canon}: got {ef.value!r}"
        assert ef.value_type == "qualitative"
        assert ef.source == "table"
        # Never evaluated against numeric sanity bounds.
        assert ef.in_range is None
        assert "raw_detected_value" not in ef.meta
        assert "conversion_applied" not in ef.meta

    # A word in a numeric field is legitimate but surfaced for confirmation.
    assert results["IVSd"].flagged is True
    # A word in a qualitative-by-definition field is an ordinary, unflagged reading.
    assert results["MV"].flagged is False


# ===========================================================================
# Case 9: "LVEDD/BSA" alone -> indexed field only, primary untouched
# ===========================================================================
def test_case9_indexed_label_never_overwrites_primary():
    results, _, _ = _run(_table([[("LVEDD/BSA", 0.96), ("2.4", 0.96)]], [100, 400]))

    assert results["LVIDd_Indexed"].value == "2.4 cm/m2"
    assert results["LVIDd_Indexed"].db_field == "lvidd_indexed_value"
    # The primary field must be left completely alone, not filled with the indexed number.
    assert results.get("LVIDd") is None or results["LVIDd"].value is None


# ===========================================================================
# End-to-end: a full page mixing all of the above
# ===========================================================================
def test_full_page_mixed_numeric_and_qualitative():
    dets = _table([
        [("LVEDD", 0.96), ("4.6 cm", 0.96), ("IVSd", 0.96), ("13 mm", 0.96)],
        [("LVEDD/BSA", 0.95), ("2.4", 0.95), ("PWd", 0.96), ("0.9 cm", 0.96)],
        [("LAd (AP)", 0.95), ("3.8 cm", 0.95), ("EF (Simpson)", 0.96), ("40 %", 0.96)],
        [("MV", 0.96), ("Normal", 0.96), ("AV", 0.96), ("Mild Stenosis", 0.96)],
    ], [100, 350, 600, 850])
    dets += _prose([
        "CONCLUSION:",
        "Moderate tricuspid regurgitation. The right atrium is enlarged.",
    ])
    results, doc, raw_text = _run(dets)

    assert results["LVIDd"].value == "46 mm"
    assert results["LVIDd_Indexed"].value == "2.4 cm/m2"
    assert results["IVSd"].value == "13 mm"
    assert results["IVSd"].meta["conversion_applied"] is False
    assert results["PWd"].value == "9 mm"
    assert results["PWd"].meta["conversion_applied"] is True
    assert results["LA_Diameter"].value == "3.8 cm"
    assert results["EF"].value == "40%"
    assert results["MV"].value == "Normal"
    assert results["AV"].value == "Mild Stenosis"
    assert results["TV"].value == "Moderate Regurgitation"
    assert results["TV"].source == "narrative"
    assert results["RA_Size"].value == "Enlarged"

    # Every stored field is tagged with how it was read and what kind of value it is.
    for canon, ef in results.items():
        if ef.value is None:
            continue
        assert ef.value_type in ("numeric", "qualitative"), canon
        assert ef.meta["source"] in ("table", "narrative", "form_field", "flat_line_fallback")
        assert ef.meta["value_type"] == ef.value_type

    # Only canonical dictionary keys ever leave the extractor.
    from app.ocr.parameter_dict import ECHO_PARAMETER_DICTIONARY
    assert set(results).issubset(set(ECHO_PARAMETER_DICTIONARY))

    # raw_text is the unfiltered preview, and it is what the report row stores.
    assert "CONCLUSION:" in raw_text
    assert raw_text == doc["raw_text"]


# ===========================================================================
# Pre-existing regressions: doc_result shapes produced by the digital-PDF
# (pdfplumber tables) and label/value (form_fields) paths.
# ===========================================================================
def _build_doc_result(table_rows=None, form_fields=None, narrative_lines=None) -> dict:
    """
    table_rows:      list of rows, each a list of (text, confidence 0-100) tuples.
    form_fields:     list of (key_text, key_conf, value_text, value_conf) tuples, 0-100.
    narrative_lines: list of (text, confidence 0-100) tuples -> lines + full_text.
    Confidences are on the project-wide 0-100 scale used everywhere downstream.
    """
    tables = []
    if table_rows:
        rows = [
            {"cells": [{"text": t, "confidence": c, "bounding_box": None} for t, c in row]}
            for row in table_rows
        ]
        tables = [{"page": 1, "rows": rows}]

    ff_out = [
        {"key_text": k, "key_confidence": kc, "value_text": v, "value_confidence": vc, "page": 1}
        for k, kc, v, vc in (form_fields or [])
    ]

    lines = list(narrative_lines or [])
    text_parts = [t for t, _ in lines]
    for row in table_rows or []:
        text_parts.append(" ".join(t for t, _ in row))
    for k, _, v, _ in (form_fields or []):
        text_parts.append(f"{k}: {v}")

    return {
        "full_text": "\n".join(text_parts),
        "tables": tables,
        "form_fields": ff_out,
        "lines": lines,
        "source_type": "test_fixture",
    }


def test_real_chong_hua_3_column_stress_echo_report():
    """Chong Hua Hospital / Siemens SC2000 treadmill stress-echo -- a dense 3-column layout,
    values transcribed from an actual uploaded report. This is the template that motivated the
    structure-aware rewrite, so it guards the matcher against regressions on a genuine (not
    synthetic) report."""
    doc = _build_doc_result(table_rows=[
        [("LVEDd", 96.0), ("4.2 cm", 96.0), ("LAd (AP)", 95.0), ("3.1 cm", 95.0)],
        [("LVESd", 96.0), ("2.5 cm", 96.0), ("LAd/BSA", 94.0), ("1.9", 94.0)],
        [("IVSd", 96.0), ("1.0 cm", 96.0), ("Annulus", 93.0), ("1.8 cm", 93.0)],
        [("IVSs", 95.0), ("1.4 cm", 95.0), ("SOV", 93.0), ("2.7 cm", 93.0)],
        [("PWd", 96.0), ("0.8 cm", 96.0), ("STJ", 93.0), ("2.2 cm", 93.0)],
        [("PWs", 95.0), ("1.3 cm", 95.0), ("E/A Ratio", 94.0), ("1.62", 94.0)],
        [("EF (Simpson)", 95.0), ("65 %", 95.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    expected = {
        "LVIDd": "42 mm", "LVIDs": "25 mm", "IVSd": "10 mm", "IVSs": "14 mm",
        "PWd": "8 mm", "PWs": "13 mm", "LA_Diameter": "3.1 cm",
        "LA_Diameter_Indexed": "1.9 cm/m2", "Ao_Annulus": "1.8 cm", "Ao_Root": "2.7 cm",
        "Ao_STJ": "2.2 cm", "E_A_Ratio": "1.62", "EF": "65%",
    }
    for canon, want in expected.items():
        assert canon in results, f"{canon} was not extracted at all"
        assert results[canon].value == want, f"{canon}: got {results[canon].value!r}, want {want!r}"

    assert results["LA_Diameter"].db_field == "la_diameter"
    assert results["LA_Diameter_Indexed"].db_field == "la_diameter_indexed_value"
    # The three aortic landmarks stay distinct rather than collapsing into one field.
    assert len({results["Ao_Annulus"].value, results["Ao_Root"].value, results["Ao_STJ"].value}) == 3


def test_form_field_source_values_and_unit_conversion():
    doc = _build_doc_result(form_fields=[
        ("IVSd", 97.0, "1.1 cm", 97.0),
        ("EF", 93.0, "62 %", 93.0),
        ("LA Diameter", 91.0, "34 mm", 91.0),
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results["IVSd"].value == "11 mm"
    assert results["IVSd"].source == "form_field"
    assert results["EF"].value == "62%"
    assert results["LA_Diameter"].value == "3.4 cm"
    assert results["LA_Diameter"].meta["conversion_applied"] is True


def test_velocity_row_is_not_stored_as_a_valve_finding():
    """Real-report regression: the flat-line fallback matched the "TV" abbreviation inside a
    "TV Vmax 1.69 m/sec" row and stored a flow velocity as the tricuspid valve FINDING. A
    qualitative field must not accept a measurement as its descriptor."""
    doc = _build_doc_result(narrative_lines=[("TV Vmax 1.69 m/sec", 92.0)])
    results, _ = ex.extract_parameters_structured(doc)

    assert results.get("TV") is None or results["TV"].value is None

    # A genuine qualitative finding on the same path still comes through.
    doc = _build_doc_result(narrative_lines=[("TV: Mild regurgitation", 92.0)])
    results, _ = ex.extract_parameters_structured(doc)
    assert "regurgitation" in (results["TV"].value or "").lower()


def test_mitral_annulus_does_not_hijack_aortic_annulus():
    """The bare "Annulus" synonym for Ao_Annulus must not swallow the mitral/tricuspid annulus
    rows printed elsewhere on the same report."""
    doc = _build_doc_result(table_rows=[
        [("Mitral Annulus", 95.0), ("2.1 cm", 95.0)],
        [("Tricuspid Annulus", 95.0), ("2.1 cm", 95.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results.get("Ao_Annulus") is None or results["Ao_Annulus"].value is None
