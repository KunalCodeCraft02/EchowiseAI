"""
Regressions pinned from the labelled real reports in tests/fixtures/ground_truth.json.

Every test here corresponds to a failure that tools/accuracy_report.py actually measured on a
real uploaded report, with the source template named in the docstring. They exist so the fix
cannot be undone by a later tuning change: the harness needs the report images to run, these
run anywhere.

Baseline when this suite was written: 66.0% correct / 8.2% wrong / 5.0% false-positive.
After the fixes below: 87.6% correct / 1.2% wrong / 0.0% false-positive (Groq disabled),
92.5% with the Groq semantic layer live.
"""
import json

import pytest

from app.ocr import extractor as ex
from app.ocr import groq_extractor as gx
from tests.test_extraction_pipeline import _build_doc_result, _prose, _run, _table


@pytest.fixture(autouse=True)
def _no_groq(monkeypatch):
    """Deterministic + offline: these pin local logic, not the LLM."""
    monkeypatch.setattr(ex, "build_semantic_label_map", lambda labels: {})
    monkeypatch.setattr(ex, "_groq_narrative_findings", lambda blocks: {})


# ===========================================================================
# Printed reference ranges must never become patient values
# ===========================================================================
def test_reference_range_column_is_not_stored_as_the_measurement():
    """[XXX] Facility + SonoNet: the normal range is printed beside a BLANK value cell.

    Reading its lower bound produced Ao_Root 2.0 / LA 1.9 / IVSd 0.6 / PWd 0.6 -- four
    plausible, in-range, entirely fictional measurements. Nothing is the correct answer.
    """
    results, _, _ = _run(_table([
        [("Aortic Root (ED)", 0.96), ("", 0.96), ("2.0-3.7 cm", 0.96)],
        [("LVPW(D)", 0.96), ("", 0.96), ("0.6-1.1 cm", 0.96)],
    ], [100, 400, 700]))

    assert results["Ao_Root"].value is None
    assert results["PWd"].value is None
    assert results["Ao_Root"].flagged is True


def test_comparator_normal_range_is_not_stored():
    """[XXX] Facility: "LVEF (est) | 55% | >50%". 55 is the patient, >50 is the range."""
    results, _, _ = _run(_table([[("LVEF (est)", 0.96), ("55%", 0.96), (">50%", 0.96)]],
                                [100, 400, 700]))
    assert results["EF"].value == "55%"


def test_range_bound_alone_yields_nothing_not_the_bound():
    ef = ex.resolve_value("EF", ">50%", 96.0, "table", "LVEF | >50%")
    assert ef.value is None
    assert "range" in (ef.meta["flag_reason"] or "").lower()


def test_value_followed_by_negative_z_score_is_not_a_range():
    """Sri Ramachandra PAEDIATRIC: "LVIDS -15 -0.41" is 15 mm plus a Z-score of -0.41.

    Without the ascending-pair rule the span "15 -0.41" reads as the range "15-0.41", the real
    measurement is discarded, and the next number on the line (an unrelated column) is stored.
    """
    match = ex.find_measurement(" -15 -0.41 MPA 7MM")
    assert match is not None and match.group(1) == "15"


# ===========================================================================
# OCR glues labels to values; labels also contain digits
# ===========================================================================
def test_glued_label_and_value_is_still_read():
    """2D-ECHO photo: OCR dropped the space, giving "PW D0.8cm" and "IVS (d0.8cm"."""
    results, _, _ = _run(_prose(["PW D0.8cm", "LEFT ATRIUM 2.1cms"]))
    assert results["PWd"].value == "8 mm"
    assert results["LA_Diameter"].value == "2.1 cm"


def test_label_ordinal_digit_is_not_read_as_the_value():
    """Duality Health: "LAd4 39.4 mm" -- the 4 names the 4-chamber view, it is not 4 cm.

    Accepting it stored LA_Diameter = "4 cm", which is both wrong and clinically plausible.
    """
    results, _, _ = _run(_prose(["LAd4 39.4 mm Last"]))
    assert results["LA_Diameter"].value != "4 cm"
    assert results["LA_Diameter"].value == "3.94 cm"


def test_digits_inside_an_identifier_are_not_a_measurement():
    """Duality Health: "LA A4Cs 9.23 cm2" -- the 4 belongs to the view name "A4Cs"."""
    assert ex.find_measurement(" A4Cs 9.23 cm2").group(1) == "9.23"
    assert ex.find_measurement("Ejection fraction (2D) (%) 68 >50").group(1) == "68"


def test_hyphenated_label_matches_its_spaced_synonym():
    """Appendix-2 PDF: "No evidence of LV segmental wall-motion abnormalities."."""
    assert any(canon == "Wall_Motion"
               for canon, _, _, _ in ex._find_all_label_matches("LV segmental wall-motion: none"))


# ===========================================================================
# Choosing between two readings
# ===========================================================================
def test_paediatric_millimetre_value_is_not_stored_as_centimetres():
    """Sri Ramachandra PAEDIATRIC: PWs 5 mm = 0.5 cm, which sits just under the 0.6 cm adult
    floor. Both readings being implausible must not default to the 10x-wrong one."""
    ef = ex.resolve_value("PWs", "5", 96.0, "table", "LVPWS | 5")
    assert ef.value == "5 mm"
    assert ef.meta["conversion_applied"] is False


def test_prose_restatement_does_not_displace_a_printed_number():
    """SonoNet: the table prints "Est. EF: 40" and the impression says "Mildly reduced ...
    ejection fraction". Storing the words loses the only number the report gave."""
    doc = _build_doc_result(
        narrative_lines=[("Est. EF: 40 (>55%)", 96.0),
                         ("IMPRESSION:", 96.0),
                         ("Mildly reduced left ventricular ejection fraction.", 96.0)],
    )
    results, _ = ex.extract_parameters_structured(doc)
    assert results["EF"].value == "40%"


def test_narrative_still_wins_for_a_qualitative_only_field():
    """The rule above must stay narrow -- MV is qualitative by definition, so prose remains its
    proper source even though a number appears elsewhere on the line."""
    doc = _build_doc_result(
        narrative_lines=[("Findings:", 96.0), ("Mild mitral regurgitation.", 96.0)],
    )
    results, _ = ex.extract_parameters_structured(doc)
    assert results["MV"].value is not None
    assert "regurgitation" in results["MV"].value.lower()


# ===========================================================================
# Not every string is a finding
# ===========================================================================
def test_ocr_noise_is_not_stored_as_a_valve_finding():
    """2D-ECHO photo: "(IVAK PEN" was read off the watermark and stored as the MV finding,
    which then feeds the 33-rule prediction engine as fact."""
    results, _, _ = _run(_table([[("Mitral Valve", 0.96), ("(IVAK PEN", 0.96)]], [100, 400]))
    assert results["MV"].value is None
    assert results["MV"].flagged is True


def test_mv_annulus_does_not_hijack_the_aortic_annulus():
    """Hi-Precision: "MV Annulus" scores 82.4 against the bare "Annulus" synonym, just over the
    82 threshold, and stored the mitral annulus 2.8 as the aortic annulus."""
    results, _, _ = _run(_table([
        [("MV Annulus", 0.96), ("2.8", 0.96)],
        [("Annulus", 0.96), ("1.9", 0.96)],
    ], [100, 400]))
    assert results["Ao_Annulus"].value == "1.9 cm"


# ===========================================================================
# Chong Hua: interleaved Normal(F/M) columns and section headers
# ===========================================================================
def test_normal_range_column_marker_blocks_the_whole_numeric_run():
    """Chong Hua: the RV value cell is blank ("continued on page 2"), and the row's own normal
    range read as "M:2.6-3.4 cm F: 2.33.1 cm" after OCR mangled "2.3+-0.3".

    Blocking only the FIRST number left the "1 cm" tail to be stored as RV_Size, so the marker
    has to cover the entire numeric run it introduces.
    """
    line = "I.Right Ventricle Ascending cm M:2.6-3.4 cm F: 2.33.1 cm Max Velocity"
    assert ex.find_measurement(line[len("I.Right Ventricle"):]) is None


def test_section_header_does_not_borrow_a_value_from_another_section():
    """Chong Hua: "IV.Right Atrium" is a header with an empty value cell. Scanning right without
    bound reached three columns across into the Diastolic Function block and took "Lateral e'".
    """
    results, _, _ = _run(_table([
        [("PWd", 0.96), ("0.8", 0.96), ("IV.Right Atrium", 0.96), ("", 0.96), ("", 0.96),
         ("Lateral e'", 0.96), ("0.11 m/sec", 0.96)],
    ], [100, 300, 500, 700, 900, 1100, 1300]))

    assert results["PWd"].value == "8 mm"
    assert results["RA_Size"].value is None
    assert results["RA_Size"].flagged is True


def test_merged_label_and_value_in_one_grid_cell_is_split():
    """Chong Hua: OCR lost the closing paren AND the column boundary, giving one cell reading
    "RAD (major3.3". Such a cell holds a number, so it is rejected as a label and the parameter
    was lost entirely -- it has to be split back into a label and a value."""
    results, _, _ = _run(_table([[("PWs", 0.96), ("1.3", 0.96), ("RAD (major3.3", 0.96)]],
                                [100, 300, 600]))
    assert results["RA_Size"].value == "3.3 cm"
    assert results["PWs"].value == "13 mm"


# ===========================================================================
# PHI must not leave the machine
# ===========================================================================
@pytest.mark.parametrize("line", [
    "Patient Name : Marshall Pamela",
    "HH NO. : 1029139",
    "Date of Birth: 06.06.1951",
    "Referred By : DR P P ASHOK",
])
def test_identifiers_are_redacted_before_the_groq_call(line):
    assert line not in gx.redact_identifiers(line)


@pytest.mark.parametrize("line", [
    "LVIDd 43.1 mm", "EF (Teich) 73.3 %", "Mild mitral regurgitation.",
    "PASP 16mm of Hg by PAT", "IVSTd 11.9 mm",
])
def test_redaction_leaves_clinical_content_untouched(line):
    assert gx.redact_identifiers(line) == line


def test_identifier_labels_are_dropped_from_the_semantic_label_batch(monkeypatch):
    """A label carrying an identifier is never a cardiac parameter, so it must not be sent."""
    sent = {}

    def _capture(prompt, **kw):
        sent["p"] = prompt
        return "[]"

    monkeypatch.setattr(gx, "_call_groq_text", _capture)
    gx.resolve_labels_semantically(["HH NO. : 1029139", "LAd (AP)"], ["LA_Diameter"])
    assert "1029139" not in sent["p"]
    assert "LAd (AP)" in sent["p"]


# ===========================================================================
# Per-report JSON export
# ===========================================================================
def test_extraction_json_records_value_confidence_and_flag(tmp_path, monkeypatch):
    """Every processed report writes its own JSON audit record, built from the SAME
    ExtractedField objects as the DB row so the two can never disagree."""
    from app import models
    from app.ocr import export as exp

    monkeypatch.setattr(exp, "EXTRACTION_JSON_DIR", tmp_path)

    doc = _build_doc_result(table_rows=[[("EF", 96.0), ("65%", 96.0)]])
    results, _ = ex.extract_parameters_structured(doc)

    report = models.CardiacReport(
        report_uid="uid-1", patient_name="Test Patient", original_filename="r.pdf",
        status="extracted",
    )
    for canon, ef in results.items():
        setattr(report, ef.db_field, ef.value)
    report.confidence_scores = {ef.db_field: ef.confidence for ef in results.values()}
    report.flagged_params = [ef.db_field for ef in results.values() if ef.flagged]
    report.extraction_meta = {ef.db_field: ef.meta for ef in results.values()}

    path = exp.write_extraction_json(report, results, doc)
    assert path is not None and path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["report_uid"] == "uid-1"
    assert payload["parameters"]["EF"]["value"] == "65%"
    assert payload["parameters"]["EF"]["detected"] is True
    assert payload["parameters"]["EF"]["source"] == "table"
    assert payload["parameters"]["LVIDd"]["detected"] is False
    # Against the dictionary, not a hardcoded count -- the export must cover every canonical
    # parameter, and that set grows.
    from app.ocr.parameter_dict import PARAMETERS
    assert payload["summary"]["parameters_total"] == len(PARAMETERS)


def test_extraction_json_is_written_even_when_nothing_was_read(tmp_path, monkeypatch):
    from app import models
    from app.ocr import export as exp

    monkeypatch.setattr(exp, "EXTRACTION_JSON_DIR", tmp_path)
    report = models.CardiacReport(report_uid="uid-2", patient_name="Test", status="failed",
                                  progress_stage="Failed: no text could be read")
    path = exp.write_extraction_json(report, {}, {})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["summary"]["parameters_detected"] == 0


# ===========================================================================
# Detector tuning: measured, and deliberately left off
# ===========================================================================
def test_upscaling_is_disabled_by_default():
    """Upscaling small pages fixes individual recognition errors but scored WORSE overall
    (141 -> 137 correct), so it is configurable and off. See the measurement table in config.py.
    """
    import numpy as np

    from app.ocr import text_extraction as tx

    small = np.zeros((452, 400, 3), dtype=np.uint8)
    assert tx.upscale_for_ocr(small, min_long_side=0) is small


def test_upscaling_enlarges_only_small_pages_when_enabled():
    import numpy as np

    from app.ocr import text_extraction as tx

    small = np.zeros((452, 400, 3), dtype=np.uint8)
    assert max(tx.upscale_for_ocr(small, min_long_side=1600).shape[:2]) == 1600

    # A page already at or above the target is never resampled for nothing.
    big = np.zeros((2000, 1500, 3), dtype=np.uint8)
    assert tx.upscale_for_ocr(big, min_long_side=1600) is big


# ===========================================================================
# Valve Doppler parameters (parameter_dict v2.1.0)
# ===========================================================================
def test_valve_velocity_is_its_own_parameter_not_the_valve_finding():
    """Hi-Precision/Duality print a peak velocity and gradient per valve. Those are numbers, and
    the *_finding fields are qualitative -- storing 1.4 as "the aortic valve finding" is the bug
    test_velocity_row_is_not_stored_as_a_valve_finding guards. They need their own fields."""
    results, _, _ = _run(_table([
        [("AV Vel", 0.96), ("134.9 cm/s", 0.96)],
        [("AV PG", 0.96), ("7.3 mmHg", 0.96)],
    ], [100, 400]))

    # cm/s is converted to the canonical m/s.
    assert results["AV_Peak_Velocity"].value == "1.349 m/s"
    assert results["AV_Peak_Gradient"].value == "7.3 mmHg"
    # The qualitative finding field is untouched.
    assert results.get("AV") is None or results["AV"].value is None


def test_velocity_units_round_trip_both_ways():
    """Reports print the same measurement as m/s or cm/s; both must land on m/s."""
    assert ex.resolve_value("AV_Peak_Velocity", "1.4 m/sec", 96.0, "table", "").value == "1.4 m/s"
    assert ex.resolve_value("AV_Peak_Velocity", "238.6 cm/s", 96.0, "table", "").value == "2.386 m/s"


def test_area_unit_is_not_read_as_a_length():
    """"9.23 cm2" is an AREA. Before cm2 was added ahead of cm in the unit alternation, the
    regex matched the "cm" inside "cm2" and stored an atrial area as a linear dimension."""
    from app.ocr.extractor import find_measurement, normalize_unit
    m = find_measurement("9.23 cm2")
    assert normalize_unit(m.group(2)) == "cm2"
    assert ex.resolve_value("MV_Area", "4.00 cm2", 96.0, "table", "").value == "4 cm2"


def test_mitral_annulus_has_its_own_field_and_does_not_touch_aortic():
    results, _, _ = _run(_table([
        [("MV Annulus", 0.96), ("2.8", 0.96)],
        [("Annulus", 0.96), ("1.9", 0.96)],
    ], [100, 400]))
    assert results["MV_Annulus"].value == "2.8 cm"
    assert results["Ao_Annulus"].value == "1.9 cm"


# ===========================================================================
# Rules 34-36: quantitative valve grading
# ===========================================================================
@pytest.mark.parametrize("vmax,expected", [
    ("4.2 m/s", "severe"), ("3.4 m/s", "moderate"), ("2.7 m/s", "mild"),
])
def test_rule_34_grades_aortic_stenosis_from_peak_velocity(vmax, expected):
    from app.predictor.rules import evaluate_all
    out = evaluate_all({"av_peak_velocity": vmax})
    hit = next(d for d in out["diseases"] if d["rule_id"] == 34)
    assert hit["severity"] == expected


@pytest.mark.parametrize("area,expected", [
    ("0.8 cm2", "severe"), ("1.2 cm2", "moderate"), ("1.8 cm2", "mild"),
])
def test_rule_35_grades_mitral_stenosis_from_valve_area(area, expected):
    from app.predictor.rules import evaluate_all
    out = evaluate_all({"mv_area": area})
    hit = next(d for d in out["diseases"] if d["rule_id"] == 35)
    assert hit["severity"] == expected


def test_rule_36_flags_elevated_pulmonary_pressure_from_tr_velocity():
    from app.predictor.rules import evaluate_all
    assert any(d["rule_id"] == 36 for d in evaluate_all({"tv_peak_velocity": "3.5 m/s"})["diseases"])
    assert not any(d["rule_id"] == 36 for d in evaluate_all({"tv_peak_velocity": "2.1 m/s"})["diseases"])


def test_normal_valve_numbers_do_not_trigger_stenosis_rules():
    """A normal study must not be graded as stenotic -- these rules add findings, not noise."""
    from app.predictor.rules import evaluate_all
    out = evaluate_all({"av_peak_velocity": "1.3 m/s", "mv_area": "4.0 cm2",
                        "tv_peak_velocity": "2.0 m/s"})
    assert not any(d["rule_id"] in (34, 35, 36) for d in out["diseases"])


def test_missing_valve_doppler_leaves_new_rules_unevaluated():
    """Missing data must stay 'insufficient', never default to normal."""
    from app.predictor.rules import evaluate_all, normalize_params
    from app.predictor import rules as R
    p = normalize_params({})
    for fn in (R.rule_34_aortic_stenosis_severity, R.rule_35_mitral_stenosis_severity,
               R.rule_36_tr_velocity_pulmonary_pressure):
        assert fn(p).matched is None


# ===========================================================================
# Report layout: sections, labels, wall condition, impression
# ===========================================================================
def test_aortic_root_has_exactly_one_owner():
    """The reported bug: "Aortic Root Diameter" was a synonym of BOTH Ao_Diameter and Ao_Root,
    and the review page rendered ao_diameter under the label "Aortic Root Diameter" while the
    value landed in ao_root. A report printing "Aortic Root 28.00 mm" showed as not-in-report."""
    from app.ocr.parameter_dict import PARAMETERS

    owners = [c for c, m in PARAMETERS.items()
              if any(s.strip().lower() == "aortic root diameter" for s in m["synonyms"])]
    assert owners == ["Ao_Root"]
    assert ex._match_label_exact_or_fuzzy("Aortic Root")[0] == "Ao_Root"


def test_no_synonym_is_claimed_by_two_canonicals():
    """One label, one owner -- otherwise which field a value lands in depends on tie ordering."""
    from collections import defaultdict

    from app.ocr.parameter_dict import PARAMETERS

    owners = defaultdict(set)
    for canon, meta in PARAMETERS.items():
        for syn in meta["synonyms"]:
            owners[syn.strip().lower()].add(canon)
    assert {s: sorted(o) for s, o in owners.items() if len(o) > 1} == {}


def test_every_parameter_appears_in_exactly_one_report_section():
    """The review page builds itself from REPORT_SECTIONS plus dedicated sections (#bsa-section).
    A parameter missing from both is extracted but invisible."""
    from app.ocr.parameter_dict import FIELD_LABELS, PARAMETERS, REPORT_SECTIONS

    section_fields = [f for s in REPORT_SECTIONS for f in s["fields"]] + ["bsa", "height", "weight"]
    db_fields = {m["db_field"] for m in PARAMETERS.values()}

    assert set(section_fields) == db_fields, "every parameter must be displayed exactly once"
    assert len(section_fields) == len(set(section_fields)), "no field listed in two sections"
    assert [f for f in section_fields if f not in FIELD_LABELS] == []


def test_report_sections_include_the_new_ao_and_wall_fields():
    from app.ocr.parameter_dict import REPORT_SECTIONS

    by_key = {s["key"]: s["fields"] for s in REPORT_SECTIONS}
    assert "ao_root" in by_key["aorta"]
    assert "wall" in by_key
    for f in ("relative_wall_thickness", "lv_mass", "rwma", "septal_wall_motion",
              "apical_wall_motion"):
        assert f in by_key["wall"]


@pytest.mark.parametrize("key,prefix", [
    ("valve_aortic", "av_"), ("valve_mitral", "mv_"),
    ("valve_tricuspid", "tv_"), ("valve_pulmonary", "pv_"),
])
def test_each_valve_section_holds_only_that_valve(key, prefix):
    """Per-valve sections: a mitral number must never surface under the aortic heading."""
    from app.ocr.parameter_dict import REPORT_SECTIONS

    fields = next(s["fields"] for s in REPORT_SECTIONS if s["key"] == key)
    assert fields and all(f.startswith(prefix) for f in fields)


def test_wall_dictionary_is_present_and_its_collisions_are_resolved():
    from app.ocr.parameter_dict import PARAMETERS, WALL_PARAMETER_DICTIONARY

    assert len(WALL_PARAMETER_DICTIONARY) == 14
    # Labels that two keys claimed in the supplied dictionary stay with their established owner.
    assert "Posterior Wall" in PARAMETERS["PWd"]["synonyms"]
    assert "Posterior Wall" not in PARAMETERS["Posterior_Wall_Motion"]["synonyms"]
    assert "RWMA" in PARAMETERS["Wall_Motion"]["synonyms"]
    assert "RWMA" not in PARAMETERS["RWMA"]["synonyms"]
    # "No RWMA" is a finding, not a label: as a synonym it outranked "RWMA" and stole the line.
    assert "No RWMA" not in PARAMETERS["RWMA"]["synonyms"]


def test_wall_thickness_label_still_reads_as_a_measurement():
    """"Posterior Wall 9.00 mm" (Spectrum) is a thickness. Routing it to a motion field would
    store a measurement as a qualitative finding."""
    results, _, _ = _run(_table([[("Posterior Wall", 0.96), ("9.00 mm", 0.96)]], [100, 400]))
    assert results["PWd"].value == "9 mm"
    assert results.get("Posterior_Wall_Motion") is None or \
        results["Posterior_Wall_Motion"].value is None


def test_impression_is_captured_verbatim():
    """The card must reproduce the cardiologist's own words -- no reordering or rewording."""
    raw = "\n".join([
        "2D ECHO:",
        "LVEF 60%",
        "IMPRESSION:",
        "Moderate concentric LVH.  Normal size LV",
        "Type I diastolic dysfunction",
    ])
    out = ex.extract_impression_text(raw)
    assert out.startswith("IMPRESSION:")
    assert "Moderate concentric LVH.  Normal size LV" in out   # double space preserved
    assert "Type I diastolic dysfunction" in out
    assert "LVEF 60%" not in out                                # section starts at the heading


def test_impression_returns_none_when_the_report_has_no_conclusion():
    assert ex.extract_impression_text("M-MODE MEASUREMENTS:\nAortic Root 28.00 mm") is None


def test_impression_is_never_reconstructed_from_the_findings():
    """The card shows the report's OWN conclusion or nothing at all. A report full of findings
    but with no Impression/Conclusion heading must yield None -- stitching those sentences
    together would manufacture a conclusion the cardiologist never wrote."""
    raw = "\n".join([
        "MITRAL",
        "Anterior Leaflet Structure   Normal",
        "AORTA",
        "Cuspal Opening   Normal",
        "Structure   Normal",
        "TRICUSPID",
        "Tricuspid Structure   Normal",
    ])
    assert ex.extract_impression_text(raw) is None


@pytest.mark.parametrize("heading", [
    "SUMMARY:", "Interpretation:", "FINAL DIAGNOSIS:", "Comment:", "Advice :",
])
def test_only_impression_and_conclusion_count_as_headings(heading):
    """These head other blocks on real templates. Treating them as the conclusion produced a
    card that looked authoritative while showing something else entirely."""
    raw = heading + "\nSome text that is not the cardiologist's conclusion."
    assert ex.extract_impression_text(raw) is None


@pytest.mark.parametrize("heading,expected", [
    ("IMPRESSION", True), ("Impression:", True), ("CONCLUSION :", True),
    ("Conclusions", True), ("FINAL IMPRESSION:", True),
    # OCR damage on the heading itself -- a real read from the 2D-ECHO photo.
    ("FINAL.DIAGNOSIS (Impresslon)-", True),
    # Prose that merely uses the word is not a heading.
    ("Findings are consistent with the clinical impression of mitral valve disease", False),
    ("The conclusion of the treadmill test was reported separately by the technician", False),
])
def test_impression_heading_detection(heading, expected):
    assert ex._is_impression_heading(heading) is expected


def test_impression_stops_at_the_next_section():
    """One appendix PDF holds several cases; without a stop rule the first Conclusion swallowed
    the whole next report (55 lines instead of 4)."""
    raw = "\n".join([
        "Conclusion:",
        "Mild mitral regurgitation.",
        "ECHOCARDIOGRAPHY REPORT SUMMARY: CASE 2",
        "LA dimension (mm) 54",
    ])
    out = ex.extract_impression_text(raw)
    assert "Mild mitral regurgitation." in out
    assert "CASE 2" not in out and "LA dimension" not in out
