"""
Parameter dictionary for 2D Echocardiogram reports.

SCHEMA (v2.0.0)
---------------
Each canonical parameter maps to:
  - synonyms: label variants used by different hospitals/machines (GE, Philips, Siemens, etc.)
  - kind: "numeric" (has a numeric value + unit), "text" (qualitative finding), or
          "chamber_size" (numeric OR qualitative -- reports vary)
  - units: acceptable unit tokens for numeric fields (used to build the extraction regex)
  - canonical_unit: the ONE unit every value for this field is normalized to and stored as,
    e.g. "cm" for linear cardiac dimensions, "%" for EF, "mmhg" for pressures, "" for unitless
    ratios. None for text-kind fields. This is what makes unit handling deterministic: a value
    is never assigned to a model field until it has been converted into this unit.
  - alternate_unit: the one other unit real reports commonly use for this field (e.g. "mm" when
    canonical_unit is "cm"). Used to disambiguate a bare number with no unit symbol by checking
    which interpretation is physiologically plausible. None if there's no common alternate.
  - valid_range: (min, max) in canonical_unit -- clinically-abnormal-but-plausible bound. A
    value outside this range is STILL STORED, just flagged for doctor review (may be a genuine
    abnormal finding, not an OCR error).
  - impossible_range: (min, max) in canonical_unit -- physiologically IMPOSSIBLE outer bound,
    always wider than valid_range. A value outside this range is REJECTED outright (not stored,
    field left null) because it almost certainly reflects an OCR error such as a dropped decimal
    point (e.g. "2.01" misread as "201") rather than a real clinical measurement.
  - disqualifying_suffixes: label tokens that, if present anywhere in a candidate label besides
    the synonym match itself, disqualify that candidate from matching THIS (base) field -- e.g.
    "LVEDD/BSA" must not satisfy the "LVEDD" synonym for the base LVIDd field, because "/BSA"
    signals this is actually the indexed variant, a clinically distinct measurement. Only
    applies to base (non-indexed) fields; indexed fields are exempt from their own suffix list.
  - indexed_variant_of: canonical name of the base field this is a BSA-indexed variant of, or
    None. Indexed fields (LVIDd_Indexed, LA_Diameter_Indexed) get their own model column instead
    of being dropped or silently overwriting the primary (non-indexed) value.
  - db_field: the CardiacReport model column this maps to.

Rationale for non-obvious synonym additions is called out inline where it isn't self-evident
from clinical terminology alone.
"""

PARAMETER_DICT_VERSION = "2.2.0"

# Suffix tokens that mark a label as referring to a BSA-indexed (or otherwise distinct
# "(i)"-suffixed) variant of a base measurement rather than the base measurement itself.
# Applied by default to every non-indexed numeric/chamber_size field so that even fields
# without a registered indexed_variant_of counterpart fail safe (unmatched, not silently wrong)
# if an indexed-looking label is encountered.
_DEFAULT_DISQUALIFYING_SUFFIXES = ("/bsa", "index", "indexed", "(i)")

PARAMETERS = {
    "EF": {
        "synonyms": ["EF", "Ejection Fraction", "LVEF", "EF (Simpson's)", "EF Simpsons",
                     "Ejection Fraction (Simpson)", "LV EF", "EF(%)", "EF %",
                     # Additional real-world shorthand: Simpson biplane / visual (eyeballed)
                     # estimation are both common ways sonographers phrase how EF was derived.
                     "EF (Simpson)", "EF Simpson Biplane", "EF (Visual)", "Eyeballed EF",
                     "LVEF (Simpson)"],
        "kind": "numeric",
        "units": ["%"],
        "canonical_unit": "%",
        "alternate_unit": None,
        "valid_range": (10, 80),
        "impossible_range": (5, 95),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ef",
    },
    "LVIDd": {
        "synonyms": ["LVIDd", "LVIDD", "LVID d", "LV ID (d)", "LVID (Diastole)",
                     "LV Internal Diameter Diastole", "LVEDD", "LV Diastolic Diameter",
                     # "Dimension" variants (frequent across Indian & European templates)
                     "Left Ventricular Internal Dimension - Diastole",
                     "Left Ventricular Internal Dimension Diastole",
                     "Left Ventricular Internal Dimension (Diastole)",
                     "Left Ventricular Internal Dimension (d)",
                     "Left Ventricular Internal Dimension - D",
                     "Left Ventricular Internal Dimension",
                     "LV Internal Dimension Diastole",
                     "LV Internal Dimension (Diastole)",
                     "LV Internal Dimension (d)",
                     "LV Internal Dimension",
                     "LVID Diastole",
                     # Additional shorthand seen across GE/Philips/Siemens report templates.
                     "LVID(d)", "LV Diam Diastole", "LV End-Diastolic Diameter",
                     # Forms seen on the real reports in tests/fixtures: the spaced-paren
                     # "LVID (d)" (2D-ECHO photo), the terse "LVd:" (SonoNet), and the
                     # spelled-out "LV end-diastolic diameter (mm)" (Appendix-2 PDF).
                     "LVID (d)", "LVd", "LV end-diastolic diameter",
                     # Spelled-out ED/ES forms (Esaote/Mindray templates). The long form
                     # "Left Ventricular ED" is deliberately NOT listed: it fuzzy-matches the
                     # bare chamber heading "Left Ventricle" at 84.8, over the 82 threshold,
                     # and stored that section's "Normal" as the LVIDd measurement.
                     "LV ED", "LVED"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "mm",
        "alternate_unit": "cm",
        "valid_range": (20.0, 70.0),
        "impossible_range": (10.0, 100.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "lvidd",
    },
    "LVIDs": {
        "synonyms": ["LVIDs", "LVIDS", "LVID s", "LV ID (s)", "LVID (Systole)",
                     "LV Internal Diameter Systole", "LVESD", "LV Systolic Diameter",
                     # "Dimension" variants
                     "Left Ventricular Internal Dimension - Systole",
                     "Left Ventricular Internal Dimension Systole",
                     "Left Ventricular Internal Dimension (Systole)",
                     "Left Ventricular Internal Dimension (s)",
                     "Left Ventricular Internal Dimension - S",
                     "LV Internal Dimension Systole",
                     "LV Internal Dimension (Systole)",
                     "LV Internal Dimension (s)",
                     "LVID Systole",
                     # Counterparts of the LVIDd additions above, same source reports.
                     "LVID (s)", "LVID (S)", "LVs", "LV end-systolic diameter",
                     "LV ES", "LVES"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "mm",
        "alternate_unit": "cm",
        "valid_range": (10.0, 60.0),
        "impossible_range": (5.0, 90.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "lvids",
    },
    "IVSd": {
        "synonyms": ["IVSd", "IVSD", "IVS d", "IVS (Diastole)", "Interventricular Septum",
                     "IVS Thickness", "Septal Wall Thickness", "IVS",
                     # "IVS (d)" and clinical-longhand variants.
                     "IVS (d)", "Interventricular Septal Thickness Diastole",
                     # Philips/Duality "thickness" infix forms.
                     "IVSTd", "IVS Td", "IVSTD",
                     # From WALL_PARAMETER_DICTIONARY (see bottom of file).
                     "IV Septum", "IV Septum Thickness", "Interventricular Septum Diastole",
                     "Interventricular Septal Thickness", "Septal Thickness (Diastole)",
                     "IVS (ED)"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "mm",
        "alternate_unit": "cm",
        "valid_range": (5.0, 25.0),
        "impossible_range": (2.0, 40.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ivsd",
    },
    "PWd": {
        "synonyms": ["PWd", "PWD", "PW d", "PW (Diastole)", "Posterior Wall Thickness",
                     "Posterior Wall", "LVPWd", "PW Thickness", "PW",
                     "PW (d)", "Posterior Wall Thickness Diastole", "LV Posterior Wall Diastole",
                     # Bare "LVPW" (SonoNet). Safe alongside "LVPWd"/"LVPWs": longest-synonym-
                     # first ordering claims those spans, and a trailing LETTER is rejected by
                     # the boundary check, so "LVPW" cannot swallow the systolic row.
                     "LVPW", "LVPWTd", "LVPWT d", "LVPWTD",
                     # From WALL_PARAMETER_DICTIONARY.
                     "Posterior Wall Diastole", "LV Posterior Wall",
                     "Left Ventricular Posterior Wall", "Posterior Wall (ED)"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "mm",
        "alternate_unit": "cm",
        "valid_range": (5.0, 25.0),
        "impossible_range": (2.0, 40.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "pwd",
    },
    "LA_Diameter": {
        "synonyms": ["LA Diameter", "LA Dia", "Left Atrium", "Left Atrial Diameter",
                     "LA Size", "LA(AP Diameter)", "LA",
                     # "LAd" family: many machines print the AP-axis LA measurement this way.
                     "LAd", "LAd (AP)", "LA AP Diameter", "Left Atrial AP Dimension",
                     # Chamber-view suffixed forms ("LAd4" = 4-chamber, "LAd2" = 2-chamber) are
                     # spelled out rather than left to "LAd" + a digit: the boundary check
                     # deliberately rejects a bare trailing integer, because on this same
                     # template "LAd4 39.4 mm" otherwise reads the ordinal 4 as the measurement.
                     "LAd4", "LAd2", "LAs", "LA (A-P)"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.5, 6.0),
        "impossible_range": (0.5, 9.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "la_diameter",
    },
    "Ao_Diameter": {
        # Generic catch-all for reports that just say "Aortic Diameter" without specifying
        # which landmark (annulus / root / STJ). Deliberately narrowed vs. the old dictionary:
        # "Aortic Root"-style synonyms have been MOVED to Ao_Root below, since real reports
        # measure and report annulus / root (sinus of Valsalva) / STJ as distinct numbers.
        # "Aortic Root Diameter" is NOT listed here -- it belongs to Ao_Root, which also claimed
        # it. Owning one label in two canonicals made the winner depend on synonym-index tie
        # ordering, and the review page compounded it by rendering ao_diameter under the LABEL
        # "Aortic Root Diameter" while the value correctly landed in ao_root. Net effect: a
        # report plainly printing "Aortic Root 28.00 mm" displayed as not-in-report.
        # One label, one owner.
        "synonyms": ["Ao Diameter", "Aortic Diameter", "AO", "Aorta"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.5, 5.0),
        "impossible_range": (0.5, 8.0),
        # Bare "Aorta" is how Hi-Precision prints the generic aortic diameter, but the same word
        # heads several rows that are NOT this measurement -- the aortic VALVE finding, and the
        # specific landmarks that have their own canonical fields (root / sinus / STJ / arch /
        # ascending). Those are disqualified explicitly so "Aorta" stays safe to match.
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + (
            "valve", "root", "sinus", "valsalva", "stj", "junction",
            "ascending", "asc", "arch", "descending", "abdominal",
            "coarctation", "dissection", "aneurysm",
        ),
        "indexed_variant_of": None,
        "db_field": "ao_diameter",
    },
    "Ao_Annulus": {
        # Bare "Annulus" appears under the "Aorta" section of Siemens/Chong Hua-style report
        # templates. The spelled-out "Mitral Annulus"/"Tricuspid Annulus" score well below the
        # fuzzy threshold against it and fail to match on their own -- but the ABBREVIATED
        # "MV Annulus" scores 82.4, just over LABEL_MATCH_THRESHOLD (82), and did hijack this
        # field on a real report (Hi-Precision: stored the mitral annulus 2.8 as the aortic
        # annulus 1.9). The other-valve prefixes below are therefore disqualified explicitly
        # rather than left to the fuzzy score. "AV Annulus" is deliberately absent from that
        # list -- the aortic VALVE annulus is exactly this measurement.
        "synonyms": ["Ao Annulus", "Aortic Annulus", "AV Annulus", "Annulus Diameter",
                     "Aortic Valve Annulus", "Annulus"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.5, 3.0),
        "impossible_range": (0.5, 5.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + (
            "mv ", "tv ", "pv ", "mitral", "tricuspid", "pulmonic", "pulmonary",
        ),
        "indexed_variant_of": None,
        "db_field": "ao_annulus",
    },
    "Ao_Root": {
        # "SOV" is the standard printed abbreviation for Sinus of Valsalva on real echo
        # report templates (observed on Chong Hua Hospital / Siemens SC2000 output).
        "synonyms": ["Aortic Root", "Ao Root", "Aortic Root Diameter", "AoRoot",
                     "Ao Root Diameter", "Sinus of Valsalva", "Aortic Sinus", "SOV"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (2.0, 4.0),
        "impossible_range": (0.5, 6.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ao_root",
    },
    "Ao_STJ": {
        "synonyms": ["STJ", "Sinotubular Junction", "Sino-tubular Junction", "ST Junction",
                     "Aortic STJ"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.5, 3.5),
        "impossible_range": (0.5, 5.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ao_stj",
    },
    "LVIDd_Indexed": {
        # BSA-indexed LV end-diastolic diameter (cm/m^2). A distinct, clinically meaningful
        # measurement from LVIDd itself -- must land in its own column, never overwrite LVIDd.
        "synonyms": ["LVEDD/BSA", "LVEDD Index", "LVEDD Indexed", "LVIDd/BSA", "LVIDd Index",
                     "LVEDD (I)", "LV End-Diastolic Diameter Index", "LVEDDI"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm/m2",
        "alternate_unit": None,
        "valid_range": (1.0, 4.0),
        "impossible_range": (0.3, 6.0),
        "disqualifying_suffixes": (),
        "indexed_variant_of": "LVIDd",
        "db_field": "lvidd_indexed_value",
    },
    "LA_Diameter_Indexed": {
        "synonyms": ["LA/BSA", "LA Diameter Index", "LA Diameter Indexed", "LAD/BSA", "LA (I)",
                     "Indexed LA Diameter", "LAd/BSA", "LAd Index"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm/m2",
        "alternate_unit": None,
        "valid_range": (1.0, 3.0),
        "impossible_range": (0.3, 5.0),
        "disqualifying_suffixes": (),
        "indexed_variant_of": "LA_Diameter",
        "db_field": "la_diameter_indexed_value",
    },
    "RV_Size": {
        "synonyms": ["RV Size", "RV Diameter", "Right Ventricle", "RV", "RVID",
                     "Right Ventricle Size", "RV Dimension",
                     "RVd", "RV (mid)", "RV Basal Diam", "RV Mid Diam"],
        # Real reports report RV size either as a raw diameter (cm/mm) or as a
        # qualitative descriptor ("Normal" / "Mildly Dilated" / "Enlarged").
        # "chamber_size" tries numeric extraction first, falls back to text.
        "kind": "chamber_size",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.0, 5.0),
        "impossible_range": (0.5, 8.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "rv_size",
    },
    "IVSs": {
        "synonyms": ["IVSs", "IVSS", "IVS s", "IVS (Systole)", "Interventricular Septum Systole",
                     "IVS Systolic Thickness", "Septal Wall Thickness Systole",
                     # From WALL_PARAMETER_DICTIONARY.
                     "Interventricular Septal Thickness Systole",
                     "Septal Thickness (Systole)", "IVS (ES)"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "mm",
        "alternate_unit": "cm",
        "valid_range": (6.0, 30.0),
        "impossible_range": (3.0, 50.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ivss",
    },
    "PWs": {
        "synonyms": ["PWs", "PWS", "PW s", "PW (Systole)", "Posterior Wall Thickness Systole",
                     "LVPWs", "PW Systolic Thickness",
                     # From WALL_PARAMETER_DICTIONARY.
                     "Posterior Wall Systole", "Posterior Wall (ES)"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "mm",
        "alternate_unit": "cm",
        "valid_range": (6.0, 30.0),
        "impossible_range": (3.0, 50.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "pws",
    },
    "RA_Size": {
        "synonyms": ["RA Size", "RA Diameter", "Right Atrium", "RA", "Right Atrial Size",
                     "Right Atrium Size", "RA Dimension",
                     # Siemens/Chong Hua prints the RA major/minor axes as "RAD (major)". Bare
                     # "RAD" is listed too because OCR merges the label into its value on that
                     # template ("RAD (major3.3", closing paren lost) and only the bare form
                     # survives. "RAD (minor)" scores 72 against it and "(major/BSA) RAD" 33, so
                     # neither the minor axis nor the indexed variant can claim this field.
                     "RAD (major)", "RAD major", "RA (R-L)", "RAD"],
        "kind": "chamber_size",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.5, 6.0),
        "impossible_range": (0.5, 9.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ra_size",
    },
    "IVC": {
        "synonyms": ["IVC", "Inferior Vena Cava", "IVC Diameter", "IVC Size",
                     "Inferior Vena Cava Diameter"],
        "kind": "chamber_size",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (0.5, 3.5),
        "impossible_range": (0.2, 5.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "ivc",
    },
    "Pericardial_Effusion": {
        "synonyms": ["Pericardial Effusion", "PE", "Pericardial Fluid", "Effusion",
                     "Pericardium", "Pericardial Space", "Pericardial Cavity",
                     "Pericardial Effusion / Fluid", "Pericardial Effusion/Fluid"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "pericardial_effusion",
    },
    "Clots_Thrombus": {
        "synonyms": [
            # Composite Clot/Vegetation/Mass variants (longest first)
            "Clots and Vegetations", "Clot and Vegetation", "Clots & Vegetations", "Clot & Vegetation",
            "Clots / Vegetations", "Clot / Vegetation", "Clots/Vegetations", "Clot/Vegetation",
            "Vegetations and Clots", "Vegetation and Clot", "Vegetations & Clots", "Vegetation & Clot",
            "Vegetations / Clots", "Vegetation / Clot", "Vegetations/Clots", "Vegetation/Clot",
            "Clots / Thrombus", "Clot / Thrombus", "Clots/Thrombus", "Clot/Thrombus", "Clots & Thrombus",
            "Mass / Thrombus", "Mass/Thrombus", "Clots / Masses", "Clots/Masses", "Clot / Mass", "Clot/Mass",
            # Intracardiac & chamber specific
            "Intracardiac Thrombus", "Intracardiac Clot", "Intracardiac Mass", "Intracardiac Masses",
            "LV Thrombus", "LA Thrombus", "RA Thrombus", "RV Thrombus",
            "LV Clot", "LA Clot", "RA Clot", "RV Clot",
            # Standalone terms
            "Vegetations", "Vegetation", "Thrombus", "Thrombi", "Clots", "Clot",
        ],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "clots_thrombus",
    },
    "MV": {
        # "MR" is the near-universal printed shorthand for the mitral finding on report tables
        # ("MR: Mild"). It is the finding, not a separate parameter, so it maps to this field.
        "synonyms": ["MV", "Mitral Valve", "MV Finding", "Mitral", "MR",
                     # Leaflet-level findings ("AML - Normal", "PML thickened") are mitral
                     # valve findings; both abbreviations appear on Indian lab templates.
                     "AML", "PML", "Anterior Mitral Leaflet", "Posterior Mitral Leaflet"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "mv_finding",
    },
    "AV": {
        # "AR" = aortic regurgitation shorthand -- see the MR note above.
        "synonyms": ["AV", "Aortic Valve", "AV Finding", "Aortic", "AR"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "av_finding",
    },
    "TV": {
        # "TR" = tricuspid regurgitation shorthand -- see the MR note above.
        "synonyms": ["TV", "Tricuspid Valve", "TV Finding", "Tricuspid", "TR"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "tv_finding",
    },
    "PV": {
        # "PR" = pulmonic regurgitation shorthand -- see the MR note above. (Unambiguous here:
        # the ECG "PR interval" sense does not appear in a 2D-Echo measurement table.)
        "synonyms": ["PV", "Pulmonary Valve", "PV Finding", "Pulmonic Valve", "PR"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "pv_finding",
    },
    "E_A_Ratio": {
        "synonyms": ["E/A Ratio", "E/A", "Mitral E/A", "E A Ratio", "E:A Ratio",
                     # OCR glues the "MV" prefix onto the ratio on Philips/Duality pages.
                     "MVE/A", "MV E/A", "MVF E/A"],
        "kind": "numeric",
        "units": ["", "ratio"],
        "canonical_unit": "",
        "alternate_unit": None,
        "valid_range": (0.3, 3.5),
        "impossible_range": (0.1, 6.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "e_a_ratio",
    },
    "PASP": {
        "synonyms": ["PASP", "Pulmonary Artery Systolic Pressure", "RVSP", "PA Pressure",
                     "Estimated PASP", "PASP (mmHg)", "RVSP/TR", "RVSP / TR", "RVSP(TR)",
                     "PASP/TR", "PASP / TR", "PASP(TR)", "RVSP/RA", "PASP/RA",
                     # Hi-Precision prints the systolic PA pressure as "sPAP (TR Jet)".
                     "sPAP", "sPAP (TR Jet)", "Systolic PA Pressure"],
        "kind": "numeric",
        "units": ["mmhg", "mm hg"],
        "canonical_unit": "mmhg",
        "alternate_unit": None,
        "valid_range": (10, 100),
        "impossible_range": (5, 150),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "pasp",
    },
    "Wall_Motion": {
        "synonyms": ["Wall Motion", "RWMA", "Regional Wall Motion", "Wall Motion Abnormality",
                     "Wall Motion Abnormalities", "LV Wall Motion",
                     # Reports often name the abnormality rather than the category.
                     "Hypokinesia", "Hypokinesis", "Akinesis", "Dyskinesis",
                     # From WALL_PARAMETER_DICTIONARY.
                     "Global Wall Motion", "Wall Motion Analysis", "Wall Kinetics",
                     "Kinetics", "Contractility", "LV Contractility"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "wall_motion",
    },

    # =======================================================================================
    # VALVE DOPPLER (v2.1.0)
    # =======================================================================================
    # WHY THESE ARE SEPARATE FIELDS, not synonyms of MV/AV/TV/PV:
    # The MV/AV/TV/PV entries above are kind="text" -- they hold a qualitative FINDING ("Mild
    # Regurgitation"). Reports also print a peak VELOCITY and a peak GRADIENT for the same
    # valve, and those are clinically distinct numbers, not a restatement of the finding.
    # Hi-Precision prints "Aortic Valve 1.4 7.5" (m/s, mmHg) with the regurgitation columns
    # blank -- there is no finding on that page at all. Folding these into the finding fields
    # would store "1.4" as the aortic valve finding, which is the exact bug
    # test_velocity_row_is_not_stored_as_a_valve_finding exists to prevent.
    #
    # VELOCITY UNITS: reports use m/s and cm/s interchangeably for the same measurement
    # ("AV Vel 134.9 cm/s" vs "Aortic Valve 1.4 m/sec"), so canonical_unit is m/s with cm/s as
    # the alternate. Ranges below are peak values across normal-to-severe disease, wide enough
    # to keep a genuine severe stenosis rather than reject it as implausible.
    "AV_Peak_Velocity": {
        "synonyms": ["AV Vel", "AVVel", "AV Peak Vel", "AV Peak Velocity", "AV Vmax",
                     "Aortic Valve Peak Velocity", "Aortic Peak Velocity", "AV V Max",
                     "Aortic Vmax", "AV Max Velocity", "Transaortic Velocity"],
        "kind": "numeric",
        "units": ["m/s", "cm/s"],
        "canonical_unit": "m/s",
        "alternate_unit": "cm/s",
        "valid_range": (0.5, 6.0),
        "impossible_range": (0.1, 10.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "av_peak_velocity",
    },
    "AV_Peak_Gradient": {
        "synonyms": ["AV PG", "AVPG", "AV Peak Gradient", "AV Max PG", "AV Gradient",
                     "Aortic Valve Gradient", "Aortic Peak Gradient", "AV Peak PG",
                     "Peak AV Gradient", "AV maxPG"],
        "kind": "numeric",
        "units": ["mmhg"],
        "canonical_unit": "mmhg",
        "alternate_unit": None,
        "valid_range": (1.0, 120.0),
        "impossible_range": (0.1, 250.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + ("mean",),
        "indexed_variant_of": None,
        "db_field": "av_peak_gradient",
    },
    "MV_Peak_Velocity": {
        # "MV E Vel" is the E-wave peak, which IS the mitral inflow peak velocity reports quote
        # here. Deliberately does NOT include "MV A Vel" -- the A wave is a separate wave, and
        # E/A is already its own canonical parameter.
        "synonyms": ["MV Vel", "MV Peak Vel", "MV Peak Velocity", "MV Vmax", "MV E Vel",
                     "Mitral Valve Peak Velocity", "Mitral Peak Velocity", "MV E Velocity",
                     "Mitral E Velocity", "MV V Max"],
        "kind": "numeric",
        "units": ["m/s", "cm/s"],
        "canonical_unit": "m/s",
        "alternate_unit": "cm/s",
        "valid_range": (0.3, 3.0),
        "impossible_range": (0.05, 6.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "mv_peak_velocity",
    },
    "MV_Peak_Gradient": {
        "synonyms": ["MV PG", "MVPG", "MV Peak Gradient", "MV Gradient", "MV Max PG",
                     "Mitral Valve Gradient", "Mitral Peak Gradient", "Peak MV Gradient"],
        "kind": "numeric",
        "units": ["mmhg"],
        "canonical_unit": "mmhg",
        "alternate_unit": None,
        "valid_range": (1.0, 40.0),
        "impossible_range": (0.1, 100.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + ("mean",),
        "indexed_variant_of": None,
        "db_field": "mv_peak_gradient",
    },
    "TV_Peak_Velocity": {
        # TR Vmax is the tricuspid REGURGITANT jet velocity -- the number PASP is derived from,
        # and the one every report prints. Kept here rather than under PASP because PASP is the
        # derived pressure, not the velocity.
        "synonyms": ["TV Vel", "TV Peak Vel", "TV Peak Velocity", "TV Vmax", "TR Vmax",
                     "TRVmax", "TR Vel", "TR Peak Velocity", "TR Jet Velocity",
                     "Tricuspid Peak Velocity", "Tricuspid Regurgitant Velocity", "TRV"],
        "kind": "numeric",
        "units": ["m/s", "cm/s"],
        "canonical_unit": "m/s",
        "alternate_unit": "cm/s",
        "valid_range": (0.5, 6.0),
        "impossible_range": (0.1, 9.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "tv_peak_velocity",
    },
    "TV_Peak_Gradient": {
        "synonyms": ["TV PG", "TVPG", "TV Peak Gradient", "TR PGmax", "TRPGmax", "TR PG",
                     "TR Peak Gradient", "TV Max PG", "Tricuspid Peak Gradient",
                     "TR Peak gradient"],
        "kind": "numeric",
        "units": ["mmhg"],
        "canonical_unit": "mmhg",
        "alternate_unit": None,
        "valid_range": (1.0, 100.0),
        "impossible_range": (0.1, 200.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + ("mean",),
        "indexed_variant_of": None,
        "db_field": "tv_peak_gradient",
    },
    "PV_Peak_Velocity": {
        "synonyms": ["PV Vel", "PV Peak Vel", "PV Peak Velocity", "PV Vmax", "PVVmax",
                     "Pulmonic Peak Velocity", "Pulmonary Valve Peak Velocity", "PV V Max"],
        "kind": "numeric",
        "units": ["m/s", "cm/s"],
        "canonical_unit": "m/s",
        "alternate_unit": "cm/s",
        "valid_range": (0.3, 5.0),
        "impossible_range": (0.05, 8.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "pv_peak_velocity",
    },
    "PV_Peak_Gradient": {
        "synonyms": ["PV PG", "PVPG", "PV Peak Gradient", "PV PGmax", "PVPGmax",
                     "PV Max PG", "Pulmonic Peak Gradient", "Pulmonary Valve Gradient"],
        "kind": "numeric",
        "units": ["mmhg"],
        "canonical_unit": "mmhg",
        "alternate_unit": None,
        "valid_range": (1.0, 100.0),
        "impossible_range": (0.1, 200.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + ("mean",),
        "indexed_variant_of": None,
        "db_field": "pv_peak_gradient",
    },
    "LVOT_Peak_Velocity": {
        "synonyms": ["LVOT Vel", "LVOT Peak Vel", "LVOT Peak Velocity", "LVOT Vmax",
                     "LVOT V MAX", "LVOTVel", "LVOT Velocity"],
        "kind": "numeric",
        "units": ["m/s", "cm/s"],
        "canonical_unit": "m/s",
        "alternate_unit": "cm/s",
        "valid_range": (0.3, 3.0),
        "impossible_range": (0.05, 6.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "lvot_peak_velocity",
    },
    "LVOT_Peak_Gradient": {
        "synonyms": ["LVOT PG", "LVOTPG", "LVOT Peak Gradient", "LVOT MAX PG",
                     "LVOT Gradient", "LVOT Max Gradient"],
        "kind": "numeric",
        "units": ["mmhg"],
        "canonical_unit": "mmhg",
        "alternate_unit": None,
        "valid_range": (0.5, 100.0),
        "impossible_range": (0.1, 200.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES + ("mean",),
        "indexed_variant_of": None,
        "db_field": "lvot_peak_gradient",
    },
    "LVOT_VTI": {
        # Velocity-time integral, printed in cm. Unitless-looking on some templates, so
        # canonical_unit is cm with no alternate.
        "synonyms": ["LVOT VTI", "LVOTVTI", "LVOT V.T.I", "LVOT Velocity Time Integral",
                     "LVOT VII"],
        "kind": "numeric",
        "units": ["cm"],
        "canonical_unit": "cm",
        "alternate_unit": None,
        "valid_range": (5.0, 40.0),
        "impossible_range": (1.0, 80.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "lvot_vti",
    },
    "MV_Area": {
        # Mitral valve area, the number mitral-stenosis severity is graded on. Printed either as
        # a planimetered area or derived from pressure half-time, "MVA (PHT)".
        "synonyms": ["MVA", "MVA (PHT)", "MV Area", "Mitral Valve Area", "MVA PHT",
                     "Mitral Area", "MV Valve Area", "MVA(PHT)"],
        "kind": "numeric",
        "units": ["cm2"],
        "canonical_unit": "cm2",
        "alternate_unit": None,
        "valid_range": (0.3, 6.0),
        "impossible_range": (0.05, 12.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "mv_area",
    },
    "AV_Area": {
        # Aortic valve area, the primary number aortic stenosis severity is graded on.
        "synonyms": ["AVA", "AVA (Planimetry)", "AVA (Doppler)", "AV Area", "Aortic Valve Area",
                     "Aortic Area", "AV Valve Area", "AVA Planimetry", "AVA Doppler", "Aortic Area (cm2)"],
        "kind": "numeric",
        "units": ["cm2"],
        "canonical_unit": "cm2",
        "alternate_unit": None,
        "valid_range": (0.3, 6.0),
        "impossible_range": (0.05, 12.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "av_area",
    },
    "MV_Annulus": {
        # Distinct from Ao_Annulus. The bare "Annulus" synonym belongs to Ao_Annulus, which
        # explicitly disqualifies "mv "/"mitral" -- these three entries are its counterparts.
        "synonyms": ["MV Annulus", "Mitral Annulus", "Mitral Valve Annulus", "MV Ann"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.5, 4.0),
        "impossible_range": (0.5, 7.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "mv_annulus",
    },
    "TV_Annulus": {
        "synonyms": ["TV Annulus", "Tricuspid Annulus", "Tricuspid Valve Annulus", "TV Ann"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.0, 4.5),
        "impossible_range": (0.4, 7.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "tv_annulus",
    },
    # =======================================================================================
    # WALL CONDITION (v2.2.0) -- see WALL_PARAMETER_DICTIONARY at the bottom of this file
    # =======================================================================================
    "RWT": {
        "synonyms": ["RWT", "Relative Wall Thickness", "Relative Wall Thickness Index",
                     "Relative LV Wall Thickness", "Relative Thickness",
                     "LV Relative Wall Thickness"],
        "kind": "numeric",
        "units": [],
        "canonical_unit": "",          # a ratio -- 2*PWd/LVIDd, no unit
        "alternate_unit": None,
        "valid_range": (0.2, 0.8),     # <=0.42 normal geometry, >0.42 concentric
        "impossible_range": (0.05, 1.5),
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "relative_wall_thickness",
    },
    "LV_Mass": {
        # NOTE (flagged deliberately): the supplied dictionary puts absolute LV mass ("LV Mass",
        # grams) and the BSA-INDEXED mass ("LVMI", "LV Mass Index", g/m2) under one key. They are
        # clinically distinct and an indexed value is roughly half the absolute one, so the two
        # cannot be compared against a single threshold. Implemented as specified, but stored
        # unit-less and NOT wired into any prediction rule for that reason -- the range below is
        # deliberately wide enough to admit either, which also means it validates neither
        # tightly. Splitting this into LV_Mass and LV_Mass_Index is the correct fix if these are
        # ever used for grading.
        "synonyms": ["LV Mass", "Left Ventricular Mass", "LVM", "LVMI", "LV Mass Index",
                     "LV Mass/BSA", "LV Mass Indexed", "Left Ventricular Mass Index"],
        "kind": "numeric",
        "units": [],
        "canonical_unit": "",
        "alternate_unit": None,
        "valid_range": (30.0, 400.0),
        "impossible_range": (5.0, 800.0),
        "disqualifying_suffixes": (),   # "/BSA" and "index" are IN the synonym list here
        "indexed_variant_of": None,
        "db_field": "lv_mass",
    },
    "RWMA": {
        # FOUR of the supplied synonyms are deliberately NOT listed:
        #   "RWMA", "Regional Wall Motion"  -- Wall_Motion already owns both and is covered by
        #       existing tests; two owners for one label makes the winner arbitrary.
        #   "No RWMA", "RWMA Present"       -- these are FINDINGS, not labels. Because matching
        #       is longest-synonym-first, "No RWMA" (8 chars) outranked Wall_Motion's "RWMA"
        #       (4) and claimed the span on the line "No RWMA at rest" -- then stored the
        #       remainder after the label, the bare word "No", as the finding, while
        #       Wall_Motion (which had read the whole phrase correctly) got nothing.
        # What is left are true label forms only.
        "synonyms": ["Regional Wall Motion Abnormality", "Regional Wall Motion Abnormalities"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "rwma",
    },
    "Septal_Wall_Motion": {
        "synonyms": ["Septal Wall Motion", "Septal Motion", "Septal Contractility",
                     "Septal Kinesis", "Septal Hypokinesia", "Septal Akinesia",
                     "Septal Dyskinesia"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "septal_wall_motion",
    },
    "Anterior_Wall_Motion": {
        "synonyms": ["Anterior Wall Motion", "Anterior Wall", "Anterior LV Wall",
                     "Anterior Contractility", "Anterior Hypokinesia", "Anterior Akinesia",
                     "Anterior Dyskinesia"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "anterior_wall_motion",
    },
    "Inferior_Wall_Motion": {
        "synonyms": ["Inferior Wall Motion", "Inferior Wall", "Inferior LV Wall",
                     "Inferior Contractility", "Inferior Hypokinesia", "Inferior Akinesia",
                     "Inferior Dyskinesia"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "inferior_wall_motion",
    },
    "Lateral_Wall_Motion": {
        "synonyms": ["Lateral Wall Motion", "Lateral Wall", "Lateral LV Wall",
                     "Lateral Contractility", "Lateral Hypokinesia", "Lateral Akinesia",
                     "Lateral Dyskinesia", "Inferolateral Hypokinesia",
                     "Anterolateral Hypokinesia"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "lateral_wall_motion",
    },
    "Posterior_Wall_Motion": {
        # "Posterior Wall" is deliberately NOT listed: PWd owns it, and on every real template
        # that phrase heads a THICKNESS measurement ("Posterior Wall 9.00 mm"), not a motion
        # finding. Claiming it here would send a wall thickness into a motion field.
        "synonyms": ["Posterior Wall Motion", "Posterior Contractility",
                     "Posterior Hypokinesia", "Posterior Akinesia", "Posterior Dyskinesia"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "posterior_wall_motion",
    },
    "Apical_Wall_Motion": {
        "synonyms": ["Apical Wall Motion", "Apical Motion", "Apex", "Apical Contractility",
                     "Apical Hypokinesia", "Apical Akinesia", "Apical Dyskinesia",
                     "Apical Segment"],
        "kind": "text",
        "units": [],
        "canonical_unit": None,
        "alternate_unit": None,
        "valid_range": None,
        "impossible_range": None,
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "apical_wall_motion",
    },
    "PV_Annulus": {
        "synonyms": ["PV Annulus", "Pulmonic Annulus", "Pulmonary Valve Annulus", "PV Ann",
                     "Pulmonary Annulus"],
        "kind": "numeric",
        "units": ["cm", "mm"],
        "canonical_unit": "cm",
        "alternate_unit": "mm",
        "valid_range": (1.0, 3.5),
        "impossible_range": (0.4, 6.0),
        "disqualifying_suffixes": _DEFAULT_DISQUALIFYING_SUFFIXES,
        "indexed_variant_of": None,
        "db_field": "pv_annulus",
    },
    "BSA": {
        "synonyms": ["BSA", "Body Surface Area", "B.S.A.", "BSA(m2)", "BSA (m2)", "BSA m2", "BSA (m²)", "BSA(m²)"],
        "kind": "numeric",
        "units": ["m2", "m²"],
        "canonical_unit": "m2",
        "alternate_unit": None,
        "valid_range": (0.5, 3.5),
        "impossible_range": (0.2, 5.0),
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "bsa",
    },
    "Height": {
        "synonyms": ["Height", "Ht", "Height (cm)", "Ht (cm)", "Height cm", "Patient Height"],
        "kind": "numeric",
        "units": ["cm", "m"],
        "canonical_unit": "cm",
        "alternate_unit": "m",
        "valid_range": (40, 250),
        "impossible_range": (20, 300),
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "height",
    },
    "Weight": {
        "synonyms": ["Weight", "Wt", "Weight (kg)", "Wt (kg)", "Weight kg", "Wt. (kg)", "Body Weight", "Patient Weight"],
        "kind": "numeric",
        "units": ["kg", "lbs"],
        "canonical_unit": "kg",
        "alternate_unit": "lbs",
        "valid_range": (2, 300),
        "impossible_range": (1, 500),
        "disqualifying_suffixes": (),
        "indexed_variant_of": None,
        "db_field": "weight",
    },
}


# Canonical name required by the extraction spec. PARAMETERS remains the in-repo name (every
# existing import uses it); this is the same object, not a copy, so the two can never drift.
ECHO_PARAMETER_DICTIONARY = PARAMETERS


# ---------------------------------------------------------------------------
# Qualitative vocabulary
# ---------------------------------------------------------------------------
# 2D-Echo reports very often print a WORD where a number could have gone ("IVSd: Normal",
# "RV: Enlarged", "MV: Mild Regurgitation"). These vocabularies are what let the extractor
# recognise such a cell as a legitimate qualitative value rather than a failed numeric parse --
# and, critically, route it AROUND the unit-conversion and physiological-range machinery, which
# is meaningless for a word.

# Severity graders, longest first so "mild to moderate" wins over a bare "mild".
SEVERITY_TERMS = (
    "grade iv", "grade iii", "grade ii", "grade i",
    "grade 4+", "grade 3+", "grade 2+", "grade 1+",
    "grade 4", "grade 3", "grade 2", "grade 1",
    "grade four", "grade three", "grade two", "grade one",
    "mild to moderate", "moderate to severe", "mild-moderate", "moderate-severe",
    "trivial", "trace", "mild", "moderate", "severe", "gross", "significant", "minimal",
    "small", "large", "massive", "none/trace", "trace/trivial",
    "4+", "3+", "2+", "1+",
)

# Named lesions/pathologies -- the noun half of a finding phrase.
LESION_TERMS = (
    "regurgitation", "regurgitant", "stenosis", "stenotic", "sclerosis", "sclerotic", "prolapse",
    "degenerative", "degeneration", "insufficiency", "incompetence", "calcification", "calcified", "thickening",
    "thickened", "vegetation", "vegetations", "effusion", "thrombus", "hypokinesia", "akinesia", "dyskinesia",
    "hypokinetic", "akinetic", "dyskinetic", "normokinetic", "hypokinesis", "akinesis", "dyskinesis",
    "hypertrophy", "coarctation", "aneurysm",
    # Valve shorthand tokens
    "mr", "tr", "ar", "pr", "ms", "as", "ts", "ps", "bav", "mvp",
)

# Standalone descriptors that fully constitute a finding on their own.
DESCRIPTOR_TERMS = (
    "normal", "abnormal", "enlarged", "dilated", "not dilated", "not enlarged",
    "hypertrophied", "thickened", "unremarkable", "wnl", "within normal limits",
    "impaired", "preserved", "reduced", "sclerotic", "calcified", "degenerative", "absent", "present",
    "nil", "none", "intact", "adequate", "good", "poor", "mildly dilated", "moderately dilated",
    "severely dilated", "grossly dilated", "borderline", "trileaflet", "bicuspid",
    "collapsing with respiration", "collapsing", "collapsible", "non-collapsing", "plethoric", "engorged",
    "not seen", "not detected", "not visualized", "not visualised", "negative", "positive",
    "detected", "seen", "visualized", "visualised",
    "clean", "clear", "no clot", "no clots", "no thrombus", "no vegetation", "none/trace",
    "no effusion", "no rwma", "no",
)

QUALITATIVE_TERMS = tuple(sorted(
    set(SEVERITY_TERMS) | set(LESION_TERMS) | set(DESCRIPTOR_TERMS),
    key=len, reverse=True,
))


def normalized_key(s: str) -> str:
    """Single normalization used everywhere two label strings are compared (form-field keys,
    table cell text, narrative headings): lowercase, collapse internal whitespace, strip
    surrounding whitespace/punctuation."""
    import re
    if s is None:
        return ""
    collapsed = re.sub(r"\s+", " ", s.strip().lower())
    return collapsed.strip(" \t:.-=")


# Longest synonyms first avoids short abbreviations (e.g. "EF") wrongly matching
# inside longer phrases (e.g. "LEFT VENTRICLE") during line scanning.
def sorted_synonym_index():
    index = []
    for canon, meta in PARAMETERS.items():
        for syn in meta["synonyms"]:
            index.append((syn, canon))
    index.sort(key=lambda x: -len(x[0]))
    return index


# ===========================================================================================
# WALL CONDITION DICTIONARY
# ===========================================================================================
# Supplied verbatim as the specification for the "Wall Condition" report section. Kept in this
# key -> [synonyms] shape (rather than rewritten into the PARAMETERS schema) so it stays
# diffable against the source it came from.
#
# HOW IT MAPS ONTO PARAMETERS
#   ivsd / ivss / pwd / pws / wall_motion  -> the EXISTING canonical entries of the same name.
#       They are the same measurements, so their synonyms here were merged into those entries
#       rather than duplicated: exactly one owner per label.
#   everything else                        -> new canonical entries added above.
#
# THREE SYNONYMS WERE NOT ADOPTED, because they collide with an established owner and an
# ambiguous label is worse than a missing one:
#   "Posterior Wall"        listed under BOTH pwd and posterior_wall_motion here. Kept on PWd:
#                           on every real template it heads a thickness ("Posterior Wall 9.00
#                           mm"), so routing it to a motion field would store a measurement as
#                           a finding.
#   "Regional Wall Motion"  listed under BOTH wall_motion and rwma here. Kept on Wall_Motion,
#                           which already owned it and is covered by existing tests.
#   "RWMA"                  already owned by Wall_Motion. RWMA keeps only the phrasings
#                           Wall_Motion does not claim ("No RWMA", "RWMA Present", ...).
WALL_PARAMETER_DICTIONARY = {
    "ivsd": ["IVSd", "IVSD", "IV Septum", "IV Septum Thickness", "Interventricular Septum",
             "Interventricular Septum Diastole", "Interventricular Septal Thickness",
             "Septal Thickness (Diastole)", "IVS Thickness", "IVS (ED)"],
    "ivss": ["IVSs", "IVSS", "Interventricular Septum Systole", "Interventricular Septal Thickness Systole",
             "Septal Thickness (Systole)", "IVS (ES)"],
    "pwd": ["PWd", "PWD", "Posterior Wall", "Posterior Wall Thickness", "Posterior Wall Diastole",
            "LV Posterior Wall", "LVPWd", "Left Ventricular Posterior Wall", "Posterior Wall (ED)"],
    "pws": ["PWs", "PWS", "Posterior Wall Systole", "Posterior Wall Thickness Systole",
            "LVPWs", "Posterior Wall (ES)"],
    "relative_wall_thickness": ["RWT", "Relative Wall Thickness", "Relative Wall Thickness Index",
                                 "Relative LV Wall Thickness", "Relative Thickness", "LV Relative Wall Thickness"],
    "lv_mass": ["LV Mass", "Left Ventricular Mass", "LVM", "LVMI", "LV Mass Index", "LV Mass/BSA",
                "LV Mass Indexed", "Left Ventricular Mass Index"],
    "wall_motion": ["Wall Motion", "LV Wall Motion", "Regional Wall Motion", "Global Wall Motion",
                     "Wall Motion Analysis", "Wall Kinetics", "Kinetics", "Contractility", "LV Contractility"],
    "rwma": ["RWMA", "Regional Wall Motion Abnormality", "Regional Wall Motion Abnormalities",
             "Regional Wall Motion", "No RWMA", "RWMA Present"],
    "septal_wall_motion": ["Septal Wall Motion", "Septal Motion", "Septal Contractility", "Septal Kinesis",
                            "Septal Hypokinesia", "Septal Akinesia", "Septal Dyskinesia"],
    "anterior_wall_motion": ["Anterior Wall Motion", "Anterior Wall", "Anterior LV Wall", "Anterior Contractility",
                              "Anterior Hypokinesia", "Anterior Akinesia", "Anterior Dyskinesia"],
    "inferior_wall_motion": ["Inferior Wall Motion", "Inferior Wall", "Inferior LV Wall", "Inferior Contractility",
                              "Inferior Hypokinesia", "Inferior Akinesia", "Inferior Dyskinesia"],
    "lateral_wall_motion": ["Lateral Wall Motion", "Lateral Wall", "Lateral LV Wall", "Lateral Contractility",
                             "Lateral Hypokinesia", "Lateral Akinesia", "Lateral Dyskinesia",
                             "Inferolateral Hypokinesia", "Anterolateral Hypokinesia"],
    "posterior_wall_motion": ["Posterior Wall Motion", "Posterior Wall", "Posterior Contractility",
                               "Posterior Hypokinesia", "Posterior Akinesia", "Posterior Dyskinesia"],
    "apical_wall_motion": ["Apical Wall Motion", "Apical Motion", "Apex", "Apical Contractility",
                            "Apical Hypokinesia", "Apical Akinesia", "Apical Dyskinesia", "Apical Segment"],
}


# ===========================================================================================
# REPORT SECTIONS -- the single source of truth for how the review page is grouped.
# ===========================================================================================
# db_field lists, in display order, served to the frontend by /api/reports/sections. Keeping the
# grouping here rather than in the HTML means a parameter added to PARAMETERS shows up on the
# review page without touching the template, and the two can never drift apart.
REPORT_SECTIONS = [
    {"key": "lv", "title": "Left Ventricle", "fields": [
        "ef", "lvidd", "lvids", "lvidd_indexed_value"]},
    {"key": "wall", "title": "Wall Condition", "fields": [
        "ivsd", "ivss", "pwd", "pws", "relative_wall_thickness", "lv_mass",
        "wall_motion", "rwma", "septal_wall_motion", "anterior_wall_motion",
        "inferior_wall_motion", "lateral_wall_motion", "posterior_wall_motion",
        "apical_wall_motion"]},
    {"key": "chambers", "title": "Chambers", "fields": [
        "la_diameter", "la_diameter_indexed_value", "ra_size", "rv_size", "ivc"]},
    {"key": "aorta", "title": "Aorta", "fields": [
        "ao_diameter", "ao_root", "ao_annulus", "ao_stj"]},
    # One section per valve, each showing only that valve's own parameters.
    {"key": "valve_aortic", "title": "Aortic Valve", "fields": [
        "av_finding", "av_peak_velocity", "av_peak_gradient", "av_area"]},
    {"key": "valve_mitral", "title": "Mitral Valve", "fields": [
        "mv_finding", "mv_peak_velocity", "mv_peak_gradient", "mv_area", "mv_annulus"]},
    {"key": "valve_tricuspid", "title": "Tricuspid Valve", "fields": [
        "tv_finding", "tv_peak_velocity", "tv_peak_gradient", "tv_annulus"]},
    {"key": "valve_pulmonary", "title": "Pulmonary Valve", "fields": [
        "pv_finding", "pv_peak_velocity", "pv_peak_gradient", "pv_annulus"]},
    {"key": "doppler", "title": "Doppler & Pressures", "fields": [
        "e_a_ratio", "pasp", "lvot_peak_velocity", "lvot_peak_gradient", "lvot_vti"]},
    {"key": "structural", "title": "Structural", "fields": [
        "pericardial_effusion", "clots_thrombus"]},
]

# Human-readable label per db_field, used by the review page.
FIELD_LABELS = {
    "ef": "Ejection Fraction (EF)",
    "lvidd": "LVIDd (LV Diastolic Diameter)",
    "lvids": "LVIDs (LV Systolic Diameter)",
    "lvidd_indexed_value": "LVIDd / BSA",
    "ivsd": "IVSd (Septal Thickness, Diastole)",
    "ivss": "IVSs (Septal Thickness, Systole)",
    "pwd": "PWd (Posterior Wall, Diastole)",
    "pws": "PWs (Posterior Wall, Systole)",
    "relative_wall_thickness": "Relative Wall Thickness (RWT)",
    "lv_mass": "LV Mass / LV Mass Index",
    "wall_motion": "Wall Motion",
    "rwma": "Regional Wall Motion Abnormality",
    "septal_wall_motion": "Septal Wall Motion",
    "anterior_wall_motion": "Anterior Wall Motion",
    "inferior_wall_motion": "Inferior Wall Motion",
    "lateral_wall_motion": "Lateral Wall Motion",
    "posterior_wall_motion": "Posterior Wall Motion",
    "apical_wall_motion": "Apical Wall Motion",
    "la_diameter": "LA Diameter",
    "la_diameter_indexed_value": "LA Diameter / BSA",
    "ra_size": "RA Size",
    "rv_size": "RV Size",
    "ivc": "IVC (Inferior Vena Cava)",
    "ao_diameter": "Aortic Diameter",
    "ao_root": "Aortic Root",
    "ao_annulus": "Aortic Annulus",
    "ao_stj": "Sinotubular Junction (STJ)",
    "av_finding": "Aortic Valve Finding",
    "av_peak_velocity": "AV Peak Velocity",
    "av_peak_gradient": "AV Peak Gradient",
    "av_area": "Aortic Valve Area",
    "mv_finding": "Mitral Valve Finding",
    "mv_peak_velocity": "MV Peak Velocity",
    "mv_peak_gradient": "MV Peak Gradient",
    "mv_area": "Mitral Valve Area",
    "mv_annulus": "Mitral Annulus",
    "tv_finding": "Tricuspid Valve Finding",
    "tv_peak_velocity": "TV Peak Velocity",
    "tv_peak_gradient": "TV Peak Gradient",
    "tv_annulus": "Tricuspid Annulus",
    "pv_finding": "Pulmonary Valve Finding",
    "pv_peak_velocity": "PV Peak Velocity",
    "pv_peak_gradient": "PV Peak Gradient",
    "pv_annulus": "Pulmonic Annulus",
    "e_a_ratio": "E/A Ratio",
    "pasp": "PASP (Pulmonary Artery Systolic Pressure)",
    "lvot_peak_velocity": "LVOT Peak Velocity",
    "lvot_peak_gradient": "LVOT Peak Gradient",
    "lvot_vti": "LVOT VTI",
    "pericardial_effusion": "Pericardial Effusion",
    "clots_thrombus": "Clots / Thrombus",
    "bsa": "BSA (Body Surface Area)",
    "height": "Height",
    "weight": "Weight",
}
