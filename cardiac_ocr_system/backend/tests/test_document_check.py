"""
Tests for app/ocr/document_check.py -- the "is this even an echo report?" gate.

The gate exists so that uploading a WhatsApp screenshot, a selfie or an unrelated PDF is
reported as the wrong FILE rather than as a bad SCAN. Getting that backwards sends the user to
re-scan a document that can never extract, so both directions are pinned here:

  NOT a report  -- chat screenshots, invoices, non-cardiac lab reports must never pass.
  IS  a report  -- a genuine echo must still pass after heavy OCR damage, because a false
                   "wrong file" verdict accuses the user of a mistake they did not make.

Fully offline: pure string matching, no OCR, no network.
"""
from app.ocr.document_check import looks_like_cardiac_report

WHATSAPP_SCREENSHOT = """
9:41 AM
Mom
Hey are you coming home for dinner tonight? I made pasta
Yes will be there by 8
Ok good. Don't be late again please
Delivered
Type a message
"""

INVOICE_PDF = """
INVOICE
Bill To: Acme Traders Pvt Ltd
GST No 27AAECS1234F1Z5
Item        Qty   Rate    Amount
Steel rods   20   450.00  9000.00
Subtotal 9000.00  Tax 18% 1620.00
Total Payable Rs 10620.00
Thank you for your business
"""

# A real medical document, but the wrong one. Shares report vocabulary ("Impression") with an
# echo report and must still be rejected -- this is the case a naive keyword list gets wrong.
BLOOD_REPORT = """
PATHOLOGY LABORATORY
Complete Blood Count
Haemoglobin 13.5 g/dL   Ref 13.0-17.0
Total Leucocyte Count 7200 /cmm
Platelet Count 2.4 lakh/cmm
Impression: Within normal limits
"""

ECHO_REPORT = """
2D ECHOCARDIOGRAPHY WITH COLOUR DOPPLER
Name: Mr Rohan Deshmukh   Age 45  Sex M
IVSd 0.9 cm   LVIDd 4.8 cm  LVIDs 3.1 cm  PWd 0.9 cm
Ejection Fraction 58%   E/A ratio 1.2
Mitral Valve: Normal. No mitral regurgitation.
No RWMA. No pericardial effusion.
IMPRESSION: Normal LV systolic function.
"""

# The same study read badly: transposed letters, dropped characters, no numbers recovered.
ECHO_REPORT_DEGRADED = """
2D ECHO REPRT
LEFT VENTRICLE norrnal
Mitrol Valve mild regurgitatlon
lmpression: normal study
Doppler
"""


def test_chat_screenshot_is_not_a_report():
    assert looks_like_cardiac_report(WHATSAPP_SCREENSHOT).is_report is False


def test_invoice_is_not_a_report():
    assert looks_like_cardiac_report(INVOICE_PDF).is_report is False


def test_non_cardiac_lab_report_is_not_a_report():
    """Shares "Impression" with an echo report; carries no cardiac content."""
    assert looks_like_cardiac_report(BLOOD_REPORT).is_report is False


def test_empty_and_trivial_text_is_not_a_report():
    assert looks_like_cardiac_report("").is_report is False
    assert looks_like_cardiac_report("IMG_20240517 143255").is_report is False


def test_genuine_echo_report_is_recognised():
    verdict = looks_like_cardiac_report(ECHO_REPORT)
    assert verdict.is_report is True
    assert "ejection fraction" in verdict.strong_hits


def test_heavily_degraded_echo_report_still_recognised():
    """A bad scan must be told to re-scan, never accused of being the wrong file."""
    assert looks_like_cardiac_report(ECHO_REPORT_DEGRADED).is_report is True


def test_evidence_is_combined_across_fragments():
    """The pipeline passes raw text, narrative blocks and flattened table cells separately: a
    report whose evidence is split across them must still clear the bar."""
    assert looks_like_cardiac_report("Left ventricle", "Mitral valve", "Doppler study").is_report


def test_terms_only_match_on_word_boundaries():
    """Short abbreviations like PASP must not fire from inside unrelated words."""
    noise = "passport application form for the passenger named above, submitted in duplicate"
    assert looks_like_cardiac_report(noise).is_report is False
