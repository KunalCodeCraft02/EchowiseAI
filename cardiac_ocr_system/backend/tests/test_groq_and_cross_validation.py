"""
Tests for the Groq vision extractor and the PaddleOCR cross-validation layer.

All HTTP is monkeypatched -- the suite never calls the live Groq API, so it runs offline,
deterministically, and without consuming the account's rate limit.
"""
import pytest

from app.ocr import groq_extractor as gx
from app.ocr import cross_validation as xv


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def test_parses_plain_json_array():
    pairs = gx.parse_groq_response('[{"label":"IVSd","value":"1.0 cm"}]')
    assert pairs == [{"label": "IVSd", "value": "1.0 cm"}]


def test_strips_think_block():
    """qwen3.6 is a reasoning model and prefixes its answer with <think>...</think>."""
    raw = '<think>Let me look at the table.\nIVSd is 1.0</think>[{"label":"IVSd","value":"1.0 cm"}]'
    assert gx.parse_groq_response(raw) == [{"label": "IVSd", "value": "1.0 cm"}]


def test_tolerates_markdown_fence_and_prose():
    raw = 'Here you go:\n```json\n[{"label":"EF","value":"65 %"}]\n```\nHope that helps.'
    assert gx.parse_groq_response(raw) == [{"label": "EF", "value": "65 %"}]


def test_unterminated_think_block_raises():
    """A truncated response has an unclosed <think> and therefore no JSON -- it must fail
    loudly rather than silently yielding a partial table."""
    with pytest.raises(gx.GroqVisionError):
        gx.parse_groq_response("<think>I am still reasoning and got cut off")


def test_malformed_json_raises():
    with pytest.raises(gx.GroqVisionError):
        gx.parse_groq_response('[{"label":"IVSd","value":')


def test_drops_incomplete_pairs():
    pairs = gx.parse_groq_response(
        '[{"label":"IVSd","value":"1.0 cm"},{"label":"","value":"x"},{"label":"PWd","value":null}]'
    )
    assert pairs == [{"label": "IVSd", "value": "1.0 cm"}]


# ---------------------------------------------------------------------------
# HTTP behaviour (mocked)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


def _ok(content, finish="stop"):
    return _Resp(200, {"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def test_truncated_completion_is_rejected(monkeypatch):
    monkeypatch.setattr(gx.httpx, "post", lambda *a, **k: _ok('[{"label":"EF"', finish="length"))
    monkeypatch.setattr(gx, "GROQ_API_KEY", "test-key")
    with pytest.raises(gx.GroqVisionError, match="truncated"):
        gx._call_groq("fakeb64")


def test_text_only_model_raises_unavailable(monkeypatch):
    resp = _Resp(400, {"error": {"message": "messages[0].content must be a string"}})
    monkeypatch.setattr(gx.httpx, "post", lambda *a, **k: resp)
    monkeypatch.setattr(gx, "GROQ_API_KEY", "test-key")
    with pytest.raises(gx.GroqVisionUnavailable):
        gx._call_groq("fakeb64")


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(429, {"error": {"message": "try again in 0.01s"}})
        return _ok('[{"label":"EF","value":"65 %"}]')

    monkeypatch.setattr(gx.httpx, "post", fake_post)
    monkeypatch.setattr(gx.time, "sleep", lambda s: None)
    monkeypatch.setattr(gx, "GROQ_API_KEY", "test-key")
    assert "65" in gx._call_groq("fakeb64")
    assert calls["n"] == 2


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(gx, "GROQ_API_KEY", "")
    with pytest.raises(gx.GroqVisionError, match="GROQ_API_KEY"):
        gx._call_groq("fakeb64")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def test_corroborated_value_uses_real_ocr_confidence():
    """When OCR independently read the same number, confidence is that token's REAL measured
    OCR confidence -- not a synthesized constant."""
    lines = [("IVSd 1.05 cm", 97.3), ("Other row 9.99", 80.0)]
    ev = xv.corroborate_value("1.05 cm", lines)
    assert ev["corroborated"] is True
    assert ev["confidence"] == 97.3
    assert ev["confidence_basis"] == xv.BASIS_CORROBORATING_OCR


def test_uncorroborated_value_is_flagged_not_discarded():
    ev = xv.corroborate_value("1.05 cm", [("totally unrelated text", 90.0)])
    assert ev["corroborated"] is False
    assert ev["confidence_basis"] == xv.BASIS_UNCORROBORATED
    # Stored (above reject) but below the flag threshold so a doctor reviews it.
    from app.config import OCR_CONFIDENCE_REJECT_THRESHOLD, OCR_CONFIDENCE_FLAG_THRESHOLD
    assert OCR_CONFIDENCE_REJECT_THRESHOLD < ev["confidence"] < OCR_CONFIDENCE_FLAG_THRESHOLD


def test_conflicting_read_is_its_own_bucket():
    """OCR reading a DIFFERENT value is far more serious than reading nothing, and must be
    distinguishable -- this is the clinically dangerous case."""
    ev = xv.corroborate_value("1.05 cm", [("IVSd 1.65 cm", 96.0)])
    assert ev["confidence_basis"] == xv.BASIS_CONFLICT
    assert ev["conflicting_value"] == "1.65"
    assert ev["confidence"] == xv.CONFLICT_CONFIDENCE


def test_decimal_drop_is_conflict_not_corroboration():
    """4.2 vs 42 is the decimal-drop OCR error class; treating it as agreement would defeat
    the impossible_range gate."""
    ev = xv.corroborate_value("4.2 cm", [("LVEDd 42 cm", 95.0)])
    assert ev["corroborated"] is False


def test_decimal_comma_is_normalized():
    """extractor._NUMERIC_VALUE_RE has no comma branch, so an un-normalized '1,62' would
    silently parse as 1 -- a 60% error on an E/A ratio."""
    assert xv.normalize_numeric_text("1,62") == "1.62"
    ev = xv.corroborate_value("1,62", [("E/A Ratio 1.62", 94.0)])
    assert ev["corroborated"] is True


def test_repeated_nondistinctive_token_does_not_falsely_corroborate():
    """A bare '65' also appears as heart rate/age; matching it anywhere in the document is
    coincidence, not corroboration."""
    lines = [("HR 65 bpm", 95.0), ("Age 65", 95.0), ("Weight 65 kg", 95.0)]
    ev = xv.corroborate_value("65", lines)
    assert ev["corroborated"] is False


def test_confidence_threshold_invariant():
    """REJECT < CONFLICT < UNCORROBORATED < FLAG must hold, or values would be silently
    dropped or silently trusted."""
    from app.config import (OCR_CONFIDENCE_REJECT_THRESHOLD, OCR_CONFIDENCE_FLAG_THRESHOLD,
                            GROQ_UNCORROBORATED_CONFIDENCE)
    assert (OCR_CONFIDENCE_REJECT_THRESHOLD < xv.CONFLICT_CONFIDENCE
            < GROQ_UNCORROBORATED_CONFIDENCE < OCR_CONFIDENCE_FLAG_THRESHOLD)


def test_cross_validate_attaches_audit_metadata():
    fields = [{"key_text": "IVSd", "key_confidence": 0.0, "value_text": "1.05 cm",
               "value_confidence": 0.0, "page": 1}]
    out = xv.cross_validate(fields, [("IVSd 1.05 cm", 97.3)])
    meta = out[0]["extra_meta"]
    assert out[0]["value_confidence"] == 97.3
    assert meta["corroborated"] is True
    assert meta["secondary_source"] == "paddleocr"
    # Never claims to be an OCR score for the LLM's own reading.
    assert meta["confidence_basis"] == xv.BASIS_CORROBORATING_OCR
