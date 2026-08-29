"""
Groq LLM integration: turns the rule engine's output (predicted disease(s) +
supporting extracted parameters) into personalized rehabilitation guidance,
via Groq's OpenAI-compatible chat completions API.

EXERCISE ONLY. Neither flow here produces dietary guidance: an
echocardiogram supports no dietary conclusion, so asking the model for one
invited content nothing in this system could justify.

Two flows live here. generate_care_plan() builds ONE combined rehabilitation
exercise plan for the whole report (the legacy path).
generate_disease_care_plans() builds one plan per predicted disease, making
ONE API CALL PER DISEASE so the operation can report measured progress
("plan 3 of 7") instead of an animated guess.

This is advisory content generation on top of the deterministic rule engine
— the rule engine (app/predictor/rules.py) is the sole source of any
"diagnosis"-shaped output. The LLM never re-derives or overrides the
predicted disease; it only explains lifestyle guidance for it.
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, GROQ_TIMEOUT_SECONDS


# What a clinician is shown when a plan cannot be generated. Deliberately free of provider names,
# HTTP status codes and response bodies: none of that is actionable at a desk, and an upstream
# error body is raw JSON that reads as a crash. The technical cause is logged instead, where the
# person who can act on it will look.
GENERIC_PLAN_ERROR = ("Something went wrong while generating the plan. Please try again in a "
                      "moment.")


class GroqError(Exception):
    """A plan could not be generated.

    Carries TWO messages on purpose. `str(exc)` is the technical one -- status codes, parse
    failures, the upstream body -- and belongs in the server log. `user_message` is the sentence
    the clinician sees, and must never contain a response body, a stack trace or a provider name.
    Routes are expected to surface `user_message` and log the rest; the split exists so that
    forgetting to do so is a visible mistake rather than a silent leak of raw JSON onto a
    clinical page.
    """

    def __init__(self, message: str, user_message: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or GENERIC_PLAN_ERROR


_SYSTEM_PROMPT = (
    "You are a cardiac rehabilitation assistant supporting a cardiologist. You are given "
    "structured 2D-Echocardiogram parameters and the output of a deterministic rule-based "
    "Clinical Decision Support System (CDSS) for a patient. "
    "Write a safe, general, patient-friendly rehabilitation exercise plan consistent with "
    "the predicted condition(s), the patient's age band and the risk level. "
    "Rules: never state a new diagnosis or contradict the given predicted disease(s); "
    "never claim to diagnose coronary artery blockage; never give dietary, nutritional or "
    "medication advice -- exercise guidance only; always include activity restrictions "
    "appropriate to the severity and risk level given; keep tone supportive "
    "and non-alarming; explicitly note this plan requires physician approval before starting. "
    "Respond ONLY with a JSON object of the exact shape: "
    '{"rehabilitation_exercise_plan": "<markdown-formatted plan text>", '
    '"key_precautions": "<short markdown bullet list>"}'
)


def _build_user_prompt(patient_context: Dict[str, Any]) -> str:
    return (
        "Patient context (from validated 2D Echo extraction + rule engine):\n\n"
        f"{json.dumps(patient_context, indent=2)}\n\n"
        "Generate the rehabilitation exercise plan and key precautions now, "
        "as the JSON object described in the system prompt."
    )


def generate_care_plan(patient_context: Dict[str, Any]) -> Dict[str, str]:
    """
    Calls Groq with the extracted cardiac parameters + predicted disease/risk
    context and returns {"rehabilitation_exercise_plan", "key_precautions"}.
    Raises GroqError on any failure (missing key, network/timeout, bad
    response shape) so the caller can surface a clean error instead of a
    stack trace.
    """
    if not GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY is not configured on the server.",
                        "The plan service is not configured on this server. Please "
                        "contact your administrator.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(patient_context)},
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(GROQ_API_URL, json=payload, headers=headers, timeout=GROQ_TIMEOUT_SECONDS)
    except httpx.TimeoutException as exc:
        raise GroqError("Groq API request timed out.",
                        "The plan service took too long to respond. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise GroqError(f"Groq API request failed: {exc}") from exc  # generic user message

    if resp.status_code != 200:
        # resp.text is an upstream JSON error body -- technical arg only, never the user one.
        raise GroqError(f"Groq API returned {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, ValueError) as exc:
        raise GroqError(f"Could not parse Groq response: {exc}",
                        "The plan service returned an unreadable response. Please "
                        "try again.") from exc

    rehab = parsed.get("rehabilitation_exercise_plan")
    if not rehab:
        raise GroqError("Groq response was missing the expected plan fields.",
                        "The plan service returned an incomplete plan. Please try again.")

    # Any diet_plan the model volunteers is dropped here rather than passed on.
    return {
        "rehabilitation_exercise_plan": rehab,
        "key_precautions": parsed.get("key_precautions", ""),
    }


# ===========================================================================================
# PER-DISEASE REHABILITATION & DIET PLANS (v4.0)
# ===========================================================================================
# Separate from generate_care_plan() above, which produces ONE markdown plan for the whole
# report. This produces one plain-text plan PER predicted disease, in a strict schema the
# frontend maps directly onto cards.
#
# EXERCISE ONLY. Dietary guidance was removed from this flow: it is prescription-adjacent advice
# the echo alone cannot support, and asking the model for it invited content no parameter in this
# system justifies. generate_care_plan() above still produces a diet plan for the legacy combined
# report; this per-disease path deliberately does not.
_DISEASE_PLAN_SYSTEM_PROMPT = (
    "You are a preventive cardiologist writing rehabilitation exercise guidance for a "
    "patient, to be reviewed by their treating physician before use.\n"
    "\n"
    "You are given, for EACH cardiac disease: its category, severity, the EXACT triggering "
    "parameter(s) and measured values with the age-group-resolved threshold each was compared "
    "against, three independent 0-100 scores (confidence/prediction/severity), the patient's age "
    "band, and other compulsory-group findings (e.g. EF, PASP) even where they did not "
    "themselves trigger a disease. You are also given the patient's Exercise Safety verdict, "
    "which is a HARD CEILING on intensity -- see CEILING RULE below.\n"
    "\n"
    "For EACH disease, write PERSONALIZED guidance that:\n"
    "  - Opens by naming the specific value(s) that triggered this finding and how they compare "
    "to the threshold for this patient's age group (e.g. 'Given an IVSd of 13mm against the "
    "11mm threshold for this age group...'). Two patients with the same disease NAME but "
    "different underlying numbers must read differently -- do not write generic boilerplate.\n"
    "  - Sets intensity, frequency and activities to avoid, scaled to how far past threshold the "
    "values sit and to the severity given, not just the disease label.\n"
    "  - References the other compulsory findings supplied (e.g. a preserved EF or elevated "
    "PASP) where they affect what is safe.\n"
    "\n"
    "CEILING RULE -- never overridden by an individual disease's own severity:\n"
    "  - If the Exercise Safety verdict supplied is 'Exercise Restricted / Supervised Only', "
    "every plan you write MUST be capped at low-intensity, medically supervised activity only, "
    "regardless of how mild the individual disease looks in isolation.\n"
    "  - You will never be asked to write a plan when the verdict is 'Exercise Contraindicated' "
    "or 'Exercise Safety Indeterminate' -- if you somehow are, refuse the intensity guidance and "
    "state only that cardiology clearance is required first.\n"
    "\n"
    "Do NOT give dietary, nutritional or medication advice. Exercise guidance only.\n"
    "\n"
    "CLINICAL RULES:\n"
    "  - Never state a new diagnosis and never contradict the disease you were given.\n"
    "  - Never claim to diagnose coronary artery blockage.\n"
    "  - Be conservative. When a condition is severe, say plainly that exercise must be "
    "supervised and medically cleared first.\n"
    "\n"
    "FORMATTING RULES -- these are strict:\n"
    "  - Output PLAIN TEXT ONLY inside every string value.\n"
    "  - Do NOT use Markdown of any kind. No hash characters, no asterisks, no bullet "
    "characters, no backticks, no underscores for emphasis.\n"
    "  - Write in complete sentences separated by full stops, as a clinician would type into a "
    "clinical note. Use plain numbered steps like '1.' only if a sequence is genuinely needed.\n"
    "\n"
    'Respond ONLY with a JSON object of the exact shape: {"plans": [ '
    '{"cardiac_disease_name": "<exactly the disease name you were given>", '
    '"rehabilitation_exercise": "<plain text>"} ]} '
    "with one array entry for EVERY disease supplied, in the same order."
)

# ---- §4 no-disease general-population guidance -------------------------------------------
# Deterministic, not LLM-generated: this is the standard ACSM/AHA baseline recommendation, not
# advice specific to this patient's findings, so there is nothing here that benefits from
# generation -- and a canned answer is guaranteed to render even when the plan service is
# unavailable, which matters most on precisely the "good news" report where nothing else went
# wrong.
def general_population_guidance(age_band_label: Optional[str] = None) -> str:
    """Standard ACSM/AHA baseline activity guidance for a patient with no predicted disease.

    Clearly labelled as general/population-level rather than condition-specific, per §4.
    """
    age_note = f" for a patient in the {age_band_label} band" if age_band_label else ""
    return (
        "GENERAL population-level activity guidance -- not condition-specific, since no cardiac "
        f"disease was predicted on this echocardiogram{age_note}. This follows the standard "
        "ACSM/AHA baseline recommendation: at least 150 minutes per week of moderate-intensity "
        "aerobic activity (e.g. brisk walking, cycling, swimming), or 75 minutes per week of "
        "vigorous-intensity aerobic activity, spread across most days of the week. Add "
        "moderate-intensity resistance training for all major muscle groups on 2 or more "
        "non-consecutive days per week. Progress intensity and duration gradually, warm up and "
        "cool down for every session, and stay within a comfortable, conversational effort "
        "level unless supervised otherwise. Stop and seek medical review for chest pain, "
        "unusual breathlessness, palpitations or dizziness during activity. This guidance "
        "assumes no other medical contraindication outside this echocardiogram and remains at "
        "the treating physician's discretion."
    )

# Markdown the model may emit despite the instruction. Stripped server-side as well as in the
# browser -- the UI must never show an asterisk or hash to a clinician, and relying on a single
# layer for that is asking to be surprised.
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_UNDERSCORE_RE = re.compile(r"(?<!\w)__(.+?)__(?!\w)", re.DOTALL)
_MD_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_MD_TICK_RE = re.compile(r"`+")
_MD_STAR_RE = re.compile(r"\*")


def _flatten_to_text(value) -> str:
    """Pull readable prose out of whatever the model actually returned.

    The schema asks for a plain string, but JSON mode delivers what it likes -- a nested object
    ({"warmup": ..., "main": ...}) or a list of steps are both common. Passing those to str()
    printed a Python dict repr onto the plan card: braces, quotes and all, which reads to a
    clinician as the system dumping its internals. Strings are collected in order and joined
    instead, so the guidance survives without any JSON syntax reaching the page. Anything with no
    text in it at all returns "", and the caller substitutes the "not generated" marker.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(t for t in (_flatten_to_text(v) for v in value.values()) if t)
    if isinstance(value, (list, tuple)):
        return " ".join(t for t in (_flatten_to_text(v) for v in value) if t)
    return ""


def strip_markdown(text) -> str:
    """Remove Markdown decoration, leaving clinician-readable plain text."""
    if not text:
        return ""
    out = _flatten_to_text(text)
    if not out:
        return ""
    out = _MD_HEADING_RE.sub("", out)
    out = _MD_BOLD_RE.sub(r"\1", out)
    out = _MD_UNDERSCORE_RE.sub(r"\1", out)
    out = _MD_BULLET_RE.sub("", out)
    out = _MD_TICK_RE.sub("", out)
    out = _MD_STAR_RE.sub("", out)
    # Collapse what the substitutions leave behind, without joining real paragraphs.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _coerce_plan_list(parsed: Any) -> List[Dict[str, Any]]:
    """Accept the shapes Groq's JSON mode actually returns.

    JSON mode always returns an OBJECT, so a bare array cannot come from the API but does arrive
    from tests and other callers; the model also varies between {"plans": [...]}, a top-level
    list, and -- for a single disease -- one bare object. Accepting all three is cheaper than
    failing a clinician's request over a wrapper key.
    """
    if isinstance(parsed, list):
        return [p for p in parsed if isinstance(p, dict)]
    if isinstance(parsed, dict):
        for key in ("plans", "care_plans", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [p for p in value if isinstance(p, dict)]
        if "cardiac_disease_name" in parsed:
            return [parsed]
    return []


_NOT_GENERATED = ("No plan was generated for this condition. "
                  "Please generate again, or advise the patient manually.")


def _post_disease_plan_prompt(user_prompt: str) -> List[Dict[str, Any]]:
    """One Groq round-trip for a disease-plan prompt -> the list of plan objects it returned."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _DISEASE_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        resp = httpx.post(GROQ_API_URL, json=payload, headers=headers,
                          timeout=GROQ_TIMEOUT_SECONDS)
    except httpx.TimeoutException as exc:
        raise GroqError("Groq API request timed out.",
                        "The plan service took too long to respond. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise GroqError(f"Groq API request failed: {exc}") from exc  # generic user message

    if resp.status_code != 200:
        # resp.text is an upstream JSON error body -- technical arg only, never the user one.
        raise GroqError(f"Groq API returned {resp.status_code}: {resp.text[:300]}")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, ValueError) as exc:
        raise GroqError(f"Could not parse Groq response: {exc}",
                        "The plan service returned an unreadable response. Please "
                        "try again.") from exc

    return _coerce_plan_list(parsed)


def generate_one_disease_care_plan(disease: Dict[str, Any],
                                   patient_context: Dict[str, Any]) -> Dict[str, str]:
    """The exercise plan for ONE disease -- a single Groq call.

    `disease` carries the FULL per-disease context (§4): category, severity, the exact
    triggering supporting_points (parameter + value + age-resolved threshold, already in that
    shape from rules_v4), and the three independent scores. `patient_context` carries what is
    shared across every disease on this report: age band, other compulsory-group findings, and
    the exercise-safety verdict as a hard ceiling the model must never write past.

    Raises GroqError if the call itself fails. A call that SUCCEEDS but returns nothing usable
    comes back as the "not generated" marker instead: an empty answer about one condition is not
    a reason to fail the other six.
    """
    if not GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY is not configured on the server.",
                        "The plan service is not configured on this server. Please "
                        "contact your administrator.")

    disease_name = disease.get("cardiac_disease_name") or disease.get("name") or "Unnamed finding"
    user_prompt = (
        "Write ONE personalized exercise plan for this patient's predicted finding. Reference "
        "the specific values in `supporting_points` and in the shared context explicitly:\n\n"
        + json.dumps(disease, indent=2, default=str)
        + "\n\nShared patient context (age band, exercise-safety ceiling, other compulsory-group "
          "findings even where they did not independently trigger a finding):\n"
        + json.dumps(patient_context, indent=2, default=str)
        + f"\n\nReturn the JSON object described in the system prompt now, with exactly one plan "
          f"entry for '{disease_name}'."
    )

    returned = _post_disease_plan_prompt(user_prompt)
    # Asked about one disease, so whatever came back is about that disease -- the model
    # paraphrasing the name is not a reason to discard its answer. The name we store is always
    # ours, never the model's, so the card still matches the prediction it belongs to.
    item = returned[0] if returned else None
    return {
        "cardiac_disease_name": disease_name,
        "rehabilitation_exercise": strip_markdown((item or {}).get("rehabilitation_exercise"))
        or _NOT_GENERATED,
    }


def generate_disease_care_plans(diseases: List[Dict[str, Any]],
                                patient_context: Dict[str, Any],
                                on_progress: Optional[Callable[[int, int, str], None]] = None
                                ) -> List[Dict[str, str]]:
    """One rehabilitation EXERCISE plan per predicted disease, as plain text.

    `diseases` is the full per-disease context list (§4) -- not bare names -- so two patients
    both labelled with the same disease name but different triggering values get different
    guidance. `patient_context` is shared across all of them and carries the exercise-safety
    verdict as the hard ceiling every plan must respect regardless of that disease's own severity.

    Generated ONE DISEASE PER CALL rather than all in a single batched request. That costs more
    round-trips, and is done for one reason: it is the only way this operation can report honest
    progress. `on_progress(done, total, next_disease_name)` fires before each disease, so the UI
    can say "plan 3 of 7" from measured state instead of animating a guess.

    Returns an entry for EVERY disease supplied, in the order supplied. A disease whose call
    failed or came back empty carries an explicit "not generated" marker rather than being
    dropped -- a silently missing card would read as "this condition needs no management".

    A failure on ONE disease does not abandon the rest. If EVERY disease failed, the first error
    is raised: that is a broken API key or a dead endpoint, not a per-condition quirk, and the
    clinician needs to see it rather than seven identical "not generated" cards.
    """
    if not GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY is not configured on the server.",
                        "The plan service is not configured on this server. Please "
                        "contact your administrator.")
    if not diseases:
        return []

    total = len(diseases)
    plans: List[Dict[str, str]] = []
    first_error: Optional[GroqError] = None
    failures = 0

    for index, disease in enumerate(diseases):
        name = disease.get("cardiac_disease_name") or disease.get("name") or "Unnamed finding"
        if on_progress is not None:
            on_progress(index, total, name)
        try:
            plans.append(generate_one_disease_care_plan(disease, patient_context))
        except GroqError as exc:
            failures += 1
            if first_error is None:
                first_error = exc
            plans.append({"cardiac_disease_name": name,
                          "rehabilitation_exercise": _NOT_GENERATED})

    if failures == total and first_error is not None:
        raise first_error

    if on_progress is not None:
        on_progress(total, total, "")
    return plans
