"""
Unified Exercise Safety & Cardiac Rehabilitation Prescription Generator.

Integrates with the Groq API (temperature=0.2) to synthesize patient echocardiographic
measurements, deterministic safety tier ceilings, and predicted pathologies into a
SINGLE, UNIFIED F.I.T.T. exercise prescription.

Strictly adheres to:
1. Single Master Plan (No fragmented disease-by-disease paragraphs).
2. Absolute Deterministic Safety Ceiling (Enforced HR and Borg RPE targets).
3. Mild Finding Absorption (Mild/incidental findings routed to Secondary Clinical Notes without false bans).
4. Rigid Output Template (Standardized Markdown format).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from app.config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
)

logger = logging.getLogger("cardiac_ocr.rehab_generator")

SYSTEM_PROMPT = """You are an expert Cardiac Rehabilitation Clinical Decision Support System.
Synthesize the provided patient data into a SINGLE, UNIFIED Exercise Prescription.

CRITICAL RULES:
1. SINGLE MASTER PLAN: Merge all findings into ONE cohesive prescription. NEVER output separate sections for individual diseases.
2. ABSOLUTE SAFETY CEILING: Strictly enforce the provided Heart Rate Ceiling and Borg RPE.
3. HANDLE MILD FINDINGS: Do not create exercise bans for mild findings (e.g., Mild MR). Mention them ONLY in the "Secondary Clinical Notes" section.
4. FORMATTING: You must output strictly in the Markdown template below.

OUTPUT TEMPLATE:
### 🏃‍♂️ Unified Cardiac Rehabilitation & Exercise Plan

**Primary Safety Ceiling:** [Insert Tier Name] (Max Heart Rate: ≤ [X] bpm | Target Borg RPE: [Y])  
**Governing Echo Findings:** [List primary conditions here]

#### 📋 The F.I.T.T. Prescription
* **Frequency:** [Specific days per week]
* **Intensity:** [Specific intensity, heart rate target, and conversational pace]
* **Time (Duration):** [Warm-up, aerobic phase, cool-down]
* **Type (Modality):** [Specific safe activities]

#### 🚫 Strict Contraindications (What to Avoid)
* [Avoidance 1 based on conditions]
* [Avoidance 2 based on conditions]

#### 💡 Secondary Clinical Notes & Monitoring
* [Acknowledge secondary/mild findings briefly]"""

# Patient-facing alternative to SYSTEM_PROMPT above (which stays the default clinical F.I.T.T.
# prompt for existing callers). Selected via the `mode` parameter on generate_unified_rehab_plan:
# "detailed" walks the patient through a full cardiac rehabilitation guide in plain language
# (Mode 1 -- "Predict Exercise based on Disease"); "general" gives a short, non-disease-specific
# conversational summary (Mode 2 -- the legacy combined plan). Neither one touches safety-tier
# math -- the deterministic ceiling computed by _resolve_safety_tier, and the primary/secondary
# conditions computed by route_conditions, are handed to the model as DATA either way. The model
# explains and formats that data; it never re-derives or overrides it.
SYSTEM_PROMPT_DETAILED = """You are a friendly, expert cardiac rehabilitation specialist explaining an
echocardiogram-based exercise plan directly to a patient, in plain, everyday language.

You are given, as DATA you must treat as already decided:
- "safety_tier_ceiling": the deterministic clinical rule engine's hard limit on intensity. This is
  a CEILING, not a suggestion -- your plan must never exceed it, no matter how mild a finding looks.
- "primary_governing_conditions": the specific cardiac finding(s) driving this plan, each with the
  exact measured value(s) that triggered it.
- "secondary_mild_findings": mild/incidental findings to mention briefly, not turn into new bans.

CRITICAL RULES:
1. USE ONLY THE SUPPLIED DATA. Never invent a measurement, threshold, disease name, or
   restriction that was not given to you. If something is not supplied, do not guess it.
2. NEVER CONTRADICT THE SAFETY CEILING. Translate "safety_tier_ceiling" into the "talk test"
   instead of Borg RPE or heart-rate math (e.g. "gentle enough that you could hold a full
   conversation the whole time") -- but never write a plan more intense than that ceiling allows.
3. PLAIN LANGUAGE ONLY. Do not use unexplained medical jargon. If a medical term is genuinely
   necessary (e.g. "ejection fraction", "hypertrophy", "regurgitation", "Valsalva"), immediately
   explain it in simple words the first time you use it.
4. NO UNSUPPORTED CLAIMS. Never say or imply that this exercise plan cures, reverses, treats, or
   fixes the underlying heart condition. It supports fitness and quality of life within the limits
   set -- nothing more.
5. BE SPECIFIC, NOT GENERIC. Reference the actual finding name(s) and, where supplied, the actual
   value(s) that triggered them -- two patients with different findings must read differently.
6. FORMATTING: Output strictly in the Markdown template below. Fill every bracketed section. Do
   not add, remove, or reorder sections.

OUTPUT TEMPLATE:
### 🏃‍♂️ Your Cardiac Rehabilitation & Exercise Guide

#### 🩺 What Your Report Shows
[Name the primary finding(s) supplied, in plain language, explaining any medical term the moment
it is used. State the specific measured value(s) supplied and how they compare to the normal
range, exactly as given -- never invent a number.]

#### ❓ Why This Matters
[1-2 short, plain-language paragraphs on why this specific finding changes how the heart responds
to exercise, and why the safety limit below exists for this patient.]

#### 📋 Your Exercise Plan
**Your Safety Limit:** [Restate the safety tier in one plain sentence using the talk test -- e.g.
"supervised, gentle activity only" vs "moderate activity at your own pace" -- consistent with
safety_tier_ceiling.]
* **How Often (Frequency):** [days per week]
* **How Hard (Intensity):** [explained via the talk test, never exceeding the ceiling]
* **How Long (Duration):** [total session length, including warm-up and cool-down]

#### 🔥 Warm-up
[Specific warm-up activity and duration, appropriate to the safety tier.]

#### 🚶 Main Exercise
[Specific recommended aerobic activity/activities and duration, appropriate to the safety tier and
the findings supplied -- e.g. walking, stationary cycling, swimming.]

#### 🧊 Cool-down
[Specific cool-down activity and duration.]

#### 💪 Strength & Flexibility
[If appropriate for this safety tier, simple resistance/strength guidance (light resistance,
higher repetitions, no straining/breath-holding); otherwise state plainly that strength training
should wait for physician clearance. Add simple flexibility or balance guidance if appropriate.]

#### 🚫 What to Avoid
* [Specific activity or exertion type to avoid, tied to this finding/tier]
* [Another, if applicable]

#### ⚠️ Warning Signs -- Stop and Seek Medical Advice If:
* [Symptom, e.g. chest pain or pressure]
* [Symptom, e.g. unusual shortness of breath, dizziness, palpitations, or fainting]
* [Tell the patient to stop the activity immediately and contact their doctor, or emergency
  services for severe symptoms, if these occur]

#### 🌟 Possible Benefits Over Time
[Plain-language paragraph on the realistic benefits of following this plan consistently --
improved stamina, exercise tolerance, heart efficiency, and general cardiovascular health -- being
careful not to claim it treats or cures the underlying finding.]

#### 🔁 Follow-up
[Plain statement that this exercise plan does not cure the underlying heart condition on its own,
and that regular cardiology/physician follow-up remains important, especially before increasing
intensity beyond what is written here.]
"""

SYSTEM_PROMPT_GENERAL = """You are a friendly doctor giving a patient a brief, GENERAL overview of
their safe exercise limits as they leave the clinic. This is the SHORT summary -- not the detailed,
disease-specific rehabilitation prescription (that is a separate, longer document); do not produce
that level of detail here.

You are given, as DATA you must treat as already decided: "safety_tier_ceiling" (the deterministic
clinical rule engine's hard limit on intensity -- a ceiling, not a suggestion), and the finding(s)
behind it ("primary_governing_conditions", "secondary_mild_findings").

CRITICAL RULES:
1. USE ONLY THE SUPPLIED DATA. Never invent a measurement, threshold, disease, or restriction.
2. NEVER CONTRADICT THE SAFETY CEILING given to you.
3. KEEP IT GENERAL, CONCISE AND SIMPLE. Do NOT give a disease-by-disease breakdown, a full F.I.T.T.
   table, exact heart-rate numbers, or Borg RPE. Avoid medical terminology entirely where possible.
4. NO UNSUPPORTED CLAIMS. Never say or imply this plan cures or reverses the underlying finding.
5. FORMATTING: Output strictly in the Markdown template below. Keep every section to 1-3 short
   sentences or bullets -- brevity is the point.

OUTPUT TEMPLATE:
### 🏃‍♂️ Your General Heart Health Overview

**The Big Picture:** [2-3 friendly, plain-language sentences summarizing overall heart health and
the general safety limit -- reassuring but consistent with the ceiling given.]

**Recommended Activities:** [1-2 general activity examples appropriate to the safety tier, e.g.
walking, light cycling, swimming.]

**General Guidance:** [One line covering roughly how often per week and how hard, in plain terms
such as "most days of the week, at a comfortable, easy pace" -- no numbers beyond that.]

**Warm-up & Cool-down:** [One short line -- a few minutes of easy movement before and after.]

**What to Avoid:**
* [Activity/exertion type to avoid for now]
* [Another, if applicable]

**When to Stop:** [One line: stop the activity and seek medical advice for chest pain or pressure,
unusual breathlessness, dizziness, palpitations, or fainting.]

*Remember: this is general guidance, not a substitute for your doctor's advice, and it does not
replace regular follow-up with your cardiologist.*
"""


def route_conditions(
    predicted_conditions: List[Union[Dict[str, Any], Any, str]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Routes predicted conditions into primary vs secondary arrays.

    - primary_conditions: Severe, Moderate, Hypertrophy, HCM, Sclerosis, Stenosis, or high-risk findings.
    - secondary_conditions: Mild or Trace findings (e.g., Mild MR, Mild TR, Trace AR, Grade I DD).
    """
    primary: List[Dict[str, Any]] = []
    secondary: List[Dict[str, Any]] = []

    for cond in predicted_conditions or []:
        if isinstance(cond, str):
            name = cond
            sev = "moderate"
            points: List[str] = []
        elif isinstance(cond, dict):
            name = cond.get("cardiac_disease_name") or cond.get("name") or "Unnamed Finding"
            sev = str(cond.get("severity") or "moderate").lower()
            points = cond.get("supporting_points") or []
        else:
            name = getattr(cond, "name", "Unnamed Finding")
            sev = str(getattr(cond, "severity", "moderate")).lower()
            points = getattr(cond, "supporting_points", [])

        name_lower = name.lower()

        is_mild = (
            sev in ("mild", "trace", "trivial")
            or "mild" in name_lower
            or "trace" in name_lower
            or "trivial" in name_lower
            or "grade i " in name_lower
            or "grade 1 " in name_lower
            or "impaired relaxation" in name_lower
        )

        is_primary = (
            sev in ("severe", "moderate")
            or "hypertrophy" in name_lower
            or "hcm" in name_lower
            or "hocm" in name_lower
            or "cardiomyopathy" in name_lower
            or "sclerosis" in name_lower
            or "stenosis" in name_lower
            or "calcification" in name_lower
            or "severe" in name_lower
            or "moderate" in name_lower
        )

        entry = {
            "name": name,
            "severity": sev,
            "supporting_points": points,
        }

        if is_mild and not is_primary:
            secondary.append(entry)
        elif is_primary:
            primary.append(entry)
        elif is_mild:
            secondary.append(entry)
        else:
            primary.append(entry)

    return primary, secondary


def _resolve_safety_tier(
    safety_tier: Optional[Union[Dict[str, Any], str]],
    patient_context: Dict[str, Any],
    primary_conditions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ensures safety tier is fully normalized into a standardized dict."""
    if isinstance(safety_tier, dict) and safety_tier.get("tier_name"):
        return safety_tier

    age = patient_context.get("age") or patient_context.get("patient_age") or 60
    try:
        age_num = float(age)
    except (ValueError, TypeError):
        age_num = 60.0

    max_hr = int(220 - age_num)

    # Derive tier if not provided or if string
    ef_val = patient_context.get("ef") or patient_context.get("ejection_fraction")
    try:
        ef_num = float(ef_val) if ef_val is not None else None
    except (ValueError, TypeError):
        ef_num = None

    ivsd_max = patient_context.get("ivsd_max") or patient_context.get("ivsd")
    try:
        ivsd_num = float(ivsd_max) if ivsd_max is not None else None
    except (ValueError, TypeError):
        ivsd_num = None

    primary_names = " ".join(p["name"].lower() for p in primary_conditions)

    # Tier 1 check
    if (ef_num is not None and ef_num < 30) or "thrombus" in primary_names or "large pericardial" in primary_names:
        return {
            "tier_level": 1,
            "tier_name": "Tier 1: Exercise Prohibited (Acute Contraindication)",
            "heart_rate_ceiling_bpm": 0,
            "borg_rpe_target": "N/A (Bedrest / Activities of Daily Living Only)",
            "exercise_allowed": False,
        }

    # Tier 2 check
    if (
        (ivsd_num is not None and ivsd_num >= 15.0)
        or "hypertrophy" in primary_names
        or "hcm" in primary_names
        or "severe" in primary_names
        or "moderate" in primary_names
        or (ef_num is not None and ef_num < 50)
    ):
        return {
            "tier_level": 2,
            "tier_name": "Tier 2: Medically Supervised Exercise Only",
            "heart_rate_ceiling_bpm": min(100, int(0.50 * max_hr)),
            "borg_rpe_target": "9–11 (Very Light to Light)",
            "exercise_allowed": True,
        }

    # Tier 3 (Default)
    return {
        "tier_level": 3,
        "tier_name": "Tier 3: Moderate Independent Exercise Permitted",
        "heart_rate_ceiling_bpm": int(0.70 * max_hr),
        "borg_rpe_target": "12–13 (Moderate / Conversational)",
        "exercise_allowed": True,
    }


# Tier-derived F.I.T.T. text reused by both patient-facing fallbacks below, so the detailed and
# general fallbacks stay consistent with each other and with the ceiling (hr_cap/rpe) computed by
# _resolve_safety_tier -- the same deterministic ceiling the live Groq prompts are handed.
_FALLBACK_TIER_TEXT = {
    1: {
        "talk_test": "resting activities of daily living only -- no structured exercise until your "
                     "cardiology team clears you",
        "frequency": "None -- formal exercise is on hold for now",
        "intensity": "Rest only; avoid physical strain and sudden positional changes",
        "duration": "Activities of daily living, as tolerated",
        "warmup": "Not applicable while exercise is on hold",
        "main": "Not applicable -- please wait for medical clearance before starting any exercise",
        "cooldown": "Not applicable while exercise is on hold",
        "strength": "No strength training until cleared by your cardiology team",
        "avoid": ["Any structured aerobic exercise, resistance training, or active sport",
                  "Sudden positional changes and heavy lifting"],
        "general_activities": "None for now -- gentle daily activities only, as advised by your doctor",
        "general_guidance": "No structured exercise right now -- your care team will tell you when "
                            "and how to safely restart",
    },
    2: {
        "talk_test": "very gentle activity -- you should be able to hold a full conversation the "
                     "entire time, without ever feeling out of breath",
        "frequency": "3 to 4 days a week, with a rest day in between",
        "intensity": "Very light and gentle, at a fully conversational pace",
        "duration": "About 5 minutes warm-up, 15-20 minutes of gentle activity, 10 minutes cool-down",
        "warmup": "5 minutes of slow, easy walking or gentle stretching",
        "main": "15-20 minutes of gentle, low-impact activity such as slow level-ground walking or "
                "easy seated stationary cycling with light resistance",
        "cooldown": "10 minutes of slow walking to bring your heart rate down gradually -- never "
                    "stop suddenly",
        "strength": "Light strength work only if your physician approves it -- no heavy lifting or "
                    "breath-holding while straining",
        "avoid": ["Straining or holding your breath while exerting (a move called the Valsalva "
                  "maneuver, which spikes pressure in the chest)",
                  "High-intensity intervals, competitive sport, or sudden bursts of effort"],
        "general_activities": "Short, gentle walks or light seated cycling",
        "general_guidance": "A few days a week, at a very easy, fully conversational pace",
    },
    3: {
        "talk_test": "moderate activity -- you should be breathing a bit harder but still able to "
                     "hold a conversation",
        "frequency": "4 to 5 days a week",
        "intensity": "Moderate, at a conversational pace",
        "duration": "5-10 minutes warm-up, 30-40 minutes of aerobic activity, 5-10 minutes cool-down",
        "warmup": "5-10 minutes of easy walking or light movement to gradually raise your heart rate",
        "main": "30-40 minutes of brisk walking, outdoor cycling on flat ground, swimming, or "
                "elliptical training",
        "cooldown": "5-10 minutes of easy walking or stretching to bring your heart rate back down",
        "strength": "Light-to-moderate resistance training 2 days a week is appropriate, focusing on "
                    "controlled movements and steady breathing rather than heavy weights",
        "avoid": ["Pushing yourself to exhaustion or lifting unaccustomed maximal weights",
                  "Exercising outdoors in extreme heat or humidity without staying well hydrated"],
        "general_activities": "Brisk walking, cycling, swimming, or similar aerobic activity",
        "general_guidance": "Most days of the week, at a comfortable pace you can hold a conversation in",
    },
}

_WARNING_SIGNS = [
    "Chest pain or pressure",
    "Unusual shortness of breath, dizziness, palpitations, or fainting",
]


def _fallback_finding_summary(primary_conditions: List[Dict[str, Any]]) -> str:
    """Plain-language, jargon-explaining summary of the supplied findings -- no invented values."""
    if not primary_conditions:
        return ("No specific abnormal finding is driving this plan; the guidance below follows "
                "the safety limit set for your overall echocardiogram results.")
    parts = []
    for cond in primary_conditions:
        points = cond.get("supporting_points") or []
        detail = f" (measured: {'; '.join(str(p) for p in points)})" if points else ""
        parts.append(f"{cond['name']}{detail}")
    return ("Your echocardiogram (heart ultrasound) showed: " + "; ".join(parts) + ". This is the "
            "finding your exercise plan below is built around.")


def _build_detailed_fallback(
    primary_conditions: List[Dict[str, Any]],
    secondary_conditions: List[Dict[str, Any]],
    tier: Dict[str, Any],
) -> str:
    """Patient-friendly, disease-specific fallback matching the Mode 1 template (§ task spec)."""
    level = tier.get("tier_level") if tier.get("tier_level") in _FALLBACK_TIER_TEXT else 3
    t = _FALLBACK_TIER_TEXT[level]
    finding_summary = _fallback_finding_summary(primary_conditions)
    sec_note = (", ".join(s["name"] for s in secondary_conditions)
               if secondary_conditions else "No other findings need special attention")
    avoid_bullets = "\n".join(f"* {a}" for a in t["avoid"])
    warning_bullets = "\n".join(f"* {w}" for w in _WARNING_SIGNS)

    return f"""### 🏃‍♂️ Your Cardiac Rehabilitation & Exercise Guide

#### 🩺 What Your Report Shows
{finding_summary} Other findings noted alongside this: {sec_note}.

#### ❓ Why This Matters
This finding affects how safely and how hard your heart can work during exercise, which is why a
specific safety limit has been set for you: {t['talk_test']}.

#### 📋 Your Exercise Plan
**Your Safety Limit:** {t['talk_test'].capitalize()}.
* **How Often (Frequency):** {t['frequency']}
* **How Hard (Intensity):** {t['intensity']}
* **How Long (Duration):** {t['duration']}

#### 🔥 Warm-up
{t['warmup']}

#### 🚶 Main Exercise
{t['main']}

#### 🧊 Cool-down
{t['cooldown']}

#### 💪 Strength & Flexibility
{t['strength']}

#### 🚫 What to Avoid
{avoid_bullets}

#### ⚠️ Warning Signs -- Stop and Seek Medical Advice If:
{warning_bullets}
* Stop the activity immediately and contact your doctor (or emergency services for severe
  symptoms) if any of these occur.

#### 🌟 Possible Benefits Over Time
Following this plan consistently can gradually improve your stamina, how well your body tolerates
activity, and your overall cardiovascular fitness -- helping you feel better day to day. This plan
supports your heart health, but it does not cure or reverse the underlying finding by itself.

#### 🔁 Follow-up
This exercise plan does not replace medical treatment. Please continue regular follow-up with your
cardiologist, especially before increasing intensity beyond what is written here."""


def _build_general_fallback(
    primary_conditions: List[Dict[str, Any]],
    secondary_conditions: List[Dict[str, Any]],
    tier: Dict[str, Any],
) -> str:
    """Brief, non-disease-specific fallback matching the Mode 2 (legacy) template (§ task spec)."""
    level = tier.get("tier_level") if tier.get("tier_level") in _FALLBACK_TIER_TEXT else 3
    t = _FALLBACK_TIER_TEXT[level]
    avoid_bullets = "\n".join(f"* {a}" for a in t["avoid"][:2])

    return f"""### 🏃‍♂️ Your General Heart Health Overview

**The Big Picture:** Based on your echocardiogram results, your safe activity level for now is:
{t['talk_test']}.

**Recommended Activities:** {t['general_activities']}.

**General Guidance:** {t['general_guidance']}.

**Warm-up & Cool-down:** A few minutes of easy movement before and after activity.

**What to Avoid:**
{avoid_bullets}

**When to Stop:** Stop and seek medical advice for chest pain or pressure, unusual breathlessness,
dizziness, palpitations, or fainting.

*Remember: this is general guidance, not a substitute for your doctor's advice, and it does not
replace regular follow-up with your cardiologist.*"""


def _build_deterministic_fallback(
    patient_context: Dict[str, Any],
    primary_conditions: List[Dict[str, Any]],
    secondary_conditions: List[Dict[str, Any]],
    tier: Dict[str, Any],
    mode: Optional[str] = None,
) -> str:
    """Deterministic fallback matching the rigid template when LLM is unavailable.

    `mode` mirrors generate_unified_rehab_plan's parameter: "detailed" and "general" route to the
    patient-facing fallbacks above so a Groq outage still respects which mode was asked for; any
    other value (including the default None) keeps the original clinical F.I.T.T. fallback below,
    unchanged, for existing callers.
    """
    if mode == "detailed":
        return _build_detailed_fallback(primary_conditions, secondary_conditions, tier)
    if mode == "general":
        return _build_general_fallback(primary_conditions, secondary_conditions, tier)

    tier_name = tier.get("tier_name", "Tier 2: Medically Supervised Exercise Only")
    hr_cap = tier.get("heart_rate_ceiling_bpm", 100)
    rpe = tier.get("borg_rpe_target", "9–11 (Very Light to Light)")

    gov_list = [p["name"] for p in primary_conditions] or ["Preserved Ventricular Function"]
    gov_str = ", ".join(gov_list)

    sec_list = [s["name"] for s in secondary_conditions] or ["No significant secondary findings"]
    sec_str = ", ".join(sec_list)

    if tier.get("tier_level") == 1:
        return f"""### 🏃‍♂️ Unified Cardiac Rehabilitation & Exercise Plan

**Primary Safety Ceiling:** {tier_name} (Max Heart Rate: N/A | Target Borg RPE: Resting ADLs only)  
**Governing Echo Findings:** {gov_str}

#### 📋 The F.I.T.T. Prescription
* **Frequency:** 0 days/week (Formal exercise prohibited).
* **Intensity:** Resting baseline only.
* **Time (Duration):** Activities of daily living as tolerated.
* **Type (Modality):** Rest and medical stabilization.

#### 🚫 Strict Contraindications (What to Avoid)
* Avoid all structured aerobic exercise, resistance lifting, and active sports until cleared by cardiology.
* Avoid sudden positional changes and physical strain.

#### 💡 Secondary Clinical Notes & Monitoring
* Background findings ({sec_str}) noted. Immediate specialist evaluation required prior to exercise clearance."""

    if tier.get("tier_level") == 2:
        return f"""### 🏃‍♂️ Unified Cardiac Rehabilitation & Exercise Plan

**Primary Safety Ceiling:** {tier_name} (Max Heart Rate: ≤ {hr_cap} bpm | Target Borg RPE: {rpe})  
**Governing Echo Findings:** {gov_str}

#### 📋 The F.I.T.T. Prescription
* **Frequency:** 3 to 4 days per week on non-consecutive days with at least 24 hours rest between sessions.
* **Intensity:** Very light to light (Borg RPE 9–11). Maintain a strictly conversational pace with heart rate ≤ {hr_cap} bpm.
* **Time (Duration):** 5-minute warm-up, 15–20 minutes continuous gentle aerobic activity, followed by a mandatory 10-minute active walking cool-down.
* **Type (Modality):** Low-impact continuous activities: level-ground walking or seated stationary recumbent cycling with minimal resistance.

#### 🚫 Strict Contraindications (What to Avoid)
* Strict avoidance of Valsalva maneuvers and heavy resistance lifting (>2–5 kg).
* Prohibition of High-Intensity Interval Training (HIIT), competitive sprinting, and burst exertion.
* Avoid abrupt termination of exertion; active 10-minute cool-down is mandatory.

#### 💡 Secondary Clinical Notes & Monitoring
* Background findings ({sec_str}) noted as stable and absorbed into overall plan; no additional exercise restrictions required beyond Tier 2 ceiling."""

    return f"""### 🏃‍♂️ Unified Cardiac Rehabilitation & Exercise Plan

**Primary Safety Ceiling:** {tier_name} (Max Heart Rate: ≤ {hr_cap} bpm | Target Borg RPE: {rpe})  
**Governing Echo Findings:** {gov_str}

#### 📋 The F.I.T.T. Prescription
* **Frequency:** 4 to 5 days per week.
* **Intensity:** Moderate intensity (Borg RPE 12–13). Conversational pace with heart rate ≤ {hr_cap} bpm.
* **Time (Duration):** 5–10 minute warm-up, 30–40 minutes aerobic conditioning, 5–10 minute cool-down.
* **Type (Modality):** Brisk walking, outdoor cycling on flat terrain, swimming, or elliptical training.

#### 🚫 Strict Contraindications (What to Avoid)
* Avoid extreme exhaustive exhaustion or unaccustomed maximal resistance straining.
* Avoid exercising in excessive environmental heat or humidity without adequate hydration.

#### 💡 Secondary Clinical Notes & Monitoring
* Secondary findings ({sec_str}) are mild and require periodic surveillance at standard follow-up intervals."""


def generate_unified_rehab_plan(
    patient_context: Dict[str, Any],
    predicted_conditions: List[Union[Dict[str, Any], Any, str]],
    safety_tier: Optional[Union[Dict[str, Any], str]] = None,
    mode: Optional[str] = None,
) -> str:
    """
    Generates a unified exercise prescription markdown plan using Groq API (temperature=0.2).

    Parameters:
    -----------
    patient_context: Dict containing patient demographics, EF, max IVSd, max PWd, gradients, etc.
    predicted_conditions: List of predicted disease dictionaries, objects, or names.
    safety_tier: Dict or string specifying the deterministic Safety Ceiling.
    mode: Which system prompt/output template to use --
        "detailed" -> patient-facing daily guide in plain language (SYSTEM_PROMPT_DETAILED).
        "general"  -> short, conversational summary (SYSTEM_PROMPT_GENERAL).
        anything else (including None, the default) -> the original clinical F.I.T.T. prompt
        (SYSTEM_PROMPT), unchanged for existing callers that don't pass `mode`.

    Returns:
    --------
    Pure Markdown string conforming to the template selected by `mode`.
    """
    if mode == "detailed":
        system_prompt = SYSTEM_PROMPT_DETAILED
    elif mode == "general":
        system_prompt = SYSTEM_PROMPT_GENERAL
    else:
        system_prompt = SYSTEM_PROMPT
    # Step 1: Data Routing
    primary_conditions, secondary_conditions = route_conditions(predicted_conditions)
    tier = _resolve_safety_tier(safety_tier, patient_context, primary_conditions)

    # If Groq API key is missing, return high-quality deterministic fallback immediately
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not configured. Using deterministic unified rehab fallback.")
        return _build_deterministic_fallback(patient_context, primary_conditions, secondary_conditions, tier, mode)

    # Step 2 & 3: Construct Payload and User Prompt
    tier_name = tier.get("tier_name", "Tier 2: Medically Supervised Exercise Only")
    hr_cap = tier.get("heart_rate_ceiling_bpm", 100)
    rpe_target = tier.get("borg_rpe_target", "9–11 (Very Light to Light)")

    user_prompt_data = {
        "patient_demographics": {
            "age": patient_context.get("age") or patient_context.get("patient_age"),
            "gender": patient_context.get("gender") or patient_context.get("patient_gender"),
        },
        "echo_measurements": {
            "ejection_fraction": patient_context.get("ef") or patient_context.get("ejection_fraction"),
            "ivsd_max_mm": patient_context.get("ivsd_max") or patient_context.get("ivsd"),
            "pwd_max_mm": patient_context.get("pwd_max") or patient_context.get("pwd"),
            "pasp_mmhg": patient_context.get("pasp"),
            "lvot_peak_gradient_mmhg": patient_context.get("lvot_peak_gradient"),
            "av_peak_gradient_mmhg": patient_context.get("av_peak_gradient"),
        },
        "safety_tier_ceiling": {
            "tier_name": tier_name,
            "max_heart_rate_bpm": hr_cap,
            "target_borg_rpe": rpe_target,
        },
        "primary_governing_conditions": primary_conditions,
        "secondary_mild_findings": secondary_conditions,
    }

    user_prompt = (
        f"Patient Clinical Context and Predicted Diseases:\n"
        f"{json.dumps(user_prompt_data, indent=2, default=str)}\n\n"
        f"Synthesize the SINGLE, UNIFIED Cardiac Rehabilitation & Exercise Plan matching the exact Markdown template now."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(GROQ_API_URL, json=payload, headers=headers, timeout=GROQ_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content and "### 🏃" in content:
                return content.strip()
            elif content:
                return content.strip()
            else:
                logger.error("Empty content received from Groq API.")
        else:
            logger.error(f"Groq API returned HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        logger.error(f"Exception during Groq unified rehab generation: {exc}")

    # Fallback if API call failed or timed out
    return _build_deterministic_fallback(patient_context, primary_conditions, secondary_conditions, tier, mode)
