"""
Is the uploaded file actually a cardiac / 2D-Echo report at all?

WHY THIS EXISTS
---------------
Uploading a WhatsApp screenshot, a selfie or an unrelated PDF used to end in:

    "text was read but no cardiac parameters could be recognised. Please re-upload a
     higher-resolution scan, or enter the values manually."

That message blames scan quality for what is really a wrong-file mistake, and it invites
exactly the wrong fix -- re-scanning the screenshot at a higher DPI, or worse, typing echo
values in by hand against a document that has none. This module separates the two cases so the
UI can say "this is not an echo report" when that is what happened.

HOW IT DECIDES
--------------
Purely local keyword evidence over the OCR'd text -- no model call, no network. Terms are
weighted: STRONG terms essentially only occur on an echocardiography report; SUPPORTING terms
are general cardiac/medical vocabulary that is meaningful in numbers but not on its own.
A document clears the bar at REPORT_SCORE_THRESHOLD.

WHERE IT IS SAFE TO USE
-----------------------
The pipeline consults this ONLY after parameter extraction has already come back empty. That
placement is deliberate: a wrong verdict can therefore never reject a report that extracted
fine, it can only change the wording of a failure that had already happened. The thresholds are
tuned accordingly -- generous towards "this is a report", because a badly OCR'd genuine echo
should still be told to re-scan rather than accused of being the wrong file.
"""
import re
from typing import List, NamedTuple, Tuple

# Terms that essentially only appear on an echocardiography / cardiac study report.
# One of these plus a little support is enough; three on their own clear the bar.
STRONG_TERMS: Tuple[str, ...] = (
    "echocardiogram", "echocardiography", "echocardiographic", "echo report",
    "2d echo", "2-d echo", "two dimensional echo", "transthoracic", "transoesophageal",
    "transesophageal", "colour doppler", "color doppler", "doppler study",
    "ejection fraction", "lvef", "lvidd", "lvids", "lvedd", "lvesd", "ivsd", "ivss",
    "lvpwd", "lvpws", "e/a ratio", "rwma", "regional wall motion", "wall motion",
    "interventricular septum", "left ventricle", "left ventricular", "right ventricle",
    "left atrium", "right atrium", "mitral valve", "aortic valve", "tricuspid valve",
    "pulmonary valve", "mitral regurgitation", "tricuspid regurgitation",
    "aortic regurgitation", "aortic stenosis", "mitral stenosis", "diastolic dysfunction",
    "pericardial effusion", "pulmonary artery systolic pressure", "pasp", "tapse",
    "parasternal", "apical four chamber", "four chamber view", "m-mode", "m mode",
    "simpson", "teichholz", "sinus of valsalva", "aortic root", "inferior vena cava",
)

# General cardiac / report vocabulary. Common on echo reports, but also on other medical
# documents -- so these only count towards the score, they never carry a verdict alone.
SUPPORTING_TERMS: Tuple[str, ...] = (
    "cardiology", "cardiologist", "cardiac", "heart", "myocardial", "myocardium",
    "ventricle", "ventricular", "atrium", "atrial", "valve", "valvular", "doppler",
    "regurgitation", "stenosis", "hypertrophy", "hypokinesia", "akinesia", "dilated",
    "systolic", "diastolic", "aorta", "aortic", "mitral", "tricuspid", "pulmonic",
    "septum", "septal", "chamber", "impression", "conclusion", "findings",
    "sinus rhythm", "ecg", "ekg", "electrocardiogram", "bpm", "heart rate",
    "normal study", "referred by", "sonologist", "dr.", "patient",
)

STRONG_WEIGHT = 3
SUPPORTING_WEIGHT = 1
REPORT_SCORE_THRESHOLD = 5

# Below this many alphabetic characters there is not enough text to judge anything -- a
# screenshot of a chat bubble, a photo of a wall. Treated as "not a report" outright.
MIN_TEXT_CHARS = 25


class DocumentVerdict(NamedTuple):
    is_report: bool
    score: int
    strong_hits: List[str]
    supporting_hits: List[str]

    @property
    def matched(self) -> List[str]:
        return self.strong_hits + self.supporting_hits


def _normalize(text: str) -> str:
    """Lowercase, and collapse everything that is not a letter/digit/`.`/`/`/`-` to a single
    space. Keeping `.`, `/` and `-` preserves the terms that carry them ("e/a ratio", "m-mode",
    "2-d echo", "dr."); collapsing the rest makes OCR punctuation noise harmless."""
    return re.sub(r"[^a-z0-9./-]+", " ", (text or "").lower()).strip()


def _hits(normalized: str, terms: Tuple[str, ...]) -> List[str]:
    """Terms present as whole words/phrases. Word boundaries matter: without them "pasp" would
    match inside an unrelated word, and short abbreviations would fire on almost any text."""
    found = []
    for term in terms:
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(term)
    return found


def looks_like_cardiac_report(*texts: str) -> DocumentVerdict:
    """Weigh the keyword evidence across every text fragment the pipeline managed to read.

    Accepts several fragments (raw OCR text, narrative blocks, flattened table cells) because a
    genuine report can carry its evidence in any one of them -- the table half of a scan may read
    cleanly while the narrative half does not.
    """
    normalized = _normalize("\n".join(t for t in texts if t))

    strong = _hits(normalized, STRONG_TERMS)
    supporting = _hits(normalized, SUPPORTING_TERMS)
    score = STRONG_WEIGHT * len(strong) + SUPPORTING_WEIGHT * len(supporting)

    # Too little text to judge -- but only when there is no strong evidence in it. A scan that
    # degraded to three words is barely readable either way; a scan that degraded to three words
    # INCLUDING "echocardiography" is a real report, and must not be called the wrong file.
    if not strong and len(re.sub(r"[^a-z]", "", normalized)) < MIN_TEXT_CHARS:
        return DocumentVerdict(False, score, strong, supporting)

    return DocumentVerdict(score >= REPORT_SCORE_THRESHOLD, score, strong, supporting)
