"""
Extraction accuracy harness.

Runs the real extraction pipeline over the labelled reports in tests/fixtures/ground_truth.json
and scores every parameter into one of four buckets:

    correct         the expected value was extracted
    WRONG           a value was extracted, but a different one    <-- clinically dangerous
    FALSE POSITIVE  the report does not state this parameter at all, yet a value was stored
                                                                  <-- clinically dangerous
    missed          the report states it, nothing was extracted   <-- safe: renders "Not detected"

WRONG and FALSE POSITIVE are reported separately from MISSED, and deliberately so. A missed
parameter shows up as "Not detected" and a doctor fills it in; a wrong one silently feeds the
33-rule prediction engine. Any change that trades misses for wrongs is a regression even if the
headline percentage improves.

USAGE
    ./venv/Scripts/python.exe tools/accuracy_report.py
    ./venv/Scripts/python.exe tools/accuracy_report.py --no-llm      # deterministic, offline
    ./venv/Scripts/python.exe tools/accuracy_report.py --no-cache    # force re-OCR
    ./venv/Scripts/python.exe tools/accuracy_report.py --save baseline.json
    ./venv/Scripts/python.exe tools/accuracy_report.py --diff baseline.json

OCR CACHE
    OCR costs ~12s/page on CPU, so the per-page doc_result is cached under
    .ocr_cache/ keyed by (file content hash, mtimes of the modules that PRODUCE doc_result).
    extractor.py is deliberately NOT part of the key -- iterating on the extractor therefore
    replays cached OCR and runs in well under a second.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

FIXTURE = BACKEND / "tests" / "fixtures" / "ground_truth.json"
UPLOADS = BACKEND / "app" / "uploads"
CACHE_DIR = BACKEND / "tools" / ".ocr_cache"

# Modules whose behaviour changes the OCR/doc_result stage. extractor.py is excluded on purpose.
_CACHE_KEY_SOURCES = [
    BACKEND / "app" / "ocr" / "text_extraction.py",
    BACKEND / "app" / "ocr" / "preprocessing.py",
    BACKEND / "app" / "ocr" / "pipeline.py",
    BACKEND / "app" / "config.py",
]

CORRECT, WRONG, FALSE_POSITIVE, MISSED = "correct", "wrong", "false_positive", "missed"


# --------------------------------------------------------------------------- comparison
def _norm(value: Optional[str]) -> str:
    """Compare on a shape that ignores cosmetic spacing only.

    "4.31 cm" == "4.31cm" == "4.31 CM", but 4.31 != 4.3 -- a dropped decimal is a real
    difference and must never be normalized away.
    """
    return "".join((value or "").split()).lower()


_NUMBER_RE = re.compile(r"\d")


def _matches(expected, actual: Optional[str]) -> bool:
    if isinstance(expected, dict):
        if "any" in expected:
            return any(_matches(alt, actual) for alt in expected["any"])
        if "contains" in expected:
            return expected["contains"].lower() in (actual or "").lower()
        if "qualitative_ok" in expected:
            # The report states this structure only in prose ("Left atrial size is normal"),
            # never as a measurement. A descriptor is a correct read; a NUMBER here would be
            # invented, so it still scores as a false positive.
            return actual is None or not _NUMBER_RE.search(actual)
        raise ValueError(f"Unrecognised expectation form: {expected!r}")
    return _norm(expected) == _norm(actual)


def classify(expected, actual: Optional[str]) -> str:
    if expected is None:
        return CORRECT if actual is None else FALSE_POSITIVE
    if isinstance(expected, dict) and "qualitative_ok" in expected:
        # Nothing extracted is acceptable too -- the prose is optional information, so an empty
        # field is a miss of something the report only implied, not a wrong answer.
        return CORRECT if _matches(expected, actual) else FALSE_POSITIVE
    if actual is None:
        return MISSED
    return CORRECT if _matches(expected, actual) else WRONG


# --------------------------------------------------------------------------- extraction
def _cache_key(file_path: Path) -> str:
    h = hashlib.md5(file_path.read_bytes()).hexdigest()
    stamp = "|".join(
        f"{p.name}:{p.stat().st_mtime_ns}" for p in _CACHE_KEY_SOURCES if p.exists()
    )
    return hashlib.md5(f"{h}|{stamp}".encode()).hexdigest()


def build_doc_result(file_path: Path, use_cache: bool = True) -> dict:
    """Mirror app/ocr/pipeline.process_report's engine selection, without the DB writes."""
    from app.ocr.pipeline import _build_doc_result
    from app.ocr.preprocessing import extract_structured_digital_pdf, is_digital_pdf
    from app.ocr.extractor import extract_parameters_structured

    cache_file = CACHE_DIR / f"{_cache_key(file_path)}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    meta: dict = {}
    if file_path.suffix.lower() == ".pdf" and is_digital_pdf(str(file_path)):
        doc = extract_structured_digital_pdf(str(file_path))
        # Same guard as pipeline.py:243 -- a junk embedded text layer falls through to OCR.
        results, _ = extract_parameters_structured(doc)
        if not any(ef.value is not None for ef in results.values()):
            doc = _build_doc_result(str(file_path), meta)
    else:
        doc = _build_doc_result(str(file_path), meta)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def extract(file_path: Path, use_cache: bool = True) -> Dict[str, Optional[str]]:
    from app.ocr.extractor import extract_parameters_structured

    doc = build_doc_result(file_path, use_cache)
    results, _ = extract_parameters_structured(doc)
    return {canon: ef.value for canon, ef in results.items()}


# --------------------------------------------------------------------------- reporting
def _tick(bucket: str) -> str:
    return {CORRECT: "ok  ", WRONG: "WRONG", FALSE_POSITIVE: "FALSE+", MISSED: "miss"}[bucket]


def run(args) -> dict:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reports = fixture["reports"]

    totals = {CORRECT: 0, WRONG: 0, FALSE_POSITIVE: 0, MISSED: 0}
    per_report = {}

    for filename, expectations in reports.items():
        if args.report and args.report not in filename:
            continue
        path = UPLOADS / filename
        if not path.exists():
            print(f"!! missing upload, skipped: {filename}")
            continue

        expectations = {k: v for k, v in expectations.items() if not k.startswith("_")}
        try:
            actual = extract(path, use_cache=not args.no_cache)
        except Exception as exc:  # noqa: BLE001 -- a crashing report is a result, not a stop
            print(f"!! {filename} raised {type(exc).__name__}: {exc}")
            per_report[filename] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        buckets = {}
        for canon, expected in expectations.items():
            got = actual.get(canon)
            bucket = classify(expected, got)
            buckets[canon] = {"expected": expected, "actual": got, "bucket": bucket}
            totals[bucket] += 1

        per_report[filename] = buckets

        scored = len(buckets)
        ok = sum(1 for b in buckets.values() if b["bucket"] == CORRECT)
        print(f"\n=== {filename}")
        print(f"    {ok}/{scored} correct")
        for canon, info in sorted(buckets.items(), key=lambda kv: kv[1]["bucket"]):
            if info["bucket"] == CORRECT and not args.verbose:
                continue
            exp = info["expected"]
            exp_s = json.dumps(exp) if isinstance(exp, dict) else repr(exp)
            print(f"      {_tick(info['bucket']):7} {canon:22} expected={exp_s:28} got={info['actual']!r}")

    scored = sum(totals.values())
    print("\n" + "=" * 66)
    print(f"  scored parameters : {scored}")
    if scored:
        print(f"  correct           : {totals[CORRECT]:4}  ({100*totals[CORRECT]/scored:.1f}%)")
        print(f"  WRONG             : {totals[WRONG]:4}  ({100*totals[WRONG]/scored:.1f}%)   <-- dangerous")
        print(f"  FALSE POSITIVE    : {totals[FALSE_POSITIVE]:4}  ({100*totals[FALSE_POSITIVE]/scored:.1f}%)   <-- dangerous")
        print(f"  missed            : {totals[MISSED]:4}  ({100*totals[MISSED]/scored:.1f}%)")
    print("=" * 66)

    return {"totals": totals, "per_report": per_report}


def diff(current: dict, baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print(f"\n--- diff vs {baseline_path.name} ---")
    for bucket in (CORRECT, WRONG, FALSE_POSITIVE, MISSED):
        was, now = baseline["totals"][bucket], current["totals"][bucket]
        delta = now - was
        if delta:
            print(f"  {bucket:15} {was:4} -> {now:4}  ({delta:+d})")

    for filename, buckets in current["per_report"].items():
        old = baseline["per_report"].get(filename, {})
        for canon, info in buckets.items():
            was = (old.get(canon) or {}).get("bucket")
            if was and was != info["bucket"]:
                arrow = "IMPROVED" if info["bucket"] == CORRECT else "REGRESSED"
                print(f"  {arrow:10} {filename[:14]} {canon:22} {was} -> {info['bucket']}"
                      f"  got={info['actual']!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", help="only score reports whose filename contains this")
    ap.add_argument("--no-cache", action="store_true", help="force re-OCR")
    ap.add_argument("--no-llm", action="store_true", help="disable the Groq semantic calls")
    ap.add_argument("--verbose", "-v", action="store_true", help="also list correct parameters")
    ap.add_argument("--save", help="write results to this JSON file")
    ap.add_argument("--diff", help="compare against a previously --saved JSON file")
    args = ap.parse_args()

    if args.no_llm:
        os.environ["GROQ_SEMANTIC_ENABLED"] = "0"

    from app import config
    print(f"Groq semantic layer: {'ON' if config.GROQ_SEMANTIC_ENABLED and config.GROQ_API_KEY else 'OFF'}")

    results = run(args)

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nsaved -> {args.save}")
    if args.diff:
        diff(results, Path(args.diff))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
