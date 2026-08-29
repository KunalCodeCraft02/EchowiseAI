"""
Clinician feedback loop + per-disease Groq care plans.

Covers the two things that would quietly hurt a user: a physician's written note being lost, and
Markdown reaching a clinical note that is supposed to read as plain typed text.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import models
from app.database import SessionLocal, init_db
from app.main import app
from app.predictor import groq_client as G


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


@pytest.fixture(scope="module")
def auth(client):
    email = "feedback-tests@example.com"
    resp = client.post("/api/auth/signup",
                       json={"full_name": "Dr Test", "email": email, "password": "secret123"})
    token = (resp.json()["access_token"] if resp.status_code == 201 else
             client.post("/api/auth/login",
                         data={"username": email, "password": "secret123"}).json()["access_token"])
    return {"headers": {"Authorization": f"Bearer {token}"}, "email": email}


def _make_report(email, **params):
    db = SessionLocal()
    doctor = db.query(models.Doctor).filter_by(email=email).one()
    report = models.CardiacReport(doctor_id=doctor.id, patient_name="Test Patient",
                                  status="extracted", **params)
    db.add(report)
    db.commit()
    uid = report.report_uid
    db.close()
    return uid


# All nine compulsory groups filled with safe/normal values EXCEPT EF, which is left for the
# caller to override -- reaching a clear "No Exercise Contraindication Found" verdict rather than
# "Indeterminate" (§4: Indeterminate is now itself a hard ceiling that withholds any plan, so a
# report this incomplete would otherwise never reach the mocked LLM call these tests assert on).
_CLEAR_EXERCISE_SAFETY = {
    "av_peak_gradient": "8 mmHg", "mv_peak_gradient": "4 mmHg", "pasp": "25 mmHg",
    "pericardial_effusion": "No effusion", "clots_thrombus": "No thrombus",
    "lvot_peak_gradient": "10 mmHg", "wall_motion": "Normal", "rwma": "No", "ivsd": "9 mm",
}


# ===========================================================================================
# Prediction endpoint now runs v4.0
# ===========================================================================================
def test_predict_uses_v4_and_returns_explainable_points(client, auth):
    uid = _make_report(auth["email"], ef="40%", ivsd="13 mm", la_diameter="44 mm")
    out = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()

    assert out["rules_total"] == 51           # v4.0 reports parameter coverage
    names = [d["cardiac_disease_name"] for d in out["diseases"]]
    assert "Moderate Left Ventricular Dysfunction" in names
    for disease in out["diseases"]:
        assert disease["supporting_points"], "every card must explain itself"


def test_prediction_is_sorted_by_priority(client, auth):
    uid = _make_report(auth["email"], ef="30%", clots_thrombus="Thrombus present", pasp="55 mmHg")
    out = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()
    assert out["diseases"][0]["cardiac_disease_name"] == "Intracardiac Thrombus"


# ===========================================================================================
# Feedback: confirm + notes
# ===========================================================================================
def test_confirm_and_notes_save_independently(client, auth):
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    name = "Moderate Left Ventricular Dysfunction"

    first = client.post(f"/api/reports/{uid}/confirmations", headers=auth["headers"],
                        json={"disease_name": name, "confirmed": True}).json()
    assert first["confirmed"] is True

    # Sending only notes must not wipe the confirmation, and vice versa.
    second = client.post(f"/api/reports/{uid}/confirmations", headers=auth["headers"],
                         json={"disease_name": name, "clinician_notes": "Agree, started ACE-i."}).json()
    assert second["confirmed"] is True
    assert second["clinician_notes"] == "Agree, started ACE-i."

    third = client.post(f"/api/reports/{uid}/confirmations", headers=auth["headers"],
                        json={"disease_name": name, "confirmed": False}).json()
    assert third["clinician_notes"] == "Agree, started ACE-i."


def test_feedback_rides_along_on_the_report_payload(client, auth):
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    client.post(f"/api/reports/{uid}/confirmations", headers=auth["headers"],
                json={"disease_name": "Moderate Left Ventricular Dysfunction", "confirmed": True})

    report = client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()
    assert any(c["confirmed"] for c in report["confirmations"])


def test_notes_survive_a_re_prediction_and_are_marked_stale(client, auth):
    """The whole point of the staleness design: a physician's written correction must never be
    destroyed because a parameter changed and renamed the disease."""
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    client.post(f"/api/reports/{uid}/confirmations", headers=auth["headers"],
                json={"disease_name": "Moderate Left Ventricular Dysfunction",
                      "confirmed": True, "clinician_notes": "Reviewed on ward round."})

    db = SessionLocal()
    report = db.query(models.CardiacReport).filter_by(report_uid=uid).one()
    report.ef = "60%"                      # now normal -> that disease is no longer predicted
    db.add(report)
    db.commit()
    db.close()
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    rows = client.get(f"/api/reports/{uid}/confirmations", headers=auth["headers"]).json()
    row = next(r for r in rows if r["disease_name"] == "Moderate Left Ventricular Dysfunction")
    assert row["is_stale"] is True
    assert row["clinician_notes"] == "Reviewed on ward round."      # never lost


def test_staleness_clears_when_the_disease_returns(client, auth):
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    name = "Moderate Left Ventricular Dysfunction"
    client.post(f"/api/reports/{uid}/confirmations", headers=auth["headers"],
                json={"disease_name": name, "confirmed": True})

    for ef in ("60%", "40%"):
        db = SessionLocal()
        report = db.query(models.CardiacReport).filter_by(report_uid=uid).one()
        report.ef = ef
        db.add(report)
        db.commit()
        db.close()
        client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    rows = client.get(f"/api/reports/{uid}/confirmations", headers=auth["headers"]).json()
    assert next(r for r in rows if r["disease_name"] == name)["is_stale"] is False


def test_another_doctor_cannot_reach_this_report(client, auth):
    uid = _make_report(auth["email"], ef="40%")
    other = client.post("/api/auth/signup", json={"full_name": "Dr Other",
                                                  "email": "other-doc@example.com",
                                                  "password": "secret123"})
    token = (other.json()["access_token"] if other.status_code == 201 else
             client.post("/api/auth/login",
                         data={"username": "other-doc@example.com",
                               "password": "secret123"}).json()["access_token"])
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"/api/reports/{uid}/confirmations", headers=headers).status_code == 403
    assert client.post(f"/api/reports/{uid}/confirmations", headers=headers,
                       json={"disease_name": "X", "confirmed": True}).status_code == 403


# ===========================================================================================
# Groq per-disease care plans
# ===========================================================================================
@pytest.mark.parametrize("payload", [
    [{"cardiac_disease_name": "A", "rehabilitation_exercise": "Walk."}],
    {"plans": [{"cardiac_disease_name": "A", "rehabilitation_exercise": "Walk."}]},
    {"cardiac_disease_name": "A", "rehabilitation_exercise": "Walk."},
])
def test_all_three_json_shapes_parse(payload):
    """Groq's JSON mode varies its wrapper. Failing a clinician's request over a key name is
    not an acceptable outcome."""
    assert len(G._coerce_plan_list(payload)) == 1


def test_markdown_is_stripped_server_side():
    messy = "## Exercise\n\n**Walk** briskly.\n- Avoid *heavy* lifting.\n`no code`"
    cleaned = G.strip_markdown(messy)
    for symbol in ("#", "*", "`"):
        assert symbol not in cleaned
    assert "Walk briskly." in cleaned
    assert "Avoid heavy lifting." in cleaned


def _diseases(*names):
    """generate_disease_care_plans() now takes the FULL per-disease context dict (§4), not a bare
    name -- these tests only care about the name, so this stands in for the rest."""
    return [{"cardiac_disease_name": n, "category": "Test", "severity": "moderate",
            "supporting_points": [], "fields": []} for n in names]


def _name_of(disease):
    """Route-level mocks receive the same dicts reports.py now passes -- pull the name back out,
    mirroring what generate_one_disease_care_plan() itself does."""
    return disease.get("cardiac_disease_name") or disease.get("name") or "Unnamed finding"


def _plan_response(plans):
    """A Groq 200 carrying `plans`. One call is now made per disease, so tests drive a SEQUENCE
    of these -- one response per expected call."""
    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": json.dumps({"plans": plans})}}]}
    return _Resp()


def _responder(monkeypatch, responses):
    """Serve `responses` in order, one per httpx.post call, and record how many calls happened."""
    calls = []

    def _post(*args, **kwargs):
        result = responses[min(len(calls), len(responses) - 1)]
        calls.append(kwargs.get("json"))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(G, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(G.httpx, "post", _post)
    return calls


def test_plans_are_returned_for_every_disease_even_if_the_model_skips_one(monkeypatch):
    """A silently missing card would read as 'this condition needs no management'."""
    # Disease A answered; the call for Disease B comes back with an empty plan list.
    _responder(monkeypatch, [
        _plan_response([{"cardiac_disease_name": "Disease A",
                         "rehabilitation_exercise": "**Walk** daily."}]),
        _plan_response([]),
    ])

    plans = G.generate_disease_care_plans(_diseases("Disease A", "Disease B"), {})
    assert [p["cardiac_disease_name"] for p in plans] == ["Disease A", "Disease B"]
    assert plans[0]["rehabilitation_exercise"] == "Walk daily."      # markdown gone
    assert "No plan was generated" in plans[1]["rehabilitation_exercise"]


def test_one_groq_call_is_made_per_disease(monkeypatch):
    """The per-disease split is what makes the progress percentage a measurement rather than an
    animation -- if this collapses back to one batched call, the bar silently becomes a guess."""
    calls = _responder(monkeypatch, [
        _plan_response([{"cardiac_disease_name": "X", "rehabilitation_exercise": "Walk."}]),
    ])

    G.generate_disease_care_plans(_diseases("Disease A", "Disease B", "Disease C"), {})
    assert len(calls) == 3


def test_progress_callback_reports_measured_counts(monkeypatch):
    """done/total must be real completed-call counts, in order, ending at total."""
    _responder(monkeypatch, [
        _plan_response([{"cardiac_disease_name": "X", "rehabilitation_exercise": "Walk."}]),
    ])

    seen = []
    G.generate_disease_care_plans(_diseases("A", "B", "C"), {},
                                  on_progress=lambda done, total, name: seen.append((done, total, name)))
    assert seen == [(0, 3, "A"), (1, 3, "B"), (2, 3, "C"), (3, 3, "")]


def test_one_failed_disease_does_not_abandon_the_others(monkeypatch):
    """A per-condition failure marks that card only; the rest of the plans still reach the page."""
    _responder(monkeypatch, [
        _plan_response([{"cardiac_disease_name": "A", "rehabilitation_exercise": "Walk."}]),
        G.GroqError("Groq API request timed out."),
        _plan_response([{"cardiac_disease_name": "C", "rehabilitation_exercise": "Swim."}]),
    ])

    plans = G.generate_disease_care_plans(_diseases("A", "B", "C"), {})
    assert plans[0]["rehabilitation_exercise"] == "Walk."
    assert "No plan was generated" in plans[1]["rehabilitation_exercise"]
    assert plans[2]["rehabilitation_exercise"] == "Swim."


def test_a_total_failure_still_raises(monkeypatch):
    """Every disease failing is a broken key or a dead endpoint -- the clinician must see the
    error, not a page of identical 'not generated' cards."""
    _responder(monkeypatch, [G.GroqError("Groq API returned 401: bad key")])

    with pytest.raises(G.GroqError):
        G.generate_disease_care_plans(_diseases("A", "B"), {})


def test_per_disease_plans_carry_no_dietary_guidance(monkeypatch):
    """This flow is exercise-only. A diet key the model volunteers must not reach a clinician."""
    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": json.dumps({"plans": [
                {"cardiac_disease_name": "Disease A",
                 "rehabilitation_exercise": "Walk daily.",
                 "recommended_diet": "Reduce salt."},
            ]})}}]}

    monkeypatch.setattr(G, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(G.httpx, "post", lambda *a, **k: _Resp())

    plans = G.generate_disease_care_plans(_diseases("Disease A"), {})
    assert plans[0]["rehabilitation_exercise"] == "Walk daily."
    assert "recommended_diet" not in plans[0]
    assert "recommended_diet" not in G._DISEASE_PLAN_SYSTEM_PROMPT


def test_disease_name_always_comes_from_our_engine_not_the_model(monkeypatch):
    """The card must match the prediction it belongs to, even if the model paraphrases."""
    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": json.dumps({"plans": [
                {"cardiac_disease_name": "LV dysfunction (moderate)",
                 "rehabilitation_exercise": "Walk."},
            ]})}}]}

    monkeypatch.setattr(G, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(G.httpx, "post", lambda *a, **k: _Resp())

    plans = G.generate_disease_care_plans(_diseases("Moderate Left Ventricular Dysfunction"), {})
    assert plans[0]["cardiac_disease_name"] == "Moderate Left Ventricular Dysfunction"


# ===========================================================================================
# Nothing developer-facing reaches a clinician
# ===========================================================================================
# An upstream error body is JSON. Printed on a clinical page it tells a doctor nothing they can
# act on, while implying the report itself is broken. These pin that the technical text stays
# server-side and only a written sentence is sent to the browser.
_UPSTREAM_JSON_BODY = ('{"error":{"message":"Invalid API Key","type":"invalid_request_error",'
                       '"code":"invalid_api_key"}}')


def _looks_developer_facing(text: str) -> bool:
    markers = ("{", "}", '"error"', "Traceback", "Groq", "401", "invalid_api_key", "httpx")
    return any(marker in (text or "") for marker in markers)


def test_upstream_error_body_never_reaches_the_response(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%", **_CLEAR_EXERCISE_SAFETY)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    def _raise(diseases, ctx, **kw):
        raise G.GroqError(f"Groq API returned 401: {_UPSTREAM_JSON_BODY}")

    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans", _raise)
    resp = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    assert resp.status_code == 502
    assert not _looks_developer_facing(resp.json()["detail"]), resp.json()["detail"]
    assert "Something went wrong" in resp.json()["detail"]

    # The two OTHER surfaces that render this text: the polled run state and the stored error
    # shown on the report page. Both must be clean as well.
    progress = client.get(f"/api/reports/{uid}/care-plans/progress",
                          headers=auth["headers"]).json()
    assert not _looks_developer_facing(progress["error"]), progress["error"]
    stored = client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()
    assert not _looks_developer_facing(stored["recommendation_error"])


def test_legacy_combined_plan_error_is_also_sanitised(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    def _raise(ctx):
        raise G.GroqError(f"Groq API returned 500: {_UPSTREAM_JSON_BODY}")

    monkeypatch.setattr("app.routers.reports.generate_care_plan", _raise)
    resp = client.post(f"/api/reports/{uid}/recommend", headers=auth["headers"])

    assert resp.status_code == 502
    assert not _looks_developer_facing(resp.json()["detail"]), resp.json()["detail"]


def test_groq_errors_carry_a_safe_user_message_by_default():
    """A raise that forgets the second argument must still be safe, not leak the first."""
    exc = G.GroqError(f"Groq API returned 401: {_UPSTREAM_JSON_BODY}")
    assert exc.user_message == G.GENERIC_PLAN_ERROR
    assert not _looks_developer_facing(exc.user_message)
    assert _UPSTREAM_JSON_BODY in str(exc)          # still available for the log


@pytest.mark.parametrize("returned,expected", [
    # JSON mode does not always honour "plain string" -- a nested object or a list of steps must
    # not reach the card as a Python repr full of braces and quotes.
    ({"warmup": "5 min walk.", "main": "Cycling 20 min."}, "5 min walk. Cycling 20 min."),
    (["Walk daily.", "Avoid heavy lifting."], "Walk daily. Avoid heavy lifting."),
    ("Plain string.", "Plain string."),
])
def test_structured_model_output_is_flattened_not_dumped(monkeypatch, returned, expected):
    _responder(monkeypatch, [
        _plan_response([{"cardiac_disease_name": "A", "rehabilitation_exercise": returned}]),
    ])
    plan = G.generate_disease_care_plans(_diseases("A"), {})[0]
    assert plan["rehabilitation_exercise"] == expected
    for symbol in ("{", "}", "[", "]", "'"):
        assert symbol not in plan["rehabilitation_exercise"]


def test_an_unusable_plan_object_falls_back_to_the_marker(monkeypatch):
    """No text at all in the response is 'not generated' -- never an empty card, never a repr."""
    _responder(monkeypatch, [
        _plan_response([{"cardiac_disease_name": "A", "rehabilitation_exercise": {}}]),
    ])
    plan = G.generate_disease_care_plans(_diseases("A"), {})[0]
    assert "No plan was generated" in plan["rehabilitation_exercise"]


# ===========================================================================================
# No disease detected is a RESULT, not a failure
# ===========================================================================================
def test_no_disease_returns_200_with_an_explanation(client, auth):
    """A healthy report used to render red, beside a Retry that could never succeed.

    All nine compulsory groups are filled and clear, so Exercise Safety is "No Exercise
    Contraindication Found" rather than "Indeterminate" -- otherwise the Indeterminate hard
    ceiling (§4) would withhold a plan for a different reason than "no disease found", which is
    not what this test is pinning.
    """
    uid = _make_report(auth["email"], ef="62%", ivsd="9 mm", pwd="9 mm", la_diameter="34 mm",
                       mv_finding="Normal", av_finding="Normal", wall_motion="Normal", rwma="No",
                       pasp="25 mmHg", pericardial_effusion="No effusion",
                       clots_thrombus="No thrombus", lvot_peak_gradient="10 mmHg")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    resp = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])
    assert resp.status_code == 200, resp.json()
    out = resp.json()
    assert out["no_disease"] is True
    assert out["care_plans"] == []
    assert "No cardiac disease was detected" in out["notice"]
    assert out["exercise_blocked"] is False


def test_no_disease_settles_the_progress_state(client, auth):
    """The run must land on done, or the polling percentage would spin forever."""
    uid = _make_report(auth["email"], ef="62%", ivsd="9 mm", pwd="9 mm", la_diameter="34 mm",
                       mv_finding="Normal", av_finding="Normal", wall_motion="Normal", rwma="No")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    out = client.get(f"/api/reports/{uid}/care-plans/progress", headers=auth["headers"]).json()
    assert out["status"] == "done"
    assert out["error"] is None


def test_running_the_prediction_first_is_still_required(client, auth):
    """A genuine misuse stays a 4xx -- only 'no disease found' stopped being an error."""
    uid = _make_report(auth["email"], ef="40%")
    resp = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])
    assert resp.status_code == 400


# ===========================================================================================
# Measured care-plan progress
# ===========================================================================================
# The percentage on the report page is polled from a SEPARATE request while the generate request
# is still running. That only works if each step is committed and visible to another session --
# an uncommitted update is a bar that never moves. These tests pin that, and the arithmetic.
def test_progress_is_committed_and_visible_to_a_concurrent_reader(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%", **_CLEAR_EXERCISE_SAFETY)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    seen = []

    def _stub(diseases, ctx, on_progress=None):
        for index, disease in enumerate(diseases):
            name = _name_of(disease)
            on_progress(index, len(diseases), name)
            # Read the row back through a DIFFERENT session, exactly as the polling endpoint
            # does. If the write were not committed, this would see the previous value.
            db = SessionLocal()
            row = db.query(models.CardiacReport).filter_by(report_uid=uid).one()
            seen.append(dict(row.care_plans_run or {}))
            db.close()
        return [{"cardiac_disease_name": _name_of(d), "rehabilitation_exercise": "Walk."}
               for d in diseases]

    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans", _stub)
    client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    assert seen, "the run published no progress at all"
    assert [s["done"] for s in seen] == list(range(len(seen)))
    assert all(s["status"] == "running" for s in seen)
    # Never claims completion while a plan is still outstanding.
    assert all(s["progress"] < 100 for s in seen)


def test_progress_endpoint_reports_measured_counts_and_finishes_at_100(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%", **_CLEAR_EXERCISE_SAFETY)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans",
                        lambda diseases, ctx, **kw: [{"cardiac_disease_name": _name_of(d),
                                                      "rehabilitation_exercise": "Walk."}
                                                     for d in diseases])
    client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    out = client.get(f"/api/reports/{uid}/care-plans/progress", headers=auth["headers"]).json()
    assert out["status"] == "done"
    assert out["progress"] == 100
    assert out["done"] == out["total"] > 0


def test_progress_is_idle_before_any_run(client, auth):
    uid = _make_report(auth["email"], ef="40%")
    out = client.get(f"/api/reports/{uid}/care-plans/progress", headers=auth["headers"]).json()
    assert out["status"] == "idle"
    assert (out["progress"], out["done"], out["total"]) == (0, 0, 0)


def test_a_failed_run_lands_on_error_rather_than_spinning(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%", **_CLEAR_EXERCISE_SAFETY)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    def _boom_groq(names, ctx, **kw):
        raise G.GroqError("Groq API returned 401: bad key")

    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans", _boom_groq)
    assert client.post(f"/api/reports/{uid}/care-plans",
                       headers=auth["headers"]).status_code == 502

    out = client.get(f"/api/reports/{uid}/care-plans/progress", headers=auth["headers"]).json()
    assert out["status"] == "error"
    # A sentence, not a status code: this field is rendered straight onto the page.
    assert out["error"] and not _looks_developer_facing(out["error"])


def test_contraindicated_run_still_settles_so_a_poller_stops(client, auth, monkeypatch):
    """No LLM call is made, but a page polling from a previous run must not spin forever."""
    uid = _make_report(auth["email"], **_CONTRAINDICATED)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans", _boom)
    client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    out = client.get(f"/api/reports/{uid}/care-plans/progress", headers=auth["headers"]).json()
    assert out["status"] == "done"


def test_re_predicting_clears_the_old_runs_progress(client, auth, monkeypatch):
    """A completed '7 of 7' left beside a prediction whose plans were just discarded would be a
    lie about work that no longer exists."""
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans",
                        lambda diseases, ctx, **kw: [{"cardiac_disease_name": _name_of(d),
                                                      "rehabilitation_exercise": "Walk."}
                                                     for d in diseases])
    client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    out = client.get(f"/api/reports/{uid}/care-plans/progress", headers=auth["headers"]).json()
    assert out["status"] == "idle"


def test_care_plans_route_requires_a_prediction_first(client, auth):
    uid = _make_report(auth["email"], ef="40%")
    resp = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])
    assert resp.status_code == 400
    assert "prediction engine" in resp.json()["detail"].lower()


def test_care_plans_route_persists_and_survives_reload(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%", **_CLEAR_EXERCISE_SAFETY)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans",
                        lambda diseases, ctx, **kw: [{"cardiac_disease_name": _name_of(d),
                                             "rehabilitation_exercise": "Walk daily."}
                                            for d in diseases])

    out = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"]).json()
    assert out["care_plans"] and out["care_plans"][0]["rehabilitation_exercise"] == "Walk daily."

    reloaded = client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()
    assert reloaded["care_plans"][0]["rehabilitation_exercise"] == "Walk daily."


# ===========================================================================================
# Exercise contraindicated -- no plan is generated at all
# ===========================================================================================
# EF 24% with an apical thrombus: two independent hard stops.
_CONTRAINDICATED = {"ef": "24%", "clots_thrombus": "LV apical thrombus seen",
                    "pericardial_effusion": "No effusion", "av_peak_gradient": "8 mmHg",
                    "mv_peak_gradient": "4 mmHg", "pasp": "25 mmHg",
                    "lvot_peak_gradient": "10 mmHg", "wall_motion": "Normal", "rwma": "No",
                    "ivsd": "10 mm"}


def _boom(*a, **k):                       # any LLM call here is a test failure
    raise AssertionError("the LLM must not be called when exercise is contraindicated")


def test_per_disease_plans_are_withheld_when_exercise_is_contraindicated(client, auth, monkeypatch):
    uid = _make_report(auth["email"], **_CONTRAINDICATED)
    pred = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()
    assert pred["exercise_safety"]["cardiac_disease_name"] == "Exercise Contraindicated"

    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans", _boom)
    out = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"]).json()

    assert out["exercise_blocked"] is True
    assert out["care_plans"] == []
    assert "Cardiology clearance required" in out["blocked_reason"]
    # The exercise section still has something to show: the verdict and the sports screening.
    assert out["exercise_safety"]["cardiac_disease_name"] == "Exercise Contraindicated"
    assert out["athlete_screening"] is not None
    # Nothing was persisted that could be read as a plan.
    assert client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()["care_plans"] is None


def test_combined_plan_is_withheld_when_exercise_is_contraindicated(client, auth, monkeypatch):
    uid = _make_report(auth["email"], **_CONTRAINDICATED)
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])

    monkeypatch.setattr("app.routers.reports.generate_care_plan", _boom)
    out = client.post(f"/api/reports/{uid}/recommend", headers=auth["headers"]).json()

    assert out["exercise_blocked"] is True
    assert out["rehab_plan"] is None
    assert "Cardiology clearance required" in out["blocked_reason"]
    assert client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()["rehab_plan"] is None


def test_a_non_contraindicated_report_still_generates_normally(client, auth, monkeypatch):
    """The block must be the verdict talking, not a blanket refusal.

    All nine compulsory groups are filled and clear, so the Exercise Safety verdict is "No
    Exercise Contraindication Found" rather than "Indeterminate" -- a report this incomplete
    would otherwise correctly withhold a plan under the §4 hard-ceiling rule tested elsewhere.
    """
    uid = _make_report(auth["email"], ef="60%", ivsd="13 mm", la_diameter="44 mm",
                       av_peak_gradient="8 mmHg", mv_peak_gradient="4 mmHg", pasp="25 mmHg",
                       pericardial_effusion="No effusion", clots_thrombus="No thrombus",
                       lvot_peak_gradient="10 mmHg", wall_motion="Normal", rwma="No")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans",
                        lambda diseases, ctx, **kw: [{"cardiac_disease_name": _name_of(d),
                                             "rehabilitation_exercise": "Walk daily."}
                                            for d in diseases])
    out = client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"]).json()
    assert out["exercise_blocked"] is False
    assert out["care_plans"]


def test_the_combined_plan_carries_no_diet_field_at_all(client, auth, monkeypatch):
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    monkeypatch.setattr("app.routers.reports.generate_care_plan",
                        lambda ctx: {"rehabilitation_exercise_plan": "Walk daily.",
                                     "key_precautions": "Stop if dizzy."})
    out = client.post(f"/api/reports/{uid}/recommend", headers=auth["headers"]).json()
    assert out["rehab_plan"] == "Walk daily."
    assert "diet_plan" not in out
    assert "diet_plan" not in client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()
    assert "diet_plan" not in G._SYSTEM_PROMPT


def test_athlete_screening_is_not_returned_among_the_predicted_diseases(client, auth):
    """It is exercise guidance, so it travels in its own key and renders after the exercise run."""
    uid = _make_report(auth["email"], ef="40%", ivsd="13 mm", pasp="50 mmHg")
    out = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()
    assert out["athlete_screening"] is not None
    for disease in out["diseases"]:
        assert disease["category"] not in ("Athlete Screening", "Exercise Safety")


def test_the_age_band_reaches_the_api_response(client, auth):
    uid = _make_report(auth["email"], ef="40%", patient_age="73 / Female")
    out = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()
    assert out["age_band"] == "older"
    assert out["age_band_label"] == "Older (> 65 years)"
    assert out["compulsory_coverage"]["groups_total"] == 9


def test_re_predicting_invalidates_stale_care_plans(client, auth, monkeypatch):
    """A plan generated for the previous prediction must not linger beside a new one."""
    uid = _make_report(auth["email"], ef="40%")
    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    monkeypatch.setattr("app.routers.reports.generate_disease_care_plans",
                        lambda diseases, ctx, **kw: [{"cardiac_disease_name": _name_of(d),
                                             "rehabilitation_exercise": "x"} for d in diseases])
    client.post(f"/api/reports/{uid}/care-plans", headers=auth["headers"])

    client.post(f"/api/reports/{uid}/predict", headers=auth["headers"])
    assert client.get(f"/api/reports/{uid}", headers=auth["headers"]).json()["care_plans"] is None
