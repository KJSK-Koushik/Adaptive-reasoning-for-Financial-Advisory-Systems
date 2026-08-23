"""Phase 8 service.

The heavy tests need the Phase 3-5 artefacts, which are regenerable and therefore not
committed, so they skip rather than fail where those are absent - CI included.
"""

from __future__ import annotations

import pytest

from adaptive_reasoning import paths

REQUIRED = (paths.UNIFIED_DATASET, paths.TRACE_DATASET, paths.RL_TRANSITIONS,
            paths.DQN_POLICY)

needs_artifacts = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED),
    reason="requires the Phase 3-5 artefacts (run the earlier phases)",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from adaptive_reasoning.app.api import create_app

    return TestClient(create_app(experiment="reported"))


# --------------------------------------------------------------------------- #
# response models - no artefacts needed
# --------------------------------------------------------------------------- #
def test_answer_response_requires_a_disclaimer():
    """Every answer must carry the disclaimer; it is not an optional field."""
    from pydantic import ValidationError

    from adaptive_reasoning.app.api import AnswerResponse

    with pytest.raises(ValidationError):
        AnswerResponse(
            question_id="q", question="?", answer="a", reference_answer="a",
            policy="dqn", stopped_at_step=0, stop_reason="policy",
            tokens_used=1, tokens_if_unrestricted=2, tokens_saved=1,
            token_reduction_pct=50.0, latency_seconds=0.1, latency_saved_seconds=0.1,
            full_reasoning_answer="a", answer_changed=False,
        )


def test_answer_response_defaults_to_replay_mode():
    """The service must never imply live generation when it is replaying."""
    from adaptive_reasoning.app.api import AnswerResponse

    response = AnswerResponse(
        question_id="q", question="?", answer="a", reference_answer="a",
        policy="dqn", stopped_at_step=0, stop_reason="policy",
        tokens_used=1, tokens_if_unrestricted=2, tokens_saved=1,
        token_reduction_pct=50.0, latency_seconds=0.1, latency_saved_seconds=0.1,
        full_reasoning_answer="a", answer_changed=False, disclaimer="x",
    )
    assert response.mode == "replay"


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@needs_artifacts
def test_health_reports_loaded_policies(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "dqn" in body["policies"]
    assert body["traced_questions"] > 0


@needs_artifacts
def test_health_reports_the_training_budget_not_the_config(client):
    """768 is what the traces were generated under; llm.max_new_tokens is 1024."""
    assert client.get("/health").json()["budget_tokens"] == 768


@needs_artifacts
def test_questions_are_limited_and_traced(client):
    body = client.get("/questions", params={"limit": 5}).json()
    assert len(body) == 5
    assert all(q["has_trace"] for q in body)


@needs_artifacts
def test_questions_can_be_filtered_by_domain(client):
    body = client.get("/questions", params={"limit": 10, "domain": "report_qa"}).json()
    assert body and all(q["domain"] == "report_qa" for q in body)


@needs_artifacts
def test_ask_returns_a_complete_answer(client):
    question_id = client.get("/questions", params={"limit": 1}).json()[0]["id"]
    body = client.post("/ask", params={"question_id": question_id}).json()

    assert body["question_id"] == question_id
    assert body["mode"] == "replay"
    assert body["disclaimer"]
    assert body["tokens_used"] <= body["tokens_if_unrestricted"]
    assert body["tokens_saved"] == body["tokens_if_unrestricted"] - body["tokens_used"]
    assert body["steps"], "the step-by-step trace is the point of the demo"
    assert body["steps"][-1]["action"] == "STOP" or \
           body["stop_reason"] == "stream_end"


@needs_artifacts
def test_ask_reports_savings_consistently(client):
    question_id = client.get("/questions", params={"limit": 1}).json()[0]["id"]
    body = client.post("/ask", params={"question_id": question_id}).json()
    expected = 100.0 * body["tokens_saved"] / body["tokens_if_unrestricted"]
    assert body["token_reduction_pct"] == pytest.approx(expected, abs=0.05)


@needs_artifacts
def test_steps_can_be_suppressed(client):
    question_id = client.get("/questions", params={"limit": 1}).json()[0]["id"]
    body = client.post("/ask", params={"question_id": question_id,
                                       "include_steps": False}).json()
    assert body["steps"] == []


@needs_artifacts
def test_both_policies_are_servable(client):
    question_id = client.get("/questions", params={"limit": 1}).json()[0]["id"]
    for policy in ("dqn", "bc"):
        body = client.post("/ask", params={"question_id": question_id,
                                           "policy": policy}).json()
        assert body["policy"] == policy


@needs_artifacts
def test_unknown_question_is_a_404(client):
    assert client.post("/ask", params={"question_id": "nope"}).status_code == 404


@needs_artifacts
def test_unknown_policy_is_a_400(client):
    question_id = client.get("/questions", params={"limit": 1}).json()[0]["id"]
    response = client.post("/ask", params={"question_id": question_id,
                                           "policy": "magic"})
    assert response.status_code == 400


@needs_artifacts
def test_stats_exposes_the_evaluation_table(client):
    body = client.get("/stats").json()
    assert body["dataset_questions"] == 30660
    assert body["test_questions"] == 599
    assert "dqn" in body["evaluation"]
    assert "oracle" in body["evaluation"]
    assert body["disclaimer"]
