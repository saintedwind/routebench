import json
import os
from pathlib import Path
from urllib.error import URLError

import pytest

from routebench.data import OUTSIDE, load, sample_test, verify
from routebench.evaluation import metrics, paired_interval, prepare
from routebench.routing import Decision, LocalModel, cascade, choose_threshold, gate, loss, predict


def test_dataset_integrity_rejects_modified_input():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(b"modified", "data/data_full.json")


@pytest.fixture(scope="module")
def corpus_path():
    path = Path(os.getenv("ROUTEBENCH_TEST_DATA", "data/clinc150"))
    if not (path / "data_full.json").is_file():
        pytest.skip("Run routebench fetch or set ROUTEBENCH_TEST_DATA")
    return path


def test_published_split_and_sample(corpus_path):
    data = load(corpus_path)
    assert len(data["train"]) == 15000
    assert len(data["test"]) == 4500
    pool = data["test"] + data["oos_test"]
    sample = sample_test(pool)
    assert len(sample) == len({row.id for row in sample}) == 240
    assert sample == sample_test(pool[::-1])
    assert sum(row.label == OUTSIDE for row in sample) == 40
    assert set(row.id for row in sample).isdisjoint(row.id for row in data["val"])


def test_real_data_training_and_held_out_decisions(corpus_path):
    data, model, threshold, _ = prepare(corpus_path)
    sample = sample_test(data["test"] + data["oos_test"], per_domain=5, outside=10)
    raw = [predict(model, row.text) for row in sample]
    results = metrics([row.label for row in sample], [gate(p, threshold) for p in raw])
    assert results["in_scope_correct_rate"] >= 0.6
    assert results["out_of_scope_false_accept_rate"] < 0.6


def test_threshold_distinguishes_risk_and_review():
    gold = ["banking", "work", OUTSIDE]
    raw = [
        Decision("banking", "linear", 0.93),
        Decision("utility", "linear", 0.4),
        Decision("home", "linear", 0.85),
    ]
    threshold = choose_threshold(gold, raw)
    assert gate(raw[0], threshold).label == "banking"
    assert gate(raw[1], threshold).label is None
    assert gate(raw[2], threshold).label is None
    assert loss(OUTSIDE, None) == 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1/v1/chat/completions",
        "http://localhost/v1/chat/completions",
        "http://127.0.0.1@outside.example/v1/chat/completions",
    ],
)
def test_model_endpoint_requires_loopback(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        LocalModel(endpoint, "qwen")


@pytest.mark.parametrize(
    ("content", "reason", "label", "via"),
    [
        ('{"route":"B"}', "stop", "banking", "local"),
        ('{"route":"K"}', "stop", OUTSIDE, "local"),
        ('{"route":"B"}', "length", None, "local_invalid"),
        ('{"route":"Z"}', "stop", None, "local_invalid"),
        ('{"route":"B","other":true}', "stop", None, "local_invalid"),
        ("null", "stop", None, "local_invalid"),
    ],
)
def test_local_output_contract(monkeypatch, content, reason, label, via):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _limit):
            return json.dumps(
                {
                    "choices": [{"finish_reason": reason, "message": {"content": content}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                }
            ).encode()

    model = LocalModel("http://127.0.0.1:8080/v1/chat/completions", "qwen")

    def response(req, **_kwargs):
        payload = json.loads(req.data)
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["route"]["enum"][-1] == "K"
        assert payload["messages"][-1] == {"role": "user", "content": "Hello"}
        return Response()

    monkeypatch.setattr(model.opener, "open", response)
    result = model.predict("Hello")
    assert (result.label, result.via, result.prompt_tokens, result.output_tokens) == (
        label,
        via,
        8,
        3,
    )


def test_backend_failure_stays_in_review(monkeypatch):
    model = LocalModel("http://127.0.0.1:8080/v1/chat/completions", "qwen")

    def offline(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr(model.opener, "open", offline)
    expensive = model.predict("Hello")
    combined = cascade(Decision(None, "review", 0.25), expensive)
    assert combined.label is None and combined.via == "local_unavailable"
    cheap = Decision("work", "linear", 0.99)
    assert cascade(cheap, expensive) is cheap


def test_metrics_preserve_wrong_routes_and_reported_usage():
    labels = ["work", "work", OUTSIDE, OUTSIDE]
    baseline = [
        Decision("work", "linear"),
        Decision(None, "review"),
        Decision(None, "review"),
        Decision(None, "review"),
    ]
    candidate = baseline[:2] + [
        Decision("work", "local", prompt_tokens=7, output_tokens=2),
        Decision(OUTSIDE, "local"),
    ]
    result = metrics(labels, candidate)
    assert result["in_scope_correct_rate"] == 0.5
    assert result["out_of_scope_false_accept_rate"] == 0.5
    assert result["llm_calls"] == 2 and result["usage_reported_calls"] == 1
    assert result["mean_cost"] == 2.25
    interval = paired_interval(labels, baseline, candidate)
    assert interval["delta_candidate_minus_baseline"] == 2
    assert interval["low95"] <= 2 <= interval["high95"]


@pytest.mark.parametrize("nested_content", [False, True])
def test_deeply_nested_response_stays_in_review(monkeypatch, nested_content):
    from io import BytesIO

    nested = "[" * 10000 + "0"
    nested += "]" * 10000
    raw = nested.encode()
    if nested_content:
        raw = json.dumps(
            {"choices": [{"finish_reason": "stop", "message": {"content": nested}}]}
        ).encode()
    assert len(raw) < 65536
    model = LocalModel("http://127.0.0.1:8080/v1/chat/completions", "qwen")
    monkeypatch.setattr(model.opener, "open", lambda *_args, **_kwargs: BytesIO(raw))
    result = model.predict("Hello")
    assert result.label is None
    assert result.via == "local_invalid"
    combined = cascade(Decision(None, "review", 0.25), result)
    assert combined.label is None


def test_empty_validation_cannot_select_a_threshold():
    with pytest.raises(ValueError, match="validation"):
        choose_threshold([], [])


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        metrics([], [])
    with pytest.raises(ValueError, match="non-empty"):
        paired_interval([], [], [])


@pytest.mark.parametrize("resamples", [0, -1, True, 1.5])
def test_invalid_bootstrap_count_rejected(resamples):
    decisions = [Decision("work", "linear")]
    with pytest.raises(ValueError, match="positive integer"):
        paired_interval(["work"], decisions, decisions, resamples=resamples)


def test_single_sample_evaluation_is_finite():
    baseline = [Decision("work", "linear")]
    candidate = [Decision(None, "review")]
    summary = metrics(["work"], baseline)
    interval = paired_interval(["work"], baseline, candidate, resamples=1)
    assert summary["mean_cost"] == 0
    assert interval["delta_candidate_minus_baseline"] == 1
    assert interval["low95"] == interval["high95"] == 1
    json.dumps({"metrics": summary, "interval": interval}, allow_nan=False)
