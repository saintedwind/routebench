"""Paired routing metrics, confidence intervals and local inference evidence."""

from collections import Counter
from dataclasses import asdict
from time import perf_counter

import numpy as np

from .data import FILES, OUTSIDE, REVISION, SAMPLE_SALT, load, sample_test
from .routing import Decision, cascade, choose_threshold, fit, gate, loss, predict


def mean(values):
    return float(np.mean(values)) if len(values) else None


def metrics(labels: list[str], predictions: list[Decision]) -> dict:
    pairs = list(zip(labels, predictions, strict=True))
    inside = [(label, p) for label, p in pairs if label != OUTSIDE]
    outside = [p for label, p in pairs if label == OUTSIDE]
    accepted = [p.label == label for label, p in inside if p.label not in (None, OUTSIDE)]
    calls = [p for p in predictions if p.via.startswith("local")]
    counts = Counter(labels)
    ms = [p.elapsed_ms for p in predictions]
    return {
        "samples": len(labels),
        "in_scope": len(inside),
        "out_of_scope": len(outside),
        "in_scope_correct_rate": mean([p.label == label for label, p in inside]),
        "in_scope_routed_rate": mean([p.label not in (None, OUTSIDE) for _, p in inside]),
        "routed_accuracy": mean(accepted),
        "out_of_scope_false_accept_rate": mean([p.label not in (None, OUTSIDE) for p in outside]),
        "abstained_count": sum(p.label is None for p in predictions),
        "outside_label_count": sum(p.label == OUTSIDE for p in predictions),
        "mean_cost": mean([loss(label, p.label) for label, p in pairs]),
        "macro_cost": sum(loss(label, p.label) / counts[label] for label, p in pairs) / len(counts),
        "llm_calls": len(calls),
        "llm_failures": sum(p.label is None for p in calls),
        "usage_reported_calls": sum(
            p.prompt_tokens is not None and p.output_tokens is not None for p in calls
        ),
        "reported_prompt_tokens": sum(p.prompt_tokens or 0 for p in calls),
        "reported_output_tokens": sum(p.output_tokens or 0 for p in calls),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
    }


def paired_interval(labels, baseline, candidate, *, seed=427, resamples=2000):
    """Stratified paired bootstrap for the delta in sample-average decision cost."""
    labels = np.asarray(labels)
    delta = np.asarray(
        [
            loss(label, b.label) - loss(label, a.label)
            for label, a, b in zip(labels, baseline, candidate, strict=True)
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    sums = np.zeros(resamples)
    for label in np.unique(labels):
        rows = delta[labels == label]
        sums += rng.choice(rows, size=(resamples, len(rows)), replace=True).sum(axis=1)
    return {
        "delta_candidate_minus_baseline": float(delta.mean()),
        "low95": float(np.percentile(sums / len(delta), 2.5)),
        "high95": float(np.percentile(sums / len(delta), 97.5)),
        "resamples": resamples,
        "seed": seed,
    }


def prepare(directory):
    data = load(directory)
    start = perf_counter()
    model = fit(data["train"])
    fit_seconds = perf_counter() - start
    validation = data["val"] + data["oos_val"]
    raw = [predict(model, row.text) for row in validation]
    threshold = choose_threshold([row.label for row in validation], raw)
    return data, model, threshold, fit_seconds


def run(directory, local=None, *, per_domain=20, outside=40, progress=None):
    data, model, threshold, fit_seconds = prepare(directory)
    test = data["test"] + data["oos_test"]
    gold = [row.label for row in test]
    raw = [predict(model, row.text) for row in test]
    gated = [gate(p, threshold) for p in raw]
    result = {
        "dataset_revision": REVISION,
        "dataset_sha256": FILES,
        "threshold": threshold,
        "fit_seconds": fit_seconds,
        "split_sizes": {split: len(rows) for split, rows in data.items()},
        "full_test": {"linear": metrics(gold, raw), "linear_with_review": metrics(gold, gated)},
    }
    if local is None:
        return result
    positions = {row.id: i for i, row in enumerate(test)}
    sample = sample_test(test, per_domain, outside)
    local_only, sample_linear, sample_gate, cascaded, rows = [], [], [], [], []
    for index, row in enumerate(sample, 1):
        i = positions[row.id]
        prediction = local.predict(row.text)
        combined = cascade(gated[i], prediction)
        local_only.append(prediction)
        sample_linear.append(raw[i])
        sample_gate.append(gated[i])
        cascaded.append(combined)
        rows.append(
            {
                "id": row.id,
                "gold": row.label,
                "linear": asdict(raw[i]),
                "gated": asdict(gated[i]),
                "local": asdict(prediction),
                "cascade": asdict(combined),
            }
        )
        if progress is not None and (index % 20 == 0 or index == len(sample)):
            progress(index, len(sample))
    labels = [row.label for row in sample]
    result["paired_sample"] = {
        "per_domain": per_domain,
        "outside": outside,
        "sample_salt": SAMPLE_SALT,
        "model_alias": local.model,
        "latency_note": "Cascade sums measured component latency. LLM outputs and usage are replayed from local-only requests on the same IDs.",
        "metrics": {
            name: metrics(labels, decisions)
            for name, decisions in (
                ("linear", sample_linear),
                ("linear_with_review", sample_gate),
                ("local_only", local_only),
                ("cascade", cascaded),
            )
        },
        "cost_delta_vs_local": paired_interval(labels, local_only, cascaded),
        "cost_delta_vs_review": paired_interval(labels, sample_gate, cascaded),
        "rows": rows,
    }
    return result
