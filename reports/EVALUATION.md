# CLINC150 evaluation · 13 November 2026

RouteBench asks how much correct routing a small CPU classifier can deliver and whether a local model improves its deferred decisions. The [raw report](clinc150-cpu.json) contains the original row IDs, ground-truth domains, model decisions, per-request times and server token counts. The dataset and checkpoint identities are pinned. The local run used Qwen3-0.6B Q8_0 with llama.cpp b11476 (988190680), a Ryzen 5 4600H, four inference threads, one slot, and prompt caching disabled. See the [run commands](../README.md).

## Full official test · 5,500 requests

| Policy | Correct of 4,500 in scope | In scope routed | Accuracy among routed in-scope cases | False accepts of 1,000 out of scope |
|---|---:|---:|---:|---:|
| Linear, always route | 4,377 | 4,500 | 97.27% | 1,000 |
| Linear + review (threshold 0.79) | 3,917 | 3,932 | 99.62% | 123 |

The threshold came from the validation split, under the declared cost schedule. The gated classifier sent 568 in-scope and 877 out-of-scope requests to review. Median decision time was 1.58 ms, with p95 of 2.72 ms on this machine after training. Model fit took about 4.60 seconds. The timings exclude startup and dataset loading.

## Paired sample · 240 requests

| Policy | Correct of 200 in scope | False accepts of 40 out of scope | Local calls | Mean cost per request | Median / p95 |
|---|---:|---:|---:|---:|---:|
| Linear, always route | 199 | 40 | 0 | 1.354 | 1.57 / 2.82 ms |
| Linear + review | 175 | 4 | 0 | 0.254 | 1.57 / 2.82 ms |
| Qwen only | 53 | 10 | 240 | 3.396 | 1,559 / 3,018 ms |
| Linear then Qwen on defer | 182 | 12 | 60 | 0.775 | 1.78 / 3,011 ms |

All 240 local responses passed the output contract after the adapter was corrected. The first attempt returned zero usable routes: 137 calls supplied usage with invalid responses, and 103 supplied no usage. A direct probe confirmed that the flat response schema did not constrain this server build, so the adapter switched to its nested `json_schema` contract. The comparison uses the corrected run on the same sample IDs.

The cascade uses 75% fewer model calls than Qwen only, with 13,565 reported input tokens versus 54,284 and 884 output tokens versus 3,482. Against linear + review, Qwen recovered 7 of the 24 deferred in-scope cases. Of the 36 deferred out-of-scope cases, it assigned 8 to supported domains. The fixed cost schedule favored review, with a paired cost increase of **0.521 units per request** for the cascade (95% stratified bootstrap interval **0.321 to 0.738**, 2,000 resamples). The interval is conditional on this fixed sample mix. Using a different cost schedule could change the preferred policy.

`Qwen only` is measured over actual local HTTP calls. The cascade reuses the same per-request model decisions and sums the measured classifier and model times for deferred cases. Latency therefore describes sequential measurements and replay on one CPU, and includes contention from local development checks. It does not establish production throughput or tail latency under concurrent load. The project also exercised the CLI against a running model and verified a local out-of-scope decision for an astronomy question.

## Check or regenerate the figure

```sh
python -m pip install -e '.[report]'
python scripts/plot_results.py reports/clinc150-cpu.json reports/clinc150-routing.png
```

The official English corpus is crowdsourced, and the task uses its ten domain labels. The recorded comparisons establish performance on the specified benchmark and CPU. Production traffic and other languages require separate validation.
