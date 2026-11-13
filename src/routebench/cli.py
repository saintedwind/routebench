"""Train and evaluate the router without hosted services or API credentials."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .data import fetch
from .evaluation import prepare, run
from .routing import LocalModel, cascade, gate, predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch", help="download and verify pinned CLINC150 files")
    evaluate = commands.add_parser(
        "evaluate", help="run full test and optional paired model comparison"
    )
    classify = commands.add_parser("classify", help="train and classify one short English request")
    for command in (get, evaluate, classify):
        command.add_argument("--data", type=Path, default=Path("data/clinc150"))
    for command in (evaluate, classify):
        command.add_argument("--local-url", help="optional loopback llama.cpp chat endpoint")
        command.add_argument("--model", default="routebench-qwen", help="local model alias")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--per-domain", type=int, default=20)
    evaluate.add_argument("--outside", type=int, default=40)
    classify.add_argument("text")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.data)
        print(f"Verified CLINC150 files in {args.data}")
        return
    local = LocalModel(args.local_url, args.model) if args.local_url else None
    if args.command == "classify":
        text = args.text.strip()
        if not 1 <= len(text) <= 2000:
            parser.error("Text must contain 1–2,000 characters")
        _, model, threshold, _ = prepare(args.data)
        decision = gate(predict(model, text), threshold)
        if decision.label is None and local is not None:
            decision = cascade(decision, local.predict(text))
        print(json.dumps(asdict(decision) | {"threshold": threshold}, indent=2))
        return
    if not 1 <= args.per_domain <= 450 or not 1 <= args.outside <= 1000:
        parser.error("Sample sizes must fit the published test groups")
    result = run(
        args.data,
        local,
        per_domain=args.per_domain,
        outside=args.outside,
        progress=lambda n, total: print(
            f"Local inference {n}/{total}", file=sys.stderr, flush=True
        ),
    )
    result["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("scikit-learn", "numpy", "scipy", "threadpoolctl")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["full_test"], indent=2))
    if local is not None:
        comparisons = result["paired_sample"]["metrics"]
        print(json.dumps(comparisons, indent=2))
        always = comparisons["local_only"]
        if always["llm_calls"] == always["llm_failures"]:
            parser.exit(2, "Local model produced zero valid routes. Inspect the saved report.\n")


if __name__ == "__main__":
    main()
