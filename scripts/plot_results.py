"""Draw the checked-in evaluation, with every plotted value read from its JSON."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ("#296d6c", "#a55832", "#584e88")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.results.read_text(encoding="utf-8"))["paired_sample"]["metrics"]
    names = ("linear_with_review", "local_only", "cascade")
    labels = ("Linear + review", "Qwen only", "Linear → Qwen")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), facecolor="#f7f5f0")
    fig.suptitle("RouteBench / 240 paired CLINC150 requests", x=0.05, ha="left", fontsize=17)
    panels = (
        ("Correct in-scope", "in_scope_correct_rate", True),
        ("Out-of-scope routed", "out_of_scope_false_accept_rate", True),
        ("Local model calls", "llm_calls", False),
    )
    for ax, (title, key, percent) in zip(axes, panels, strict=True):
        ax.set_facecolor(fig.get_facecolor())
        values = [metrics[name][key] * (100 if percent else 1) for name in names]
        bars = ax.barh(labels, values, color=PALETTE, height=0.55)
        ax.invert_yaxis()
        ax.set_xlim(0, 115 if percent else 285)
        ax.set_title(title, loc="left", fontsize=11, pad=13)
        ax.bar_label(
            bars, labels=[f"{x:.1f}%" if percent else str(int(x)) for x in values], padding=4
        )
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.05, 0.07, "200 in-scope / 40 out-of-scope · Qwen3-0.6B Q8_0 · Windows CPU", fontsize=9
    )
    fig.subplots_adjust(left=0.135, right=0.98, bottom=0.18, top=0.73, wspace=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
