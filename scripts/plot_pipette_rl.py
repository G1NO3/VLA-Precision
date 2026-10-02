#!/usr/bin/env python3
"""One-shot, CPU-only plots of pipette ACoB metrics (requires numpy/matplotlib)."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = (Path(__file__).resolve().parents[2]
               / "outputs/pipette-hil293-10epochs-acob-gamma0999-reward10/learner")


def rolling_mean(values, window):
    valid = np.isfinite(values)
    sums = np.r_[0, np.cumsum(np.where(valid, values, 0))]
    counts = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    count = counts[end] - counts[start]
    return np.divide(sums[end] - sums[start], count,
                     out=np.full(len(values), np.nan), where=count > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-dir", type=Path, default=DEFAULT_DIR,
                        help="Default: HIL293 10-epoch BC + ACoB, gamma=0.999, reward10 experiment")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Trailing window in logged updates (default: 50)")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: LEARNER/plots-step-NNNNNNNN")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")
    directory = args.learner_dir.expanduser().resolve()
    try:
        raw = (directory / "metrics.jsonl").read_text()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not rows:
            raise ValueError("metrics.jsonl is empty")
        x = np.array([row["step"] for row in rows], dtype=float)
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError("steps must be finite and strictly increasing; separate overlapping runs first")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    def values(key):
        return np.array([row.get(key, np.nan) for row in rows], dtype=float)

    # Checkpoint metadata records actual resumes; do not assume a fixed run length.
    boundaries = set()
    for path in directory.glob("step-*/metadata.json"):
        try:
            parent = json.loads(path.read_text()).get("parent_checkpoint")
            if parent and Path(parent).name.startswith("step-"):
                step = int(Path(parent).name.removeprefix("step-"))
                if x[0] <= step < x[-1]:
                    boundaries.add(step)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Warning: cannot read resume metadata {path}: {exc}", file=sys.stderr)
    cuts = [0] + sorted({int(np.searchsorted(x, b, side="right")) for b in boundaries}) + [len(x)]

    def smooth(y):
        # Reset the average after each resume rather than mix training rounds.
        return np.concatenate([rolling_mean(y[a:b], args.smooth)
                               for a, b in zip(cuts[:-1], cuts[1:])])

    def line(ax, y, label, raw_line=True):
        averaged = smooth(y)
        for a, b in zip(cuts[:-1], cuts[1:]):
            curve, = ax.plot(x[a:b], averaged[a:b], lw=1.6,
                             label=label if a == 0 else None,
                             color=color if a else None)
            color = curve.get_color()
        if raw_line:
            ax.plot(x, y, lw=0.5, alpha=0.12, color=color)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
    panels = [
        (axes[0, 0], "Critic losses", [("critic_loss", "Total critic"), ("td_loss", "TD")]),
        (axes[0, 1], "Actor total loss", [("actor_loss", "Actor")]),
        (axes[1, 0], "Actor components (unweighted)",
         [("bc_loss", "BC / flow"), ("imp_loss", "Improvement"), ("ref_loss", "Reference")]),
        (axes[1, 1], "Q estimate and TD target",
         [("predicted_qs", "Predicted Q"), ("target_qs", "TD target")]),
    ]
    for ax, title, metrics in panels:
        for key, label in metrics:
            line(ax, values(key), label)
        ax.set_title(title, loc="left", weight="bold")
    losses = np.r_[values("critic_loss"), values("td_loss")]
    finite_losses = losses[np.isfinite(losses)]
    if finite_losses.size and np.all(finite_losses > 0):
        axes[0, 0].set_yscale("log")

    pairs = values("intervention_pref_pair_count")
    for key, label in [("intervention_pref_accuracy", "Human ranked above proposal"),
                       ("intervention_pref_margin_satisfied_rate", "Required margin met")]:
        rate = values(key)
        valid = np.isfinite(rate) & np.isfinite(pairs) & (pairs > 0)
        weights = np.where(valid, pairs, 0)
        denominator = smooth(weights)
        numerator = smooth(np.where(valid, rate, 0) * weights)
        weighted = np.divide(numerator, denominator, out=np.full(len(x), np.nan),
                             where=denominator > 0)
        # Already smoothed and pair-weighted: do not apply a second average.
        for a, b in zip(cuts[:-1], cuts[1:]):
            curve, = axes[2, 0].plot(x[a:b], weighted[a:b] * 100,
                                     label=label if a == 0 else None,
                                     color=color if a else None)
            color = curve.get_color()
    axes[2, 0].set(title="Preference quality (weighted by valid pair count)",
                   ylabel="% of sampled pairs", ylim=(-2, 102))
    for key, label in [("sampled_success_terminal_count", "Successful terminal frames"),
                       ("sampled_success_tail_count", "Successful tail frames (includes terminal)")]:
        line(axes[2, 1], values(key), label, raw_line=False)
    for key, label in [("sampled_failure_terminal_count", "Failed terminal frames"),
                       ("sampled_terminal_tail_count", "All terminal tails (success + failure)")]:
        if np.any(np.isfinite(values(key))):
            line(axes[2, 1], values(key), label, raw_line=False)
    if np.any(np.isfinite(values("sampled_recent_episode_count"))):
        line(axes[2, 1], values("sampled_recent_episode_count"),
             "Recent episodes (overlaps success samples)", raw_line=False)
    if np.any(np.isfinite(values("sampled_priority_episode_count"))):
        line(axes[2, 1], values("sampled_priority_episode_count"),
             "Priority collection (overlaps success samples)", raw_line=False)
    axes[2, 1].set(title="Replay sampling", ylabel="Mean sampled count per batch")

    for ax in axes.flat:
        for boundary in sorted(boundaries):
            ax.axvline(boundary, color="gray", linestyle="--", lw=1)
        if x[-1] > x[0]:
            ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Learner step")
        ax.legend(fontsize=8)
    discounts = values("discount")
    gamma = ", ".join(f"{v:g}" for v in np.unique(discounts[np.isfinite(discounts)])) or "unknown"
    resumes = ", ".join(str(b) for b in sorted(boundaries)) or "none recorded"
    fig.suptitle(f"{directory.parent.name} | gamma = {gamma}\n"
                 f"Steps {int(x[0]):,}–{int(x[-1]):,} | {args.smooth}-update trailing means; faint = raw\n"
                 f"Dashed: resume at {resumes}. Training metrics are not robot success rates.",
                 fontsize=12, weight="bold")
    output = (args.output_dir or directory / f"plots-step-{int(x[-1]):08d}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output / f"rl_curves.{extension}"
        fig.savefig(path, dpi=170)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
