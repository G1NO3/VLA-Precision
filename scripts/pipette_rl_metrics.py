#!/usr/bin/env python3
"""Read recent ACoB metrics using only the Python standard library."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time


DEFAULT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs/pipette-hil293-10epochs-acob-gamma0999/learner"
)
METRICS = {
    "actor_loss": "Actor loss",
    "critic_loss": "Critic loss",
    "td_loss": "TD loss",
    "bc_loss": "BC / flow loss",
    "imp_loss": "Improvement loss",
    "ref_loss": "Reference loss",
    "predicted_qs": "Predicted Q",
    "target_qs": "Target Q",
    "local_delta_advantage": "Action advantage",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path, notes):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        notes.append(f"Cannot read {path.name}: {error}")
        return {}


def recent_rows(path, limit, notes):
    """Read a bounded tail, ignoring an unfinished final line during append."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            position = stream.tell()
            data = b""
            while position and data.count(b"\n") <= limit:
                size = min(position, 65536)
                position -= size
                stream.seek(position)
                data = stream.read(size) + data
    except FileNotFoundError:
        return []
    except OSError as error:
        notes.append(f"Cannot read metrics: {error}")
        return []
    lines = data.split(b"\n")
    if position:
        lines = lines[1:]  # A leading fragment from the backwards read.
    if lines[-1]:
        notes.append("Ignored unfinished final metrics line; retry on next refresh.")
    rows = []
    for line in lines[:-1]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not finite(row.get("step")):
                raise ValueError("missing numeric step")
            rows.append(row)
        except (ValueError, UnicodeError):
            notes.append("Ignored a malformed metrics line.")
    return rows[-limit:]


def paired_rate(rows, metric):
    eligible = [r for r in rows if finite(r.get(metric))
                and finite(r.get("intervention_pref_pair_count"))
                and r["intervention_pref_pair_count"] > 0]
    count = sum(r["intervention_pref_pair_count"] for r in eligible)
    return {
        "rate": sum(r[metric] * r["intervention_pref_pair_count"] for r in eligible) / count if count else None,
        "pair_samples": count,
    }


def snapshot(directory, window):
    directory = directory.expanduser().resolve()
    notes = []
    status = read_json(directory / "status.json", notes)
    latest = read_json(directory / "latest.json", notes)
    rows = recent_rows(directory / "metrics.jsonl", window, notes)
    checkpoint = Path(latest["checkpoint"]) if latest.get("checkpoint") else None
    if checkpoint is not None:
        if not checkpoint.is_absolute():
            checkpoint = directory / checkpoint
        if not checkpoint.exists():
            checkpoint = directory / checkpoint.name  # Relocated downloaded run.
    metadata = read_json(checkpoint / "metadata.json", notes) if checkpoint else {}
    last = rows[-1] if rows else {}
    status_metrics = status.get("metrics") or {}
    gamma = next((v for v in [status.get("discount"), status_metrics.get("discount"),
                              last.get("discount"), metadata.get("discount")] if finite(v)), None)
    replay = status.get("replay") or {}
    # An explicit null sampling value means legacy uniform sampling.
    if "sampling" in status:
        sampling, sampling_known = status["sampling"], True
    elif "sampling" in replay:
        sampling, sampling_known = replay["sampling"], True
    else:
        sampling, sampling_known = metadata.get("replay_sampling"), "replay_sampling" in metadata
    metrics = {}
    for key in METRICS:
        values = [r[key] for r in rows if finite(r.get(key))]
        metrics[key] = {"latest": last.get(key) if finite(last.get(key)) else None,
                        "mean": statistics.mean(values) if values else None, "samples": len(values)}
    nonfinite = sorted({key for row in rows for key, value in row.items()
                        if isinstance(value, (int, float)) and not math.isfinite(value)})
    if nonfinite:
        notes.append("Non-finite values in this window: " + ", ".join(nonfinite))
    discounts = sorted({r["discount"] for r in rows if finite(r.get("discount"))})
    if len(discounts) > 1 or (discounts and gamma is not None and discounts != [gamma]):
        notes.append("Window discount differs from current settings; statistics include earlier settings.")
    metrics_path = directory / "metrics.jsonl"
    mtime = metrics_path.stat().st_mtime if metrics_path.exists() else None
    recorded_time = status.get("time")
    historical = bool(rows and status.get("state") in ("loading", "waiting_completed_episodes")
                      and finite(recorded_time) and mtime is not None and mtime < recorded_time)
    if historical:
        notes.append("Metrics are from the previous run; the new run has not written updates yet.")
    updates = [r["seconds"] for r in rows if finite(r.get("seconds"))]
    sample_counts = {}
    for key in ("sampled_success_terminal_count", "sampled_success_tail_count"):
        values = [r[key] for r in rows if finite(r.get(key))]
        sample_counts[key] = sum(values) if values else None
    return {
        "learner_dir": str(directory), "recorded_state": status.get("state", "unknown"),
        "status_age_seconds": max(0, time.time() - recorded_time) if finite(recorded_time) else None,
        "metrics_age_seconds": max(0, time.time() - mtime) if mtime is not None else None,
        "latest_logged_step": last.get("step"), "checkpoint_step": latest.get("step"),
        "checkpoint_ready": bool(checkpoint and (checkpoint / "READY").is_file()),
        "gamma": gamma, "window_discounts": discounts, "sampling": sampling,
        "sampling_known": sampling_known, "rows": len(rows),
        "first_window_step": rows[0]["step"] if rows else None, "metrics": metrics,
        "preference_ranking": paired_rate(rows, "intervention_pref_accuracy"),
        "preference_margin_met": paired_rate(rows, "intervention_pref_margin_satisfied_rate"),
        "sampling_counts": sample_counts,
        "median_update_seconds": statistics.median(updates) if updates else None,
        "replay": {k: replay[k] for k in ("episodes", "transitions", "corrections", "discarded_episodes") if k in replay},
        "nonfinite_fields": nonfinite, "notes": notes,
    }


def fmt(value):
    return f"{value:.6g}" if finite(value) else "n/a"


def render(report):
    r = report
    lines = [f"Learner: {r['learner_dir']}",
             f"Recorded status: {r['recorded_state']} (updated {fmt(r['status_age_seconds'])}s ago)",
             f"Latest logged step: {fmt(r['latest_logged_step'])} | Checkpoint: {fmt(r['checkpoint_step'])}"
             f" ({'READY' if r['checkpoint_ready'] else 'no READY'}) | Gamma: {fmt(r['gamma'])}"]
    sampling = r["sampling"]
    if sampling:
        all_outcomes = sampling.get('schema_version') == 4
        prefix = 'terminal' if all_outcomes else 'success'
        outcome = 'successful/failed' if all_outcomes else 'successful'
        detail = (' uniform tail frames, no special terminal quota' if all_outcomes else
                  f" terminal probability within that branch: {100 * sampling['terminal_fraction_within_tail']:g}%")
        lines.append(f"Sampling: {100 * sampling[prefix+'_tail_fraction']:g}% of non-correction draws"
                     f" from {outcome} last {sampling[prefix+'_tail_seconds']:g}s;" + detail)
    else:
        lines.append("Sampling: " + ("legacy uniform + corrections" if r["sampling_known"] else "not recorded"))
    if r["replay"]:
        lines.append("Replay (last status): " + ", ".join(f"{k}={v}" for k, v in r["replay"].items()))
    lines += [f"Window: {r['rows']} logged steps ({fmt(r['first_window_step'])}–{fmt(r['latest_logged_step'])})",
              f"{'Metric':<23} {'Latest':>12} {'Window mean':>14}"]
    for key, label in METRICS.items():
        item = r["metrics"][key]
        lines.append(f"{label:<23} {fmt(item['latest']):>12} {fmt(item['mean']):>14}")
    for key, label in [("preference_ranking", "Human > proposal"), ("preference_margin_met", "Preference margin met")]:
        value = r[key]
        rate = "n/a" if value["rate"] is None else f"{100 * value['rate']:.1f}%"
        lines.append(f"{label}: {rate} ({fmt(value['pair_samples'])} pair samples, repeats included)")
    counts = r["sampling_counts"]
    lines += [f"Window samples: success terminal={fmt(counts['sampled_success_terminal_count'])},"
              f" success tail incl. terminal={fmt(counts['sampled_success_tail_count'])}",
              f"Median update time: {fmt(r['median_update_seconds'])}s (excludes input preparation and saving)",
              "Training metrics are not robot success rates. Status is recorded, not a process-liveness check."]
    lines.extend("NOTE: " + note for note in r["notes"])
    return "\n".join(lines)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--window", type=positive_int, default=100, help="Recent learner steps to summarize (default: 100)")
    parser.add_argument("--watch", type=positive_seconds, nargs="?", const=5.0, metavar="SECONDS",
                        help="Refresh continuously, every 5 seconds by default; Ctrl+C exits")
    parser.add_argument("--json", action="store_true", help="Print JSON; watch mode emits one object per line")
    args = parser.parse_args()
    if not args.learner_dir.expanduser().is_dir():
        parser.error(f"learner directory does not exist: {args.learner_dir}")
    try:
        while True:
            report = snapshot(args.learner_dir, args.window)
            if args.watch and sys.stdout.isatty() and not args.json:
                print("\033[2J\033[H", end="")
            print(json.dumps(report, ensure_ascii=False, allow_nan=False) if args.json else render(report), flush=True)
            if args.watch is None:
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
