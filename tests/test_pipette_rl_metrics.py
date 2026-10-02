"""CPU-only checks for live-log boundaries and preference statistics."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pipette_rl_metrics.py"
spec = importlib.util.spec_from_file_location("pipette_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class MetricsTests(unittest.TestCase):
    def test_uniform_all_outcome_tail_settings_render_without_terminal_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sampling=dict(schema_version=4,terminal_tail_fraction=.5,
                          terminal_tail_seconds=5.,latest_collection_fraction=.5)
            (root/'status.json').write_text(json.dumps(dict(state='training',sampling=sampling)))
            text=metrics.render(metrics.snapshot(root,100))
            self.assertIn('50% of non-correction draws',text)
            self.assertIn('successful/failed last 5s',text)
            self.assertIn('no special terminal quota',text)

    def test_tail_ignores_inflight_row_and_handles_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            lines = [json.dumps({"step": i, "padding": "x" * 4000}) for i in range(1, 51)]
            path.write_text("\n".join(lines) + '\n{"step": 51')
            notes = []
            rows = metrics.recent_rows(path, 20, notes)
            self.assertEqual([r["step"] for r in rows], list(range(31, 51)))
            self.assertTrue(any("unfinished" in n for n in notes))

    def test_preference_uses_pair_weight_and_excludes_empty_batches(self):
        rows = [
            {"intervention_pref_pair_count": 0, "intervention_pref_accuracy": 0},
            {"intervention_pref_pair_count": 1, "intervention_pref_accuracy": 1},
            {"intervention_pref_pair_count": 3, "intervention_pref_accuracy": 0},
        ]
        result = metrics.paired_rate(rows, "intervention_pref_accuracy")
        self.assertEqual(result, {"rate": 0.25, "pair_samples": 4})
        self.assertIsNone(metrics.paired_rate(rows[:1], "intervention_pref_accuracy")["rate"])

    def test_new_loading_status_does_not_present_old_settings_as_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(json.dumps({"step": 1000, "discount": 0.99}) + "\n")
            os.utime(root / "metrics.jsonl", (100, 100))
            (root / "status.json").write_text(json.dumps({"state": "loading", "time": 200,
                                                        "discount": 0.999, "sampling": None}))
            report = metrics.snapshot(root, 100)
            self.assertEqual(report["gamma"], 0.999)
            self.assertTrue(report["sampling_known"])
            self.assertTrue(any("previous run" in n for n in report["notes"]))
            self.assertTrue(any("discount differs" in n for n in report["notes"]))

    def test_nonfinite_is_visible_but_json_is_standard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(json.dumps({"step": 1, "actor_loss": float("nan")}) + "\n")
            report = metrics.snapshot(root, 100)
            self.assertEqual(report["nonfinite_fields"], ["actor_loss"])
            self.assertIsNone(report["metrics"]["actor_loss"]["mean"])
            json.dumps(report, allow_nan=False)

    def test_missing_log_is_no_data_not_zero_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            report = metrics.snapshot(Path(directory), 100)
            self.assertEqual(report["rows"], 0)
            self.assertIsNone(report["metrics"]["td_loss"]["latest"])
            self.assertIsNone(report["preference_ranking"]["rate"])
            self.assertIn("n/a", metrics.render(report))


if __name__ == "__main__":
    unittest.main()
