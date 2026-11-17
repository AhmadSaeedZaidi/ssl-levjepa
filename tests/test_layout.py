"""Regression checks for artifact relocation and safe CLI inspection."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ssl_levjepa import cli, experiment, layout


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        outputs = self.root / "outputs"
        outputs.mkdir()
        (outputs / "migration.json").write_text(json.dumps({"paths": {
            "runs/old": "outputs/dated",
            "runs/old/baseline": "outputs/dated/training",
            "data/pilot": "outputs/datasets/pilot",
        }}))
        self.patch = patch.multiple(layout, ROOT=self.root, OUTPUTS=outputs)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_saved_paths_use_longest_mapping_across_platforms(self):
        expected = self.root / "outputs/dated/training/checkpoints/final.ckpt"
        for saved in ["runs/old/baseline/checkpoints/final.ckpt",
                      "C:\\Users\\old\\code\\ssl-levjepa\\runs\\old\\baseline\\checkpoints\\final.ckpt",
                      "/mnt/c/Users/old/code/ssl-levjepa/runs/old/baseline/checkpoints/final.ckpt"]:
            with self.subTest(saved=saved):
                self.assertEqual(layout.resolve_path(saved), expected)

    def test_existing_paths_are_never_redirected(self):
        path = self.root / "data/pilot"
        path.mkdir(parents=True)
        self.assertEqual(layout.resolve_path("data/pilot"), path)

    def test_relative_dataset_reference_survives_bundle_relocation(self):
        bundle = self.root / "outputs/dated"
        bundle.mkdir()
        (bundle / "dataset.json").write_text(json.dumps({"path": "../datasets/pilot"}))
        self.assertEqual(layout.experiment_dataset(bundle), self.root / "outputs/datasets/pilot")

    def test_plan_requires_no_inputs_and_creates_no_output(self):
        destination = self.root / "outputs/planned"
        result = experiment.run(config=self.root / "absent.yaml", dataset=self.root / "absent_dataset",
                                output=destination, plan=True)
        self.assertFalse(destination.exists())
        self.assertEqual([s["stage"] for s in result["stages"]],
                         ["validate_train", "validate_validation", "validate_test", "benchmark",
                          "training", "checkpoint", "evaluation", "report"])
        self.assertTrue(all("ssl_levjepa" in s["command"] for s in result["stages"]))

    def test_existing_evidence_is_rejected_before_loading_inputs(self):
        from ssl_levjepa import dataset, evaluation
        evidence = self.root / "evidence.json"
        evidence.write_text("preserve me")
        for check in (dataset.audit, dataset.validate):
            with self.assertRaises(FileExistsError):
                check(self.root / "missing", output=evidence)
        with self.assertRaises(FileExistsError):
            evaluation.evaluate(self.root / "missing", self.root / "missing", evidence)
        self.assertEqual(evidence.read_text(), "preserve me")

    def test_info_resolves_each_seed_checkpoint_in_a_v2_bundle(self):
        bundle = self.root/"outputs/expanded"
        bundle.mkdir()
        (bundle/"dataset.json").write_text(json.dumps({"path":"../datasets/pilot"}))
        (bundle/"protocol.json").write_text(json.dumps({"version":2,"seeds":[42,43,44]}))
        for seed in (42,43,44):
            checkpoint = bundle/"seeds"/str(seed)/"training/checkpoints/final.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"path test only")
        result = experiment.info(bundle)
        self.assertNotIn("checkpoint",result)
        self.assertEqual(set(result["checkpoints"]),{"42","43","44"})
        self.assertTrue(all(Path(path).exists() for path in result["checkpoints"].values()))

    def test_fresh_source_checkout_is_pinned_before_dirty_check(self):
        from ssl_levjepa import simulator
        calls = []
        with patch.object(simulator, "SUMATRA_SOURCE", self.root / "new_source"):
            with patch.object(simulator, "_run", side_effect=lambda args, **kwargs: calls.append(args)):
                simulator._ensure_source()
        self.assertEqual(calls[-1], ["git", "checkout", "--detach", simulator.SUMATRA_COMMIT])
        self.assertFalse(any("status" in command for command in calls))

    def test_bundle_selects_measured_batch_and_records_failure(self):
        from omegaconf import OmegaConf
        source = self.root / "source.yaml"
        source.write_text("dataset: {path: old, episodes_path: old}\nvalidation_dataset: {path: old, episodes_path: old}\ndata: {batch_size: 8}\n")
        data = self.root / "outputs/datasets/pilot"
        data.mkdir(parents=True)
        for name, contents in [("audit.json", '{"passed": true}'), ("metadata.json", '{}'), ("clips.jsonl", '')]:
            (data / name).write_text(contents)
        output = self.root / "outputs/trial"

        def child(command, log):
            if "benchmark" in command:
                (output / "benchmark").mkdir()
                (output / "benchmark/recommendation.json").write_text('{"batch_size": 64}')
            if "train" in command:
                cfg = OmegaConf.load(output / "config.yaml")
                self.assertEqual(cfg.data.batch_size, 64)
                self.assertEqual(Path(cfg.dataset.path), data / "train/train.lance")
                self.assertEqual(Path(cfg.validation_dataset.path), data / "validation/train.lance")
                raise RuntimeError("simulated child failure")

        with patch.multiple(experiment, ROOT=self.root, OUTPUTS=self.root / "outputs"):
            with patch.object(experiment, "execute", side_effect=child):
                with self.assertRaisesRegex(RuntimeError, "simulated child failure"):
                    experiment.run(config=source, dataset=data, output=output)
        status = json.loads((output / "status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["failed_stage"], "training")
        self.assertEqual(status["stages"]["benchmark"], "complete")
        self.assertFalse((self.root / "outputs/latest.json").exists())
        self.assertEqual((output / "provenance/source_config.yaml").read_bytes(), source.read_bytes())


class CliTests(unittest.TestCase):
    def test_help_has_no_ml_import_or_artifact_dependency(self):
        code = """import sys
from ssl_levjepa.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'lightning' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dataset", result.stdout)

    def test_console_entrypoint_returns_success_and_prints_results(self):
        with patch.object(cli, "dispatch", return_value={"passed": True}):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(cli.main([]), 0)
        self.assertEqual(json.loads(stream.getvalue()), {"passed": True})


if __name__ == "__main__":
    unittest.main()
