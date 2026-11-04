import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile


SCRIPTS = Path(__file__).resolve().parents[1] / "inductor" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import gpu_guard
import inspect_model
from benchmark_model import summarize, validate_contract, validate_runtime_source


def sample(processes=(), memory="120", utilization="0"):
    return {"device": {"uuid": "GPU-test", "memory.used": memory,
                       "utilization.gpu": utilization}, "processes": list(processes)}


class AdmissionTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Requires Linux process namespace metadata")
    def test_profiler_descendant_in_separate_session_is_owned(self):
        with tempfile.TemporaryDirectory() as root:
            child_file = Path(root) / "child.pid"
            code = ("import subprocess,sys,pathlib; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                    f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); p.wait()")
            parent = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            child = None
            try:
                deadline = time.monotonic() + 5
                while not child_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                child = int(child_file.read_text())
                self.assertNotEqual(os.getpgid(child), parent.pid)
                owned = gpu_guard.process_group_pids(parent.pid)
                self.assertIn(str(child), owned)
                self.assertNotIn(str(os.getpid()), owned)
            finally:
                if child is not None:
                    os.kill(child, signal.SIGTERM)
                parent.terminate()
                parent.wait()

    def test_zero_utilization_does_not_allow_foreign_process(self):
        busy = sample([{"pid": "42", "name": "[Not Found]"}])
        with tempfile.TemporaryDirectory() as root, \
             patch.object(gpu_guard, "snapshot", return_value=busy), \
             patch.object(gpu_guard.time, "sleep"):
            evidence = Path(root) / "admission.json"
            with self.assertRaisesRegex(RuntimeError, "foreign compute"):
                gpu_guard.admit("GPU-test", evidence, 256)
            self.assertEqual(len(json.loads(evidence.read_text())), 2)

    def test_second_check_catches_new_occupant(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(gpu_guard, "snapshot", side_effect=[sample(), sample([{"pid": "42"}])]), \
             patch.object(gpu_guard.time, "sleep"):
            with self.assertRaises(RuntimeError):
                gpu_guard.admit("GPU-test", Path(root) / "admission.json", 256)

    def test_idle_memory_and_utilization_are_both_required(self):
        self.assertTrue(gpu_guard.idle_errors(sample(memory="500"), 256))
        self.assertTrue(gpu_guard.idle_errors(sample(utilization="20"), 256))
        self.assertFalse(gpu_guard.idle_errors(sample(), 256))

    def test_unmapped_process_stops_only_worker_group(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(gpu_guard, "snapshot", return_value=sample([{"pid": "unmapped"}])), \
             patch.object(gpu_guard, "process_group_pids", return_value=set()):
            result = gpu_guard.supervise(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                env=None, cwd=root, gpu="GPU-test", log_path=Path(root) / "worker.log",
                telemetry_path=Path(root) / "telemetry.jsonl",
            )
            self.assertEqual(result["returncode"], 1)
            self.assertIn("unmapped", result["error"])
            self.assertNotEqual(result["worker_returncode"], 0)


class ArchiveTests(unittest.TestCase):
    def test_export_and_aot_are_distinguished_without_torch(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "export.pt2"
            graph = {"schema_version": {"major": 1}, "graph_module": {"graph": {
                "nodes": [{"target": "aten.mm.default"}], "inputs": [], "outputs": [],
            }}}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/serialized_exported_program.json", json.dumps(graph))
            result = inspect_model.inspect(path)
            self.assertEqual(result["exported_graphs"][0]["targets"], {"aten.mm.default": 1})
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data/aotinductor/model.so", b"example")
            self.assertEqual(inspect_model.inspect(path)["exported_graphs"], [])


class MeasurementTests(unittest.TestCase):
    def test_relocated_runtime_requires_identical_source(self):
        expected = {"root":"/original", "commit":"abc", "status":"", "diff_sha256":"empty",
                    "untracked_python_sha256":{}}
        actual = {**expected, "root":"/immutable", "untracked_template_sha256":{}}
        validate_runtime_source(actual, expected)
        with self.assertRaisesRegex(RuntimeError, "diff_sha256"):
            validate_runtime_source({**actual,"diff_sha256":"changed"}, expected)
        with self.assertRaisesRegex(RuntimeError, "untracked_template_sha256"):
            validate_runtime_source({**actual,"untracked_template_sha256":{"new.jinja":"hash"}}, expected)

    def test_failed_accuracy_cannot_silently_enter_measurement(self):
        compilation = {"status": "failed", "validation": {"passed": False},
                       "package": {"sha256": "package"}, "saved_inputs": {"sha256": "inputs"}}
        with self.assertRaises(RuntimeError):
            validate_contract(compilation, None)
        contract = {"kind": "baseline_equivalence", "package_sha256": "package", "inputs_sha256": "inputs", "rtol": 0.001, "atol": 0.00001}
        with self.assertRaisesRegex(ValueError, "explicit diagnostic authorization"):
            validate_contract(compilation, contract)
        contract["diagnostic_timing_authorization"] = "Explicit user instruction in this task"
        validate_contract(compilation, contract)
        contract["package_sha256"] = "other package"
        with self.assertRaisesRegex(ValueError, "another package"):
            validate_contract(compilation, contract)

    def test_nearest_rank_and_forward_normalization(self):
        rows = [{"mode": mode, "round": 0, "k": 10, "event_ms": value, "wall_ms": value}
                for mode in ("ordinary", "graph") for value in range(1, 21)]
        result = summarize(rows)["ordinary"]
        self.assertEqual(result["samples"], 20)
        self.assertEqual(result["forwards"], 200)
        self.assertEqual(result["event_ms"]["median"], 10.5)
        self.assertEqual(result["event_ms"]["p95_nearest_rank"], 19)
        self.assertAlmostEqual(result["loop_forwards_per_second"], 200 / 2.1)


if __name__ == "__main__":
    unittest.main()
