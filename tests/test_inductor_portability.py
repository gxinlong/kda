"""CPU-only regression tests for portable task contracts and evidence tools."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'inductor' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import analyze_graph
import analyze_profile
import audit_codegen
import compile_aot
import task_workspace
import run_guarded
import validation_contract


def context():
    return {'model_name': 'synthetic', 'model_path': 'user-supplied-export.pt2',
            'execution': {'location': 'test-only', 'python': sys.executable,
                          'gpu': 'GPU-supplied', 'hardware': 'user-supplied', 'max_idle_memory_mib': 0},
            'accuracy': {'reference_policy': 'prepared_export_eager', 'rtol': 0.001, 'atol': 0.00001}}


def artifact(root, name, data):
    path = root / name
    path.write_bytes(data)
    return {'path': str(path), 'sha256': compile_aot.file_hash(path)}


def baseline(root):
    return {'status': 'succeeded', 'validation': {'passed': True},
            'package': artifact(root, 'model.pt2', b'fake package'),
            'saved_inputs': artifact(root, 'inputs.pt', b'fake inputs'),
            'reference': artifact(root, 'reference.pt', b'fake reference'),
            'request': {'torch_backends': {'cuda.matmul.allow_tf32': True}, 'rtol': 0.001, 'atol': 0.00001}}


class ContractTests(unittest.TestCase):
    def test_baseline_profile_and_overrides_are_explicit(self):
        profiles, name, profile, configs = compile_aot.resolve_profile(compile_aot.PROFILES)
        self.assertEqual(name, 'default')
        self.assertEqual(profiles['baseline_id'], 'inductor-aot-baseline-v1')
        self.assertEqual(configs['combo_kernel_max_num_nodes'], 8)
        self.assertEqual(configs['max_autotune_gemm_backends'], 'ATEN,CUTLASS,TRITON')
        self.assertTrue(profile['torch_backends']['cuda.matmul.allow_tf32'])
        _, _, _, changed = compile_aot.resolve_profile(compile_aot.PROFILES, overrides=['combo_kernel_max_num_nodes=5'])
        self.assertEqual(changed['combo_kernel_max_num_nodes'], 5)
        self.assertEqual(configs['combo_kernel_max_num_nodes'], 8)
        with self.assertRaises(KeyError):
            compile_aot.resolve_profile(compile_aot.PROFILES, 'missing-profile')

    def test_accuracy_tolerance_is_not_inherited_from_another_task(self):
        for pair in ((None, None), (-1, 0), (float('nan'), 0), (0, float('inf'))):
            with self.assertRaises(ValueError):
                compile_aot.resolve_tolerances(*pair)
        with self.assertRaisesRegex(ValueError, 'differs'):
            compile_aot.resolve_tolerances(0.1, None, {'rtol': 0.01, 'atol': 0.01})

    def test_successful_reference_contract_needs_no_waiver(self):
        with tempfile.TemporaryDirectory() as folder:
            model = baseline(Path(folder))
            contract = validation_contract.build(model)
            self.assertEqual(contract['kind'], 'validated_reference')
            compile_aot.validate_reference_contract(model, contract)
            self.assertNotIn('diagnostic_timing_authorization', contract)
            altered = copy.deepcopy(contract)
            altered['torch_backends']['cuda.matmul.allow_tf32'] = False
            with self.assertRaisesRegex(ValueError, 'does not match'):
                compile_aot.validate_reference_contract(model, altered)
            Path(model['saved_inputs']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                validation_contract.build(model)

    def test_failed_baseline_needs_explicit_reference_and_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            model = baseline(Path(folder))
            model.update(status='failed', validation={'passed': False})
            with self.assertRaises(ValueError):
                validation_contract.build(model)
            with self.assertRaises(ValueError):
                validation_contract.build(model, model['reference']['path'], rtol=0.001, atol=0.00001)
            contract = validation_contract.build(model, model['reference']['path'],
                'Explicit instruction for this test task', rtol=0.001, atol=0.00001)
            self.assertEqual(contract['kind'], 'baseline_equivalence')
            self.assertNotIn('passed', contract)

    def test_installed_source_identity_detects_changes_without_a_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'torch').mkdir()
            source = root / 'torch' / '__init__.py'
            source.write_text('VERSION = 1\n')
            before = compile_aot.source_identity(root)
            source.write_text('VERSION = 2\n')
            with patch.object(compile_aot.shutil, 'which', return_value=None):
                after = compile_aot.source_identity(root)
            self.assertEqual(before['kind'], 'installed_package')
            self.assertNotEqual(before['package_source_sha256'], after['package_source_sha256'])


class WorkspaceTests(unittest.TestCase):
    def test_cold_start_cli_and_relocated_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            supplied = root / 'context.json'
            supplied.write_text(json.dumps(context()))
            task = root / 'task'
            entry = [sys.executable, str(SCRIPTS / 'task_workspace.py')]
            subprocess.run([*entry, 'init', '--context', str(supplied), '--task-root', str(task)], check=True, capture_output=True)
            evidence = task / 'runs' / 'manifest.json'
            evidence.write_text('{"status": "test-only"}')
            subprocess.run([*entry, 'checkpoint', '--task-root', str(task), '--phase', 'offline-check',
                            '--next', 'Verify supplied runtime', '--decision', 'No GPU work executed',
                            '--artifact', f'baseline={evidence}'], check=True, capture_output=True)
            moved = root / 'moved'
            shutil.copytree(task, moved)
            check = subprocess.run([*entry, 'status', '--task-root', str(moved)], check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(check.stdout)['verified'])
            (moved / 'runs' / 'manifest.json').write_text('changed')
            changed = subprocess.run([*entry, 'status', '--task-root', str(moved)], capture_output=True, text=True)
            self.assertEqual(changed.returncode, 1)
            self.assertEqual(json.loads(changed.stdout)['artifact_checks']['baseline'], 'changed')

    def test_missing_hardware_and_implicit_waiver_rejected_without_creating_task(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, value in enumerate((context(), context())):
                task = Path(folder) / str(index)
                if index == 0:
                    value['execution']['hardware'] = None
                else:
                    value['accuracy']['reference_policy'] = 'baseline_equivalence'
                with self.assertRaises(ValueError):
                    task_workspace.initialize(task, value)
                self.assertFalse(task.exists())

    def test_baseline_checkpoint_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / 'task'
            task_workspace.initialize(task, context())
            original = task / 'runs' / 'original.json'
            newer = task / 'runs' / 'newer.json'
            original.write_text('original')
            newer.write_text('newer')
            task_workspace.checkpoint(task, 'baseline', 'measure', 'valid', [f'baseline={original}'])
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                task_workspace.checkpoint(task, 'candidate', 'compare', 'valid', [f'baseline={newer}'])
            task_workspace.checkpoint(task, 'candidate', 'compare', 'valid', [f'incumbent={newer}'])
            with self.assertRaises(FileExistsError):
                task_workspace.initialize(task, context())


class GuardedSourceTests(unittest.TestCase):
    def test_source_mutation_invalidates_an_otherwise_successful_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'worker.py'
            source.write_text('before')
            def mutate(*args, **kwargs):
                source.write_text('after')
                return {'returncode': 0}
            argv = ['run_guarded.py', '--gpu', 'GPU-test', '--max-idle-memory-mib', '0',
                    '--output-root', str(root / 'runs'), '--label', 'mutation',
                    '--', sys.executable, str(source)]
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_guarded, 'admit', return_value={'device': {'uuid': 'GPU-test'}}), \
                 patch.object(run_guarded, 'supervise', side_effect=mutate), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_guarded.main(), 1)
            run, = (root / 'runs').iterdir()
            result = json.loads((run / 'exit.json').read_text())
            self.assertIn('sources changed', result['error'])


class GraphTests(unittest.TestCase):
    def test_value_edges_ignore_literals_and_keep_nested_scopes(self):
        tensor = lambda name: {'as_tensor': {'name': name}}
        child = {'inputs': [tensor('x')], 'outputs': [tensor('z')], 'nodes': [
            {'target': 'child', 'inputs': [{'name': 'a', 'arg': tensor('x')}], 'outputs': [tensor('z')]}]}
        graph = {'inputs': [tensor('x'), {'as_sym_int': {'as_name': 'n'}}],
                 'outputs': [tensor('out')], 'nodes': [
                     {'target': 'parent', 'inputs': [{'name': 'a', 'arg': tensor('x')},
                         {'name': 'literal', 'arg': {'as_string': 'x'}},
                         {'name': 'size', 'arg': {'as_sym_int': {'as_name': 'n'}}},
                         {'name': 'body', 'arg': {'as_graph': {'name': 'branch', 'graph': child}}}],
                      'outputs': [tensor('out')], 'metadata': {'stack_trace': 'x'}}]}
        scopes = analyze_graph.graph_scope(graph, 'root')
        self.assertEqual(len(scopes), 2)
        self.assertEqual([edge['value'] for edge in scopes[0]['edges']], ['x', 'n', 'out'])
        self.assertEqual([edge['value'] for edge in scopes[1]['edges']], ['x', 'z'])
        self.assertTrue(scopes[1]['scope'].startswith('root/node:0/'))
        self.assertEqual(scopes[0]['unresolved_references'], [])

    def test_unknown_schema_variant_fails_instead_of_dropping_edges(self):
        with self.assertRaisesRegex(ValueError, 'export schema'):
            list(analyze_graph.references({'as_future_value': {'name': 'x'}}))

    def test_unresolved_value_is_reported_not_invented(self):
        graph = {'inputs': [], 'outputs': [{'as_tensor': {'name': 'missing'}}], 'nodes': []}
        result, = analyze_graph.graph_scope(graph, 'scope')
        self.assertEqual(result['edges'], [])
        self.assertEqual(result['unresolved_references'][0]['value'], 'missing')


def wrapper(name='triton_test_0', warps=4, fusion=True):
    module = ("import triton\n@heuristics.template(num_warps=" + str(warps)
              + ", triton_meta={'backend_options': {'enable_fp_fusion': " + str(fusion)
              + "}}, inductor_meta={'backend_hash': 'cache-key'})\n@triton.jit\ndef "
              + name + "(x):\n    value = x + 1\n")
    return ("async_compile.triton('" + name + "', '''" + module + "''', device_str='cuda')\n"
            + 'static __attribute__((noinline)) void call_' + name
            + '(int arg) { int var_9 = 1; loadKernel("generated.cubin"); '
            + 'launchKernel(kernels_.' + name + ', 1, ' + str(warps) + ', 0, stream);\n}\n')


class CodeAuditTests(unittest.TestCase):
    def test_renaming_is_allowed_but_precision_and_actual_launch_changes_are_visible(self):
        before = audit_codegen.inspect_wrapper(wrapper())
        renamed = audit_codegen.inspect_wrapper(wrapper('triton_test_9'))
        self.assertEqual(audit_codegen.compare(before, renamed)['pairs'][0]['changed'], [])
        precision = audit_codegen.inspect_wrapper(wrapper(fusion=False))
        self.assertIn('module_ast', audit_codegen.compare(before, precision)['pairs'][0]['changed'])
        launch = audit_codegen.inspect_wrapper(wrapper(warps=8))
        changed = audit_codegen.compare(before, launch)['pairs'][0]['changed']
        self.assertIn('launch_parameters', changed)
        self.assertIn('module_ast', changed)

    def test_one_candidate_cannot_match_two_baseline_definitions(self):
        before = audit_codegen.inspect_wrapper(wrapper('triton_test_0') + wrapper('triton_test_1'))
        after = audit_codegen.inspect_wrapper(wrapper('triton_test_2'))
        pairs = audit_codegen.compare(before, after)['pairs']
        matched = [row['candidate'] for row in pairs if row['candidate'] is not None]
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(pairs), 2)

    def test_unsupported_wrapper_cannot_report_no_changes(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            audit_codegen.inspect_wrapper('unrecognized wrapper')


class CaptureTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'Host C++ compiler unavailable')
    def test_host_capture_filter_and_span_bounds(self):
        code = r'''
#include "capture_bmm_config.h"
#include <cassert>
int main() {
  auto values = kda_capture::shapes("2,3,4,5;1,2,3,4");
  assert(values.size() == 2 && values[0][3] == 5);
  assert(kda_capture::span_bytes({2,3},{4,1},28) == 28);
  assert(kda_capture::span_bytes({2,3},{0,1},12) == 12);
  bool rejected = false;
  try { kda_capture::shapes("2,3,-4,5"); } catch (...) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { kda_capture::span_bytes({2,3},{4,1},24); } catch (...) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { kda_capture::span_bytes({INT64_MAX,INT64_MAX},{INT64_MAX,1},1024); }
  catch (...) { rejected = true; }
  assert(rejected);
}
'''
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'check.cpp').write_text(code)
            flags = []
            if sys.platform == 'darwin' and shutil.which('xcrun'):
                sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
                headers = Path(sdk) / 'usr' / 'include' / 'c++' / 'v1'
                if headers.is_dir():
                    flags = ['-isystem', str(headers)]
            compiled = subprocess.run(['c++', '-std=c++17', *flags, '-I', str(SCRIPTS.parent / 'csrc'),
                                       str(root / 'check.cpp'), '-o', str(root / 'check')], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(root / 'check')], check=True)


def profile_fixture(root):
    result = {'status': 'succeeded', 'pid': 123, 'request': {'gpu': {'uuid': 'GPU-test', 'index': '7'}, 'k': 2},
              'profile_blocks': {'ordinary': 3, 'graph': 3}}
    (root / 'result.json').write_text(json.dumps(result))
    (root / 'exit.json').write_text('{"returncode": 0}')
    (root / 'trace.nsys-rep').write_bytes(b'synthetic trace marker')
    with sqlite3.connect(root / 'trace.sqlite') as db:
        db.executescript('''
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, end INT, globalPid INT, deviceId INT,
 demangledName INT, graphNodeId INT, gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT, streamId INT);
CREATE TABLE NVTX_EVENTS(start INT, end INT, text TEXT, globalTid INT);
CREATE TABLE PROCESSES(globalPid INT, pid INT);
CREATE TABLE TARGET_INFO_GPU(uuid TEXT, cuDevice INT);
CREATE TABLE StringIds(id INT, value TEXT);
CREATE TABLE DIAGNOSTIC_EVENT(severity INT);
INSERT INTO PROCESSES VALUES(1,123);
INSERT INTO TARGET_INFO_GPU VALUES('test',7);
INSERT INTO StringIds VALUES(1,'elementwise_long'),(2,'bmm_repeated');
''')
        for mode_index, mode in enumerate(('ordinary', 'graph')):
            for block in range(3):
                start = (mode_index * 3 + block + 1) * 100000
                db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)', (start, start + 90000, f'kda/{mode}/block/{block}', 1))
                cursor = start + 1000
                for forward in range(2):
                    for name, duration in ((1, 9000), (2, 4000), (2, 4000), (2, 4000)):
                        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (cursor, cursor + duration, 1, 0, name, mode_index, 1, 1, 1, 32, 1, 1, 1))
                        cursor += duration + 1000


class ProfileTests(unittest.TestCase):
    def test_ranking_uses_aggregate_cost_not_per_call_or_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                analyze_profile.analyze(root, 2)
            data = json.loads((root / 'hotspots.json').read_text())
            for mode in ('ordinary', 'graph'):
                ranking = data['modes'][mode]['ranking']
                self.assertEqual(ranking[0]['name'], 'bmm_repeated')
                self.assertEqual(ranking[0]['calls_per_forward'], 3)
                self.assertEqual(ranking[0]['us_per_forward'], 12)
                self.assertAlmostEqual(ranking[0]['kernel_time_fraction'], 12 / 21)
                self.assertEqual(data['modes'][mode]['forwards'], 6)

    def test_incomplete_graph_coverage_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile_fixture(root)
            with sqlite3.connect(root / 'trace.sqlite') as db:
                db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=0')
            with self.assertRaisesRegex(RuntimeError, 'Graph node coverage'):
                analyze_profile.analyze(root)

    def test_empty_complete_block_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile_fixture(root)
            with sqlite3.connect(root / 'trace.sqlite') as db:
                db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start < 200000')
            with self.assertRaisesRegex(RuntimeError, 'complete forwards'):
                analyze_profile.analyze(root)


if __name__ == '__main__':
    unittest.main()
