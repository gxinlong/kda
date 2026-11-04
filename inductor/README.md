# Inductor model optimization

This is the model-level loop around KDA's single-kernel workflow. Start with the
[skill](../skills/inductor-model-optimization/SKILL.md) or the
[starter prompt](../prompts/inductor-model.md). No previous chat or private task
files are required. The objective is complete-model inference performance.

1. Supply the model, execution environment, hardware, GPU selection, representative
   input contract and numerical acceptance policy for this task.
2. Initialize a workspace; inspect and compile the immutable baseline.
3. Measure ordinary AOT and CUDA Graph separately; profile in a separate process.
4. Rank all executed kernels by summed device time, including external libraries.
5. Optimize selected hotspots through KDA's kernel loop, starting with a measured
   bottleneck and a mechanism hypothesis. Translate wins into guarded templates.
6. Recompile, audit selection, validate and measure the whole model against both
   the original baseline and the last accepted candidate. Retain or reject.

## Read by stage

| Stage | Guide |
|---|---|
| Start, resume, promotion and deliverables | [Model workflow](docs/workflow.md) |
| Exact versioned baseline and runtime identity | [Compilation](docs/compilation.md) |
| Validation, ordinary/Graph timing and top-k | [Benchmarking](docs/benchmarking.md) |
| Export/FX/scheduler/code generation analysis | [Compiler analysis](docs/compiler-analysis.md) |
| Runtime kernel to boundary tensors and template | [Provenance and capture](docs/provenance.md) |
| Bottleneck diagnosis and structural implementations | [Kernel mechanisms](docs/kernel-mechanisms.md) |
| Copyable command sequence and parameter meanings | [Tool recipes](docs/tool-recipes.md) |

## Reusable tools

All Python entrypoints provide `--help`. Inspection, task management, graph
analysis, contract creation, extraction and code auditing use only the Python
standard library. GPU tools need the user-supplied compatible Torch/Triton/CUDA
runtime; profiling additionally needs Nsight Systems or Compute. They do not
install or rebuild these dependencies.

| Tool under `scripts/` | Purpose |
|---|---|
| `task_workspace.py` | Initialize context/profile snapshot; checkpoint evidence and resume with hash checks |
| `inspect_model.py` | Archive identity, companion files and export summary without Torch |
| `analyze_graph.py` | Ordered graph boundaries, scoped value dependencies and module metadata |
| `compile_aot.py` | Explicit baseline profile, prepared inputs/reference, source/runtime identity and guarded compilation |
| `validation_contract.py` | Bind successful reference validation or explicitly authorized baseline equivalence to exact artifacts |
| `benchmark_model.py` | Unprofiled ordinary/Graph timing, separate short nsys capture or diagnostic tensor capture |
| `analyze_profile.py` | Validate complete trace blocks and rank kernel groups by aggregate GPU duration |
| `extract_kernels.py` | Extract exact embedded source, compiler metadata and a kernel-body starting point |
| `run_guarded.py`, `gpu_guard.py` | Fresh GPU admission and external worker supervision |
| `compare_models.py` | Paired or three-way whole-model validation and ordinary/Graph timing |
| `audit_codegen.py` | Audit embedded Triton code, compiler metadata and actual launch helpers |
| `export_autotune_configs.py`, `rebind_autotune_configs.py` | Source-bound configuration replay for controlled compiler experiments |

`csrc/` contains optional Linux diagnostic capture helpers. Review their ABI,
explicit selectors and limitations before building; see [provenance](docs/provenance.md).

Reusable files live in `inductor/` and `skills/`. Model archives, inputs, task-local
configuration, private paths, generated kernels, profiles and measurements belong
in ignored `agent_space/<model>_<task-date>/`. Each kernel subtask follows the
existing [KDA layout and loop](../docs/agent-flow.md). The bundled
[baseline profile](configs/compile_profiles.json) contains compiler policy, not a
model, remote host, card type or selected GPU. Compatibility with the supplied
compiler must be checked explicitly; `safe` is never an automatic fallback.

## Offline validation

From the repository root, run:

```sh
python3 -m unittest discover -s tests -p 'test_inductor*.py' -v
```

These tests cover GPU admission policy with mocks, immutable reference contracts,
task initialization/relocation/resume, export graph scopes, synthetic Nsight
ranking, generated-code audits and host-side capture bounds. They require no
Torch or CUDA; the capture bounds test uses a host C++ compiler when available,
and the process-tree test requires Linux. All tool `--help` paths are usable
without importing Torch. GPU compilation, capture ABI and numerical/performance
validation must still be exercised with each task's supplied runtime and device.
