# Reproducible AOT compilation

Use the original export archive and its adjacent model configuration/schema.
An existing AOT package is a measurement input, not an ExportedProgram. Inspect
archive members before choosing a loader. Preserve dynamic constraints, request
versus note batch dimensions, input order, storage semantics and custom ops.

The baseline is the project `default` profile in `../configs/compile_profiles.json`:
max autotune, epilogue/permute/aggressive fusion, pointwise/GEMM autotune, combo
kernels with dynamic foreach handling, no runtime constant folding, and
`ATEN,CUTLASS,TRITON` GEMM candidates. The combo limit is explicitly 8. TF32 is
enabled for matmul and cuDNN. The profile has a stable baseline ID and schema
version; snapshot its JSON and hash in each task. `safe` requires an explicit
selection and is never an automatic retry.

Before importing Torch set the memory-planning, combined-pool, freezing,
fixed-workspace, dynamic-concat and debug environment from the profile. Record
both requested and effective options. The bundled profile was exercised with the
recorded Torch series, not every version. Missing requested config keys fail;
optional observed keys may be recorded as unsupported. Custom switches such as
fixed workspace, dynamic concat and unbacked handling are branch/adapter-specific:
verify their readers in the supplied source. If incompatible, make a task-local
profile revision, document its differences and establish a new baseline explicitly.
Do not silently drop keys or treat an ignored environment variable as an active
optimization. The backend list does not prove every backend participated.

Specify one physical GPU or UUID. Two fresh checks must show no foreign compute
processes and utilization/memory compatible with the host's idle baseline. Pin
the UUID, retain external 1 Hz telemetry and stop this task's worker on interference.
Admission is not a scheduler reservation. Do not switch cards, kill foreign
processes, reset devices, lock clocks or alter power settings to make a run pass.

Record interpreter, imported Torch/Inductor paths, wheel binary and source commit,
dirty diff, Triton/CUDA/driver, adapter commit and helper hashes. A Python source
overlay and installed C extension can have different revisions. Save every exact
command, model/config/schema hash, effective environment, log, exit code and
package hash. Use separate caches and ample temporary space for every run.
The bundled native identity check hashes `torch._C`; it is not a complete wheel
or shared-library inventory. If native libraries/custom ops can change between
runs, retain and verify their identities in the task environment manifest too.

The standard `exported` loader consumes an ExportedProgram with prepared example
inputs. The optional Fourier adapter externalizes weights and re-exports on CUDA;
it requires an explicit `--fourier-root`. Verify its input ABI against the supplied
schema. `--allow-triton-version-override` is explicit, process-local and recorded;
it does not certify arbitrary adapter compatibility. Do not change installed
packages to satisfy a pin. Other preparation/custom-op paths require a task-local
adapter with the same explicit input/reference contract.

Save the exact prepared example inputs with serialization that preserves storage
relationships. Compare the compiled outputs against eager execution of the
prepared ExportedProgram with explicitly recorded tolerances. Clone outputs before
another call can overwrite a fixed workspace. Save shape/dtype, finite values,
per-output errors and output distribution. Report model preparation, compile call,
package loading, smoke forward and numerical comparison as separate outcomes.
Saturated examples or a prepared-program reference limit the conclusion: neither
proves all business inputs nor independently validates the export transformation.

Background launches must retain logs and exit status. Before resuming, check for
the existing supervisor and worker; never duplicate a long compilation because
its log is quiet. Source changes during a run invalidate its source identity.

Create a contract with `validation_contract.py` after successful baseline
validation; it reuses the saved eager reference without requiring a precision
waiver. For a known mismatch, an explicitly authorized baseline-equivalence
contract instead names a separately saved compiled-baseline output. Pass
`--baseline-run` and `--validation-contract` for subsequent candidates. The compiler checks the original model hash, prepared
graph and input ABI, reuses the baseline's exact serialized inputs, and compares
all outputs to the immutable chosen reference. For baseline equivalence, keep the
original eager failure; that contract does not establish independent model accuracy. Record untracked
template sources as well as tracked diffs and Python files.

`--seed-timing-cache` copies the baseline's persistent GEMM timing records,
content-addressed `.kernel_perf` fusion timing records, and `.best_config` choice
records into the fresh cache. Record every copied hash. New template choices are
still timed. Never copy graph/code caches to hide template changes, and do not
compare compilation latency between a cold baseline and a seeded run.

Copying `.best_config` paths alone does not freeze unchanged kernels when Torch
source changes: their cache keys incorporate `torch_key()`. Reduction launch
choices can then change numerical results even when generated source is identical.
Export baseline source/config associations with `export_autotune_configs.py` in
the original source/runtime environment, and pass the resulting manifest through
`--baseline-autotune-configs`. The compiler verifies baseline package, native
runtime, generated-source and configuration hashes, then places the configuration
JSON under the current Torch key. Only identical generated-source filenames can
hit; Inductor's own `configs_hash` check still applies. Inspect the resulting
wrapper launch settings to verify actual reuse rather than assuming a cache hit.

Reconstructing the original `torch_key()` requires the original imported Python
module set, including ignored generated modules such as
`torch/testing/_internal/generated/annotated_fn_args.py`. A clean Git worktree alone
may be insufficient. Preserve those files and the wheel runtime without modifying
the original baseline. Also account for editable-install `.pth` files that prepend
another checkout: select the intended source root before importing Torch and verify
`torch.__file__`. A changed key is not permission to reuse compiled binaries.

Use the [tool recipes](tool-recipes.md) for commands and parameter definitions.
`--rtol`, `--atol` (or a reference contract), GPU selection and the host's idle
memory ceiling are explicit task inputs. Local or remote execution is supported;
no host or hardware defaults are embedded in the profile.
