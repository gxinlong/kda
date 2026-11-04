# Kernel-to-template provenance

Start with the final generated implementation. Escalate to compiler tracing only
when it cannot explain the template, boundary inputs or selection conditions.

Follow: measured event -> package wrapper call -> argument/view/offset/lifetime
mapping -> generated load/compute/store code -> selected template -> lowered IR
and FX nodes -> export nodes and model module. Names alone are not identity.
Record the exact package/source hashes and observed call sites; mappings may be
many-to-many. A generated `fused` name is not proof of fusion semantics.

Inspect PT2 archive JSON without loading weights when sufficient. Preserve graph
scope, ordered boundaries, dynamic expressions, tensor metadata, multi-output
ports, aliases and mutation. Build dependencies from value references, not node
adjacency. An AOT binary cannot restore missing ExportedProgram provenance.

For a new compile, verify supported trace/log options in that checkout before
enabling `TORCH_TRACE`, `TORCH_LOGS` or `INDUCTOR_PROVENANCE`. Capture cache behavior,
pre/post-fusion IR, wrapper and actual selection diagnostics. Source inspection
shows possible conditions; it does not prove why a historical compile rejected
a candidate. Instrumented builds need comparison with the baseline before their
boundaries can be attributed to it.

A template addition must encode real requirements: GPU capability, dtype,
strides/alignment, static or guarded dynamic shape, TF32 policy and supported
prologue/epilogue. Preserve fallback behavior and output ordering. Inspect the
fresh generated package to prove routing and selection, then validate numerics
and measure the complete model. Keep compile time, kernel time and model time
separate.

## Capturing an existing package boundary

`csrc/capture_kernel.cpp` can intercept one named `cuLaunchKernel` call from the
AOT wrapper in a separate diagnostic process. It synchronizes the target stream,
copies the declared input spans, launches the original kernel, synchronizes and
copies outputs. It does not edit the compiled package. This changes timing and
must not be used for acceptance measurements.

Build the helper as a shared library with the installed CUDA headers. Use
`benchmark_model.py --mode capture --capture-spec <json>` only after reviewing
the actual wrapper argument list and generated indexing. The specification has
`library`, `kernel` (exact name) and `arguments` (comma-separated `i:<bytes>`,
`o:<bytes>`, `s:<bytes>` for input pointer, output pointer and scalar parameters).
Scalar sizes are at most eight bytes. The helper records pointer/allocation
relationships and checks pointer spans against CUDA allocations. It does not
infer logical shapes, strides or argument count; save these from the wrapper.
The current helper is intended for pure kernels with distinct input/output roles.

Require a capture record, compare the instrumented full-model output against the
same approved reference, then reproduce the extracted kernel against the saved
boundary output before using it in a KDA search. Preserve the package and helper
hashes. A capture that fails to observe the exact target is an error, not evidence
that the kernel did not execute.

External library kernels may pass an opaque parameter structure to the CUDA
driver, so the pointer/scalar interceptor is not appropriate without a proven ABI.
For supported FP32 BMM calls, `csrc/capture_aten_bmm.cpp` intercepts
`aoti_torch_cuda_bmm_out`. Verify this symbol/ABI in the supplied runtime before use.
It obtains shape, stride, dtype and pointers through AOTI accessors, validates
allocation spans and captures a reviewed number of matching calls in execution
order. Supply a JSON specification with `kind: "aten_bmm"`, `library`, `shapes`
(a list of `[batch, M, K, N]` rows), `expected_cases` and `max_tensor_bytes`.
Every selector and limit is explicit; no model shape or call count is built in.
The byte limit applies to each captured tensor, including gaps in strided storage.
Only positive shapes, nonnegative strides and FP32 are supported. The worker pins
one physical device to visible CUDA device zero.

The diagnostic performs one complete forward and fails if the expected number of
cases is not captured. It captures the first matching calls up to that count;
review call order and metadata to establish which operators they represent. It
does not prove that no additional same-shape calls exist. Equal shapes alone are
not operator identity. Each case retains input/output pointers and allocation
metadata; full-model reference validation remains required. Capture helpers do
not change the original math but synchronization/host copies invalidate timing.
Keep large binaries at the execution location if appropriate and synchronize
metadata and SHA256 records for review.

## Template integration checks

An MM-shaped generated body may originate from AddMM after graph rewriting. Check
the package's post-gradient provenance nodes before choosing a lowering hook.
Preserve bias, alpha/beta handling and supported prologue/epilogue fusion in that
actual lowering, then verify the selected source in the new package.

Small isolated floating-point errors do not guarantee whole-model equivalence.
When the model reference is strict, test candidates that retain the original K
grouping and accumulator semantics. Inspect the installed backend's actual shape
limits; a tuning heuristic's tile sizes are not necessarily compiler minimums.
Keep tolerance-based candidates separate from candidates that pass bitwise checks.

TF32 permission alone does not specify identical rounding, reduction grouping
or accumulation order. Inspect actual generated instructions and target support,
and test the captured reference before claiming numerical equivalence. Do not
transfer an instruction/architecture-specific recipe to an unverified device.

Compare extracted sources and launch parameters for every non-target kernel.
Persistent GEMM timings and content-addressed fusion timing memos reduce unrelated
selection changes, but copying them is not proof that the resulting choices match.
Keep failed compilations and source comparisons as part of the outer loop.

Adding a template can renumber later generated kernels even when their bodies
are unchanged. In that case, filename-based configuration replay alone misses
those kernels. `scripts/rebind_autotune_configs.py` can derive a new replay
manifest from the candidate cache: it compares the **complete module AST**,
including decorators and precision/device metadata, after normalizing only one
Triton kernel name and its references. It rejects ambiguous baseline configs,
records both source hashes and the normalized AST hash, and copies only the
baseline configuration record. Normal `configs_hash` validation remains active.
Recompile with the derived manifest, then audit actual launch helpers and model
outputs again. A name-only match is not sufficient evidence of unchanged code.

For deeper graph and compiler investigation, use [compiler analysis](compiler-analysis.md).
