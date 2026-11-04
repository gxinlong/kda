# From a model hotspot to a structural kernel change

Rank actual summed GPU time first, including external-library calls. Preserve
original baseline and incumbent separately. Capture real boundary operands and
downstream uses before entering the [KDA loop](../../docs/agent-flow.md).

## Diagnose before implementing

Choose a concrete question for a profiler: bytes moved, instruction issue,
occupancy/resources, dependency waits, synchronization or insufficient parallelism.
Record the supplied hardware/tool versions, exact kernel/shape, cache policy and
clock policy. Use NCU with `--clock-control none` when the task does not authorize
clock changes. Keep raw reports and source/PC correlation. Profiling replay time
is not acceptance timing. A wave count alone does not establish a dominant tail;
different tensor-pipeline activity metrics are not interchangeable.

| Evidence to investigate | Structural hypothesis | Evidence required after the change |
|---|---|---|
| High read traffic and bandwidth utilization with operand-load waits | Reuse operands across rows/projections, change traversal/layout, reduce intermediates | Lower measured bytes/latency at equivalent work; include any transforms |
| Low occupancy due to registers/shared memory | Shorten live ranges, change pipeline stages or accumulator organization | Actual generated resources, resident work, spills and model timing |
| Underfilled device or long final wave | Repartition work or test split/persistent scheduling | Concurrency/tail improvement including reduction/synchronization cost |
| Repeated intermediate writes and launches | Fuse a producer/consumer or epilogue with supported semantics | Preserved boundary semantics, removed traffic/launches, whole-model gain |
| Layout or packing overhead | Specialize loading or explicitly amortize packing | Packing cost and valid weight-update/ABI contract |

Externalized weights remain runtime inputs unless the contract guarantees
immutability. Never cache a packed representation by pointer identity alone.
Include packing cost when required and test in-place weight changes. Sharing an
operand between projections is only useful when the actual graph and layouts
permit it. Preserve side effects and independently consumed intermediate outputs.

## Match the prototype to the generated candidate

A static microbenchmark can screen ideas, but generated symbolic scalars, address
arithmetic, launch metadata, integer types, precision and epilogue can change
resource usage. Compare registers, spills, shared memory and PTX/SASS. A small
register increase can cross an occupancy boundary; register caps can introduce
spills. Measure the complete tradeoff instead of applying a universal cap.

Follow compiler-option forwarding in the installed version. For example, a cap
may require `triton_meta.backend_options` rather than a top-level field. Assert the
actual Triton compiler JSON contains the intended option. A generated decorator
is not evidence that the backend used it. Keep all arithmetic options and
qualifiers when adapting an extracted kernel to a standalone runner.

For direct wrapper benchmarks, obtain the current CUDA stream inside each launch.
Saving a stream before entering Graph capture can leave work outside the measured
Graph. Poison outputs before validation, verify replay actually executes the target,
and sanity-check launch count and timing.

TF32 permission does not specify rounding mode, K grouping or accumulator order.
Inspect the generated arithmetic and the target backend's supported instructions;
validate captured data and model outputs. Do not copy a conversion instruction or
architecture constraint from another task. Keep bitwise comparisons distinct from
tolerance-based validation. Backend algorithm choice can change with shape.

Test captured inputs, masked tails, supported aliases, input/weight mutations and
negative guard matches. Preserve failures and state validated input scope. One
successful example does not establish all dynamic shapes or business accuracy.

## Accept in the model

Preserve unrelated code and launch settings through controlled source-bound
configuration replay when needed. Kernel renumbering can invalidate path-only
replay. Use complete-module identity for rebinds and audit actual launch helpers;
never use the audit's looser display normalization to authorize cache reuse.

Compare the fresh package against the original baseline and incumbent with
`compare_models.py --incumbent-run`. Keep profilers out of acceptance timing and
profile separately to explain the result. Report pure-kernel, fused-region and
model timing with their respective boundaries. Removing launches or improving a
microbenchmark does not justify promotion when complete-model performance loses.
