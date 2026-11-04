# Export, compiler and generated-code analysis

Use this guide when final code and wrapper inspection cannot establish a kernel's
origin or integration point. Analyze the exact artifact first; a new compilation
is a separate experiment and may change fusion or selection.

## Inspect the model without executing it

`inspect_model.py` distinguishes serialized ExportedPrograms from AOT packages and
records archive/companion identities. `analyze_graph.py` extracts graph scopes,
ordered input/output ports, node metadata, typed value edges, tensor/symbol
metadata, constraints, graph signature and module-call information using only the
standard library. It supports the inspected export JSON schema; review schema
version and unresolved references before interpreting results. No weight loading
is required. An AOT binary alone cannot restore missing export provenance.

Read the signature to distinguish parameters, buffers, user inputs, constants,
mutations and outputs. Preserve TreeSpec and positional/keyword ordering. Follow
value references, not adjacency in serialized node arrays. Keep nested control
flow graphs in separate scopes and preserve multi-output ports. Tensor shape and
stride metadata do not establish storage aliases, offsets or lifetimes.

Use `nn_module_stack`, `stack_trace` and module-call metadata when present to map
operators to model components. Derive module boundaries from values crossing them,
including shared values and multiple outputs. Absence of metadata means missing
provenance, not absence of a source module. Do not assume a matrix batch dimension
is the request batch; inspect lengths, indices, masks and dynamic relationships.

Report counts with their stage: serialized nodes, FX nodes, scheduler nodes,
generated definitions, wrapper call sites and executed GPU launches are different
quantities. One external operator can launch several kernels; one generated
kernel can contain multiple logical regions or be called several times.

## Follow the compiler in the supplied checkout

Conceptual flow:

```text
ExportedProgram / preparation / decomposition
  -> AOTAutograd and Inductor FX passes
  -> GraphLowering: buffers, loops and template choices
  -> Scheduler: dependencies, legality, fusion and scheduling
  -> kernel and wrapper code generation / autotune selection
  -> AOT package -> runtime launches
```

The stages are not a strict chronological trace: construction can perform work,
selection may be deferred, and caches may bypass steps. In supported checkouts,
`GraphLowering.codegen` constructs a Scheduler, then runs scheduler codegen and
wrapper generation; inspect the actual implementation before placing probes.

| Question | Starting point in the supplied PyTorch source |
|---|---|
| Archive schema and serialization | `torch/_export/serde/schema.py`, serialization code, `torch/export/pt2_archive` |
| Preparation and FX passes | `torch/_inductor/compile_fx.py`, `fx_passes/` |
| Lowered buffers, layouts and nodes | `graph.py`, `lowering.py`, `ir.py` |
| Dependencies, fusion legality and profitability | `scheduler.py`, including initialization, dependency computation, `can_fuse` and `speedup_by_fusion` |
| MM/BMM/AddMM choices and selection | `kernel/`, `select_algorithm.py`, relevant template implementations |
| Generated loops and fusion groups | `codegen/simd.py`, Triton and combo codegen |
| Allocation, argument mapping and lifetimes | Wrapper codegen and the exact generated C++ wrapper |

Trace the actual lowering, not just the generated operation's apparent shape. An
MM-like body may originate from AddMM after rewriting; preserve bias, alpha/beta,
prologue and epilogue semantics at that hook. `MultiTemplateBuffer` can represent
several implementation choices for one operation; it does not itself mean several
GEMMs were fused. Distinguish producer-consumer fusion, prologue/epilogue fusion,
combo dispatch, graph batching, CSE and CUDA Graph replay.

Separate three claims: fusion is legal, fusion was considered, fusion was selected.
Current source proves available conditions; only diagnostics from that compile
can establish why a historical candidate was rejected.

## Collect additional evidence only when needed

Check the supplied version's support before enabling diagnostics such as:

```sh
TORCH_TRACE="$TRACE_DIR"
TORCH_LOGS="+inductor,fusion,schedule,ir_pre_fusion,ir_post_fusion,output_code"
INDUCTOR_PROVENANCE=1
```

Set these in a recorded diagnostic compilation environment, not an acceptance
measurement. If available, use `tlparse "$TRACE_DIR" -o "$TRACE_REPORT" --latest`.
Retain cache-hit information, pre/post-fusion IR, transformed FX, output code,
selection logs and source mapping. A trace cannot recreate an earlier compile's
missing history. Before attributing an instrumented graph to the baseline, compare
generated boundaries, shapes and launch metadata.

## Map an observed kernel back to the model

Follow measured event -> exact package wrapper call -> argument/view/offset and
lifetime -> generated code -> template/lowered IR -> FX/export nodes -> module.
Record package and source hashes at every boundary. Nsight row IDs, correlation
IDs and Graph-node IDs have distinct meanings and are not stable across reports.
Chronological position is supporting evidence only with verified stable
single-stream order; mixed streams invalidate adjacency-based mapping.

Read pointer-offset units at the API boundary before translating into element
indices. Check partial writes into concatenated buffers and reused storage across
lifetimes. Empty `Source Nodes` comments and fused names are not semantic proofs.
Capture true boundary tensors when necessary; see [provenance](provenance.md).

A runnable extracted FX subgraph needs exact boundary values, constants, symbolic
integers, alias/mutation behavior and side effects. Same-shape random inputs are
only a limited synthetic test. Recompiling the subgraph may select different
kernels, so its standalone performance is not evidence of the packaged path.
