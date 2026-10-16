# Metal compiler crash in surface classification

Investigation: 2026-10-16, base commit `72d4e01`.

## Finding

The Ganesha GPU failure occurs while compiling `classify_surface_scatter`,
before any rendering dispatch. On this machine, changing the material-kind
`if / else if` chain to `switch` avoids the failure while retaining the same
material cases, scattering queues, direct-light eligibility tests, ray-origin
calculation, and direct-light queue append. The shader's resource bindings and
stage boundaries remain the same.

This is evidence of a Metal compiler control-flow optimization bug. It does
not establish the exact faulty optimizer transformation. In particular, the
nested branch structure and its merged `supports_direct` value form a
reproducible trigger; that does not mean mutable booleans or storage writes
are generally unsupported.

The failure is separate from the earlier 29-buffer limit. A standalone wgpu
program reproduces it with only 14 storage buffers and 3 uniform buffers. That
program requests no optional features, loads no scene, creates no rendering
buffers, and never dispatches the shader. Changing the dispatch to `switch`
also fixes that reproduction.

## Environment and failure

- Apple M3, Metal backend.
- macOS 26.5.1, build 25F80.
- `wgpu` 30.0.0; `wgpu-core`, `wgpu-hal`, and `naga` 30.0.1.
- API failure: `new_compute_pipeline_state`,
  `XPC_ERROR_CONNECTION_INTERRUPTED` after retries.
- `MTLCompilerService` diagnostics: `EXC_BAD_ACCESS` at address `0x8`;
  the failing stack includes `_platform_memmove` and
  `llvm::DominatorTreeBase<llvm::MachineBasicBlock, false>::eraseNode`,
  followed by LLVM machine-function passes and the AGX compile plan.

The internal LLVM crash strongly locates the failure in Apple's AGX backend.
The investigation does not determine whether a different Naga lowering could
also avoid it. Removing `VALIDATION_INDIRECT_CALL` did not avoid the original
failure.

## Controlled comparisons

All compile-only comparisons use the same wgpu version and device. The full
classifier candidates retain its queue writes and direct-light processing,
unless the table explicitly says otherwise.

| Classifier variant | Result |
| --- | --- |
| Original nested material branches and merged boolean | Metal compiler crash |
| Final direct-light conjunction expanded into early returns | Same crash |
| Boolean replaced by a `u32` eligibility flag | Same crash |
| Ineligible cases return from their material branches, eliminating the flag | Compiles; Ganesha 1 spp renders |
| Material dispatch changed to `switch`, retaining the boolean | Compiles; Ganesha 1 and 16 spp render |
| Reduced standalone WGSL with original material branches | Same crash |
| Reduced standalone WGSL with `switch` | Compiles |

Earlier partial variants that removed the origin calculation or some writes
ran for tens of seconds inside the compiler before being stopped. Those runs
are inconclusive about whether compilation would eventually complete and do
not establish that any individual origin calculation or write is faulty.

An even smaller synthetic shader with a similar branch pattern compiled
successfully. The failure therefore depends on more than the branch pattern
alone. The reduced shader retains the relevant work-item types and helpers;
it is not a minimal reproducer.

## Rendering verification

The `switch` candidate completed all nine tiles of the original Ganesha scene
with `--gpu --spp 16`, producing a 1440 by 1440 EXR. Its preview was inspected.
A separate 1 spp render was compared against the independently working
branch-local-return candidate:

- RGB RMS error: `2.3206e-08`.
- Maximum absolute channel error: `5.6877732e-05`.
- Two pixels out of 2,073,600 exceeded `1e-06`.

This comparison is close but not bit-identical. The failing original shader
cannot provide a rendered reference on this machine, and this investigation
has not performed a CPU/GPU image agreement test or a performance benchmark.

## Local investigation artifacts

The self-contained reproduction project is saved in
`/tmp/pbrt-metal-investigation/repro/`, with a Cargo lockfile, WGSL failure and
success variants, and instructions. The parent directory contains the
compile logs, original/composed shaders, diagnostic source, and rendered EXRs.
These are local temporary artifacts and are not required by the renderer.

The production workaround is limited to the material dispatch in
`src/gpu/webgpu/shaders/classify_surface_scatter.wgsl`. Keep its `switch`
structure: restoring the equivalent nested chain reproduces the driver bug
on the tested environment. No upstream report has been submitted.
