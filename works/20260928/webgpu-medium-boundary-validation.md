# WebGPU homogeneous medium boundary check

## Setup

- Scene: `tests/scenes/crown-step6-gem-media-fast16.pbrt` (128×128); a temporary copy replaced the sapphire shape's `NamedMaterial` with `Material ""` while keeping its `MediumInterface`.
- Integrator: `volpath`; pbrt-r4 CPU and GPU used `--spp 8 --maxdepth 2`; pbrt-v4 used the same scene settings and `--spp 8`.
- pbrt-r4 base: `4b4d1dcd215e48e8d3ba2a27ea21e6b83873cb76` (`origin/feature/gpu`), with the current uncommitted implementation.
- pbrt-v4: `cdccb71cb1e153b63e538f624efcc13ab0f9bda2`.
- Output EXRs and the temporary scene were kept under `/tmp`, outside the repository.

## Measurements

`imgtool diff --metric MAE` over the full frame:

| Comparison | R | G | B | Overall MAE |
|---|---:|---:|---:|---:|
| pbrt-r4 GPU vs pbrt-v4 | 0.028148 | 0.023449 | 0.023924 | 0.025174 |
| pbrt-r4 CPU vs pbrt-v4 | 0.005839 | 0.004492 | 0.003906 | 0.004745 |

The same interface-only scene completed on the GPU. These low-sample measurements are not a pass criterion; the GPU-v4 difference has not been attributed to a specific implementation difference yet.

## Crown scene check

With the original material-bearing crown scene at 128×128, 16 spp, `maxdepth=32`, overall MAE was 0.029572 for pbrt-r4 GPU vs CPU, 0.028617 for pbrt-r4 GPU vs pbrt-v4, and 0.015002 for pbrt-r4 CPU vs pbrt-v4. These values are measurements only and do not establish image equivalence.
