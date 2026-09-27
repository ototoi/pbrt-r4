# WebGPU homogeneous medium boundary check

## Setup

- Scene: `tests/scenes/crown-step6-gem-media-fast16.pbrt` (128×128); a temporary copy changed only line 122 from `NamedMaterial "saphire"` to `Material ""` while keeping `MediumInterface "sapphire-medium" ""`.
- Integrator: `volpath`; pbrt-r4 CPU and GPU used `--spp 8 --maxdepth 2`; pbrt-v4 used the same scene settings and `--spp 8`.
- pbrt-r4 base: `4b4d1dcd215e48e8d3ba2a27ea21e6b83873cb76` (`origin/feature/gpu`); the first comparison used the implementation later committed as `009b988`.
- pbrt-v4: `cdccb71cb1e153b63e538f624efcc13ab0f9bda2`.
- Output EXRs and the temporary scene were kept under `/tmp`, outside the repository.

## Measurements

`imgtool diff --metric MAE` over the full frame:

| Comparison | R | G | B | Overall MAE |
|---|---:|---:|---:|---:|
| pbrt-r4 GPU vs pbrt-v4 | 0.028148 | 0.023449 | 0.023924 | 0.025174 |
| pbrt-r4 CPU vs pbrt-v4 | 0.005839 | 0.004492 | 0.003906 | 0.004745 |

The same interface-only scene completed on the GPU. These low-sample measurements are not a pass criterion; the GPU-v4 difference has not been attributed to a specific implementation difference yet.

## Active segment queue follow-up

The follow-up to `009b988` sends only boundary continuations into the next active index queue. Intersection and Medium sampling now read that queue, and escaped rays are classified after Medium sampling. A 128×128, 1 spp, maxdepth 2 GPU render of the same temporary interface-only crown scene completed. Its full-frame MAE against the earlier GPU render was 0.000178; this measures the code change, not agreement with v4.

The ignored GPU integration tests in `tests/gpu_medium_segment_integration.rs` were run on the available ray-query adapter. The 8×8, 1 spp absorption scene uses a camera inside homogeneous medium and adds a transparent boundary over only half the image: pixels that do not cross the boundary matched the no-boundary render within 1e-6, while the affected half changed. A second 8×8, 1 spp scene inserted a material-less boundary between a diffuse surface and point light; its output matched the no-boundary render within 1e-5. Both tests passed. These inputs are embedded in the test file so the comparison is reproducible.

## Crown scene check

With the original material-bearing crown scene at 128×128, 16 spp, `maxdepth=32`, overall MAE was 0.029572 for pbrt-r4 GPU vs CPU, 0.028617 for pbrt-r4 GPU vs pbrt-v4, and 0.015002 for pbrt-r4 CPU vs pbrt-v4. These values are measurements only and do not establish image equivalence.
