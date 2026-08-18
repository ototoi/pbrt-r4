# pbrt-r4
[![Rust](https://github.com/ototoi/pbrt-r4/actions/workflows/rust.yml/badge.svg)](https://github.com/ototoi/pbrt-r4/actions/workflows/rust.yml)
[![License](https://img.shields.io/github/license/ototoi/pbrt-r4)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ototoi/pbrt-r4)](https://github.com/ototoi/pbrt-r4/releases/latest)
[![Crates.io Version](https://img.shields.io/crates/v/pbrt-r4?color=%20%23ecc57b)
](https://crates.io/crates/pbrt-r4)

![sportscar-area-lights rendered with pbrt-r4](https://github.com/user-attachments/assets/14b19b8f-46eb-4d53-9280-69291c24e306)

## What is pbrt-r4

pbrt-r4 is a Rust implementation of [pbrt-v4](https://github.com/mmp/pbrt-v4),
evolved from a pbrt-r3 (pbrt-v3 Rust port) foundation. It is a CPU renderer
whose primary compatibility target is pbrt-v4 behavior and input files.

The codebase now mirrors pbrt-v4's class structure on every major axis — `Spectrum` / `SampledSpectrum` are 4-wavelength packets, `BSDF`/`BxDF` carry per-path `SampledSpectrum` reflectance, the `Light` trait matches pbrt-v4's `SampleLi` / `PDF_Li` / `SampleLe` / `Phi` shapes, and integrators (Path / VolPath / SPPM / BDPT / MLT / LightPath / SimpleVolPath / RandomWalk / SimplePath / AO / Function) translate their pbrt-v4 counterparts line-by-line. pbrt-v4 scene files render through the same pipeline; legacy pbrt-v3 input should first be upgraded to pbrt-v4 format.

A broad set of the official pbrt-v4 scenes is renderable, but pixel-perfect
image equality, Monte Carlo noise, render time, and memory usage are not
guaranteed yet.

## License
pbrt-r4 is distributed under [the Apache License 2.0](LICENSE), matching the license of the upstream [pbrt-v4](https://github.com/mmp/pbrt-v4/blob/master/LICENSE.txt).

## Build

From the pbrt-r4 repository root:

```sh
cargo build --release
cargo test
```

## How to use

Render a scene with the following command:

```sh
./target/release/pbrt-r4 \
  --pixelsamples 64 \
  --nthreads 16 \
  --outfile /tmp/output.exr \
  <example.pbrt>
```

Useful options include `--pixelsamples`, `--nthreads`, `--outfile`, and
`--display-server`. The `--quick` option is intended for fast
visual checks and changes the rendering configuration; do not use it for a
controlled comparison with pbrt-v4.

## Tev display
pbrt-r4 supports the [tev](https://github.com/Tom94/tev) display implemented
in pbrt-v4. After starting tev, display rendering progress with:

```sh
./target/release/pbrt-r4 \
  --display-server localhost:14158 \
  <example.pbrt>
```

## Build Options
pbrt-r4 has several options as Rust features.
| Feature | Description |
|---------|-------------|
| `stats` | Enables the pbrt-v4 `pixelstats` scene option. |
| `float-as-double` | Uses double precision (64-bit) for floating-point calculations. This increases precision but also increases execution time and memory usage. |

## Example scenes
pbrt-r4 takes pbrt-v3 and pbrt-v4 scene files as input. The compatibility
comparison set used by this project comes from the official
[pbrt-v4-scenes repository](https://github.com/mmp/pbrt-v4-scenes).

Legacy pbrt-v3 input should be converted before rendering:

```sh
./target/release/pbrt-r4 \
  --upgrade \
  --outfile /tmp/scene-v4.pbrt \
  path/to/scene-v3.pbrt

./target/release/pbrt-r4 \
  --pixelsamples 64 \
  --outfile /tmp/scene.exr \
  /tmp/scene-v4.pbrt
```

`--upgrade` writes the converted pbrt-v4 input and does not render an image.

## Implementation status

The tables below describe the current implementation scope, not a claim of
complete pixel-level parity for every feature combination. Known limitations
include GPU/wavefront rendering, PTex, medium vertices in BDPT/MLT, complete
volumetric BSSRDF parity, photometric film calibration, and the standalone
pbrt-v4 command-line utilities.

### Integrators (`Integrator "<name>" ...`)
| Name | pbrt-v4 reference | Status in pbrt-r4 |
|------|-------------------|-------------------|
| `path`           | `PathIntegrator`           | Full surface MIS path tracer. |
| `simplepath`     | `SimplePathIntegrator`     | Toggleable `samplelights` / `samplebsdf` arms (no MIS). |
| `volpath`        | `VolPathIntegrator`        | Volumetric path tracer with `r_u`/`r_l` rescaled-density MIS; BSSRDF subsurface scattering deferred. |
| `simplevolpath`  | `SimpleVolPathIntegrator`  | Delta-tracking volume path tracer, no surface scattering. |
| `lightpath`      | `LightPathIntegrator`      | Light tracing with `Camera::SampleWi` splat. |
| `sppm`           | `SPPMIntegrator`           | Stochastic Progressive Photon Mapping. |
| `bdpt`           | `BDPTIntegrator`           | Bidirectional path tracing (surface-only; medium vertices deferred). |
| `mlt`            | `MLTIntegrator`            | Primary-sample-space MLT on top of BDPT (single-threaded; medium vertices deferred). |
| `randomwalk`     | `RandomWalkIntegrator`     | Minimal uniform-sphere random walk (no MIS / RR / direct lighting). |
| `ambientocclusion` | `AmbientOcclusionIntegrator` | Single AO sample per path. |
| `function`       | `FunctionIntegrator`       | r4 extension (formerly `aov`) — surface attribute outputs: position, normal, uv, depth, etc. |

### BxDFs (`Material "<name>" ...`)
| Material | pbrt-v4 reference | Notes |
|---|---|---|
| `diffuse` / `diffusetransmission` | `DiffuseBxDF`, `DiffuseTransmissionBxDF` | Per-path `SampledSpectrum` reflectance / transmittance. |
| `dielectric` / `thindielectric` | `DielectricBxDF`, `ThinDielectricBxDF` | Smooth + rough dielectric, full Fresnel + microfacet. |
| `conductor` | `ConductorBxDF` | Smooth + rough metal. |
| `coateddiffuse` / `coatedconductor` | `LayeredBxDF<Dielectric, Diffuse, ...>` | Pbrt-v4 layered model with random-walk through coat. |
| `measured` | `MeasuredBxDF` | RGL `.bsdf` tensor format reader (`PiecewiseLinear2D` NDF/VNDF/sigma/spectra). |
| `hair` | `HairBxDF` | Marschner three-lobe hair scattering. |
| `mix` | composite | Mix material via index/probability. |
| `subsurface` / `kdsubsurface` | `SubsurfaceMaterial` | `NormalizedFresnelBxDF` surface + tabulated BSSRDF. |
| `interface` | — | r4 helper material that exposes a stored `BxDF`. |

### Lights
`point`, `distant`, `spot`, `goniometric`, `projection`, `infinite` (uniform), `infinite` with `mapname` (image-based), and `diffuse` (area). Each is translated verbatim from `lights.h:189–625`. The light's `scale` parameter is a `Float` (pbrt-v4 shape); the `SpectrumToPhotometric` normalization is deferred until the film's photometric calibration lands.

### Films / Samplers / Accelerators
- **Films**: `rgb` (`RGBFilm`), `gbuffer` (`GBufferFilm`), `spectral` (`SpectralFilm` with Fichet et al. spectral-EXR layout).
- **Samplers**: `independent`, `stratified`, `halton`, `02sequence`, `sobol`, `zsobol`, `paddedsobol`, `pmj02bn`, plus the internal `MLTSampler` driven by the `mlt` integrator.
- **Accelerators**: `bvh` (default; r4-specific QBVH SIMD aggregator), `kdtree`, `exhaustive`.

## Differences from pbrt-v4

### Relationship to upstream pbrt
pbrt-r4 began as a Rust port of pbrt-v3 (pbrt-r3) and has been progressively rewritten so its public class structure now matches pbrt-v4. Code paths are translated from pbrt-v4 line-by-line wherever practical; remaining pbrt-r3 inheritances are explicitly called out in source comments.

### Implementation choices
- **Language**: Rust instead of C++. Ownership / lifetimes replace manual memory management. No `unsafe` outside SIMD intrinsics and a small set of FFI / raw-pointer helpers.
- **Class inheritance**: pbrt-v4 inheritance becomes composition — a Rust struct holds the parent struct as a `base` field. pbrt-v4 `TaggedPointer<...>` polymorphism becomes Rust `enum`s with the same variant names.
- **Parallel execution**: pbrt-v4's `ParallelFor` / hand-rolled thread pool becomes [`rayon`](https://crates.io/crates/rayon). The QBVH aggregator uses SSE / NEON intrinsics directly.
- **Parser**: pbrt-v4's hand-written recursive-descent parser becomes the [`nom`](https://crates.io/crates/nom) crate.
- **Progress / CLI / logging**: [`indicatif`](https://crates.io/crates/indicatif), [`clap`](https://crates.io/crates/clap), [`log`](https://crates.io/crates/log) + [`env_logger`](https://crates.io/crates/env_logger).
- **Other crates**: `image` (EXR/PNG/etc.), `ply-rs` (PLY mesh I/O), `rust-crypto` (hashing), `serde` + `serde_json` (scene JSON), `flate2` (`.ply.gz` decompression).
- **AtomicDouble → Mutex/tile aggregation**: pbrt-v4's `AtomicDouble` splat accumulators are replaced by a per-tile `SplatTile` that merges into the per-pixel splat buffer at the end of rendering. The same pattern is used for the photon-pass accumulators in SPPM.

### Additional features
- **Tev display**: real-time progressive display through the [tev](https://github.com/Tom94/tev) viewer.
- **QBVH accelerator**: an r4-original quad-BVH aggregator using SSE intrinsics. v4-faithful in that it now passes the unclamped `t_max` to per-primitive intersect (fixes flat-AABB area-light hit loss).
- **Function (AOV) integrator**: render scene attributes (position, normal, uv, depth, ...) for compositing or debugging.

### Post-release scope

The following areas are intentionally outside the initial release scope:

- **GPU / wavefront rendering**: pbrt-v4's CUDA / OptiX backend is out of scope. pbrt-r4 is CPU-only.
- **PTex textures**: not implemented.
- **pbrt-v4 CLI utilities (`imgtool` etc.)**: pbrt-v4's stand-alone tools under `src/pbrt/cmd` are unported; priority is low.

### r4-specific fixes
- Negative `pdf` from numerical drift is clamped (`pdf.max(0.0)`).
- `shading.dpdu` / `shading.dpdv` orientation fix (see [#97](https://github.com/ototoi/pbrt-r4/issues/97)).
- QBVH SIMD aggregator: pass the original `t_max` (not the bbox-clamped one) to per-primitive intersect, fixing a 20% hit-loss for flat AABB area lights (see [#29](https://github.com/ototoi/pbrt-r4/pull/29)).

## Development and compatibility

Behavioral changes are developed against the local pbrt-v4 implementation.
When a v4 path is not implemented, it must not be silently replaced by an
unrelated fallback merely to keep a scene rendering. Compatibility gaps should
be documented, covered by focused tests, and resolved by implementing the
corresponding v4 behavior or by making the limitation explicit.

Porting and compatibility decisions should be documented alongside the
corresponding implementation and tests.

## Future plans

**Near term**
- Parallelize `mlt` chains and `sppm` photon pass via `rayon`.
- Continue to harden pbrt-v4-scenes rendering and investigate high memory usage on large `.ply` scenes.

**Mid term**
- Benchmarking against pbrt-v3 / pbrt-v4 to guide optimization (current pbrt-r4 is typically slower than the C++ reference; we want a public benchmark suite to track this).
- Comprehensive `cargo doc` API surface and rustdoc examples.

**Long term**
- Port pbrt-v4 comment text to the corresponding pbrt-r4 functions to honour the original implementation.

## Acknowledgments
Thanks to Matt Pharr, Wenzel Jakob, and Greg Humphreys — the authors of pbrt-v3 and pbrt-v4 — for releasing their reference implementations. Thanks also to the Rust community and the maintainers of the crates listed above.
