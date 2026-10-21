# pbrt-r4
[![Rust](https://github.com/ototoi/pbrt-r4/actions/workflows/rust.yml/badge.svg)](https://github.com/ototoi/pbrt-r4/actions/workflows/rust.yml)
[![License](https://img.shields.io/github/license/ototoi/pbrt-r4)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ototoi/pbrt-r4)](https://github.com/ototoi/pbrt-r4/releases/latest)
[![Crates.io Version](https://img.shields.io/crates/v/pbrt-r4?color=%20%23ecc57b)
](https://crates.io/crates/pbrt-r4)

![sportscar-area-lights rendered with pbrt-r4](https://github.com/user-attachments/assets/14b19b8f-46eb-4d53-9280-69291c24e306)

## What is pbrt-r4

pbrt-r4 is a Rust implementation of [pbrt-v4](https://github.com/mmp/pbrt-v4),
evolved from a pbrt-r3 (pbrt-v3 Rust port) foundation. It provides CPU and
WebGPU renderers. Its primary compatibility target is
pbrt-v4 behavior and input files.

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

## Installation

Prebuilt binaries for Windows x86_64, macOS Apple Silicon, and macOS Intel are
available from [GitHub Releases](https://github.com/ototoi/pbrt-r4/releases/latest),
along with `SHA256SUMS` for verifying the downloads. Each archive contains the
renderer and the five command-line utilities listed below.

Install the published crate from crates.io with:

```sh
cargo install pbrt-r4
```

To install all command-line utilities in addition to the renderer, use:

```sh
cargo install pbrt-r4 --bins
```

WebGPU support is enabled by default. For a CPU-only installation, use
`cargo install pbrt-r4 --no-default-features`.

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
| `webgpu` | Enables the WebGPU renderer. Enabled by default. |

## GPU rendering

The default build enables WebGPU rendering. To build a CPU-only renderer, disable
default features:

```sh
cargo build --release --no-default-features
```

Render with WebGPU using the default build:

```sh
cargo build --release
./target/release/pbrt-r4 \
  --gpu \
  --spp 64 \
  --outfile /tmp/output.exr \
  <example.pbrt>
```

`--wavefront` also selects the GPU renderer. The GPU path currently supports
the `path` and `volpath` scene integrators. If a scene specifies another
integrator, such as `bdpt`, `mlt`, or `sppm`, the GPU renderer warns and uses
`volpath`, retaining the scene's `maxdepth` and `lightsampler` settings. This
matches pbrt-v4's GPU integrator selection behavior.

`--gpu-tile-size` controls the square tile size and defaults to 512 pixels.
Reduce it to limit memory used by per-pixel rendering buffers. Scene geometry,
textures, and acceleration structures must still fit in available GPU memory.

The WebGPU backend requires an adapter that provides experimental ray-query
support and the resource limits required by the scene. Cargo enables Vulkan on
Linux, Direct3D 12 or Vulkan on Windows, and Metal on macOS. Project CI currently
runs build and test checks on Linux and release builds on Windows x86_64,
macOS Apple Silicon, and macOS Intel. GPU runtime checks have been performed
on Linux/Vulkan and Windows; CI builds do not validate rendering on GPU devices.

### Large scenes

Since v4.5.0, the WebGPU backend splits scene geometry into pages sized to
the adapter's buffer limits. This allows total geometry to exceed the limit
of a single buffer. Each geometry must fit within one page; an oversized
geometry produces an error.

Single-page scenes use ordinary vertex/index storage bindings. Multi-page
scenes require the adapter's `BUFFER_BINDING_ARRAY` feature, and the maximum
page count depends on the adapter's binding limits and the scene's other
resources. Adapters without the required features or limits report an error.
Paging does not reduce the total GPU memory required by the scene.

Kroken `camera-1.pbrt` from the official pbrt-v4 scenes was rendered at its
full 1500×1500 resolution and 256 samples per pixel on an NVIDIA RTX 3060
(12 GiB) using Linux/Vulkan. Its geometry occupied three pages. The render
used 128-pixel tiles:

```sh
./target/release/pbrt-r4 \
  --gpu --spp 256 --gpu-tile-size 128 --nthreads 4 \
  --outfile /tmp/kroken.exr \
  path/to/pbrt-v4-scenes/kroken/camera-1.pbrt
```

### Resource diagnostics

To inspect per-stage buffer requirements and adapter headroom, enable the
WebGPU debug log:

```sh
RUST_LOG=pbrt_r4::gpu::webgpu=debug ./target/release/pbrt-r4 --gpu <example.pbrt>
```

The binding report lists storage buffers, uniform buffers, acceleration
structures, and resource names for each composed wavefront stage. Texture and
sampler bindings are listed separately. If adapter buffer limits are exceeded,
the error includes the over-budget stages even without debug logging. Counts
are binding counts, not buffer sizes in bytes. The report identifies its noise
and NanoVDB variants and covers wavefront stages; the separately constructed
BSSRDF probe is outside its scope.

## Example scenes
pbrt-r4 renders pbrt-v4 scene files directly. The compatibility comparison
set used by this project comes from the official
[pbrt-v4-scenes repository](https://github.com/mmp/pbrt-v4-scenes).

Legacy pbrt-v3 input must be converted before rendering:

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

## Command-line utilities

The pbrt-v4 CPU command-line utilities are built as separate binaries:

| Binary | Purpose |
|--------|---------|
| `imgtool` | Inspect, convert, compare, and post-process images. |
| `plytool` | Inspect, displace, concatenate, and split triangle PLY meshes. |
| `nanovdb2pbrt` | Convert a NanoVDB grid into pbrt input. |
| `cyhair2pbrt` | Convert CyHair data into pbrt input. |
| `pspec` | Compute sampler power-spectrum data. |

For example, after a release build:

```sh
./target/release/imgtool help
./target/release/plytool help
./target/release/nanovdb2pbrt --help
```

`cyhair2pbrt` and `pspec` print their usage when invoked without the required
input. These utilities are CPU implementations. GPU-only utilities such as
`pspec_gpu` and `denoise-optix` are not included.

## Implementation status

The tables below describe the current implementation scope, not a claim of
complete pixel-level parity for every feature combination. Known limitations
include partial GPU feature support, PTex, and photometric film calibration.

### Integrators (`Integrator "<name>" ...`)
| Name | pbrt-v4 reference | Status in pbrt-r4 |
|------|-------------------|-------------------|
| `path`           | `PathIntegrator`           | Full surface MIS path tracer. |
| `simplepath`     | `SimplePathIntegrator`     | Toggleable `samplelights` / `samplebsdf` arms (no MIS). |
| `volpath`        | `VolPathIntegrator`        | Volumetric path tracer with `r_u`/`r_l` rescaled-density MIS and the pbrt-v4 BSSRDF probe/reservoir path. |
| `simplevolpath`  | `SimpleVolPathIntegrator`  | Delta-tracking volume path tracer, no surface scattering. |
| `lightpath`      | `LightPathIntegrator`      | Light tracing with `Camera::SampleWi` splat. |
| `sppm`           | `SPPMIntegrator`           | Stochastic Progressive Photon Mapping with parallel camera, photon, and pixel-update passes. |
| `bdpt`           | `BDPTIntegrator`           | Bidirectional path tracing with surface and medium vertices. |
| `mlt`            | `MLTIntegrator`            | Primary-sample-space MLT on top of BDPT with parallel bootstrap sampling and Markov chains. |
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
`point`, `distant`, `spot`, `goniometric`, `projection`, `infinite` (uniform), `infinite` with `mapname` (image-based), and `diffuse` (area). Each is translated verbatim from `lights.h:189–625`. The light's `scale` parameter is a `Float` (pbrt-v4 shape), and light emission is normalized with `SpectrumToPhotometric`. Photometric film calibration remains separate work.

### Films / Samplers / Accelerators
- **Films**: `rgb` (`RGBFilm`), `gbuffer` (`GBufferFilm`), `spectral` (`SpectralFilm` with Fichet et al. spectral-EXR layout).
- **Samplers**: `independent`, `stratified`, `halton`, `02sequence`, `sobol`, `zsobol`, `paddedsobol`, `pmj02bn`, plus the internal `MLTSampler` driven by the `mlt` integrator.
- **Accelerators**: `bvh` (default; r4-specific QBVH SIMD aggregator), `kdtree`, `exhaustive`.

## Differences from pbrt-v4

### Relationship to upstream pbrt
pbrt-r4 began as a Rust port of pbrt-v3 (pbrt-r3) and has been progressively rewritten so its public class structure now matches pbrt-v4. Code paths are translated from pbrt-v4 line-by-line wherever practical; remaining compatibility differences are documented in source or project documentation.

### Implementation choices
- **Language**: Rust instead of C++. Ownership / lifetimes replace manual memory management. No `unsafe` outside SIMD intrinsics and a small set of FFI / raw-pointer helpers.
- **Class inheritance**: pbrt-v4 inheritance becomes composition — a Rust struct holds the parent struct as a `base` field. pbrt-v4 `TaggedPointer<...>` polymorphism becomes Rust `enum`s with the same variant names.
- **Parallel execution**: pbrt-v4's `ParallelFor` / hand-rolled thread pool becomes [`rayon`](https://crates.io/crates/rayon). The QBVH aggregator uses SSE / NEON intrinsics directly.
- **Parser**: pbrt-v4's hand-written recursive-descent parser becomes the [`nom`](https://crates.io/crates/nom) crate.
- **Progress / CLI / logging**: [`indicatif`](https://crates.io/crates/indicatif), [`clap`](https://crates.io/crates/clap), [`log`](https://crates.io/crates/log) + [`env_logger`](https://crates.io/crates/env_logger).
- **Other crates**: `image` (EXR/PNG/etc.), `ply-rs` (PLY mesh I/O), `rust-crypto` (hashing), `serde` + `serde_json` (scene JSON), `flate2` (`.ply.gz` decompression).
- **Splat accumulation**: `RGBFilm`, `GBufferFilm`, and `SpectralFilm` use per-pixel `AtomicDouble` splat buffers, corresponding to pbrt-v4's film splat accumulators. The Rust implementation stores `f64` bits in an `AtomicU64` and adds contributions with a compare-and-exchange loop. SPPM photon contributions currently use Mutex-protected accumulators.

### Additional features
- **Tev display**: real-time progressive display through the [tev](https://github.com/Tom94/tev) viewer.
- **QBVH accelerator**: an r4-original quad-BVH aggregator using SSE intrinsics. v4-faithful in that it now passes the unclamped `t_max` to per-primitive intersect (fixes flat-AABB area-light hit loss).
- **Function (AOV) integrator**: render scene attributes (position, normal, uv, depth, ...) for compositing or debugging.

### Out of scope

The following features and tools remain unimplemented or are not shipped:

- **PTex textures**: not implemented.
- **GPU-only command-line utilities**: `pspec_gpu` and `denoise-optix` are not included in the release.
- **Build and test utilities**: `soac`, `rgb2spec_opt`, and `pbrt_test` are handled by build generation or Rust's test infrastructure rather than shipped as user-facing binaries.

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
- Continue to harden pbrt-v4-scenes rendering and investigate high memory usage on large `.ply` scenes.

**Mid term**
- Benchmarking against pbrt-v3 / pbrt-v4 to guide optimization (current pbrt-r4 is typically slower than the C++ reference; we want a public benchmark suite to track this).
- Comprehensive `cargo doc` API surface and rustdoc examples.

**Long term**
- Port pbrt-v4 comment text to the corresponding pbrt-r4 functions to honour the original implementation.

## Acknowledgments
Thanks to Matt Pharr, Wenzel Jakob, and Greg Humphreys — the authors of pbrt-v3 and pbrt-v4 — for releasing their reference implementations.

Thanks also to the Rust community and the maintainers of the crates listed above.
