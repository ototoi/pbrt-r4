#!/usr/bin/env python3
"""Extract pbrt-v4 named PiecewiseLinearSpectrum arrays + name mappings
from `pbrt-v4/src/pbrt/util/spectrum.cpp` and emit a Rust source file
(`src/util/spectrum/named_arrays.rs`) containing:

  - one `const NAME: [Float; N]` per inline numeric array
  - `pub fn named_piecewise_data(name: &str) -> Option<&'static [Float]>`

The Rust side feeds these into `PiecewiseLinearSpectrum::from_interleaved`.

Run from anywhere; both source and destination paths are hard-coded
relative to the devkit layout (matching tools/extract_pmj02_tables.py).
"""
import re
from pathlib import Path

V4_SPECTRUM_CPP = Path(
    "/mnt/hdd1/src/other/pbrt-r4-devkit/pbrt-v4/src/pbrt/util/spectrum.cpp"
)
OUT_RS = Path(
    "/mnt/hdd1/src/other/pbrt-r4-devkit/pbrt-r4/src/util/spectrum/named_arrays.rs"
)

# Arrays we do NOT want to port: CIE base curves are handled by the
# regular `CIE_*` constants in r4's spectrum module, and the bare lambda
# table is shared by all of them.
SKIP_ARRAYS = {
    "CIE_X",
    "CIE_Y",
    "CIE_Z",
    "CIE_lambda",
    "CIE_S0",
    "CIE_S1",
    "CIE_S2",
    "CIE_S_lambda",
}


def parse_arrays(text: str) -> dict[str, list[float]]:
    """Return {array_name: [floats...]} for every top-level
    `const Float NAME[...] = { ... };` definition."""
    out: dict[str, list[float]] = {}
    pattern = re.compile(
        r"^const\s+Float\s+([A-Za-z0-9_]+)\s*\[[^\]]*\]\s*=\s*\{(.*?)\};",
        re.MULTILINE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        name = m.group(1)
        if name in SKIP_ARRAYS:
            continue
        body = m.group(2)
        # Strip C++ comments inside the body just in case.
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        body = re.sub(r"//[^\n]*", "", body)
        nums = [
            float(t)
            for t in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+", body)
        ]
        if not nums:
            continue
        out[name] = nums
    return out


def parse_named_spectra_map(text: str) -> list[tuple[str, str]]:
    """Return [(public_name, variable_name)] from the `namedSpectra = { ... }`
    initializer."""
    map_start = text.find("namedSpectra = {")
    if map_start == -1:
        raise RuntimeError("`namedSpectra = {` not found")
    # Find matching closing brace.
    depth = 0
    end = None
    i = text.find("{", map_start)
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end is None:
        raise RuntimeError("Could not find end of namedSpectra map")
    body = text[map_start:end]
    entries = re.findall(r'\{\s*"([^"]+)"\s*,\s*([a-zA-Z0-9_]+)\s*\}', body)
    return entries


def parse_camera_sensor_map(
    text: str, arrays: dict[str, list[float]]
) -> list[tuple[str, str]]:
    """Return [(public_name_with_channel, array_name)] for camera response
    spectra. v4 stores them inline as
    `{"canon_eos_5d", {"R", PiecewiseLinearSpectrum::FromInterleaved(canon_eos_5d_r, ...)}, ...}`
    but we just want a flat list of (public_name, array_name) entries we
    can expose via the same lookup table."""
    out: list[tuple[str, str]] = []
    for name in arrays:
        lower = name.lower()
        # Heuristic: camera response arrays are lowercased and end in
        # _r/_g/_b (the only arrays in the cpp that follow this pattern).
        if lower.endswith(("_r", "_g", "_b")):
            channel = lower[-1].upper()
            base = lower[:-2]
            # public string used by `Camera "spectrum sensor"`-style
            # lookups in v4 is the array's lowercase base name.
            out.append((f"{base}_{channel.lower()}", name))
    return out


def _fmt(v: float) -> str:
    # Force every literal to carry a decimal point so rustc reads it as a
    # float — `300` alone would parse as an integer in a `[Float; N]`
    # initializer and trigger a type error.
    s = f"{v:.10g}"
    if any(c in s for c in ".eE"):
        return s
    return s + ".0"


def to_rust_array(name: str, values: list[float]) -> str:
    body_lines: list[str] = []
    chunk = 6
    for i in range(0, len(values), chunk):
        body_lines.append(
            "    " + ", ".join(_fmt(v) for v in values[i : i + chunk]) + ","
        )
    return (
        f"pub(super) const {name}: [Float; {len(values)}] = [\n"
        + "\n".join(body_lines)
        + "\n];\n"
    )


def main() -> int:
    text = V4_SPECTRUM_CPP.read_text()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    arrays = parse_arrays(text)
    named_pairs = parse_named_spectra_map(text)
    # Strip the CIE_X/Y/Z lambda arrays from `arrays`; not all of them get
    # caught by the [N] regex above because some declare an explicit size.
    arrays = {k: v for k, v in arrays.items() if k not in SKIP_ARRAYS}

    # Build var-name -> (array-name, normalize_flag). v4 spectrum.cpp
    # introduces local Spectrum vars (illuma, ageta, glassbk7eta, ...) that
    # the namedSpectra map references. Illuminants pass `true` for
    # normalize; metals/glasses pass `false`.
    var_to_array: dict[str, tuple[str, bool]] = {}
    var_re = re.compile(
        r"Spectrum\s+(\w+)\s*=\s*PiecewiseLinearSpectrum::FromInterleaved\(\s*(\w+)\s*,\s*(true|false)",
    )
    for m in var_re.finditer(text):
        var_to_array[m.group(1)] = (m.group(2), m.group(3) == "true")

    # Resolve public_name -> (array_name, normalize) through var_to_array.
    public_to_array: list[tuple[str, str, bool]] = []
    missing: list[tuple[str, str]] = []
    for pub, var in named_pairs:
        info = var_to_array.get(var)
        if info is None or info[0] not in arrays:
            missing.append((pub, var))
            continue
        public_to_array.append((pub, info[0], info[1]))

    # Camera sensor spectra are inlined in v4's sensorSpectra map with
    # `normalize=false` (response curves are not luminance-normalized).
    for pub, arr in parse_camera_sensor_map(text, arrays):
        public_to_array.append((pub, arr, False))

    # Sort arrays by name so the emitted file is stable.
    array_items = sorted(arrays.items())

    lines: list[str] = []
    lines.append(
        "// AUTOGENERATED by tools/extract_named_spectra.py — do not edit by hand."
    )
    lines.append("// Source: pbrt-v4 src/pbrt/util/spectrum.cpp (PiecewiseLinearSpectrum data).")
    lines.append("")
    lines.append("#![allow(non_upper_case_globals)]")
    lines.append("")
    lines.append("use crate::util::base::Float;")
    lines.append("")
    for name, values in array_items:
        lines.append(to_rust_array(name, values))
    lines.append("/// Lookup an interleaved (lambda, value) table plus the v4")
    lines.append("/// `normalize` flag (true for illuminants; false for metal /")
    lines.append("/// glass eta-k and camera response curves).")
    lines.append(
        "pub(super) fn named_piecewise_data(name: &str) -> Option<(&'static [Float], bool)> {"
    )
    lines.append("    match name {")
    for pub, arr, norm in sorted(public_to_array):
        flag = "true" if norm else "false"
        lines.append(f'        "{pub}" => Some((&{arr}, {flag})),')
    lines.append("        _ => None,")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    OUT_RS.write_text("\n".join(lines))

    print(f"wrote {OUT_RS}")
    print(f"  {len(array_items)} arrays, {len(public_to_array)} lookup entries")
    if missing:
        print(f"  skipped (no matching variable in cpp): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
