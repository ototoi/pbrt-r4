#!/usr/bin/env python3
"""Extract pbrt-v4 rgb2spec_opt C++ tables as little-endian f32 binaries.

This is a Phase 0 reproducibility tool.  The eventual build-time generator
will be implemented in Rust; until then, this script provides an explicit and
verifiable conversion from pbrt-v4's generated C++ tables to the binary
layout consumed by pbrt-r4.
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path


RESOLUTION = 64
SCALE_COUNT = RESOLUTION
ILLUMINANT_SAMPLE_COUNT = 107
COEFFICIENT_COUNT = 3 * RESOLUTION**3 * 3
EXPECTED_FLOAT_COUNT = SCALE_COUNT + 2 * ILLUMINANT_SAMPLE_COUNT + COEFFICIENT_COUNT

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)

TABLES = {
    "srgb": "sRGBToSpectrumTable",
    "aces": "ACES2065_1ToSpectrumTable",
    "dci_p3": "DCI_P3ToSpectrumTable",
    "rec2020": "REC2020ToSpectrumTable",
}


def array_body(source: str, symbol: str) -> str:
    pattern = re.compile(
        rf"(?:extern const float|pub(?:\([^)]*\))?\s+const)\s+"
        rf"{re.escape(symbol)}[^=]*=\s*[\{{\[]",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"array not found: {symbol}")

    start = match.end()
    opening = source[start - 1]
    closing = "}" if opening == "{" else "]"
    depth = 1
    index = start
    while index < len(source) and depth:
        if source[index] == opening:
            depth += 1
        elif source[index] == closing:
            depth -= 1
        index += 1
    if depth:
        raise ValueError(f"unterminated array: {symbol}")
    return source[start : index - 1]


def parse_floats(source: str, symbol: str) -> list[float]:
    return [float(value) for value in FLOAT_RE.findall(array_body(source, symbol))]


def extract(
    source_path: Path,
    illuminant_source_path: Path,
    output_path: Path,
    table_prefix: str,
    illuminant_symbol: str,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    scale = parse_floats(source, f"{table_prefix}_Scale")
    data = parse_floats(source, f"{table_prefix}_Data")
    if len(scale) != SCALE_COUNT:
        raise ValueError(f"{source_path}: expected {SCALE_COUNT} scale values, got {len(scale)}")
    if len(data) != COEFFICIENT_COUNT:
        raise ValueError(
            f"{source_path}: expected {COEFFICIENT_COUNT} coefficient values, got {len(data)}"
        )

    illuminant_source = illuminant_source_path.read_text(encoding="utf-8")
    interleaved_illuminant = parse_floats(illuminant_source, illuminant_symbol)
    if len(interleaved_illuminant) != 2 * ILLUMINANT_SAMPLE_COUNT:
        raise ValueError(
            f"{illuminant_symbol}: expected {2 * ILLUMINANT_SAMPLE_COUNT} values, "
            f"got {len(interleaved_illuminant)}"
        )
    # pbrt-v4 stores wavelength/value pairs, while pbrt-r4's existing bin
    # layout stores all wavelengths followed by all values.
    illuminant = interleaved_illuminant[::2] + interleaved_illuminant[1::2]
    values = scale + illuminant + data
    if len(values) != EXPECTED_FLOAT_COUNT:
        raise ValueError(f"{source_path}: unexpected output length: {len(values)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(struct.pack(f"<{len(values)}f", *values))
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-build", type=Path, required=True)
    parser.add_argument("--r4-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    illuminants = {
        "srgb": "CIE_Illum_D6500",
        "aces": "ACES_Illum_D60",
        "dci_p3": "CIE_Illum_D6500",
        "rec2020": "CIE_Illum_D6500",
    }
    for name, prefix in TABLES.items():
        source_path = args.v4_build / f"rgbspectrum_{name}.cpp"
        extract(
            source_path,
            args.r4_source,
            args.output / f"rgb_to_spectrum_{name}.bin",
            prefix,
            illuminants[name],
        )


if __name__ == "__main__":
    main()
