#!/usr/bin/env python3
"""Extract rgb2spec_opt constants from pbrt-v4 into Rust source."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ARRAYS = {
    "CIE_X": "cie_x",
    "CIE_Y": "cie_y",
    "CIE_Z": "cie_z",
    "CIE_D65": "cie_d65",
    "CIE_D60": "cie_d60",
    "SRGB_TO_XYZ": "srgb_to_xyz",
    "XYZ_TO_SRGB": "xyz_to_srgb",
    "ACES2065_1_TO_XYZ": "aces2065_1_to_xyz",
    "XYZ_TO_ACES2065_1": "xyz_to_aces2065_1",
    "REC2020_TO_XYZ": "rec2020_to_xyz",
    "XYZ_TO_REC2020": "xyz_to_rec2020",
    "DCI_P3_TO_XYZ": "dcip3_to_xyz",
    "XYZ_TO_DCI_P3": "xyz_to_dcip3",
}


def body(source: str, name: str) -> str:
    match = re.search(rf"const double {name}\[[^=]*\]\s*=\s*\{{", source)
    if match is None:
        raise ValueError(f"array not found: {name}")
    index = match.end()
    depth = 1
    while index < len(source) and depth:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
        index += 1
    if depth:
        raise ValueError(f"unterminated array: {name}")
    return source[match.end() : index - 1]


def values(source: str, name: str) -> list[str]:
    return re.findall(FLOAT, body(source, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    output = [
        "// Generated from pbrt-v4/src/pbrt/cmd/rgb2spec_opt.cpp.",
        "// Do not edit by hand; regenerate with tools/extract_rgb2spec_constants.py.",
        "",
    ]
    for rust_name, cpp_name in ARRAYS.items():
        data = values(source, cpp_name)
        if cpp_name == "cie_d65":
            data = [f"({value} / 10566.864005283874576)" for value in data]
        elif cpp_name == "cie_d60":
            data = [f"({value} / 10536.3)" for value in data]
        # The v4 source declares CIE_D60 with 95 samples but provides 94
        # initializers; C++ zero-initializes the remaining element.
        if cpp_name == "cie_d60" and len(data) == 94:
            data.append("0.0")
        output.append(f"pub const {rust_name}: [f64; {len(data)}] = [")
        for value in data:
            output.append(f"    {value},")
        output.extend(["];", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    main()
