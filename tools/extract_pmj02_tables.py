#!/usr/bin/env python3
"""Extract pmj02bnSamples and BlueNoiseTextures from pbrt-v4 C++ source
to raw little-endian binary files."""
import argparse
import re
import struct
import sys
from pathlib import Path

DEVKIT_ROOT = Path(__file__).resolve().parents[2]
R4_ROOT = DEVKIT_ROOT / "pbrt-r4"
DEFAULT_V4_ROOT = DEVKIT_ROOT / "pbrt-v4/src/pbrt/util"
DEFAULT_OUT_ROOT = R4_ROOT / "src/samplers/data"


def extract_ints(path: Path) -> list[int]:
    """Pull every decimal integer from the C++ source's array body,
    skipping the type declaration's `[N]` literals."""
    text = path.read_text()
    # Strip C-style comments to avoid picking up integers in them.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    # Start parsing after the first `= {` so the `[2]` etc. in the type
    # declaration don't slip into the data stream.
    eq_pos = text.find("= {")
    if eq_pos == -1:
        raise RuntimeError(f"`= {{` not found in {path}")
    body = text[eq_pos + len("= {"):]
    return [int(m.group(0)) for m in re.finditer(r"\b\d+\b", body)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract PMJ02 and blue-noise tables from pbrt-v4 sources."
    )
    parser.add_argument(
        "--v4-util",
        type=Path,
        default=DEFAULT_V4_ROOT,
        help=f"pbrt-v4 util source directory (default: {DEFAULT_V4_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"output directory for binary tables (default: {DEFAULT_OUT_ROOT})",
    )
    args = parser.parse_args()

    for filename in ("pmj02tables.cpp", "bluenoise.cpp"):
        if not (args.v4_util / filename).is_file():
            parser.error(f"source file does not exist: {args.v4_util / filename}")

    # --- pmj02bnSamples: 5 * 65536 * 2 = 655360 u32 ---
    p = args.v4_util / "pmj02tables.cpp"
    ints = extract_ints(p)
    expect = 5 * 65536 * 2
    # The header also has integers (nPMJ02bnSets=5 etc.) but that's in the
    # separate header file. The .cpp source should contain only table values.
    print(f"pmj02tables.cpp: parsed {len(ints)} integers, expected {expect}")
    if len(ints) != expect:
        print(f"  WARN: count mismatch — trailing {len(ints) - expect} ignored")
        ints = ints[:expect]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "pmj02bn_samples.bin"
    with out.open("wb") as f:
        for v in ints:
            f.write(struct.pack("<I", v))
    print(f"  wrote {out} ({out.stat().st_size} bytes)")

    # --- BlueNoiseTextures: 48 * 128 * 128 = 786432 u16 ---
    p = args.v4_util / "bluenoise.cpp"
    ints = extract_ints(p)
    expect = 48 * 128 * 128
    print(f"bluenoise.cpp: parsed {len(ints)} integers, expected {expect}")
    if len(ints) != expect:
        print(f"  WARN: count mismatch — trailing {len(ints) - expect} ignored")
        ints = ints[:expect]
    out = args.output_dir / "bluenoise_textures.bin"
    with out.open("wb") as f:
        for v in ints:
            f.write(struct.pack("<H", v))
    print(f"  wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
