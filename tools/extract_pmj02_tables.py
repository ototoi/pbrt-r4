#!/usr/bin/env python3
"""Extract pmj02bnSamples and BlueNoiseTextures from pbrt-v4 C++ source
to raw little-endian binary files."""
import re
import struct
import sys
from pathlib import Path

V4_ROOT = Path("/mnt/hdd1/src/other/pbrt-r4-devkit/pbrt-v4/src/pbrt/util")
OUT_ROOT = Path("/mnt/hdd1/src/other/pbrt-r4-devkit/pbrt-r4/src/samplers/data")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


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
    # --- pmj02bnSamples: 5 * 65536 * 2 = 655360 u32 ---
    p = V4_ROOT / "pmj02tables.cpp"
    ints = extract_ints(p)
    expect = 5 * 65536 * 2
    # The header also has integers (nPMJ02bnSets=5 etc.) but that's in the
    # separate header file. The .cpp source should contain only table values.
    print(f"pmj02tables.cpp: parsed {len(ints)} integers, expected {expect}")
    if len(ints) != expect:
        print(f"  WARN: count mismatch — trailing {len(ints) - expect} ignored")
        ints = ints[:expect]
    out = OUT_ROOT / "pmj02bn_samples.bin"
    with out.open("wb") as f:
        for v in ints:
            f.write(struct.pack("<I", v))
    print(f"  wrote {out} ({out.stat().st_size} bytes)")

    # --- BlueNoiseTextures: 48 * 128 * 128 = 786432 u16 ---
    p = V4_ROOT / "bluenoise.cpp"
    ints = extract_ints(p)
    expect = 48 * 128 * 128
    print(f"bluenoise.cpp: parsed {len(ints)} integers, expected {expect}")
    if len(ints) != expect:
        print(f"  WARN: count mismatch — trailing {len(ints) - expect} ignored")
        ints = ints[:expect]
    out = OUT_ROOT / "bluenoise_textures.bin"
    with out.open("wb") as f:
        for v in ints:
            f.write(struct.pack("<H", v))
    print(f"  wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
