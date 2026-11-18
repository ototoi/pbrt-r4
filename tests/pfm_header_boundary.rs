use pbrt_r4::prelude::*;
use pbrt_r4::util::imageio::read_image_pfm::read_image_pfm;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "pbrt-r4-pfm-{}-{}-{}",
        std::process::id(),
        nanos,
        id
    ));
    fs::create_dir_all(&dir).unwrap();
    dir.join("image.pfm")
}

fn write_pfm(path: &PathBuf, header: &[u8], floats: &[f32], little: bool) {
    let mut bytes = header.to_vec();
    for &value in floats {
        let bytes4 = if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        bytes.extend_from_slice(&bytes4);
    }
    fs::write(path, bytes).unwrap();
}

fn rgb(path: &PathBuf) -> Vec<[Float; 3]> {
    let (pixels, _) = read_image_pfm(path.to_str().unwrap()).unwrap();
    pixels.iter().map(|p| p.to_rgb()).collect()
}

#[test]
fn whitespace_payload_bytes_are_pixel_data() {
    // The first float's low byte is 0x09/0x0a/0x0d/0x20: all valid binary
    // bytes that the old multispace-based header parser consumed as padding.
    for low_byte in [0x09u32, 0x0a, 0x0d, 0x20] {
        let bits = 0x3f80_0000 | low_byte; // 1.0 + a tiny bit pattern
        let path = temp_path();
        write_pfm(&path, b"Pf\n1 1\n-1.0\n", &[f32::from_bits(bits)], true);
        let pixels = rgb(&path);
        assert_eq!(pixels.len(), 1);
        assert_eq!(pixels[0][0], f32::from_bits(bits) as Float);
        assert_eq!(pixels[0][1], f32::from_bits(bits) as Float);
        assert_eq!(pixels[0][2], f32::from_bits(bits) as Float);
    }
}

#[test]
fn three_channel_big_endian_rows_and_crlf_header() {
    let path = temp_path();
    // PF, big endian (positive scale), scale 2.0, CRLF separators.
    // Rows are stored bottom-to-top; after the Y flip, row 0 is (7,8,9)*2.
    let floats = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // bottom row (stored first)
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // top row (stored second)
    ];
    write_pfm(&path, b"PF\r\n2 2\r\n2.0\r\n", &floats, false);
    let pixels = rgb(&path);
    assert_eq!(pixels.len(), 4);
    assert_eq!(pixels[0], [14.0 as Float, 16.0 as Float, 18.0 as Float]);
    assert_eq!(pixels[1], [20.0 as Float, 22.0 as Float, 24.0 as Float]);
    assert_eq!(pixels[2], [2.0 as Float, 4.0 as Float, 6.0 as Float]);
}

#[test]
fn truncated_and_invalid_dimensions_fail() {
    let path = temp_path();
    fs::write(&path, b"Pf\n1 1\n-1.0\n\x00\x00\x80").unwrap();
    assert!(read_image_pfm(path.to_str().unwrap()).is_err());

    for header in [
        b"Pf\n0 1\n-1.0\n".as_slice(),
        b"Pf\n-1 1\n-1.0\n",
        b"Pf\n1 0\n-1.0\n",
        b"Pf\n2147483648 1\n-1.0\n",
        b"Pf\n65536 65536\n-1.0\n",
    ] {
        write_pfm(&path, header, &[1.0], true);
        assert!(
            read_image_pfm(path.to_str().unwrap()).is_err(),
            "header {:?} should be rejected",
            String::from_utf8_lossy(header)
        );
    }
}
