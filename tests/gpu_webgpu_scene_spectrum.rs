#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{DenseSpectrum, DENSE_SAMPLE_COUNT, SPECTRUM_FLAG_CONSTANT};
use pbrt_r4::gpu::webgpu::scene::spectrum::convert_spectra;

#[test]
fn preserves_spectrum_order_sample_bits_and_flags() {
    let mut samples = std::array::from_fn(|index| index as f32 - 200.0);
    samples[0] = -0.0;
    samples[DENSE_SAMPLE_COUNT - 1] = f32::MAX;
    let source = [
        DenseSpectrum::new(samples, 0x80),
        DenseSpectrum::new([1.5; DENSE_SAMPLE_COUNT], SPECTRUM_FLAG_CONSTANT),
    ];
    let records = convert_spectra(&source).unwrap();
    assert_eq!(records.len(), source.len());
    for (record, spectrum) in records.iter().zip(&source) {
        assert_eq!(record.flags, spectrum.flags);
        for (actual, expected) in record.samples.iter().zip(&spectrum.samples) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn rejects_non_finite_samples_in_any_record() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [0, DENSE_SAMPLE_COUNT / 2, DENSE_SAMPLE_COUNT - 1] {
            let mut spectra = vec![DenseSpectrum::new([1.0; DENSE_SAMPLE_COUNT], 0); 2];
            spectra[1].samples[index] = value;
            let error = convert_spectra(&spectra).err().unwrap().to_string();
            assert!(
                error.contains("Flat spectrum table contains a non-finite sample"),
                "{error}"
            );
        }
    }
}

#[test]
fn accepts_empty_spectrum_table() {
    assert!(convert_spectra(&[]).unwrap().is_empty());
}
