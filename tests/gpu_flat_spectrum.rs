use pbrt_r4::gpu::ir::flat::{
    SpectrumTableBuilder, DENSE_LAMBDA_MAX, DENSE_LAMBDA_MIN, DENSE_SAMPLE_COUNT,
    SPECTRUM_FLAG_CONSTANT,
};
use pbrt_r4::util::spectrum::{DenselySampledSpectrum, Spectrum};

#[test]
fn dense_spectrum_domain_matches_the_renderer_domain() {
    assert_eq!(DENSE_LAMBDA_MIN, 360);
    assert_eq!(DENSE_LAMBDA_MAX, 830);
    assert_eq!(DENSE_SAMPLE_COUNT, DenselySampledSpectrum::N_SAMPLES);
}

#[test]
fn spectrum_ids_are_stable_and_deduplicate_values() {
    let first = Spectrum::from_rgb_albedo(&[0.25, 0.5, 0.75]);
    let same = first.clone();
    let different = Spectrum::from_rgb_unbounded(&[0.25, 0.5, 0.75]);
    let mut builder = SpectrumTableBuilder::default();

    let first_id = builder.intern(&first).unwrap();
    assert_eq!(builder.intern(&same).unwrap(), first_id);
    assert_ne!(builder.intern(&different).unwrap(), first_id);

    let table = builder.finish();
    assert_eq!(table.spectrum_count(), 2);
    assert_eq!(table.samples.len(), 2 * DENSE_SAMPLE_COUNT);
    table.validate().unwrap();
}

#[test]
fn constant_metadata_is_semantic_and_part_of_the_dedup_key() {
    let constant = Spectrum::from(0.5);
    let dense = Spectrum::from(constant.to_dense());
    let mut builder = SpectrumTableBuilder::default();

    let constant_id = builder.intern(&constant).unwrap();
    let dense_id = builder.intern(&dense).unwrap();
    assert_ne!(constant_id, dense_id);

    let table = builder.finish();
    assert_eq!(
        table.metadata[constant_id as usize].flags,
        SPECTRUM_FLAG_CONSTANT
    );
    assert_eq!(table.metadata[dense_id as usize].flags, 0);
}

#[test]
fn dense_intern_normalizes_negative_zero() {
    let positive = DenselySampledSpectrum::new(0.0);
    let negative = DenselySampledSpectrum::new(-0.0);
    let mut builder = SpectrumTableBuilder::default();

    let positive_id = builder.intern_dense(&positive, 0).unwrap();
    assert_eq!(builder.intern_dense(&negative, 0).unwrap(), positive_id);
}
