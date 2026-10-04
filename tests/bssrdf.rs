use pbrt_r4::bssrdf::{
    compute_beam_diffusion_bssrdf, subsurface_from_diffuse_sampled, BSSRDFTable,
};
use pbrt_r4::util::spectrum::SampledSpectrum;

#[test]
fn subsurface_from_diffuse_clamps_zero_mfp_to_finite_coefficients() {
    let mut table = BSSRDFTable::new(100, 64);
    compute_beam_diffusion_bssrdf(0.0, 1.33, &mut table);
    let (sigma_a, sigma_s) = subsurface_from_diffuse_sampled(
        &table,
        &SampledSpectrum::new(0.5),
        &SampledSpectrum::zero(),
    );
    assert!(sigma_a.is_valid());
    assert!(sigma_s.is_valid());
}
