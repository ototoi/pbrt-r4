use pbrt_r4::base::{
    BxDF, BXDF_DIFFUSE, BXDF_GLOSSY, BXDF_REFL_TRANS_REFLECTION, BXDF_REFL_TRANS_TRANSMISSION,
    BXDF_SPECULAR,
};
use pbrt_r4::materials::{MaterialEvalContext, MixMaterial};
use pbrt_r4::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

trait TestMaterialGetBxDF {
    fn test_get_bxdf(&self, si: &SurfaceInteraction, lambda: &SampledWavelengths) -> BxDF;
    fn test_get_bsdf(&self, si: &SurfaceInteraction, lambda: &SampledWavelengths) -> BSDF;
    fn test_get_bssrdf(
        &self,
        si: &SurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDF>;
}

impl TestMaterialGetBxDF for Material {
    fn test_get_bxdf(&self, si: &SurfaceInteraction, lambda: &SampledWavelengths) -> BxDF {
        let tex_eval = UniversalTextureEvaluator;
        let ctx = MaterialEvalContext::from(si);
        self.get_bxdf(&tex_eval, &ctx, lambda)
    }

    fn test_get_bsdf(&self, si: &SurfaceInteraction, lambda: &SampledWavelengths) -> BSDF {
        let tex_eval = UniversalTextureEvaluator;
        let ctx = MaterialEvalContext::from(si);
        self.get_bsdf(&tex_eval, &ctx, lambda)
    }

    fn test_get_bssrdf(
        &self,
        si: &SurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDF> {
        let tex_eval = UniversalTextureEvaluator;
        let ctx = MaterialEvalContext::from(si);
        self.get_bssrdf(&tex_eval, &ctx, lambda)
    }
}

fn sampled_transmission_eta(
    material: &Material,
    si: &SurfaceInteraction,
    lambda: &SampledWavelengths,
) -> Float {
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    material
        .test_get_bxdf(si, lambda)
        .sample_f(
            &wo,
            0.99,
            &Point2f::new(0.5, 0.5),
            TransportMode::Radiance,
            BXDF_REFL_TRANS_TRANSMISSION,
        )
        .expect("transmission sample should succeed")
        .eta
}

fn hair_test_si() -> SurfaceInteraction {
    let mut si = SurfaceInteraction::default();
    si.uv = Point2f::new(0.0, 0.5);
    si
}

fn surface_test_si() -> SurfaceInteraction {
    let mut si = SurfaceInteraction::default();
    si.n = Normal3f::new(0.0, 0.0, 1.0);
    si.shading.n = si.n;
    si.dpdu = Vector3f::new(1.0, 0.0, 0.0);
    si.shading.dpdu = si.dpdu;
    si.wo = Vector3f::new(0.0, 0.0, 1.0);
    si
}

fn hair_test_signature(material: &Material, si: &SurfaceInteraction) -> Float {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let bxdf = material.test_get_bxdf(si, &lambda);
    let directions = [
        (
            Vector3f::new(0.54986966, 0.03359017, 0.83457476),
            Vector3f::new(-0.37383357, -0.91920084, 0.12376696),
        ),
        (
            Vector3f::new(0.31622776, 0.84327406, 0.4330127),
            Vector3f::new(-0.4472136, 0.7745967, -0.4472136),
        ),
        (
            Vector3f::new(-0.2, 0.4, 0.8944272),
            Vector3f::new(0.45, -0.55, 0.70400524),
        ),
        (
            Vector3f::new(0.15, -0.72, 0.67742187),
            Vector3f::new(-0.35, 0.28, -0.8930848),
        ),
    ];

    directions
        .iter()
        .map(|(wo, wi)| bxdf.f(wo, wi, TransportMode::Radiance).y(&lambda) * wi.z.abs())
        .sum()
}

#[test]
fn material_create_coateddiffuse_uses_coated_diffuse_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("coateddiffuse", &tp);
    assert!(matches!(material, Ok(Material::CoatedDiffuse(_))));
}

#[test]
fn material_coated_diffuse_errors_when_named_albedo_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture albedo", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("coateddiffuse", &tp).is_err());
}

#[test]
fn material_coated_diffuse_uses_uv_roughness_anisotropy_when_remap_disabled() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut iso_params = ParameterDictionary::new();
    let mut uv_params = ParameterDictionary::new();
    iso_params.add_float("roughness", 0.5);
    iso_params.add_bool("remaproughness", false);
    uv_params.add_float("uroughness", 0.0);
    uv_params.add_float("vroughness", 1.0);
    uv_params.add_bool("remaproughness", false);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let iso_tp = TextureParameterDictionary::new(&iso_params, &f_tex, &s_tex);
    let uv_tp = TextureParameterDictionary::new(&uv_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let iso =
        Material::create("coateddiffuse", &iso_tp).expect("isotropic material should be created");
    let uv =
        Material::create("coateddiffuse", &uv_tp).expect("uv-roughness material should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let iso_f = iso
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let uv_f = uv
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((iso_f - uv_f).abs() > 1e-6);
}

#[test]
fn material_coated_diffuse_remaproughness_changes_response() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut no_remap_params = ParameterDictionary::new();
    let mut remap_params = ParameterDictionary::new();
    no_remap_params.add_float("roughness", 0.5);
    no_remap_params.add_bool("remaproughness", false);
    remap_params.add_float("roughness", 0.5);
    remap_params.add_bool("remaproughness", true);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let no_remap_tp = TextureParameterDictionary::new(&no_remap_params, &f_tex, &s_tex);
    let remap_tp = TextureParameterDictionary::new(&remap_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let no_remap = Material::create("coateddiffuse", &no_remap_tp)
        .expect("coateddiffuse without remap should be created");
    let remap = Material::create("coateddiffuse", &remap_tp)
        .expect("coateddiffuse with remap should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let no_remap_f = no_remap
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let remap_f = remap
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((no_remap_f - remap_f).abs() > 1e-3);
}

#[test]
fn material_coated_diffuse_flags_include_medium_diffuse_scattering() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut vacuum_params = ParameterDictionary::new();
    let mut medium_params = ParameterDictionary::new();
    vacuum_params.add_spectrum("reflectance", &Spectrum::from(0.0));
    medium_params.add_spectrum("reflectance", &Spectrum::from(0.0));
    medium_params.add_spectrum("albedo", &Spectrum::from(0.5));
    medium_params.add_float("g", 0.35);

    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let vacuum_tp = TextureParameterDictionary::new(&vacuum_params, &f_tex, &s_tex);
    let medium_tp = TextureParameterDictionary::new(&medium_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let vacuum = Material::create("coateddiffuse", &vacuum_tp)
        .expect("vacuum coateddiffuse should be created");
    let medium = Material::create("coateddiffuse", &medium_tp)
        .expect("medium coateddiffuse should be created");

    let vacuum_flags = vacuum.test_get_bxdf(&si, &lambda).flags();
    let medium_flags = medium.test_get_bxdf(&si, &lambda).flags();
    assert_eq!(vacuum_flags & BXDF_GLOSSY, BXDF_GLOSSY);
    assert_eq!(vacuum_flags & BXDF_DIFFUSE, 0);
    assert_eq!(medium_flags & BXDF_DIFFUSE, BXDF_DIFFUSE);
}

#[test]
fn material_coated_diffuse_defaults_match_explicit_defaults() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_spectrum("reflectance", &Spectrum::from(0.5));
    explicit_params.add_spectrum("eta", &Spectrum::from(1.5));
    explicit_params.add_float("roughness", 0.0);
    explicit_params.add_float("thickness", 0.01);
    explicit_params.add_float("g", 0.0);
    explicit_params.add_spectrum("albedo", &Spectrum::from(0.0));
    explicit_params.add_bool("remaproughness", true);
    explicit_params.add_int("maxdepth", 10);
    explicit_params.add_int("nsamples", 1);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default = Material::create("coateddiffuse", &default_tp)
        .expect("default coateddiffuse should create");
    let explicit = Material::create("coateddiffuse", &explicit_tp)
        .expect("explicit coateddiffuse should create");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let default_f = default
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let explicit_f = explicit
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((default_f - explicit_f).abs() < 1e-6);
}

#[test]
fn material_create_unknown_still_returns_error() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("not_a_real_material", &tp);
    assert!(material.is_err());
}

#[test]
fn material_create_rejects_legacy_names_with_upgrade_hint() {
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    for name in [
        "uber",
        "matte",
        "kdsubsurface",
        "plastic",
        "substrate",
        "translucent",
        "mirror",
        "glass",
        "disney",
        "metal",
        "fourier",
    ] {
        let error = match Material::create(name, &tp) {
            Err(error) => error,
            Ok(_) => panic!("legacy material {name:?} should not be created"),
        };
        assert!(error.msg.contains("legacy material"));
        assert!(error.msg.contains(name));
    }
}

#[test]
fn material_create_dielectric_uses_dielectric_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("dielectric", &tp);
    assert!(matches!(material, Ok(Material::Dielectric(_))));
}

#[test]
fn material_create_thindielectric_uses_thin_dielectric_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("thindielectric", &tp);
    assert!(matches!(material, Ok(Material::ThinDielectric(_))));
}

#[test]
fn material_create_conductor_uses_conductor_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("conductor", &tp);
    assert!(matches!(material, Ok(Material::Conductor(_))));
}

#[test]
fn material_create_coatedconductor_uses_coated_conductor_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("coatedconductor", &tp);
    assert!(matches!(material, Ok(Material::CoatedConductor(_))));
}

#[test]
fn material_create_mix_requires_scene_builder_resolution() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("mix", &tp);
    assert!(material.is_err());
}

#[test]
fn material_create_hair_uses_hair_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("hair", &tp);
    assert!(matches!(material, Ok(Material::Hair(_))));
}

#[test]
fn material_hair_returns_hair_bxdf() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let material = Material::create("hair", &tp).expect("hair material should create");
    assert!(matches!(
        material.test_get_bxdf(&si, &lambda),
        BxDF::Hair(_)
    ));
}

#[test]
fn material_create_interface_uses_interface_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let interface = Material::create("interface", &tp).expect("interface material should create");
    assert!(matches!(interface, Material::Interface(_)));
    assert!(matches!(
        interface.test_get_bxdf(&si, &lambda),
        BxDF::Dielectric(_)
    ));
}

#[test]
fn material_create_empty_or_none_maps_to_interface() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let empty = Material::create("", &tp).expect("empty material should map to interface");
    let none = Material::create("none", &tp).expect("none material should map to interface");
    assert!(matches!(empty, Material::Interface(_)));
    assert!(matches!(none, Material::Interface(_)));
}

#[test]
fn material_hair_accepts_color_alias() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut reflectance_params = ParameterDictionary::new();
    let mut color_params = ParameterDictionary::new();
    reflectance_params.add_spectrum("reflectance", &Spectrum::from(0.35));
    color_params.add_spectrum("color", &Spectrum::from(0.35));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let reflectance_tp = TextureParameterDictionary::new(&reflectance_params, &f_tex, &s_tex);
    let color_tp = TextureParameterDictionary::new(&color_params, &f_tex, &s_tex);
    let si = hair_test_si();

    let reflectance_hair =
        Material::create("hair", &reflectance_tp).expect("hair with reflectance should create");
    let color_hair = Material::create("hair", &color_tp).expect("hair with color should create");

    let reflectance_f = hair_test_signature(&reflectance_hair, &si);
    let color_f = hair_test_signature(&color_hair, &si);
    assert!((reflectance_f - color_f).abs() < 1e-6);
}

#[test]
fn material_hair_defaults_to_brownish_sigma_a() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_spectrum(
        "sigma_a",
        &Spectrum::from_rgb_unbounded(&[0.419 * 1.3, 0.697 * 1.3, 1.37 * 1.3]),
    );
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = hair_test_si();

    let default_hair = Material::create("hair", &default_tp).expect("default hair should create");
    let explicit_hair =
        Material::create("hair", &explicit_tp).expect("explicit hair should create");

    let default_f = hair_test_signature(&default_hair, &si);
    let explicit_f = hair_test_signature(&explicit_hair, &si);
    assert!((default_f - explicit_f).abs() < 1e-6);
}

#[test]
fn material_hair_sigma_a_takes_precedence_over_reflectance() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut sigma_params = ParameterDictionary::new();
    let mut mixed_params = ParameterDictionary::new();
    let mut reflectance_params = ParameterDictionary::new();
    sigma_params.add_spectrum("sigma_a", &Spectrum::from([1.2, 0.8, 0.5]));
    mixed_params.add_spectrum("sigma_a", &Spectrum::from([1.2, 0.8, 0.5]));
    mixed_params.add_spectrum("reflectance", &Spectrum::from(0.95));
    reflectance_params.add_spectrum("reflectance", &Spectrum::from(0.95));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let sigma_tp = TextureParameterDictionary::new(&sigma_params, &f_tex, &s_tex);
    let mixed_tp = TextureParameterDictionary::new(&mixed_params, &f_tex, &s_tex);
    let reflectance_tp = TextureParameterDictionary::new(&reflectance_params, &f_tex, &s_tex);
    let si = hair_test_si();

    let sigma_hair = Material::create("hair", &sigma_tp).expect("hair with sigma_a should create");
    let mixed_hair = Material::create("hair", &mixed_tp)
        .expect("hair with sigma_a and reflectance should create");
    let reflectance_hair =
        Material::create("hair", &reflectance_tp).expect("hair with reflectance should create");

    let sigma_f = hair_test_signature(&sigma_hair, &si);
    let mixed_f = hair_test_signature(&mixed_hair, &si);
    let reflectance_f = hair_test_signature(&reflectance_hair, &si);

    assert!((sigma_f - mixed_f).abs() < 1e-6);
    assert!((mixed_f - reflectance_f).abs() > 1e-3);
}

#[test]
fn material_hair_melanin_parameters_reduce_reflectance() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut light_params = ParameterDictionary::new();
    let mut dark_params = ParameterDictionary::new();
    light_params.add_float("eumelanin", 0.1);
    light_params.add_float("pheomelanin", 0.1);
    dark_params.add_float("eumelanin", 2.0);
    dark_params.add_float("pheomelanin", 2.0);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let light_tp = TextureParameterDictionary::new(&light_params, &f_tex, &s_tex);
    let dark_tp = TextureParameterDictionary::new(&dark_params, &f_tex, &s_tex);
    let si = hair_test_si();

    let light_hair = Material::create("hair", &light_tp).expect("light hair should create");
    let dark_hair = Material::create("hair", &dark_tp).expect("dark hair should create");

    let light_f = hair_test_signature(&light_hair, &si);
    let dark_f = hair_test_signature(&dark_hair, &si);
    assert!(dark_f < light_f);
}

#[test]
fn material_hair_beta_n_changes_bxdf_response() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut low_beta_n_params = ParameterDictionary::new();
    let mut high_beta_n_params = ParameterDictionary::new();
    low_beta_n_params.add_spectrum("reflectance", &Spectrum::from(0.6));
    low_beta_n_params.add_float("beta_m", 0.2);
    low_beta_n_params.add_float("beta_n", 0.2);
    high_beta_n_params.add_spectrum("reflectance", &Spectrum::from(0.6));
    high_beta_n_params.add_float("beta_m", 0.2);
    high_beta_n_params.add_float("beta_n", 1.0);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let low_tp = TextureParameterDictionary::new(&low_beta_n_params, &f_tex, &s_tex);
    let high_tp = TextureParameterDictionary::new(&high_beta_n_params, &f_tex, &s_tex);
    let si = hair_test_si();

    let low_hair = Material::create("hair", &low_tp).expect("low beta_n hair should create");
    let high_hair = Material::create("hair", &high_tp).expect("high beta_n hair should create");

    let low_f = hair_test_signature(&low_hair, &si);
    let high_f = hair_test_signature(&high_hair, &si);
    assert!((low_f - high_f).abs() > 1e-4);
}

#[test]
fn material_create_subsurface_uses_subsurface_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("subsurface", &tp);
    assert!(matches!(material, Ok(Material::Subsurface(_))));
}

#[test]
fn material_subsurface_get_bxdf_returns_dielectric() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = surface_test_si();

    let material =
        Material::create("subsurface", &tp).expect("subsurface material should be created");
    assert!(matches!(
        material.test_get_bxdf(&si, &lambda),
        BxDF::Dielectric(_)
    ));
}

#[test]
fn material_subsurface_get_bssrdf_returns_bssrdf() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = surface_test_si();
    let tex_eval = UniversalTextureEvaluator;
    let ctx = MaterialEvalContext::from(&si);

    let material =
        Material::create("subsurface", &tp).expect("subsurface material should be created");
    assert!(material.get_bssrdf(&tex_eval, &ctx, &lambda).is_some());
}

#[test]
fn material_subsurface_accepts_albedo_alias() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("albedo", &Spectrum::from(0.25));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let subsurface =
        Material::create("subsurface", &tp).expect("subsurface material should be created");
    let si = surface_test_si();
    let bsdf = subsurface.test_get_bsdf(&si, &lambda);
    assert!(matches!(&bsdf.bxdf, BxDF::Dielectric(_)));
    assert!(subsurface.test_get_bssrdf(&si, &lambda).is_some());
}

#[test]
fn material_subsurface_name_preset_populates_bssrdf() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut preset_params = ParameterDictionary::new();
    preset_params.add_string("name", "Skin1");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let preset_tp = TextureParameterDictionary::new(&preset_params, &f_tex, &s_tex);

    let default_subsurface =
        Material::create("subsurface", &default_tp).expect("default subsurface should create");
    let preset_subsurface =
        Material::create("subsurface", &preset_tp).expect("preset subsurface should create");

    let default_si = surface_test_si();
    let preset_si = surface_test_si();
    assert!(default_subsurface
        .test_get_bssrdf(&default_si, &lambda)
        .is_some());
    assert!(preset_subsurface
        .test_get_bssrdf(&preset_si, &lambda)
        .is_some());
}

#[test]
fn material_subsurface_accepts_sigma_a_and_sigma_s_pair() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("sigma_a", &Spectrum::from([0.03, 0.07, 0.2]));
    mat_params.add_spectrum("sigma_s", &Spectrum::from([0.7, 0.9, 1.1]));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let subsurface =
        Material::create("subsurface", &tp).expect("subsurface with sigma pair should create");
    let si = surface_test_si();
    assert!(subsurface.test_get_bssrdf(&si, &lambda).is_some());
}

#[test]
fn material_subsurface_named_preset_ignores_nonzero_g_like_v4() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut preset_params = ParameterDictionary::new();
    let mut preset_with_g_params = ParameterDictionary::new();
    preset_params.add_string("name", "Cream");
    preset_with_g_params.add_string("name", "Cream");
    preset_with_g_params.add_float("g", 0.35);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let preset_tp = TextureParameterDictionary::new(&preset_params, &f_tex, &s_tex);
    let preset_with_g_tp = TextureParameterDictionary::new(&preset_with_g_params, &f_tex, &s_tex);

    let preset_material =
        Material::create("subsurface", &preset_tp).expect("preset subsurface should create");
    let preset_with_g_material = Material::create("subsurface", &preset_with_g_tp)
        .expect("preset subsurface with g should create");
    let si = surface_test_si();

    let base = preset_material
        .test_get_bssrdf(&si, &lambda)
        .expect("preset bssrdf should exist");
    let with_g = preset_with_g_material
        .test_get_bssrdf(&si, &lambda)
        .expect("preset bssrdf with g should exist");
    let u1 = 0.31;
    let u2 = Point2f::new(0.17, 0.63);
    let base_sp = base
        .sample_sp(u1, &u2, &lambda)
        .expect("preset sample_sp should exist");
    let with_g_sp = with_g
        .sample_sp(u1, &u2, &lambda)
        .expect("preset sample_sp with g should exist");
    assert_eq!(base_sp.p0, with_g_sp.p0);
    assert_eq!(base_sp.p1, with_g_sp.p1);
}

#[test]
fn material_subsurface_errors_when_named_sigma_a_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture sigma_a", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("subsurface", &tp).is_err());
}

#[test]
fn material_subsurface_errors_when_named_reflectance_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture reflectance", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("subsurface", &tp).is_err());
}

#[test]
fn material_subsurface_get_bsdf_and_bssrdf_returns_interface_scattering() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("reflectance", &Spectrum::from(0.6));
    mat_params.add_spectrum("mfp", &Spectrum::from([0.001, 0.0008, 0.0006]));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material =
        Material::create("subsurface", &tp).expect("subsurface material should be created");
    let si = surface_test_si();

    let bsdf = material.test_get_bsdf(&si, &lambda);
    assert!(matches!(&bsdf.bxdf, BxDF::Dielectric(_)));
    assert!(material.test_get_bssrdf(&si, &lambda).is_some());
}

#[test]
fn material_create_measured_requires_filename() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("measured", &tp);
    assert!(material.is_err());
}

#[test]
fn material_create_measured_uses_measured_material() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("filename", "tests/bsdfs/metallic_spec.bsdf");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("measured", &tp);
    assert!(matches!(material, Ok(Material::Measured(_))));
}

#[test]
fn material_get_bxdf_uses_v4_style_material_api() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let diffuse = Material::create("diffuse", &tp).expect("diffuse material should be created");
    assert!(matches!(
        diffuse.test_get_bxdf(&si, &lambda),
        BxDF::Diffuse(_)
    ));

    let conductor =
        Material::create("conductor", &tp).expect("conductor material should be created");
    assert!(matches!(
        conductor.test_get_bxdf(&si, &lambda),
        BxDF::Conductor(_)
    ));
}

#[test]
fn material_diffuse_accepts_albedo_alias() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("albedo", &Spectrum::from(0.2));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let diffuse = Material::create("diffuse", &tp).expect("diffuse material should be created");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let f = diffuse
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance);
    assert!(f.y(&lambda) > 0.0);
}

#[test]
fn material_diffuse_errors_when_named_reflectance_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture reflectance", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("diffuse", &tp).is_err());
}

#[test]
fn material_diffuse_defaults_to_half_reflectance() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_spectrum("reflectance", &Spectrum::from(0.5));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default = Material::create("diffuse", &default_tp).expect("default diffuse should create");
    let explicit =
        Material::create("diffuse", &explicit_tp).expect("explicit diffuse should create");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let default_f = default
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let explicit_f = explicit
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((default_f - explicit_f).abs() < 1e-6);
}

#[test]
fn material_diffuse_uses_explicit_reflectance() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("reflectance", &Spectrum::from(0.8));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let diffuse = Material::create("diffuse", &tp).expect("diffuse material should create");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let f = diffuse
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let expected = SampledSpectrum::new(0.8).y(&lambda) / PI;
    assert!((f - expected).abs() < 1e-6);
}

#[test]
fn material_diffuse_transmission_scale_modulates_reflectance_and_transmittance() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut base_params = ParameterDictionary::new();
    let mut scaled_params = ParameterDictionary::new();
    base_params.add_float("scale", 1.0);
    scaled_params.add_float("scale", 0.5);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let base_tp = TextureParameterDictionary::new(&base_params, &f_tex, &s_tex);
    let scaled_tp = TextureParameterDictionary::new(&scaled_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let base = Material::create("diffusetransmission", &base_tp)
        .expect("base diffuse transmission should be created");
    let scaled = Material::create("diffusetransmission", &scaled_tp)
        .expect("scaled diffuse transmission should be created");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let base_f = base
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let scaled_f = scaled
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((scaled_f - 0.5 * base_f).abs() < 1e-6);
}

#[test]
fn material_diffuse_transmission_defaults_to_quarter_reflectance_and_transmittance() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_spectrum("reflectance", &Spectrum::from(0.25));
    explicit_params.add_spectrum("transmittance", &Spectrum::from(0.25));
    explicit_params.add_float("scale", 1.0);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default = Material::create("diffusetransmission", &default_tp)
        .expect("default diffuse transmission should create");
    let explicit = Material::create("diffusetransmission", &explicit_tp)
        .expect("explicit diffuse transmission should create");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let default_f = default
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let explicit_f = explicit
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((default_f - explicit_f).abs() < 1e-6);
}

#[test]
fn material_diffuse_transmission_errors_when_named_reflectance_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture reflectance", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("diffusetransmission", &tp).is_err());
}

#[test]
fn material_diffuse_transmission_errors_when_named_transmittance_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture transmittance", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("diffusetransmission", &tp).is_err());
}

#[test]
fn material_diffuse_transmission_accepts_legacy_kd_kt_aliases() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut alias_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    alias_params.add_spectrum("Kd", &Spectrum::from(0.25));
    alias_params.add_spectrum("Kt", &Spectrum::from(0.25));
    explicit_params.add_spectrum("reflectance", &Spectrum::from(0.25));
    explicit_params.add_spectrum("transmittance", &Spectrum::from(0.25));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let alias_tp = TextureParameterDictionary::new(&alias_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let alias = Material::create("diffusetransmission", &alias_tp)
        .expect("diffuse transmission with aliases should be created");
    let explicit = Material::create("diffusetransmission", &explicit_tp)
        .expect("explicit diffuse transmission should be created");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.0, 0.0, 1.0);
    let alias_f = alias
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    let explicit_f = explicit
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi, TransportMode::Radiance)
        .y(&lambda);
    assert!((alias_f - explicit_f).abs() < 1e-6);
}

#[test]
fn material_diffuse_transmission_negative_scale_clamps_to_black() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_float("scale", -1.0);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let material = Material::create("diffusetransmission", &tp)
        .expect("diffuse transmission with negative scale should be created");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi_reflect = Vector3f::new(0.0, 0.0, 1.0);
    let wi_transmit = Vector3f::new(0.0, 0.0, -1.0);
    let fr = material
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi_reflect, TransportMode::Radiance)
        .y(&lambda);
    let ft = material
        .test_get_bxdf(&si, &lambda)
        .f(&wo, &wi_transmit, TransportMode::Radiance)
        .y(&lambda);
    assert!(fr.abs() < 1e-6);
    assert!(ft.abs() < 1e-6);
}

#[test]
fn dielectric_sample_reports_v4_eta_convention() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let dielectric =
        Material::create("dielectric", &tp).expect("dielectric material should be created");
    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let sample = dielectric
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.99,
            &Point2f::new(0.5, 0.5),
            TransportMode::Radiance,
            BXDF_REFL_TRANS_TRANSMISSION,
        )
        .expect("dielectric transmission sample should succeed");

    assert!(sample.is_transmission());
    let expected_eta =
        pbrt_r4::util::spectrum::eta_from_spectrum(Spectrum::from(1.5), &lambda, 1.5);
    assert!((sample.eta - expected_eta).abs() < 1e-6);
}

#[test]
fn material_dielectric_defaults_to_eta_1_point_5() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_spectrum("eta", &Spectrum::from(1.5));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default =
        Material::create("dielectric", &default_tp).expect("default dielectric should create");
    let explicit =
        Material::create("dielectric", &explicit_tp).expect("explicit dielectric should create");
    assert!(
        (sampled_transmission_eta(&default, &si, &lambda)
            - sampled_transmission_eta(&explicit, &si, &lambda))
        .abs()
            < 1e-6
    );
}

#[test]
fn material_dielectric_eta_accepts_named_spectrum() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("spectrum eta", "glass-BK7");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let dielectric =
        Material::create("dielectric", &tp).expect("dielectric material should be created");
    let expected_eta = pbrt_r4::util::spectrum::lookup_named_spectrum("glass-BK7")
        .map(|s| pbrt_r4::util::spectrum::eta_from_spectrum(s, &lambda, 1.5))
        .expect("glass-BK7 should resolve");
    assert!((sampled_transmission_eta(&dielectric, &si, &lambda) - expected_eta).abs() < 1e-6);
}

#[test]
fn material_dielectric_ignores_index_named_spectrum() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("spectrum index", "glass-F11");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let dielectric =
        Material::create("dielectric", &tp).expect("dielectric material should be created");
    let expected_eta =
        pbrt_r4::util::spectrum::eta_from_spectrum(Spectrum::from(1.5), &lambda, 1.5);
    assert!((sampled_transmission_eta(&dielectric, &si, &lambda) - expected_eta).abs() < 1e-6);
}

#[test]
fn material_thin_dielectric_index_accepts_named_spectrum() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("index", &lookup_named_spectrum("glass-BK7").unwrap());
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let dielectric = Material::create("thindielectric", &tp)
        .expect("thin dielectric material should be created");
    let expected_eta = pbrt_r4::util::spectrum::lookup_named_spectrum("glass-BK7")
        .map(|s| pbrt_r4::util::spectrum::eta_from_spectrum(s, &lambda, 1.5))
        .expect("glass-BK7 should resolve");
    match dielectric.test_get_bxdf(&si, &lambda) {
        BxDF::ThinDielectric(bxdf) => {
            assert!((bxdf.eta() - expected_eta).abs() < 1e-6);
        }
        _ => panic!("expected thin dielectric bxdf"),
    }
}

#[test]
fn material_conductor_named_spectrum_changes_bxdf_response() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut named_params = ParameterDictionary::new();
    named_params.add_string("spectrum eta", "metal-Au-eta");
    named_params.add_string("spectrum k", "metal-Au-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let named_tp = TextureParameterDictionary::new(&named_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default_conductor =
        Material::create("conductor", &default_tp).expect("default conductor should be created");
    let named_conductor =
        Material::create("conductor", &named_tp).expect("named conductor should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let default_sample = default_conductor
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("default conductor should sample");
    let named_sample = named_conductor
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("named conductor should sample");

    assert!((named_sample.f.y(&lambda) - default_sample.f.y(&lambda)).abs() > 1e-3);
}

#[test]
fn material_conductor_default_matches_named_copper_eta_k() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut copper_params = ParameterDictionary::new();
    copper_params.add_string("spectrum eta", "metal-Cu-eta");
    copper_params.add_string("spectrum k", "metal-Cu-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let copper_tp = TextureParameterDictionary::new(&copper_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default_conductor =
        Material::create("conductor", &default_tp).expect("default conductor should be created");
    let copper_conductor =
        Material::create("conductor", &copper_tp).expect("copper conductor should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let default_sample = default_conductor
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("default conductor should sample");
    let copper_sample = copper_conductor
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("copper conductor should sample");

    assert!((default_sample.f.y(&lambda) - copper_sample.f.y(&lambda)).abs() < 1e-6);
}

#[test]
fn material_conductor_rejects_reflectance_with_eta_k() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("reflectance", &Spectrum::from(0.8));
    mat_params.add_string("spectrum eta", "metal-Au-eta");
    mat_params.add_string("spectrum k", "metal-Au-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("conductor", &tp);
    assert!(material.is_err());
}

#[test]
fn material_conductor_errors_when_named_roughness_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture roughness", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();

    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let material = Material::create("conductor", &tp);
    assert!(material.is_err());
}

#[test]
fn material_conductor_errors_when_named_reflectance_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture reflectance", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();

    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let material = Material::create("conductor", &tp);
    assert!(material.is_err());
}

#[test]
fn material_conductor_reflectance_mode_matches_v4_eta_k_conversion() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut reflectance_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    reflectance_params.add_spectrum("reflectance", &Spectrum::from(0.5));
    explicit_params.add_spectrum("eta", &Spectrum::one());
    explicit_params.add_spectrum("k", &Spectrum::from(2.0));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let reflectance_tp = TextureParameterDictionary::new(&reflectance_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let reflectance_conductor = Material::create("conductor", &reflectance_tp)
        .expect("reflectance conductor should be created");
    let explicit_conductor = Material::create("conductor", &explicit_tp)
        .expect("explicit eta/k conductor should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let reflectance_sample = reflectance_conductor
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("reflectance conductor should sample");
    let explicit_sample = explicit_conductor
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("explicit eta/k conductor should sample");

    assert!((reflectance_sample.f.y(&lambda) - explicit_sample.f.y(&lambda)).abs() < 1e-6);
}

#[test]
fn material_conductor_wavelengths_change_named_spectrum_response() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("spectrum eta", "metal-Au-eta");
    mat_params.add_string("spectrum k", "metal-Au-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();
    let conductor = Material::create("conductor", &tp).expect("conductor should be created");

    let blue_lambda = SampledWavelengths::sample_uniform(0.0, 360.0, 830.0);
    let red_lambda = SampledWavelengths::sample_uniform(0.95, 360.0, 830.0);

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let blue_sample = conductor
        .test_get_bxdf(&si, &blue_lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("blue conductor sample should succeed");
    let red_sample = conductor
        .test_get_bxdf(&si, &red_lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("red conductor sample should succeed");

    assert!(
        !SampledSpectrum::near_equal(&blue_sample.f, &red_sample.f, 1e-6),
        "expected wavelength-conditioned conductor samples to differ"
    );
}

#[test]
fn material_dielectric_terminates_secondary_wavelengths_for_spectral_eta() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("eta", &Spectrum::from_sampled(&[200.0, 900.0], &[3.5, 3.3]));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let material = Material::create("dielectric", &tp).expect("dielectric should be created");
    let si = SurfaceInteraction::default();
    let lambda = SampledWavelengths::sample_visible(0.31);

    assert!(!lambda.secondary_terminated());
    let updated = match &material {
        Material::Dielectric(m) => m.maybe_terminate_secondary_wavelengths(&si, &lambda),
        _ => panic!("expected dielectric material"),
    };
    let terminated = updated.expect("spectral eta should terminate secondary wavelengths");
    assert!(terminated.secondary_terminated());
}

#[test]
fn material_dielectric_keeps_secondary_wavelengths_for_constant_eta() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("eta", &Spectrum::from(1.5));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let material = Material::create("dielectric", &tp).expect("dielectric should be created");
    let si = SurfaceInteraction::default();
    let lambda = SampledWavelengths::sample_visible(0.31);

    assert!(!lambda.secondary_terminated());
    let updated = match &material {
        Material::Dielectric(m) => m.maybe_terminate_secondary_wavelengths(&si, &lambda),
        _ => panic!("expected dielectric material"),
    };
    assert!(
        updated.is_none(),
        "constant eta should not terminate secondary wavelengths"
    );
}

#[test]
fn material_coated_diffuse_terminates_secondary_wavelengths_for_spectral_eta() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("eta", &Spectrum::from_sampled(&[200.0, 900.0], &[3.5, 3.3]));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let material = Material::create("coateddiffuse", &tp).expect("coateddiffuse should be created");
    let si = SurfaceInteraction::default();
    let lambda = SampledWavelengths::sample_visible(0.31);

    assert!(!lambda.secondary_terminated());
    let updated = match &material {
        Material::CoatedDiffuse(m) => m.maybe_terminate_secondary_wavelengths(&si, &lambda),
        _ => panic!("expected coateddiffuse material"),
    };
    let terminated = updated.expect("spectral eta should terminate secondary wavelengths");
    assert!(terminated.secondary_terminated());
}

#[test]
fn material_coated_diffuse_keeps_secondary_wavelengths_for_constant_eta() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("eta", &Spectrum::from(1.5));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);
    let material = Material::create("coateddiffuse", &tp).expect("coateddiffuse should be created");
    let si = SurfaceInteraction::default();
    let lambda = SampledWavelengths::sample_visible(0.31);

    assert!(!lambda.secondary_terminated());
    let updated = match &material {
        Material::CoatedDiffuse(m) => m.maybe_terminate_secondary_wavelengths(&si, &lambda),
        _ => panic!("expected coateddiffuse material"),
    };
    assert!(
        updated.is_none(),
        "constant eta should not terminate secondary wavelengths"
    );
}

#[test]
fn material_conductor_uses_glossy_lobe_for_nonzero_roughness() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut smooth_params = ParameterDictionary::new();
    smooth_params.add_string("spectrum eta", "metal-Au-eta");
    smooth_params.add_string("spectrum k", "metal-Au-k");

    let mut rough_params = ParameterDictionary::new();
    rough_params.add_string("spectrum eta", "metal-Au-eta");
    rough_params.add_string("spectrum k", "metal-Au-k");
    rough_params.add_float("roughness", 0.25);
    rough_params.add_bool("remaproughness", false);

    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let smooth_tp = TextureParameterDictionary::new(&smooth_params, &f_tex, &s_tex);
    let rough_tp = TextureParameterDictionary::new(&rough_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let smooth =
        Material::create("conductor", &smooth_tp).expect("smooth conductor should be created");
    let rough =
        Material::create("conductor", &rough_tp).expect("rough conductor should be created");

    let smooth_bxdf = smooth.test_get_bxdf(&si, &lambda);
    let rough_bxdf = rough.test_get_bxdf(&si, &lambda);
    assert_eq!(smooth_bxdf.flags() & BXDF_SPECULAR, BXDF_SPECULAR);
    assert_eq!(smooth_bxdf.flags() & BXDF_GLOSSY, 0);
    assert_eq!(rough_bxdf.flags() & BXDF_GLOSSY, BXDF_GLOSSY);

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let wi = Vector3f::new(0.3, 0.0, Float::sqrt(1.0 - 0.3 * 0.3));
    assert_eq!(
        smooth_bxdf.f(&wo, &wi, TransportMode::Radiance).y(&lambda),
        0.0
    );
    assert!(rough_bxdf.f(&wo, &wi, TransportMode::Radiance).y(&lambda) > 0.0);

    let rough_sample = rough_bxdf
        .sample_f(
            &wo,
            0.5,
            &Point2f::new(0.37, 0.61),
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("rough conductor should sample");
    assert_eq!(rough_sample.flags & BXDF_GLOSSY, BXDF_GLOSSY);
    assert!(rough_sample.pdf.is_finite() && rough_sample.pdf > 0.0);
    assert!(rough_sample.wi.z > 0.0);
    assert!(rough_sample.f.y(&lambda).is_finite() && rough_sample.f.y(&lambda) > 0.0);
}

#[test]
fn material_coated_conductor_rejects_reflectance_with_eta_k() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("reflectance", &Spectrum::from(0.8));
    mat_params.add_string("spectrum conductor.eta", "metal-Au-eta");
    mat_params.add_string("spectrum conductor.k", "metal-Au-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("coatedconductor", &tp);
    assert!(material.is_err());
}

#[test]
fn material_coated_conductor_ignores_unprefixed_eta_k() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_spectrum("reflectance", &Spectrum::from(0.8));
    mat_params.add_string("spectrum eta", "metal-Au-eta");
    mat_params.add_string("spectrum k", "metal-Au-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let material = Material::create("coatedconductor", &tp);
    assert!(matches!(material, Ok(Material::CoatedConductor(_))));
}

#[test]
fn material_coated_conductor_errors_when_named_reflectance_texture_cannot_be_resolved() {
    let geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture reflectance", "missing");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(Material::create("coatedconductor", &tp).is_err());
}

#[test]
fn material_coated_conductor_reflectance_mode_matches_v4_eta_k_conversion() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let mut reflectance_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    reflectance_params.add_spectrum("reflectance", &Spectrum::from(0.5));
    explicit_params.add_spectrum("conductor.eta", &Spectrum::one());
    explicit_params.add_spectrum("conductor.k", &Spectrum::from(2.0));
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let reflectance_tp = TextureParameterDictionary::new(&reflectance_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let reflectance_coated = Material::create("coatedconductor", &reflectance_tp)
        .expect("reflectance coated conductor should be created");
    let explicit_coated = Material::create("coatedconductor", &explicit_tp)
        .expect("explicit eta/k coated conductor should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let reflectance_sample = reflectance_coated
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("reflectance coated conductor should sample");
    let explicit_sample = explicit_coated
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("explicit eta/k coated conductor should sample");

    assert!((reflectance_sample.f.y(&lambda) - explicit_sample.f.y(&lambda)).abs() < 1e-6);
}

#[test]
fn material_coated_conductor_defaults_to_named_copper_eta_k() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_string("spectrum conductor.eta", "metal-Cu-eta");
    explicit_params.add_string("spectrum conductor.k", "metal-Cu-k");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default_coated = Material::create("coatedconductor", &default_tp)
        .expect("default coated conductor should be created");
    let explicit_coated = Material::create("coatedconductor", &explicit_tp)
        .expect("explicit coated conductor should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let default_sample = default_coated
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("default coated conductor should sample");
    let explicit_sample = explicit_coated
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("explicit coated conductor should sample");

    assert!((default_sample.f.y(&lambda) - explicit_sample.f.y(&lambda)).abs() < 1e-6);
}

#[test]
fn material_coated_conductor_defaults_match_explicit_v4_parameters() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_spectrum("conductor.eta", &Spectrum::one());
    explicit_params.add_spectrum("conductor.k", &Spectrum::from(2.0));
    explicit_params.add_spectrum("interface.eta", &Spectrum::from(1.5));
    explicit_params.add_float("interface.roughness", 0.0);
    explicit_params.add_float("conductor.roughness", 0.0);
    explicit_params.add_float("thickness", 0.01);
    explicit_params.add_spectrum("albedo", &Spectrum::from(0.0));
    explicit_params.add_float("g", 0.0);
    explicit_params.add_bool("remaproughness", true);
    explicit_params.add_int("maxdepth", 10);
    explicit_params.add_int("nsamples", 1);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let default_coated = Material::create("coatedconductor", &default_tp)
        .expect("default coated conductor should be created");
    let explicit_coated = Material::create("coatedconductor", &explicit_tp)
        .expect("explicit coated conductor should be created");

    let wo = Vector3f::new(0.0, 0.0, 1.0);
    let u = Point2f::new(0.5, 0.5);
    let default_sample = default_coated
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("default coated conductor should sample");
    let explicit_sample = explicit_coated
        .test_get_bxdf(&si, &lambda)
        .sample_f(
            &wo,
            0.0,
            &u,
            TransportMode::Radiance,
            BXDF_REFL_TRANS_REFLECTION,
        )
        .expect("explicit coated conductor should sample");

    assert!((default_sample.f.y(&lambda) - explicit_sample.f.y(&lambda)).abs() < 1e-6);
}

#[test]
fn mix_material_amount_selects_between_two_materials() {
    let mut lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let base_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let base_tp = TextureParameterDictionary::new(&base_params, &f_tex, &s_tex);
    let si = SurfaceInteraction::default();

    let diffuse = Arc::new(Material::create("diffuse", &base_tp).expect("diffuse should exist"));
    let conductor =
        Arc::new(Material::create("conductor", &base_tp).expect("conductor should exist"));

    let mut mix0_params = ParameterDictionary::new();
    mix0_params.add_float("amount", 0.0);
    let mix0_tp = TextureParameterDictionary::new(&mix0_params, &f_tex, &s_tex);
    let mix0 =
        MixMaterial::create(&mix0_tp, &diffuse, &conductor).expect("mix material should create");
    let material0 = Material::Mix(mix0);
    assert!(matches!(
        material0.test_get_bxdf(&si, &lambda),
        BxDF::Diffuse(_)
    ));

    let mut mix1_params = ParameterDictionary::new();
    mix1_params.add_float("amount", 1.0);
    let mix1_tp = TextureParameterDictionary::new(&mix1_params, &f_tex, &s_tex);
    let mix1 =
        MixMaterial::create(&mix1_tp, &diffuse, &conductor).expect("mix material should create");
    let material1 = Material::Mix(mix1);
    assert!(matches!(
        material1.test_get_bxdf(&si, &lambda),
        BxDF::Conductor(_)
    ));
}

#[test]
fn mix_material_get_bsdf_uses_selected_material_at_zero_amount() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let base_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let base_tp = TextureParameterDictionary::new(&base_params, &f_tex, &s_tex);

    let diffuse = Arc::new(Material::create("diffuse", &base_tp).expect("diffuse should exist"));
    let dielectric =
        Arc::new(Material::create("dielectric", &base_tp).expect("dielectric should exist"));

    let mut mix_params = ParameterDictionary::new();
    mix_params.add_float("amount", 0.0);
    let mix_tp = TextureParameterDictionary::new(&mix_params, &f_tex, &s_tex);
    let mix = MixMaterial::create(&mix_tp, &diffuse, &dielectric).expect("mix should exist");
    let material = Material::Mix(mix);

    let si = SurfaceInteraction::default();

    let bsdf = material.test_get_bsdf(&si, &lambda);
    assert!(matches!(&bsdf.bxdf, BxDF::Diffuse(_)));
}

#[test]
fn mix_material_get_bsdf_uses_selected_material_at_one_amount() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let base_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let base_tp = TextureParameterDictionary::new(&base_params, &f_tex, &s_tex);

    let diffuse = Arc::new(Material::create("diffuse", &base_tp).expect("diffuse should exist"));
    let dielectric =
        Arc::new(Material::create("dielectric", &base_tp).expect("dielectric should exist"));

    let mut mix_params = ParameterDictionary::new();
    mix_params.add_float("amount", 1.0);
    let mix_tp = TextureParameterDictionary::new(&mix_params, &f_tex, &s_tex);
    let mix = MixMaterial::create(&mix_tp, &diffuse, &dielectric).expect("mix should exist");
    let material = Material::Mix(mix);

    let si = SurfaceInteraction::default();

    let bsdf = material.test_get_bsdf(&si, &lambda);
    assert!(matches!(&bsdf.bxdf, BxDF::Dielectric(_)));
}

#[test]
fn mix_material_defaults_to_half_amount() {
    let lambda = SampledWavelengths::sample_visible(0.5);
    let geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let explicit_params = {
        let mut p = ParameterDictionary::new();
        p.add_float("amount", 0.5);
        p
    };
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);

    let diffuse = Arc::new(Material::create("diffuse", &default_tp).expect("diffuse should exist"));
    let dielectric =
        Arc::new(Material::create("dielectric", &default_tp).expect("dielectric should exist"));

    let default_mix =
        MixMaterial::create(&default_tp, &diffuse, &dielectric).expect("default mix should exist");
    let explicit_mix = MixMaterial::create(&explicit_tp, &diffuse, &dielectric)
        .expect("explicit mix should exist");
    let material_default = Material::Mix(default_mix);
    let material_explicit = Material::Mix(explicit_mix);
    let si = SurfaceInteraction::default();

    let bsdf_default = material_default.test_get_bsdf(&si, &lambda);
    let bsdf_explicit = material_explicit.test_get_bsdf(&si, &lambda);
    assert_eq!(
        format!("{:?}", bsdf_default.bxdf),
        format!("{:?}", bsdf_explicit.bxdf)
    );
}
