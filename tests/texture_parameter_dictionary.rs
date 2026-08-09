use pbrt_r4::paramdict::{ParameterDictionary, TextureParameterDictionary};
use pbrt_r4::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn texture_parameters_do_not_fall_back_to_another_parameter_source() {
    let mut shape_params = ParameterDictionary::new();
    shape_params.add_float("displacement", 0.25);
    let material_params = ParameterDictionary::new();
    let float_textures = HashMap::<String, Arc<FloatTexture>>::new();
    let spectrum_textures = HashMap::<String, Arc<SpectrumTexture>>::new();

    let shape_texture_params =
        TextureParameterDictionary::new(&shape_params, &float_textures, &spectrum_textures);
    assert!(shape_texture_params
        .get_float_texture_or_null("displacement")
        .unwrap()
        .is_some());

    let material_texture_params =
        TextureParameterDictionary::new(&material_params, &float_textures, &spectrum_textures);
    assert!(material_texture_params
        .get_float_texture_or_null("displacement")
        .unwrap()
        .is_none());
}
