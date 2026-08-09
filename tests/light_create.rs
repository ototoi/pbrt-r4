use pbrt_r4::base::light::Light;
use pbrt_r4::media::MediumInterface;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::transform::Transform;

fn infinite_light_params() -> ParameterDictionary {
    ParameterDictionary::new()
}

#[test]
fn infinite_light_accepts_filename() {
    let params = {
        let mut params = infinite_light_params();
        params.add_string(
            "filename",
            "tests/scenes/crown-step13-textured-material-panels.exr",
        );
        params
    };
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("infinite light with filename should create");
    assert!(matches!(&*light, Light::Infinite(_)));
}

#[test]
fn infinite_light_accepts_mapname_when_filename_is_missing() {
    let params = {
        let mut params = infinite_light_params();
        params.add_string(
            "mapname",
            "tests/scenes/crown-step13-textured-material-panels.exr",
        );
        params
    };
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("infinite light with mapname should create");
    assert!(matches!(&*light, Light::Infinite(_)));
}

#[test]
fn infinite_light_without_filename_or_mapname_stays_uniform() {
    let params = infinite_light_params();
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("default infinite light should create");
    assert!(matches!(&*light, Light::Infinite(_)));
}
