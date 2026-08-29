use pbrt_r4::base::film::Film;
use pbrt_r4::base::filter::Filter;
use pbrt_r4::cpu::integrators::bdpt::compute_light_tracing_splat_scale;
use pbrt_r4::paramdict::ParameterDictionary;

fn create_film(params: &ParameterDictionary) -> std::sync::Arc<std::sync::RwLock<Film>> {
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    Film::create("rgb", params, &filter).expect("film should create")
}

#[test]
fn light_tracing_splat_scale_is_one_without_crop() {
    let mut params = ParameterDictionary::new();
    params.add_int("xresolution", 400);
    params.add_int("yresolution", 200);
    let film = create_film(&params);
    let film = film.read().unwrap();

    assert_eq!(compute_light_tracing_splat_scale(&film), 1.0);
}

#[test]
fn light_tracing_splat_scale_uses_full_to_cropped_area_ratio() {
    let mut params = ParameterDictionary::new();
    params.add_int("xresolution", 400);
    params.add_int("yresolution", 200);
    params.add_floats("cropwindow", &[0.25, 0.75, 0.25, 0.75]);
    let film = create_film(&params);
    let film = film.read().unwrap();

    assert_eq!(compute_light_tracing_splat_scale(&film), 4.0);
}
