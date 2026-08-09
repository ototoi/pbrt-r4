use pbrt_r4::base::film::Film;
use pbrt_r4::base::filter::Filter;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::geometry::Bounds2i;

#[test]
fn create_film_accepts_rgb_gbuffer_and_spectral() {
    let params = ParameterDictionary::new();
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    assert!(Film::create("rgb", &params, &filter).is_ok());
    assert!(Film::create("gbuffer", &params, &filter).is_ok());
    assert!(Film::create("spectral", &params, &filter).is_ok());
}

#[test]
fn create_filter_rejects_unknown_type() {
    assert!(Filter::create("not-a-real-filter", &ParameterDictionary::new()).is_err());
}

#[test]
fn create_film_rejects_unknown_type() {
    let params = ParameterDictionary::new();
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    assert!(Film::create("unknown_film_type", &params, &filter).is_err());
}

#[test]
fn create_film_prefers_cropwindow_over_pixelbounds() {
    let mut params = ParameterDictionary::new();
    params.add_ints("pixelbounds", &[100, 200, 50, 150]);
    params.add_floats("cropwindow", &[0.25, 0.75, 0.1, 0.9]);
    params.add_int("xresolution", 400);
    params.add_int("yresolution", 200);
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let film = Film::create("rgb", &params, &filter).expect("film should create");
    let film = film.read().unwrap();
    assert_eq!(
        film.get_pixel_bounds(),
        Bounds2i::from(((100, 20), (300, 180)))
    );
}

#[test]
fn create_film_supports_pixelbounds() {
    let mut params = ParameterDictionary::new();
    params.add_ints("pixelbounds", &[10, 30, 20, 50]);
    params.add_int("xresolution", 64);
    params.add_int("yresolution", 64);
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let film = Film::create("rgb", &params, &filter).expect("film should create");
    let film = film.read().unwrap();
    assert_eq!(
        film.get_pixel_bounds(),
        Bounds2i::from(((10, 20), (30, 50)))
    );
}

#[test]
fn create_film_respects_sensor_and_iso_parameters() {
    let mut params = ParameterDictionary::new();
    params.add_string("sensor", "canon_eos_5d_mkiv");
    params.add_float("iso", 150.0);
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let film = Film::create("rgb", &params, &filter).expect("film should create");
    let film = film.read().unwrap();
    let sensor = film.base().pixel_sensor();
    assert_eq!(sensor.sensor_name(), "canon_eos_5d_mkiv");
    assert!((sensor.imaging_ratio() - 1.5).abs() < 1e-6);
}

#[test]
fn create_rgb_film_defaults_to_cie1931_sensor_and_pbrt_exr_filename() {
    let params = ParameterDictionary::new();
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let film = Film::create("rgb", &params, &filter).expect("film should create");
    let film = film.read().unwrap();
    assert_eq!(film.base().pixel_sensor().sensor_name(), "cie1931");
    assert!(film.base().filename().ends_with("pbrt.exr"));
}

#[test]
fn create_spectral_film_defaults_to_v4_lambda_range_and_bucket_count() {
    let params = ParameterDictionary::new();
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let film = Film::create("spectral", &params, &filter).expect("film should create");
    let film = film.read().unwrap();
    match &*film {
        Film::Spectral(s) => {
            assert_eq!(s.lambda_min(), 360.0);
            assert_eq!(s.lambda_max(), 830.0);
            assert_eq!(s.n_buckets(), 16);
        }
        _ => panic!("expected spectral film"),
    }
}

#[test]
fn create_film_threads_maxcomponentvalue_into_film_tiles() {
    let mut params = ParameterDictionary::new();
    params.add_float("maxcomponentvalue", 12.5);
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let film = Film::create("rgb", &params, &filter).expect("film should create");
    let film = film.read().unwrap();
    let tile = film.get_film_tile(&Bounds2i::from(((0, 0), (1, 1))));
    assert!((tile.max_sample_luminance - 12.5).abs() < 1e-6);
}

#[test]
fn create_film_rejects_unknown_sensor() {
    let mut params = ParameterDictionary::new();
    params.add_string("sensor", "not-a-sensor");
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    assert!(Film::create("rgb", &params, &filter).is_err());
}
