use pbrt_r4::base::filter::Filter;
use pbrt_r4::film::film_base::FilmBaseParameters;
use pbrt_r4::film::pixel_sensor::PixelSensor;
use pbrt_r4::film::spectral_film::SpectralFilm;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::base::{Float, Point2f, Point2i};
use pbrt_r4::util::geometry::Bounds2i;
use pbrt_r4::util::spectrum::{SampledSpectrum, SampledWavelengths};

fn make_film(lambda_min: Float, lambda_max: Float, n_buckets: usize) -> SpectralFilm {
    let filter =
        Filter::create("box", &ParameterDictionary::new()).expect("box filter should create");
    let sensor = PixelSensor::create("cie1931", 100.0, 0.0).unwrap();
    let params = FilmBaseParameters {
        full_resolution: Point2i::from((4, 4)),
        pixel_bounds: Bounds2i::from(((0, 0), (4, 4))),
        filter,
        diagonal: 35.0,
        filename: "test.exr".to_string(),
        pixel_sensor: sensor,
        scale: 1.0,
        max_sample_luminance: Float::INFINITY,
    };
    SpectralFilm::with_spectral_range(params, lambda_min, lambda_max, n_buckets)
}

#[test]
fn lambda_to_bucket_clamps_below_range() {
    let film = make_film(400.0, 700.0, 4);
    assert_eq!(film.lambda_to_bucket(350.0), 0);
}

#[test]
fn lambda_to_bucket_clamps_above_range() {
    let film = make_film(400.0, 700.0, 4);
    assert_eq!(film.lambda_to_bucket(800.0), 3);
}

#[test]
fn lambda_to_bucket_distributes_midrange() {
    let film = make_film(400.0, 700.0, 3);
    assert_eq!(film.lambda_to_bucket(450.0), 0);
    assert_eq!(film.lambda_to_bucket(550.0), 1);
    assert_eq!(film.lambda_to_bucket(650.0), 2);
}

#[test]
fn sample_wavelengths_uses_uniform_distribution() {
    let film = make_film(400.0, 700.0, 16);
    let lambda = film.sample_wavelengths(0.5);
    let lambda0 = lambda[0];
    assert!(
        (lambda0 - 550.0).abs() < 1e-3,
        "expected ~550, got {}",
        lambda0
    );
}

#[test]
fn spectral_buckets_receive_sample_contribution() {
    let film = make_film(400.0, 700.0, 3);
    let pixel_bounds = film.base().pixel_bounds();
    let mut tile = film.get_film_tile(&pixel_bounds);
    let tile_area = tile.pixel_bounds.area() as usize;
    assert_eq!(tile.spectral_buckets.len(), tile_area * 3);
    assert_eq!(tile.spectral_bucket_weights.len(), tile_area * 3);

    let lambda = SampledWavelengths::sample_uniform(0.5, 400.0, 700.0);
    let l_packet = SampledSpectrum::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let _l_spectrum = l_packet.to_dense(&lambda);
    let p_film = Point2f::new(0.5, 0.5);
    tile.add_sample(&p_film, l_packet, &lambda, None, 1.0);

    let n_buckets = 3;
    let tile_width = (tile.pixel_bounds.max.x - tile.pixel_bounds.min.x) as usize;
    let local_x = (-tile.pixel_bounds.min.x) as usize;
    let local_y = (-tile.pixel_bounds.min.y) as usize;
    let off = (local_y * tile_width + local_x) * n_buckets;
    assert!(tile.spectral_buckets[off] > 0.0);
    assert!(tile.spectral_buckets[off + 1] > 0.0);
    assert!(tile.spectral_buckets[off + 2] > 0.0);
    assert!(tile.spectral_buckets[off + 2] > tile.spectral_buckets[off + 1]);
}
