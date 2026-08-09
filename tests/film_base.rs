use pbrt_r4::base::filter::{Filter, GaussianFilter, TriangleFilter};
use pbrt_r4::film::film_base::{
    add_splat_packet_into_tiles, make_splat_tiles, normalize_pixel, FilmBase, FilmBaseParameters,
};
use pbrt_r4::film::pixel_sensor::PixelSensor;
use pbrt_r4::util::base::{Float, Point2i};
use pbrt_r4::util::geometry::{Bounds2i, Vector2};
use pbrt_r4::util::spectrum::{SampledSpectrum, SampledWavelengths};

type Vector2f = Vector2<f32>;

#[test]
fn sample_bounds_match_v4_filter_center_bounds() {
    let film = FilmBase::new(&FilmBaseParameters {
        full_resolution: Point2i::new(100, 80),
        pixel_bounds: Bounds2i::new(&Point2i::new(0, 0), &Point2i::new(100, 80)),
        filter: Filter::Gaussian(GaussianFilter::new(&Vector2f::new(1.5, 1.5), 0.5)),
        diagonal: 35.0,
        filename: "test.exr".to_string(),
        pixel_sensor: PixelSensor::default(),
        scale: 1.0,
        max_sample_luminance: Float::INFINITY,
    });

    assert_eq!(
        film.sample_bounds(),
        Bounds2i::new(&Point2i::new(-1, -1), &Point2i::new(101, 81))
    );
}

#[test]
fn normalize_pixel_preserves_negative_weighted_rgb() {
    let splat_pixel = [0.0, 0.0, 0.0];

    let c = normalize_pixel([-4.0, 6.0, -2.0], 2.0, &splat_pixel, 1.0);

    assert_eq!(c, [-2.0, 3.0, -1.0]);
}

#[test]
fn normalize_pixel_preserves_negative_splats_after_scaling() {
    let splat_pixel = [-0.25, 0.5, -1.0];

    let c = normalize_pixel([1.0, -1.0, 0.0], 1.0, &splat_pixel, 2.0);

    assert_eq!(c, [1.5, -1.0, -2.0]);
}

#[test]
fn add_splat_packet_filters_footprint_and_clamps_sensor_rgb() {
    let bounds = Bounds2i::new(&Point2i::new(0, 0), &Point2i::new(4, 4));
    let (splat_tiles, splat_size) = make_splat_tiles(&bounds);
    let sensor = PixelSensor::create("cie1931", 100.0, 0.0).unwrap();
    let filter = Filter::Triangle(TriangleFilter::new(&Vector2f::new(1.0, 1.0)));
    let lambda = SampledWavelengths::sample_visible(0.37);
    let spectrum = SampledSpectrum::new(4.0);
    let sensor_rgb = sensor.to_sensor_rgb_from_packet(&spectrum, &lambda);
    let max_component = sensor_rgb[0].max(sensor_rgb[1]).max(sensor_rgb[2]);
    let max_sample_luminance = max_component * 0.5;

    add_splat_packet_into_tiles(
        &splat_tiles,
        splat_size,
        bounds,
        &sensor,
        &filter,
        max_sample_luminance,
        &Vector2f::new(1.25, 1.25),
        &spectrum,
        &lambda,
    );

    let tile = splat_tiles[0].read().unwrap();
    assert!(tile
        .pixels
        .iter()
        .any(|pixel| pixel.iter().any(|value| *value != 0.0)));
    assert_eq!(tile.pixels[2 * 4 + 2], [0.0, 0.0, 0.0]);
}
