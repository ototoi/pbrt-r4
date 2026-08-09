use pbrt_r4::base::camera::CameraSample;
use pbrt_r4::base::Filter;
use pbrt_r4::cameras::base_camera::CameraBaseParameters;
use pbrt_r4::cameras::spherical::{SphericalCamera, SphericalMapping};
use pbrt_r4::film::film_base::FilmBaseParameters;
use pbrt_r4::film::rgb_film::RGBFilm;
use pbrt_r4::filters::box_filter::BoxFilter;
use pbrt_r4::prelude::*;

use std::sync::{Arc, RwLock};

fn test_film() -> Arc<RwLock<Film>> {
    let filter = BoxFilter::new(&Vector2f::new(0.5, 0.5));
    let params = FilmBaseParameters {
        full_resolution: Point2i::new(2, 2),
        pixel_bounds: Bounds2i::new(&Point2i::new(0, 0), &Point2i::new(2, 2)),
        filter: Filter::Box(filter),
        diagonal: 35.0,
        filename: "camera_spherical.exr".to_string(),
        pixel_sensor: PixelSensor::default(),
        scale: 1.0,
        max_sample_luminance: Float::INFINITY,
    };
    Arc::new(RwLock::new(Film::Rgb(RGBFilm::new(params))))
}

fn test_base_params() -> CameraBaseParameters {
    let film = test_film();
    CameraBaseParameters::new(
        &AnimatedTransform::with_identity(0.0, 1.0),
        0.0,
        1.0,
        &film,
        &None,
    )
}

#[test]
fn spherical_camera_equirectangular_variant_generates_expected_direction() {
    let camera = SphericalCamera::new(test_base_params(), SphericalMapping::Equirectangular);
    let sample = CameraSample {
        p_film: Point2f::new(1.0, 1.0),
        time: 0.0,
        p_lens: Point2f::new(0.0, 0.0),
        filter_weight: 1.0,
    };
    let lambda = SampledWavelengths::sample_visible(0.5);

    let ray = camera
        .generate_ray(&sample, &lambda)
        .expect("ray should be generated");
    assert!((ray.ray.d.x + 1.0).abs() < 1e-6);
    assert!((ray.ray.d.y - 0.0).abs() < 1e-6);
    assert!((ray.ray.d.z - 0.0).abs() < 1e-6);
}

#[test]
fn spherical_camera_equalarea_default_remains_distinct_from_equirectangular() {
    let equirectangular =
        SphericalCamera::new(test_base_params(), SphericalMapping::Equirectangular);
    let equalarea = SphericalCamera::new(test_base_params(), SphericalMapping::EqualArea);
    let sample = CameraSample {
        p_film: Point2f::new(1.0, 0.0),
        time: 0.0,
        p_lens: Point2f::new(0.0, 0.0),
        filter_weight: 1.0,
    };
    let lambda = SampledWavelengths::sample_visible(0.5);

    let eq_ray = equirectangular
        .generate_ray(&sample, &lambda)
        .expect("ray should be generated");
    let ea_ray = equalarea
        .generate_ray(&sample, &lambda)
        .expect("ray should be generated");
    assert_ne!(eq_ray.ray.d, ea_ray.ray.d);
}

#[test]
fn equal_area_mapping_round_trips_unit_square() {
    for uv in [
        Point2f::new(0.1, 0.2),
        Point2f::new(0.5, 0.5),
        Point2f::new(0.9, 0.8),
    ] {
        let d = equal_area_square_to_sphere(&uv);
        let mapped = equal_area_sphere_to_square(&d);
        assert!((mapped.x - uv.x).abs() < 2e-4);
        assert!((mapped.y - uv.y).abs() < 2e-4);
    }
}
