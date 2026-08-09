use std::sync::Arc;

use pbrt_r4::film::film_tile::FT_SZ;
use pbrt_r4::film::pixel_sensor::PixelSensor;
use pbrt_r4::film::FilmTile;
use pbrt_r4::util::base::{Float, Point2i};
use pbrt_r4::util::geometry::vector2::Vector2;
use pbrt_r4::util::geometry::Bounds2i;
use pbrt_r4::util::spectrum::sampled::{SampledSpectrum, SampledWavelengths};

type Vector2f = Vector2<f32>;

fn make_tile() -> FilmTile {
    let pixel_bounds = Bounds2i::from(((0, 0), (3, 3)));
    let filter_radius = Vector2f::new(1.5, 1.5);
    let filter_table = Arc::new([1.0; FT_SZ]);
    let sensor = PixelSensor::create("cie1931", 100.0, 0.0).unwrap();
    FilmTile::new(
        &pixel_bounds,
        &filter_radius,
        &filter_table,
        Float::INFINITY,
        sensor,
    )
}

#[test]
fn add_sample_pixel_only_updates_target_pixel() {
    let mut tile = make_tile();
    let lambda = SampledWavelengths::sample_visible(0.5);
    let l = SampledSpectrum::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let p = Point2i::new(1, 1);
    tile.add_sample_pixel(&p, l, &lambda, None, 2.0);

    for y in 0..3 {
        for x in 0..3 {
            let index = tile.get_pixel_index(&Point2i::new(x, y));
            let pixel = &tile.pixels[index];
            if x == 1 && y == 1 {
                assert_eq!(pixel.filter_weight_sum, 2.0);
                assert!(pixel.contrib_sum.iter().any(|v| *v > 0.0));
            } else {
                assert_eq!(pixel.filter_weight_sum, 0.0);
                assert_eq!(pixel.contrib_sum, [0.0; 3]);
            }
        }
    }
}
