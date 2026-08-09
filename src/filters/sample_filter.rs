use super::filter_sample::FilterSample;
use crate::util::base::{lerp, Float, Point2f, Vector2f};
use crate::util::sampling::Distribution2D;

pub trait SampledFilter {
    fn radius(&self) -> Vector2f;
    fn evaluate(&self, p: &Point2f) -> Float;
}

#[derive(Clone)]
pub struct FilterSampler {
    domain_min: Point2f,
    domain_max: Point2f,
    nx: usize,
    ny: usize,
    f: Vec<Float>,
    distrib: Distribution2D,
}

impl FilterSampler {
    pub fn new(filter: &impl SampledFilter) -> Self {
        let radius = filter.radius();
        let domain_min = Point2f::new(-radius.x, -radius.y);
        let domain_max = Point2f::new(radius.x, radius.y);

        let nx = usize::max(1, (32.0 * radius.x) as usize);
        let ny = usize::max(1, (32.0 * radius.y) as usize);

        let mut f = vec![0.0; nx * ny];
        let mut abs_f = vec![0.0; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let ux = (x as Float + 0.5) / nx as Float;
                let uy = (y as Float + 0.5) / ny as Float;
                let p = Point2f::new(
                    lerp(ux, domain_min.x, domain_max.x),
                    lerp(uy, domain_min.y, domain_max.y),
                );
                let value = filter.evaluate(&p);
                f[y * nx + x] = value;
                abs_f[y * nx + x] = value.abs();
            }
        }
        let distrib = Distribution2D::new(&abs_f, nx, ny);

        FilterSampler {
            domain_min,
            domain_max,
            nx,
            ny,
            f,
            distrib,
        }
    }

    pub fn sample(&self, u: &Point2f) -> FilterSample {
        let (d, pdf_unit, ix, iy) = self.distrib.sample_continuous_with_indices(u);
        let p = Point2f::new(
            lerp(d.x, self.domain_min.x, self.domain_max.x),
            lerp(d.y, self.domain_min.y, self.domain_max.y),
        );

        let area =
            (self.domain_max.x - self.domain_min.x) * (self.domain_max.y - self.domain_min.y);
        let pdf = pdf_unit / area;
        let i = usize::min(iy, self.ny - 1) * self.nx + usize::min(ix, self.nx - 1);
        let weight = if pdf > 0.0 { self.f[i] / pdf } else { 0.0 };
        FilterSample { p, weight }
    }
}
