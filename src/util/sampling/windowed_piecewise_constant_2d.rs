// pbrt-v4 verbatim port of `WindowedPiecewiseConstant2D`
// (util/sampling.h:895-983). Lets `PortalImageInfiniteLight` sample
// directions inside the portal's image bounds with the correct
// importance density.

use crate::util::base::*;
use crate::util::geometry::Bounds2f;

use super::summed_area_table::SummedAreaTable;

#[derive(Debug, Clone)]
pub struct WindowedPiecewiseConstant2D {
    sat: SummedAreaTable,
    /// Row-major `width × height` per-pixel weights.
    func: Vec<Float>,
    width: usize,
    height: usize,
}

impl WindowedPiecewiseConstant2D {
    pub fn new(values: Vec<Float>, width: usize, height: usize) -> Self {
        assert_eq!(values.len(), width * height);
        let sat = SummedAreaTable::new(&values, width, height);
        WindowedPiecewiseConstant2D {
            sat,
            func: values,
            width,
            height,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Sample `(p, pdf)` from the windowed function over `b`. Returns
    /// `None` if the integral over `b` is zero.
    pub fn sample(&self, u: &Point2f, b: &Bounds2f) -> Option<(Point2f, Float)> {
        let b_int = self.sat.integral(b);
        if b_int <= 0.0 {
            return None;
        }
        let inv_b_int = 1.0 / b_int;

        // Marginal CDF in x: Px(x) = ∫ b with bx.max.x = x divided by total.
        let px = |x: Float| -> Float {
            let bx = Bounds2f::new(&Point2f::new(b.min.x, b.min.y), &Point2f::new(x, b.max.y));
            self.sat.integral(&bx) * inv_b_int
        };
        let p_x = Self::sample_bisection(&px, u.x, b.min.x, b.max.x, self.width as i64);

        // Conditional CDF in y over the x-slab containing p_x.
        let nx = self.width as Float;
        let mut b_cond = Bounds2f::new(
            &Point2f::new(Float::floor(p_x * nx) / nx, b.min.y),
            &Point2f::new(Float::ceil(p_x * nx) / nx, b.max.y),
        );
        if b_cond.min.x == b_cond.max.x {
            b_cond.max.x += 1.0 / nx;
        }
        let cond_int = self.sat.integral(&b_cond);
        if cond_int <= 0.0 {
            return None;
        }
        let inv_cond = 1.0 / cond_int;
        let py = |y: Float| -> Float {
            let by = Bounds2f::new(
                &Point2f::new(b_cond.min.x, b_cond.min.y),
                &Point2f::new(b_cond.max.x, y),
            );
            self.sat.integral(&by) * inv_cond
        };
        let p_y = Self::sample_bisection(&py, u.y, b.min.y, b.max.y, self.height as i64);

        let p = Point2f::new(p_x, p_y);
        let pdf = self.eval(&p) * inv_b_int;
        Some((p, pdf))
    }

    /// PDF of point `p` under the windowed distribution over `b`.
    pub fn pdf(&self, p: &Point2f, b: &Bounds2f) -> Float {
        let func_int = self.sat.integral(b);
        if func_int <= 0.0 {
            return 0.0;
        }
        self.eval(p) / func_int
    }

    fn eval(&self, p: &Point2f) -> Float {
        let xi = ((p.x * self.width as Float) as i64)
            .max(0)
            .min(self.width as i64 - 1) as usize;
        let yi = ((p.y * self.height as Float) as i64)
            .max(0)
            .min(self.height as i64 - 1) as usize;
        self.func[yi * self.width + xi]
    }

    fn sample_bisection<F: Fn(Float) -> Float>(
        f: &F,
        u: Float,
        mut lo: Float,
        mut hi: Float,
        n: i64,
    ) -> Float {
        // Match v4's bracket-shrinking termination: keep halving until
        // `lo` and `hi` are within the same integer grid cell of width 1/n.
        while (Float::ceil(n as Float * hi) - Float::floor(n as Float * lo)) > 1.0 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > u {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let p_lo = f(lo);
        let p_hi = f(hi);
        let t = if p_hi - p_lo > 0.0 {
            (u - p_lo) / (p_hi - p_lo)
        } else {
            0.0
        };
        let v = lerp(t, lo, hi);
        v.clamp(lo, hi)
    }
}

fn lerp(t: Float, a: Float, b: Float) -> Float {
    (1.0 - t) * a + t * b
}
