use crate::util::base::*;
use crate::util::geometry::Bounds2f;

#[derive(Debug, Default, Clone)]
pub struct Distribution1D {
    pub func: Vec<Float>,
    pub cdf: Vec<Float>,
    pub func_int: Float,
    pub min: Float,
    pub max: Float,
}

impl Distribution1D {
    pub fn new(f: &[Float]) -> Self {
        Self::new_with_domain(f, 0.0, 1.0)
    }

    pub fn new_with_domain(f: &[Float], min: Float, max: Float) -> Self {
        assert!(max > min);
        let n = f.len();
        // pbrt-v4's PiecewiseConstant1D samples the non-negative function
        // represented by the absolute values of the input samples.
        let func: Vec<Float> = f.iter().map(|value| value.abs()).collect();
        let mut cdf = vec![0.0; n + 1];
        cdf[0] = 0.0;
        for i in 1..(n + 1) {
            cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / (n as Float);
        }
        let func_int = cdf[n];
        if func_int == 0.0 {
            for i in 1..(n + 1) {
                cdf[i] = (i as Float) / (n as Float);
            }
        } else {
            for i in 1..(n + 1) {
                cdf[i] /= func_int;
            }
        }
        Distribution1D {
            func,
            cdf,
            func_int,
            min,
            max,
        }
    }

    pub fn count(&self) -> usize {
        return self.func.len();
    }

    pub fn sample_continuous(&self, u: Float) -> (Float, Float, usize) {
        let offset = find_interval(&self.cdf, &|vv, i| -> bool {
            return vv[i] <= u;
        });

        let mut du = u - self.cdf[offset];
        if (self.cdf[offset + 1] - self.cdf[offset]) > 0.0 {
            du /= self.cdf[offset + 1] - self.cdf[offset];
        }
        let pdf = if self.func_int > 0.0 {
            self.func[offset] / self.func_int
        } else {
            0.0
        };
        let r =
            (((offset as Float) + du) / (self.count() as Float)) * (self.max - self.min) + self.min;
        return (r, pdf, offset);
    }

    pub fn sample_discrete(&self, u: Float) -> (usize, Float, Float) {
        let offset = find_interval(&self.cdf, &|vv, i| -> bool {
            return vv[i] <= u;
        });
        assert!(self.cdf[offset] <= u);
        assert!(u <= self.cdf[offset + 1]);
        let pdf = if self.func_int > 0.0 {
            self.func[offset] / (self.func_int * self.count() as Float)
        } else {
            0.0
        };
        let interval = self.cdf[offset + 1] - self.cdf[offset];
        let remapped = if interval > 0.0 {
            ((u - self.cdf[offset]) / interval).min(ONE_MINUS_EPSILON)
        } else {
            0.0
        };
        return (offset, pdf, remapped);
    }

    pub fn discrete_pdf(&self, index: usize) -> Float {
        if self.func_int > 0.0 {
            self.func[index] / (self.func_int * self.count() as Float)
        } else {
            0.0
        }
    }

    pub fn invert(&self, x: Float) -> Option<Float> {
        if self.count() == 0 || x < self.min || x > self.max {
            return None;
        }
        let c = (x - self.min) / (self.max - self.min) * self.count() as Float;
        let offset = usize::clamp(c as usize, 0, self.count() - 1);
        let delta = c - offset as Float;
        Some((1.0 - delta) * self.cdf[offset] + delta * self.cdf[offset + 1])
    }
}

#[derive(Clone, Debug)]
pub struct Distribution2D {
    pub conditional_v: Vec<Box<Distribution1D>>,
    pub marginal: Box<Distribution1D>,
    pub domain: Bounds2f,
}

impl Distribution2D {
    pub fn new(data: &[Float], nu: usize, nv: usize) -> Self {
        Self::new_with_domain(data, nu, nv, Bounds2f::from(((0.0, 0.0), (1.0, 1.0))))
    }

    pub fn new_with_domain(data: &[Float], nu: usize, nv: usize, domain: Bounds2f) -> Self {
        let mut conditional_v = Vec::with_capacity(nv);
        for v in 0..nv {
            let a = v * nu;
            let b = a + nu;
            let s = &data[a..b];
            conditional_v.push(Box::new(Distribution1D::new_with_domain(
                s,
                domain.min.x,
                domain.max.x,
            )));
        }
        let mut marginal_func = Vec::with_capacity(nv);
        for v in 0..nv {
            marginal_func.push(conditional_v[v].func_int);
        }
        Distribution2D {
            conditional_v,
            marginal: Box::new(Distribution1D::new_with_domain(
                &marginal_func,
                domain.min.y,
                domain.max.y,
            )),
            domain,
        }
    }

    pub fn sample_continuous(&self, u: &Point2f) -> (Point2f, Float) {
        let (d1, pdf1, v) = self.marginal.sample_continuous(u[1]);
        let (d0, pdf0, _) = self.conditional_v[v].as_ref().sample_continuous(u[0]);
        return (Point2f::new(d0, d1), pdf0 * pdf1);
    }

    pub fn sample_continuous_with_indices(&self, u: &Point2f) -> (Point2f, Float, usize, usize) {
        let (d1, pdf1, iv) = self.marginal.sample_continuous(u[1]);
        let (d0, pdf0, iu) = self.conditional_v[iv].as_ref().sample_continuous(u[0]);
        (Point2f::new(d0, d1), pdf0 * pdf1, iu, iv)
    }
    pub fn pdf(&self, p: &Point2f) -> Float {
        if self.conditional_v.is_empty() || self.marginal.func_int <= 0.0 {
            return 0.0;
        }
        let ucount = self.conditional_v[0].count();
        let vcount = self.marginal.count();
        if ucount == 0 || vcount == 0 {
            return 0.0;
        }
        let pu = self.domain.offset(&Point2f::new(p[0], p[1]));
        let iu = usize::clamp((pu.x * ucount as Float) as usize, 0, ucount - 1);
        let iv = usize::clamp((pu.y * vcount as Float) as usize, 0, vcount - 1);
        self.conditional_v[iv].func[iu] / self.marginal.func_int
    }

    pub fn invert(&self, p: &Point2f) -> Option<Point2f> {
        let v = self.marginal.invert(p.y)?;
        let offset = usize::clamp(
            ((p.y - self.domain.min.y) / (self.domain.max.y - self.domain.min.y)
                * self.conditional_v.len() as Float) as usize,
            0,
            self.conditional_v.len().checked_sub(1)?,
        );
        let u = self.conditional_v[offset].invert(p.x)?;
        Some(Point2f::new(u, v))
    }
}
