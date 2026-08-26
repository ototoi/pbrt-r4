use crate::util::base::*;
use crate::util::Bounds2f;

#[derive(Clone, Debug)]
pub struct PiecewiseConstant1D {
    func: Vec<Float>,
    cdf: Vec<Float>,
    domain: (Float, Float),
    integral: Float,
}

impl PiecewiseConstant1D {
    pub fn new(func: &[Float], a: Float, b: Float) -> Self {
        let n = func.len();
        let values: Vec<Float> = func.iter().map(|value| value.abs()).collect();
        let mut cdf = vec![0.0; n + 1];
        for i in 1..=n {
            cdf[i] = cdf[i - 1] + values[i - 1] * (b - a) / n as Float;
        }
        let integral = cdf[n];
        if integral == 0.0 {
            for i in 1..=n {
                cdf[i] = i as Float / n as Float;
            }
        } else {
            for i in 1..=n {
                cdf[i] /= integral;
            }
        }
        Self {
            func: values,
            cdf,
            domain: (a, b),
            integral,
        }
    }

    pub fn integral(&self) -> Float {
        self.integral
    }

    pub fn sample(&self, u: Float) -> (Float, Float, Float) {
        if self.func.is_empty() {
            return (self.domain.0, 0.0, 0.0);
        }
        let mut offset = 0usize;
        // Keep the offset in the final interval when u is exactly one,
        // matching v4's clamped FindInterval result.
        while offset + 2 < self.cdf.len() && self.cdf[offset + 1] <= u {
            offset += 1;
        }
        let du = if self.cdf[offset + 1] > self.cdf[offset] {
            (u - self.cdf[offset]) / (self.cdf[offset + 1] - self.cdf[offset])
        } else {
            0.0
        };
        let remapped = du.min(1.0 - 1e-7);
        let pdf = if self.integral > 0.0 {
            self.func[offset] / self.integral
        } else {
            0.0
        };
        let value = self.domain.0
            + (offset as Float + remapped) * (self.domain.1 - self.domain.0)
                / self.func.len() as Float;
        (value, pdf, remapped)
    }

    pub fn pdf(&self, x: Float) -> Float {
        if x < self.domain.0 || x > self.domain.1 || self.func.is_empty() {
            return 0.0;
        }
        let u = (x - self.domain.0) / (self.domain.1 - self.domain.0);
        let i = usize::clamp(
            (u * self.func.len() as Float) as usize,
            0,
            self.func.len() - 1,
        );
        if self.integral > 0.0 {
            // `integral` already includes the width of the domain, so this
            // ratio is a density with respect to x. Dividing by the domain
            // width again would make the PDF integrate to 1 / (b - a).
            self.func[i] / self.integral
        } else {
            0.0
        }
    }

    pub fn invert(&self, x: Float) -> Option<Float> {
        if x < self.domain.0 || x > self.domain.1 || self.func.is_empty() {
            return None;
        }
        let u = (x - self.domain.0) / (self.domain.1 - self.domain.0);
        let i = usize::clamp(
            (u * self.func.len() as Float) as usize,
            0,
            self.func.len() - 1,
        );
        let du = u * self.func.len() as Float - i as Float;
        Some(lerp(du, self.cdf[i], self.cdf[i + 1]))
    }
}

#[derive(Clone, Debug)]
pub struct PiecewiseConstant2D {
    domain: Bounds2f,
    p_conditional_v: Vec<PiecewiseConstant1D>,
    p_marginal: PiecewiseConstant1D,
}

impl PiecewiseConstant2D {
    pub fn new(data: &[Float], nx: usize, ny: usize) -> Self {
        Self::new_with_domain(
            data,
            nx,
            ny,
            Bounds2f::new(&Point2f::zero(), &Point2f::new(1.0, 1.0)),
        )
    }

    pub fn new_with_domain(data: &[Float], nx: usize, ny: usize, domain: Bounds2f) -> Self {
        assert_eq!(data.len(), nx * ny);
        let mut p_conditional_v = Vec::with_capacity(ny);
        for v in 0..ny {
            let row = &data[v * nx..(v + 1) * nx];
            p_conditional_v.push(PiecewiseConstant1D::new(row, domain.min.x, domain.max.x));
        }
        let mut marginal = Vec::with_capacity(ny);
        for row in &p_conditional_v {
            marginal.push(row.integral());
        }
        let p_marginal = PiecewiseConstant1D::new(&marginal, domain.min.y, domain.max.y);
        Self {
            domain,
            p_conditional_v,
            p_marginal,
        }
    }

    pub fn sample_continuous(&self, u: &Point2f) -> (Point2f, Float) {
        let (v, pdf_v, _) = self.p_marginal.sample(u.y);
        let iv = usize::clamp(
            (((v - self.domain.min.y) / (self.domain.max.y - self.domain.min.y))
                * self.p_conditional_v.len() as Float) as usize,
            0,
            self.p_conditional_v.len() - 1,
        );
        let (u0, pdf_u, _) = self.p_conditional_v[iv].sample(u.x);
        (Point2f::new(u0, v), pdf_u * pdf_v)
    }

    pub fn pdf(&self, p: &Point2f) -> Float {
        if !self.domain.inside(p) {
            return 0.0;
        }
        let u = self.domain.offset(p);
        let iu = usize::clamp(
            (u.x * self.p_conditional_v[0].func.len() as Float) as usize,
            0,
            self.p_conditional_v[0].func.len() - 1,
        );
        let iv = usize::clamp(
            (u.y * self.p_conditional_v.len() as Float) as usize,
            0,
            self.p_conditional_v.len() - 1,
        );
        let denom = self.p_marginal.integral();
        if denom > 0.0 {
            self.p_conditional_v[iv].func[iu] / denom
        } else {
            0.0
        }
    }
}
