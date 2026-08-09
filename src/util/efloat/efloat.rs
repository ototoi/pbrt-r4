use crate::util::base::Float;
use crate::util::misc::*;

use std::ops;

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub struct EFloat {
    pub v: Float,
    pub low: Float,
    pub high: Float,
}

impl EFloat {
    pub fn from_float(v: Float, err: Float) -> Self {
        if err == 0.0 {
            EFloat { v, low: v, high: v }
        } else {
            EFloat {
                v,
                low: next_float_down(v - err),
                high: next_float_up(v + err),
            }
        }
    }

    pub fn from_f32(v: f32, err: f32) -> Self {
        return Self::from_float(v as Float, err as Float);
    }

    pub fn from_f64(v: f64, err: f64) -> Self {
        return Self::from_float(v as Float, err as Float);
    }

    pub fn upper_bound(&self) -> Float {
        self.high
    }

    pub fn lower_bound(&self) -> Float {
        self.low
    }

    pub fn get_absolute_error(&self) -> Float {
        return next_float_up(Float::max(
            Float::abs(self.high - self.v),
            Float::abs(self.v - self.low),
        ));
    }

    pub fn sqrt(&self) -> Self {
        let v = Float::sqrt(self.v);
        let low = next_float_down(Float::sqrt(self.low));
        let high = next_float_up(Float::sqrt(self.high));
        EFloat { v, low, high }
    }

    pub fn abs(&self) -> Self {
        if self.low >= 0.0 {
            return self.clone();
        } else if self.high <= 0.0 {
            let v = -self.v;
            let low = -self.high;
            let high = -self.low;
            EFloat { v, low, high }
        } else {
            let v = Float::abs(self.v);
            let low = 0.0;
            let high = Float::max(-self.low, self.high);
            EFloat { v, low, high }
        }
    }

    pub fn quadratic(a: EFloat, b: EFloat, c: EFloat) -> Option<(EFloat, EFloat)> {
        const EPS: f64 = f64::EPSILON;
        let av = a.v as f64;
        let bv = b.v as f64;
        let cv = c.v as f64;
        let discrim = bv * bv - 4.0 * av * cv;
        if discrim < 0.0 {
            return None;
        }
        let root_discrim = discrim.sqrt();
        let float_root_discrim = EFloat::from_f64(root_discrim, EPS * root_discrim);

        let q = if b.v < 0.0 {
            (b - float_root_discrim) * -0.5
        } else {
            (b + float_root_discrim) * -0.5
        };
        let t0 = q / a;
        let t1 = c / q;
        if t0.v <= t1.v {
            return Some((t0, t1));
        } else {
            return Some((t1, t0));
        }
    }
}

impl ops::Add<EFloat> for EFloat {
    type Output = EFloat;
    fn add(self, ef: EFloat) -> EFloat {
        return EFloat {
            v: self.v + ef.v,
            low: next_float_down(self.low + ef.low),
            high: next_float_up(self.high + ef.high),
        };
    }
}

impl ops::Sub<EFloat> for EFloat {
    type Output = EFloat;
    fn sub(self, ef: EFloat) -> EFloat {
        return EFloat {
            v: self.v - ef.v,
            low: next_float_down(self.low - ef.high),
            high: next_float_up(self.high - ef.low),
        };
    }
}

impl ops::Mul<EFloat> for EFloat {
    type Output = EFloat;
    fn mul(self, ef: EFloat) -> EFloat {
        let v = self.v * ef.v;
        let prod = [
            self.low * ef.low,
            self.high * ef.low,
            self.low * ef.high,
            self.high * ef.high,
        ];
        let low = next_float_down(Float::min(
            Float::min(prod[0], prod[1]),
            Float::min(prod[2], prod[3]),
        ));
        let high = next_float_up(Float::max(
            Float::max(prod[0], prod[1]),
            Float::max(prod[2], prod[3]),
        ));
        return EFloat { v, low, high };
    }
}

impl ops::Div<EFloat> for EFloat {
    type Output = EFloat;
    fn div(self, ef: EFloat) -> EFloat {
        let v = self.v / ef.v;
        if ef.low < 0.0 && ef.high > 0.0 {
            let low = -Float::INFINITY;
            let high = Float::INFINITY;
            return EFloat { v, low, high };
        } else {
            let div = [
                self.lower_bound() / ef.lower_bound(),
                self.upper_bound() / ef.lower_bound(),
                self.lower_bound() / ef.upper_bound(),
                self.upper_bound() / ef.upper_bound(),
            ];
            let low = next_float_down(Float::min(
                Float::min(div[0], div[1]),
                Float::min(div[2], div[3]),
            ));
            let high = next_float_up(Float::max(
                Float::max(div[0], div[1]),
                Float::max(div[2], div[3]),
            ));
            return EFloat { v, low, high };
        }
    }
}

impl ops::Mul<Float> for EFloat {
    type Output = EFloat;
    fn mul(self, f: Float) -> EFloat {
        return self * EFloat::from_float(f, 0.0);
    }
}

impl ops::Mul<EFloat> for Float {
    type Output = EFloat;
    fn mul(self, rhs: EFloat) -> EFloat {
        return EFloat::from_float(self, 0.0) * rhs;
    }
}

impl From<f32> for EFloat {
    fn from(value: f32) -> Self {
        EFloat::from_f32(value, 0.0)
    }
}

impl From<(f32, f32)> for EFloat {
    fn from(value: (f32, f32)) -> Self {
        EFloat::from_f32(value.0, value.1)
    }
}

impl From<f64> for EFloat {
    fn from(value: f64) -> Self {
        EFloat::from_f64(value, 0.0)
    }
}

impl From<(f64, f64)> for EFloat {
    fn from(value: (f64, f64)) -> Self {
        EFloat::from_f64(value.0, value.1)
    }
}

impl From<EFloat> for f32 {
    fn from(value: EFloat) -> f32 {
        value.v as f32
    }
}

impl From<EFloat> for f64 {
    fn from(value: EFloat) -> f64 {
        value.v as f64
    }
}
