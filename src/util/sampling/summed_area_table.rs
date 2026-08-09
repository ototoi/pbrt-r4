use crate::util::base::*;
use crate::util::geometry::Bounds2f;

#[derive(Debug, Clone)]
pub struct SummedAreaTable {
    sum: Vec<f64>,
    width: usize,
    height: usize,
}

impl SummedAreaTable {
    /// `values` is a row-major `width × height` array.
    pub fn new(values: &[Float], width: usize, height: usize) -> Self {
        assert_eq!(values.len(), width * height);
        let mut sum = vec![0.0_f64; width * height];
        let at = |x: usize, y: usize| (y * width + x);
        sum[at(0, 0)] = values[at(0, 0)] as f64;
        for x in 1..width {
            sum[at(x, 0)] = values[at(x, 0)] as f64 + sum[at(x - 1, 0)];
        }
        for y in 1..height {
            sum[at(0, y)] = values[at(0, y)] as f64 + sum[at(0, y - 1)];
        }
        for y in 1..height {
            for x in 1..width {
                sum[at(x, y)] = values[at(x, y)] as f64 + sum[at(x - 1, y)] + sum[at(x, y - 1)]
                    - sum[at(x - 1, y - 1)];
            }
        }
        SummedAreaTable { sum, width, height }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// `∫_b f(x, y) dx dy` -- area-weighted average over the
    /// closed sub-rectangle `b ⊆ [0, 1]²` of the original 2D
    /// function. Returns max(0, …) to clamp negative drift.
    pub fn integral(&self, b: &Bounds2f) -> Float {
        let s = (self.lookup(b.max.x, b.max.y) - self.lookup(b.min.x, b.max.y))
            + (self.lookup(b.min.x, b.min.y) - self.lookup(b.max.x, b.min.y));
        let denom = (self.width * self.height) as f64;
        Float::max(0.0, (s / denom) as Float)
    }

    fn lookup(&self, x: Float, y: Float) -> f64 {
        let xs = x * self.width as Float;
        let ys = y * self.height as Float;
        let x0 = xs as i64;
        let y0 = ys as i64;
        let dx = (xs - x0 as Float) as f64;
        let dy = (ys - y0 as Float) as f64;
        let v00 = self.lookup_int(x0, y0);
        let v10 = self.lookup_int(x0 + 1, y0);
        let v01 = self.lookup_int(x0, y0 + 1);
        let v11 = self.lookup_int(x0 + 1, y0 + 1);
        (1.0 - dx) * (1.0 - dy) * v00
            + (1.0 - dx) * dy * v01
            + dx * (1.0 - dy) * v10
            + dx * dy * v11
    }

    fn lookup_int(&self, x: i64, y: i64) -> f64 {
        // v4 returns zero at x==0 or y==0 boundaries: those represent
        // the "before the first cell" edge of the cumulative table.
        if x <= 0 || y <= 0 {
            return 0.0;
        }
        let xi = (x as usize - 1).min(self.width - 1);
        let yi = (y as usize - 1).min(self.height - 1);
        self.sum[yi * self.width + xi]
    }
}
