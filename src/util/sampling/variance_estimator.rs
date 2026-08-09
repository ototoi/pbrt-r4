use crate::util::base::Float;

#[derive(Debug, Clone, Copy, Default)]
pub struct VarianceEstimator {
    mean: Float,
    sum_squared_deltas: Float,
    count: i64,
}

impl VarianceEstimator {
    pub fn add(&mut self, value: Float) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as Float;
        let delta2 = value - self.mean;
        self.sum_squared_deltas += delta * delta2;
    }

    pub fn mean(&self) -> Float {
        self.mean
    }

    pub fn variance(&self) -> Float {
        if self.count > 1 {
            self.sum_squared_deltas / (self.count - 1) as Float
        } else {
            0.0
        }
    }

    pub fn count(&self) -> i64 {
        self.count
    }

    pub fn relative_variance(&self) -> Float {
        if self.count < 1 || self.mean == 0.0 {
            0.0
        } else {
            self.variance() / self.mean
        }
    }

    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        let total = self.count + other.count;
        self.sum_squared_deltas += other.sum_squared_deltas
            + (other.mean - self.mean).powi(2) * self.count as Float * other.count as Float
                / total as Float;
        self.mean =
            (self.count as Float * self.mean + other.count as Float * other.mean) / total as Float;
        self.count = total;
    }
}
