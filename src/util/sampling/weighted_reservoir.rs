//! pbrt-v4 `WeightedReservoirSampler<T>` (util/sampling.h:524-602).
//!
//! Online reservoir sampling with weights: feed `(sample, weight)`
//! pairs in any order, the reservoir keeps one item whose probability
//! of being kept is proportional to its weight. Uses O(1) memory
//! regardless of how many samples are added — that's the reason v4
//! reaches for it inside `VolPathIntegrator::Li`'s BSSRDF block, where
//! walking the BSSRDF probe segment can produce an unbounded number of
//! candidate intersections.

use crate::util::base::Float;
use crate::util::rng::RNG;

pub struct WeightedReservoirSampler<T> {
    rng: RNG,
    weight_sum: Float,
    reservoir_weight: Float,
    reservoir: Option<T>,
}

impl<T> WeightedReservoirSampler<T> {
    pub fn new() -> Self {
        Self {
            rng: RNG::new(),
            weight_sum: 0.0,
            reservoir_weight: 0.0,
            reservoir: None,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: RNG::new_sequence(seed),
            weight_sum: 0.0,
            reservoir_weight: 0.0,
            reservoir: None,
        }
    }

    pub fn seed(&mut self, seed: u64) {
        self.rng.set_sequence(seed);
    }

    /// Offer a sample with the given weight; returns `true` if it
    /// replaced the current reservoir entry. `weight` must be `>= 0`.
    pub fn add(&mut self, sample: T, weight: Float) -> bool {
        if weight <= 0.0 {
            return false;
        }
        self.weight_sum += weight;
        let p = weight / self.weight_sum;
        if self.rng.uniform_float() < p {
            self.reservoir = Some(sample);
            self.reservoir_weight = weight;
            true
        } else {
            false
        }
    }

    pub fn has_sample(&self) -> bool {
        self.weight_sum > 0.0 && self.reservoir.is_some()
    }

    pub fn sample(&self) -> Option<&T> {
        self.reservoir.as_ref()
    }

    /// pbrt-v4 `SampleProbability()` — probability that the reservoir
    /// entry was selected, i.e. `weight / total_weight`. Returns 0 if
    /// no sample has been added.
    pub fn sample_probability(&self) -> Float {
        if self.weight_sum > 0.0 {
            self.reservoir_weight / self.weight_sum
        } else {
            0.0
        }
    }

    pub fn weight_sum(&self) -> Float {
        self.weight_sum
    }

    pub fn reset(&mut self) {
        self.weight_sum = 0.0;
        self.reservoir_weight = 0.0;
        self.reservoir = None;
    }

    /// Take the reservoir entry out of the sampler (consumes it).
    pub fn take(&mut self) -> Option<T> {
        let s = self.reservoir.take();
        // The probability fields remain so callers can still query
        // `sample_probability()` after extraction, matching v4 which
        // stores the sample by value.
        s
    }
}

impl<T> Default for WeightedReservoirSampler<T> {
    fn default() -> Self {
        Self::new()
    }
}
