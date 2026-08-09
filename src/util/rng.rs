use crate::util::base::*;

const PCG32_DEFAULT_STATE: u64 = 0x853c49e6748fea9b;
const PCG32_DEFAULT_STREAM: u64 = 0xda3e39cb94b95bdb;
const PCG32_MULT: u64 = 0x5851f42d4c957f2d;

#[derive(Debug, PartialEq, Clone)]
pub struct RNG {
    pub state: u64,
    pub inc: u64,
}

impl RNG {
    pub fn new() -> Self {
        RNG {
            state: PCG32_DEFAULT_STATE,
            inc: PCG32_DEFAULT_STREAM,
        }
    }

    pub fn new_sequence(initseq: u64) -> Self {
        let mut r = Self::new();
        r.set_sequence(initseq);
        return r;
    }

    pub fn set_sequence(&mut self, initseq: u64) {
        self.set_sequence_with_seed(initseq, mix_bits(initseq));
    }

    pub fn set_sequence_with_seed(&mut self, sequence_index: u64, seed: u64) {
        self.state = 0;
        self.inc = (sequence_index << 1) | 1;
        self.uniform_uint32();
        self.state = self.state.wrapping_add(seed);
        self.uniform_uint32();
    }

    #[inline]
    pub fn uniform_uint32(&mut self) -> u32 {
        let oldstate: u64 = self.state;
        self.state = oldstate.wrapping_mul(PCG32_MULT).wrapping_add(self.inc);
        let xorshifted: u32 = ((oldstate.wrapping_shr(18) ^ oldstate).wrapping_shr(27)) as u32;
        let rot: u32 = (oldstate.wrapping_shr(59)) as u32;
        return (xorshifted.wrapping_shr(rot))
            | (xorshifted.wrapping_shl(((!rot).wrapping_add(1)) & 31));
    }

    /// pbrt-v4 `RNG::Uniform<uint64_t>`: concatenate two consecutive PCG32
    /// outputs without changing the underlying sequence semantics.
    pub fn uniform_uint64(&mut self) -> u64 {
        (u64::from(self.uniform_uint32()) << 32) | u64::from(self.uniform_uint32())
    }

    /// pbrt-v4 `RNG::Uniform<int32_t>`.
    pub fn uniform_int32(&mut self) -> i32 {
        let value = self.uniform_uint32();
        if value <= i32::MAX as u32 {
            value as i32
        } else {
            (value.wrapping_sub(i32::MIN as u32)) as i32 + i32::MIN
        }
    }

    /// pbrt-v4 `RNG::Uniform<int64_t>`.
    pub fn uniform_int64(&mut self) -> i64 {
        let value = self.uniform_uint64();
        if value <= i64::MAX as u64 {
            value as i64
        } else {
            (value.wrapping_sub(i64::MIN as u64)) as i64 + i64::MIN
        }
    }

    pub fn uniform_uint32_threshold(&mut self, b: u32) -> u32 {
        let threshold = (!b + 1) % b;
        loop {
            let r = self.uniform_uint32();
            if r >= threshold {
                return r % b;
            }
        }
    }

    #[inline]
    pub fn uniform_float(&mut self) -> Float {
        return self.uniform_float32() as Float;
    }

    pub fn uniform_float32(&mut self) -> f32 {
        let f: f32 = self.uniform_uint32() as f32 * 2.3283064365386963e-10;
        return FLOAT_ONE_MINUS_EPSILON.min(f);
    }

    /// pbrt-v4 `RNG::Uniform<double>` with the same half-open upper bound.
    pub fn uniform_float64(&mut self) -> f64 {
        (self.uniform_uint64() as f64 * 5.421010862427522e-20).min(1.0 - f64::EPSILON)
    }

    pub fn advance(&mut self, idelta: i64) {
        let mut cur_mult = PCG32_MULT;
        let mut cur_plus = self.inc;
        let mut acc_mult = 1u64;
        let mut acc_plus = 0u64;
        let mut delta = idelta as u64;

        while delta > 0 {
            if (delta & 1) != 0 {
                acc_mult = acc_mult.wrapping_mul(cur_mult);
                acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            }
            cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
            delta >>= 1;
        }

        self.state = acc_mult.wrapping_mul(self.state).wrapping_add(acc_plus);
    }

    /// Return the number of PCG steps from `other` to this RNG state.
    /// The v4 operator requires both generators to use the same stream;
    /// Rust callers receive an explicit error for a mismatched stream.
    pub fn distance(&self, other: &RNG) -> Result<i64, String> {
        if self.inc != other.inc {
            return Err("RNG distance requires identical streams.".to_string());
        }
        let mut cur_mult = PCG32_MULT;
        let mut cur_plus = self.inc;
        let mut cur_state = other.state;
        let mut bit = 1u64;
        let mut distance = 0u64;
        while self.state != cur_state {
            if (self.state & bit) != (cur_state & bit) {
                cur_state = cur_state.wrapping_mul(cur_mult).wrapping_add(cur_plus);
                distance |= bit;
            }
            if (self.state & bit) != (cur_state & bit) {
                return Err("RNG states are not in the same reachable sequence.".to_string());
            }
            bit <<= 1;
            cur_plus = (cur_mult.wrapping_add(1)).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
        }
        Ok(distance as i64)
    }
}

/// pbrt-v4 `MixBits` (hash.h:39-46). Splitmix64-style finalizer used in
/// `RNG::SetSequence` and various MLT/sampler seed paths.
pub fn mix_bits(mut v: u64) -> u64 {
    v ^= v >> 31;
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v >> 27;
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v >> 33;
    v
}

impl Default for RNG {
    fn default() -> Self {
        Self::new()
    }
}
