use super::digit_permutation::DigitPermutation;
use super::primes::PRIMES;
use crate::util::base::*;

pub fn reverse_bits32(mut n: u32) -> u32 {
    n = (n.wrapping_shl(16)) | (n.wrapping_shr(16));
    n = ((n & 0x00ff00ff).wrapping_shl(8)) | ((n & 0xff00ff00).wrapping_shr(8));
    n = ((n & 0x0f0f0f0f).wrapping_shl(4)) | ((n & 0xf0f0f0f0).wrapping_shr(4));
    n = ((n & 0x33333333).wrapping_shl(2)) | ((n & 0xcccccccc).wrapping_shr(2));
    n = ((n & 0x55555555).wrapping_shl(1)) | ((n & 0xaaaaaaaa).wrapping_shr(1));
    return n;
}

pub fn reverse_bits64(n: u64) -> u64 {
    let n0 = reverse_bits32(n as u32) as u64;
    let n1 = reverse_bits32((n.wrapping_shr(32)) as u32) as u64;
    return (n0.wrapping_shl(32)) | n1;
}

fn radical_inverse_specialized(base: u64, mut a: u64) -> Float {
    let inv_base = 1.0 / base as Float;
    let mut reversed_digits = 0;
    let mut inv_base_n = 1.0;
    while a != 0 {
        let next = a / base;
        let digit = a - next * base;
        reversed_digits = reversed_digits * base + digit;
        inv_base_n *= inv_base;
        a = next;
    }
    return Float::min(reversed_digits as Float * inv_base_n, ONE_MINUS_EPSILON);
}

pub fn radical_inverse(base_index: u32, a: u64) -> Float {
    assert!(base_index < 1024);
    return match base_index {
        0 => reverse_bits64(a) as Float * 5.4210108624275222e-20,
        _ => radical_inverse_specialized(PRIMES[base_index as usize], a),
    };
}

pub fn inverse_radical_inverse(base: u64, inverse: u64, ndigits: usize) -> u64 {
    let mut inverse = inverse;
    let mut index = 0;
    for _ in 0..ndigits {
        let digit = inverse % base;
        inverse /= base;
        index = index * base + digit;
    }
    return index;
}

/// pbrt-v4 `ScrambledRadicalInverse` (lowdiscrepancy.h:115-134).
/// Loops over `digitIndex` positions until the next digit's
/// contribution is below the precision of `Float`, permuting each
/// digit through the per-digit `DigitPermutation` (not a single
/// per-prime permutation).
pub fn scrambled_radical_inverse(base_index: u32, a: u64, perm: &DigitPermutation) -> Float {
    let base = PRIMES[base_index as usize];
    // Guard against overflow when shifting digits up by `base`.
    let limit = (!0u64) / base - base;
    let inv_base = 1.0 / base as Float;
    let mut inv_base_m: Float = 1.0;
    let mut reversed_digits: u64 = 0;
    let mut digit_index: u32 = 0;
    let mut a = a;
    // pbrt-v4's loop terminates when the *next* iteration's
    // contribution would be smaller than `Float` can represent — that
    // matches `1 - (base - 1) * inv_base_m < 1` after the iteration.
    while 1.0 - ((base - 1) as Float) * inv_base_m < 1.0 && reversed_digits < limit {
        let next = a / base;
        let digit_value = (a - next * base) as u32;
        reversed_digits = reversed_digits * base + perm.permute(digit_index, digit_value) as u64;
        inv_base_m *= inv_base;
        digit_index += 1;
        a = next;
    }
    Float::min(inv_base_m * reversed_digits as Float, ONE_MINUS_EPSILON)
}

/// pbrt-v4 `ComputeRadicalInversePermutations` (lowdiscrepancy.cpp:47).
/// One `DigitPermutation` per prime in the table, each independently
/// seeded via `Hash(base, digitIndex, seed)`.
pub fn compute_radical_inverse_permutations(seed: u32) -> Vec<DigitPermutation> {
    let mut perms = Vec::with_capacity(PRIMES.len());
    for i in 0..PRIMES.len() {
        perms.push(DigitPermutation::new(PRIMES[i] as u32, seed));
    }
    perms
}
