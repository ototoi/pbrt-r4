// pbrt-v4 verbatim translation of `class DigitPermutation`
// (util/lowdiscrepancy.h:25-64). Each prime base has `nDigits × base`
// permutation entries — a unique permutation per digit position seeded
// by `Hash(base, digitIndex, seed)`.

/// pbrt-v4 `class DigitPermutation` (lowdiscrepancy.h:25). The
/// per-digit permutations are stored as a flat `Vec<u16>` of size
/// `n_digits * base`; `permute(digit_index, digit_value)` indexes
/// into it as `permutations[digit_index * base + digit_value]`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DigitPermutation {
    pub base: u32,
    pub n_digits: u32,
    pub permutations: Vec<u16>,
}

impl DigitPermutation {
    /// pbrt-v4 `DigitPermutation::DigitPermutation(int base, uint32_t
    /// seed, Allocator alloc)` (lowdiscrepancy.h:30-49). For each digit
    /// position, the per-digit seed is `Hash(base, digitIndex, seed)`
    /// and the permutation entry is
    /// `PermutationElement(digitValue, base, dseed)`.
    pub fn new(base: u32, seed: u32) -> Self {
        assert!(base < 65536, "DigitPermutation::base must fit in u16");

        // pbrt-v4 (lowdiscrepancy.h:33-38): pick `nDigits` such that
        // `1 - (base - 1) * (1 / base)^nDigits < 1` no longer holds in
        // floating-point arithmetic — i.e. enough digits to represent
        // any number in `[0, 1)` at `Float` precision.
        let mut n_digits: u32 = 0;
        let inv_base = 1.0_f64 / base as f64;
        let mut inv_base_m = 1.0_f64;
        while 1.0_f64 - ((base - 1) as f64) * inv_base_m < 1.0_f64 {
            n_digits += 1;
            inv_base_m *= inv_base;
        }

        let total = (n_digits * base) as usize;
        let mut permutations = vec![0u16; total];
        for digit_index in 0..n_digits {
            // pbrt-v4 `Hash(base, digitIndex, seed)` — see
            // `pbrt::Hash<Args...>` (util/hash.h:99-107). The variadic
            // hash packs each arg as its native byte representation
            // into a buffer of total `sizeof(args...)` bytes and feeds
            // it through `MurmurHash64A` with `seed = 0`.
            let dseed = hash_base_digit_seed(base as i32, digit_index as i32, seed);
            for digit_value in 0..base {
                let idx = (digit_index * base + digit_value) as usize;
                permutations[idx] = permutation_element(digit_value, base, dseed as u32) as u16;
            }
        }

        DigitPermutation {
            base,
            n_digits,
            permutations,
        }
    }

    /// pbrt-v4 `DigitPermutation::Permute(int digitIndex, int
    /// digitValue) const` (lowdiscrepancy.h:51-56). Indexes into the
    /// per-digit permutation table.
    #[inline]
    pub fn permute(&self, digit_index: u32, digit_value: u32) -> u32 {
        debug_assert!(digit_index < self.n_digits);
        debug_assert!(digit_value < self.base);
        self.permutations[(digit_index * self.base + digit_value) as usize] as u32
    }
}

/// pbrt-v4 variadic `Hash(int, int, uint32_t)` specialised for the
/// `DigitPermutation` ctor: packs (i32, i32, u32) = 12 bytes
/// little-endian and feeds them through `MurmurHash64A(buf, 12, 0)`.
fn hash_base_digit_seed(base: i32, digit_index: i32, seed: u32) -> u64 {
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&base.to_ne_bytes());
    buf[4..8].copy_from_slice(&digit_index.to_ne_bytes());
    buf[8..12].copy_from_slice(&seed.to_ne_bytes());
    murmur_hash_64a(&buf, 0)
}

/// pbrt-v4 `PermutationElement(int i, int l, uint32_t p)` — Andrew
/// Kensler 2013 randomised permutation. Identical bit-mixing constants
/// to the existing copies in `samplers/halton.rs`,
/// `samplers/pmj02bn.rs` and `samplers/paddedsobol.rs`.
fn permutation_element(mut i: u32, l: u32, p: u32) -> u32 {
    let mut w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    loop {
        i ^= p;
        i = i.wrapping_mul(0xe170893d);
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i = i.wrapping_mul(0x0929eb3f);
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i = i.wrapping_mul(1 | (p >> 27));
        i = i.wrapping_mul(0x6935fa69);
        i ^= (i & w) >> 11;
        i = i.wrapping_mul(0x74dcb303);
        i ^= (i & w) >> 2;
        i = i.wrapping_mul(0x9e501cc3);
        i ^= (i & w) >> 2;
        i = i.wrapping_mul(0xc860a3df);
        i &= w;
        i ^= i >> 5;
        if i < l {
            break;
        }
    }
    (i + p) % l
}

/// MurmurHash64A (Austin Appleby), as used by pbrt-v4
/// `pbrt::MurmurHash64A` (util/hash.h:19-66). Inlined here so the
/// `DigitPermutation` ctor doesn't depend on any sampler-local copy.
fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    let m: u64 = 0xc6a4a7935bd1e995;
    let r: u32 = 47;

    let len = key.len() as u64;
    let mut h = seed ^ len.wrapping_mul(m);

    let nblocks = key.len() / 8;
    for i in 0..nblocks {
        let start = i * 8;
        let mut k = u64::from_ne_bytes(key[start..start + 8].try_into().unwrap());
        k = k.wrapping_mul(m);
        k ^= k >> r;
        k = k.wrapping_mul(m);

        h ^= k;
        h = h.wrapping_mul(m);
    }

    let tail = &key[nblocks * 8..];
    match tail.len() {
        7 => {
            h ^= (tail[6] as u64) << 48;
            h ^= (tail[5] as u64) << 40;
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        6 => {
            h ^= (tail[5] as u64) << 40;
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        5 => {
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        4 => {
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        3 => {
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        2 => {
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        1 => {
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        _ => {}
    }

    h ^= h >> r;
    h = h.wrapping_mul(m);
    h ^= h >> r;
    h
}
