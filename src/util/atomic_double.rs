use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// An atomically updated `f64` implemented with its IEEE-754 bit pattern.
///
/// Stable Rust does not provide `AtomicF64`, so the value is stored in an
/// `AtomicU64` and updated with a compare-and-exchange loop.
pub struct AtomicDouble {
    bits: AtomicU64,
}

impl AtomicDouble {
    pub fn new(value: f64) -> Self {
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }

    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.bits.load(ordering))
    }

    pub fn store(&self, value: f64, ordering: Ordering) {
        self.bits.store(value.to_bits(), ordering);
    }

    /// Atomically adds `value` and returns the previous value.
    pub fn fetch_add(&self, value: f64, ordering: Ordering) -> f64 {
        let mut old_bits = self.bits.load(ordering);
        loop {
            let old_value = f64::from_bits(old_bits);
            let new_bits = (old_value + value).to_bits();
            match self.bits.compare_exchange_weak(
                old_bits,
                new_bits,
                ordering,
                compare_exchange_failure_ordering(ordering),
            ) {
                Ok(_) => return old_value,
                Err(actual_bits) => old_bits = actual_bits,
            }
        }
    }
}

impl Default for AtomicDouble {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl fmt::Debug for AtomicDouble {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AtomicDouble")
            .field("value", &self.load(Ordering::Relaxed))
            .finish()
    }
}

fn compare_exchange_failure_ordering(ordering: Ordering) -> Ordering {
    match ordering {
        Ordering::Relaxed | Ordering::Release => Ordering::Relaxed,
        Ordering::Acquire | Ordering::AcqRel => Ordering::Acquire,
        Ordering::SeqCst => Ordering::SeqCst,
        _ => unreachable!(),
    }
}
