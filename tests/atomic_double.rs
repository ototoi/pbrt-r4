use pbrt_r4::util::AtomicDouble;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

#[test]
fn atomic_double_preserves_value_bits() {
    let value = AtomicDouble::new(-3.25);
    assert_eq!(value.load(Ordering::Relaxed), -3.25);

    value.store(1.5, Ordering::Relaxed);
    assert_eq!(value.load(Ordering::Relaxed), 1.5);
}

#[test]
fn atomic_double_fetch_add_is_thread_safe() {
    const THREADS: usize = 8;
    const INCREMENTS: usize = 10_000;
    let value = Arc::new(AtomicDouble::default());
    let mut workers = Vec::with_capacity(THREADS);

    for _ in 0..THREADS {
        let value = Arc::clone(&value);
        workers.push(thread::spawn(move || {
            for _ in 0..INCREMENTS {
                value.fetch_add(1.0, Ordering::Relaxed);
            }
        }));
    }

    for worker in workers {
        worker.join().unwrap();
    }

    assert_eq!(value.load(Ordering::Relaxed), (THREADS * INCREMENTS) as f64);
}
