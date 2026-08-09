#[cfg(feature = "stats")]
mod _impl {
    use super::super::stats_accumlator::*;
    use crate::options::PbrtOptions;

    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::RwLock;

    pub trait StatReporter: Send + Sync {
        fn report(&self, accum: &mut StatsAccumulator);
        fn clear(&mut self);
    }

    pub type StatReporterRef = Arc<RwLock<dyn StatReporter>>;

    static REGISTER_REPORTERS: Mutex<Vec<StatReporterRef>> = Mutex::new(Vec::new());

    pub fn init_stats() {}

    pub fn register_stat_reporter(reporter: StatReporterRef) {
        let options = PbrtOptions::get();
        if !options.stats {
            return;
        }
        let mut reporters = REGISTER_REPORTERS.lock().unwrap();
        reporters.push(reporter);
    }

    pub fn print_stats() {
        let options = PbrtOptions::get();
        if !options.stats {
            return;
        }
        let mut accum = StatsAccumulator::new();
        let reporters = REGISTER_REPORTERS.lock().unwrap();
        for reporter in reporters.iter() {
            let reporter = reporter.read().unwrap();
            reporter.report(&mut accum);
        }
        println!("{}", accum);
    }

    pub fn clear_stats() {
        let reporters = REGISTER_REPORTERS.lock().unwrap();
        for reporter in reporters.iter() {
            let mut reporter = reporter.write().unwrap();
            reporter.clear();
        }
    }
}

#[cfg(not(feature = "stats"))]
mod _impl {
    pub fn init_stats() {
        log::warn!("Stats is not enabled.");
    }
    pub fn print_stats() {}
    pub fn clear_stats() {}
}

pub use _impl::*;
