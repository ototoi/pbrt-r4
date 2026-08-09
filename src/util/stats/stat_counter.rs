#[cfg(feature = "stats")]
mod _impl {

    use super::super::stat_reporter::*;
    use super::super::stats_accumlator::StatsAccumulator;
    use crate::options::PbrtOptions;

    use std::sync::Arc;
    use std::sync::RwLock;

    #[derive(Debug, Clone)]
    struct BaseCountReporter {
        pub name: String,
        pub value: u64,
    }

    impl BaseCountReporter {
        pub fn new(name: &str) -> Self {
            BaseCountReporter {
                name: name.to_string(),
                value: 0,
            }
        }
        pub fn add(&mut self, val: u64) {
            self.value += val;
        }
    }

    #[derive(Debug, Clone)]
    struct CountReporter(BaseCountReporter);
    impl CountReporter {
        pub fn new(name: &str) -> Self {
            CountReporter(BaseCountReporter::new(name))
        }
        fn add(&mut self, val: u64) {
            self.0.add(val);
        }
    }
    impl StatReporter for CountReporter {
        fn report(&self, accum: &mut StatsAccumulator) {
            accum.report_counter(&self.0.name, self.0.value);
        }
        fn clear(&mut self) {
            self.0.value = 0;
        }
    }

    #[derive(Debug, Clone)]
    struct MemoryReporter(BaseCountReporter);
    impl MemoryReporter {
        pub fn new(name: &str) -> Self {
            MemoryReporter(BaseCountReporter::new(name))
        }
        fn add(&mut self, val: u64) {
            self.0.add(val);
        }
    }
    impl StatReporter for MemoryReporter {
        fn report(&self, accum: &mut StatsAccumulator) {
            accum.report_memory_counter(&self.0.name, self.0.value);
        }
        fn clear(&mut self) {
            self.0.value = 0;
        }
    }

    #[derive(Debug, Clone)]
    pub struct IntDistributionReporter {
        pub name: String,
        pub sum: u64,
        pub count: u64,
        pub min: u64,
        pub max: u64,
    }

    impl IntDistributionReporter {
        pub fn new(name: &str) -> Self {
            IntDistributionReporter {
                name: name.to_string(),
                sum: 0,
                count: 0,
                min: std::u64::MAX,
                max: std::u64::MIN,
            }
        }

        pub fn add(&mut self, val: u64) {
            self.sum += val;
            self.count += 1;
            self.min = self.min.min(val);
            self.max = self.max.max(val);
        }
    }

    impl StatReporter for IntDistributionReporter {
        fn report(&self, accum: &mut StatsAccumulator) {
            accum.report_int_distribution(&self.name, self.sum, self.count, self.min, self.max);
        }
        fn clear(&mut self) {
            self.sum = 0;
            self.count = 0;
            self.min = std::u64::MAX;
            self.max = std::u64::MIN;
        }
    }

    #[derive(Debug, Clone)]
    pub struct FloatDistributionReporter {
        pub name: String,
        pub sum: f64,
        pub count: u64,
        pub min: f64,
        pub max: f64,
    }

    impl FloatDistributionReporter {
        pub fn new(name: &str) -> Self {
            FloatDistributionReporter {
                name: name.to_string(),
                sum: 0.0,
                count: 0,
                min: std::f64::MAX,
                max: f64::NEG_INFINITY,
            }
        }

        pub fn add(&mut self, val: f64) {
            self.sum += val;
            self.count += 1;
            self.min = self.min.min(val);
            self.max = self.max.max(val);
        }
    }

    impl StatReporter for FloatDistributionReporter {
        fn report(&self, accum: &mut StatsAccumulator) {
            if self.count != 0 {
                accum.report_float_distribution(
                    &self.name, self.sum, self.count, self.min, self.max,
                );
            }
        }
        fn clear(&mut self) {
            self.sum = 0.0;
            self.count = 0;
            self.min = std::f64::MAX;
            self.max = f64::NEG_INFINITY;
        }
    }

    #[derive(Debug, Clone)]
    pub struct BaseFractionReporter {
        pub name: String,
        pub num: u64,
        pub denom: u64,
    }

    impl BaseFractionReporter {
        pub fn new(name: &str) -> Self {
            BaseFractionReporter {
                name: name.to_string(),
                num: 0,
                denom: 0,
            }
        }
        fn clear(&mut self) {
            self.num = 0;
            self.denom = 0;
        }
        pub fn add_num(&mut self, val: u64) {
            self.num += val;
        }
        pub fn add_denom(&mut self, val: u64) {
            self.denom += val;
        }
    }

    #[derive(Debug, Clone)]
    struct PercentageReporter(BaseFractionReporter);

    impl PercentageReporter {
        pub fn new(name: &str) -> Self {
            PercentageReporter(BaseFractionReporter::new(name))
        }
        fn add_num(&mut self, val: u64) {
            self.0.add_num(val);
        }
        fn add_denom(&mut self, val: u64) {
            self.0.add_denom(val);
        }
    }

    impl StatReporter for PercentageReporter {
        fn report(&self, accum: &mut StatsAccumulator) {
            accum.report_percentage(&self.0.name, self.0.num, self.0.denom);
        }
        fn clear(&mut self) {
            self.0.clear();
        }
    }

    #[derive(Debug, Clone)]
    struct RatioReporter(BaseFractionReporter);

    impl RatioReporter {
        pub fn new(name: &str) -> Self {
            RatioReporter(BaseFractionReporter::new(name))
        }
        fn add_num(&mut self, val: u64) {
            self.0.add_num(val);
        }
        fn add_denom(&mut self, val: u64) {
            self.0.add_denom(val);
        }
    }

    impl StatReporter for RatioReporter {
        fn report(&self, accum: &mut StatsAccumulator) {
            accum.report_percentage(&self.0.name, self.0.num, self.0.denom);
        }
        fn clear(&mut self) {
            self.0.clear();
        }
    }

    //-----------------------------------------------------------------------

    pub struct StatCounter {
        reporter: Arc<RwLock<CountReporter>>,
        is_enabled: bool,
    }

    impl StatCounter {
        pub fn new(name: &str) -> Self {
            let reporter: Arc<RwLock<CountReporter>> =
                Arc::new(RwLock::new(CountReporter::new(name)));
            register_stat_reporter(reporter.clone());
            let options = PbrtOptions::get();
            let is_enabled = options.stats;
            StatCounter {
                reporter,
                is_enabled,
            }
        }
        pub fn inc(&self) {
            self.add(1);
        }
        pub fn add(&self, val: u64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add(val);
        }
    }

    pub struct StatMemoryCounter {
        reporter: Arc<RwLock<MemoryReporter>>,
        is_enabled: bool,
    }

    impl StatMemoryCounter {
        pub fn new(name: &str) -> Self {
            let reporter: Arc<RwLock<MemoryReporter>> =
                Arc::new(RwLock::new(MemoryReporter::new(name)));
            register_stat_reporter(reporter.clone());
            let options = PbrtOptions::get();
            let is_enabled = options.stats;
            StatMemoryCounter {
                reporter,
                is_enabled,
            }
        }
        pub fn inc(&self) {
            self.add(1);
        }
        pub fn add(&self, val: usize) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add(val as u64);
        }
    }

    pub struct StatIntDistribution {
        reporter: Arc<RwLock<IntDistributionReporter>>,
        is_enabled: bool,
    }

    impl StatIntDistribution {
        pub fn new(name: &str) -> Self {
            let reporter: Arc<RwLock<IntDistributionReporter>> =
                Arc::new(RwLock::new(IntDistributionReporter::new(name)));
            register_stat_reporter(reporter.clone());
            let options = PbrtOptions::get();
            let is_enabled = options.stats;
            StatIntDistribution {
                reporter,
                is_enabled,
            }
        }
        pub fn add(&self, val: u64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add(val);
        }
    }

    pub struct StatFloatDistribution {
        reporter: Arc<RwLock<FloatDistributionReporter>>,
        is_enabled: bool,
    }

    impl StatFloatDistribution {
        pub fn new(name: &str) -> Self {
            let reporter: Arc<RwLock<FloatDistributionReporter>> =
                Arc::new(RwLock::new(FloatDistributionReporter::new(name)));
            register_stat_reporter(reporter.clone());
            let options = PbrtOptions::get();
            let is_enabled = options.stats;
            StatFloatDistribution {
                reporter,
                is_enabled,
            }
        }
        pub fn add(&self, val: f64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add(val);
        }
    }

    pub struct StatPercent {
        reporter: Arc<RwLock<PercentageReporter>>,
        is_enabled: bool,
    }

    impl StatPercent {
        pub fn new(name: &str) -> Self {
            let reporter: Arc<RwLock<PercentageReporter>> =
                Arc::new(RwLock::new(PercentageReporter::new(name)));
            register_stat_reporter(reporter.clone());
            let options = PbrtOptions::get();
            let is_enabled = options.stats;
            StatPercent {
                reporter,
                is_enabled,
            }
        }
        pub fn add_num(&self, val: u64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add_num(val);
        }
        pub fn add_denom(&self, val: u64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add_denom(val);
        }
    }

    pub struct StatRatio {
        reporter: Arc<RwLock<RatioReporter>>,
        is_enabled: bool,
    }

    impl StatRatio {
        pub fn new(name: &str) -> Self {
            let reporter: Arc<RwLock<RatioReporter>> =
                Arc::new(RwLock::new(RatioReporter::new(name)));
            register_stat_reporter(reporter.clone());
            let options = PbrtOptions::get();
            let is_enabled = options.stats;
            StatRatio {
                reporter,
                is_enabled,
            }
        }
        pub fn add_num(&self, val: u64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add_num(val);
        }
        pub fn add_denom(&self, val: u64) {
            if !self.is_enabled {
                return;
            }
            let mut reporter = self.reporter.write().unwrap();
            reporter.add_denom(val);
        }
    }
}

#[cfg(not(feature = "stats"))]
mod _impl {
    pub struct StatCounter {}
    impl StatCounter {
        pub fn new(_name: &str) -> Self {
            StatCounter {}
        }
        pub fn inc(&self) {}
        pub fn add(&self, _val: u64) {}
    }

    pub struct StatMemoryCounter {}
    impl StatMemoryCounter {
        pub fn new(_name: &str) -> Self {
            StatMemoryCounter {}
        }
        pub fn inc(&self) {}
        pub fn add(&self, _val: usize) {}
    }

    pub struct StatIntDistribution {}
    impl StatIntDistribution {
        pub fn new(_name: &str) -> Self {
            StatIntDistribution {}
        }
        pub fn add(&self, _val: u64) {}
    }

    pub struct StatFloatDistribution {}
    impl StatFloatDistribution {
        pub fn new(_name: &str) -> Self {
            StatFloatDistribution {}
        }
        pub fn add(&self, _val: f64) {}
    }

    pub struct StatPercent {}
    impl StatPercent {
        pub fn new(_name: &str) -> Self {
            StatPercent {}
        }
        pub fn add_num(&self, _val: u64) {}
        pub fn add_denom(&self, _val: u64) {}
    }

    pub struct StatRatio {}
    impl StatRatio {
        pub fn new(_name: &str) -> Self {
            StatRatio {}
        }
        pub fn add_num(&self, _val: u64) {}
        pub fn add_denom(&self, _val: u64) {}
    }
}

pub use _impl::*;
