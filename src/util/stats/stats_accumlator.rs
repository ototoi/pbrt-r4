#[cfg(feature = "stats")]
mod _impl {
    use std::collections::HashMap;
    use std::fmt::*;

    pub struct StatsAccumulator {
        counters: HashMap<String, u64>,
        memory_counters: HashMap<String, u64>,

        int_distribution_sums: HashMap<String, u64>,
        int_distribution_counts: HashMap<String, u64>,
        int_distribution_mins: HashMap<String, u64>,
        int_distribution_maxs: HashMap<String, u64>,

        float_distribution_sums: HashMap<String, f64>,
        float_distribution_counts: HashMap<String, u64>,
        float_distribution_mins: HashMap<String, f64>,
        float_distribution_maxs: HashMap<String, f64>,

        percentages: HashMap<String, (u64, u64)>,
        ratios: HashMap<String, (u64, u64)>,
    }

    type CategoryMap = HashMap<String, Vec<String>>;
    fn get_category_and_title(name: &str) -> (String, String) {
        let mut parts = name.splitn(2, '/');
        let category = parts.next().unwrap();
        let title = if let Some(t) = parts.next() { t } else { "" };
        (category.to_string(), title.to_string())
    }

    impl StatsAccumulator {
        pub fn new() -> Self {
            StatsAccumulator {
                counters: HashMap::new(),
                memory_counters: HashMap::new(),
                int_distribution_sums: HashMap::new(),
                int_distribution_counts: HashMap::new(),
                int_distribution_mins: HashMap::new(),
                int_distribution_maxs: HashMap::new(),
                float_distribution_sums: HashMap::new(),
                float_distribution_counts: HashMap::new(),
                float_distribution_mins: HashMap::new(),
                float_distribution_maxs: HashMap::new(),
                percentages: HashMap::new(),
                ratios: HashMap::new(),
            }
        }

        pub fn report_counter(&mut self, name: &str, val: u64) {
            let counter = self.counters.entry(name.to_string()).or_insert(0);
            *counter += val;
        }

        pub fn report_memory_counter(&mut self, name: &str, val: u64) {
            let counter = self.memory_counters.entry(name.to_string()).or_insert(0);
            *counter += val;
        }

        pub fn report_int_distribution(
            &mut self,
            name: &str,
            sum: u64,
            count: u64,
            min: u64,
            max: u64,
        ) {
            {
                let counter = self
                    .int_distribution_sums
                    .entry(name.to_string())
                    .or_insert(0);
                *counter += sum;
            }
            {
                let counter = self
                    .int_distribution_counts
                    .entry(name.to_string())
                    .or_insert(0);
                *counter += count;
            }
            {
                let counter = self
                    .int_distribution_mins
                    .entry(name.to_string())
                    .or_insert(min);
                *counter = (*counter).min(min);
            }
            {
                let counter = self
                    .int_distribution_maxs
                    .entry(name.to_string())
                    .or_insert(max);
                *counter = (*counter).max(max);
            }
        }

        pub fn report_float_distribution(
            &mut self,
            name: &str,
            value_sum: f64,
            value_count: u64,
            value_min: f64,
            value_max: f64,
        ) {
            let sum = self
                .float_distribution_sums
                .entry(name.to_string())
                .or_insert(0.0);
            *sum += value_sum;
            let count = self
                .float_distribution_counts
                .entry(name.to_string())
                .or_insert(0);
            *count += value_count;
            let min = self
                .float_distribution_mins
                .entry(name.to_string())
                .or_insert(value_min);
            *min = (*min).min(value_min);
            let max = self
                .float_distribution_maxs
                .entry(name.to_string())
                .or_insert(value_max);
            *max = (*max).max(value_max);
        }

        pub fn report_percentage(&mut self, name: &str, num: u64, denom: u64) {
            let percentage = self.percentages.entry(name.to_string()).or_insert((0, 0));
            percentage.0 += num;
            percentage.1 += denom;
        }

        pub fn report_ratio(&mut self, name: &str, num: u64, denom: u64) {
            let ratio = self.ratios.entry(name.to_string()).or_insert((0, 0));
            ratio.0 += num;
            ratio.1 += denom;
        }

        fn get_category_map(&self) -> CategoryMap {
            let mut to_print: CategoryMap = CategoryMap::new();

            for (key, val) in &self.counters {
                if *val == 0 {
                    continue;
                }
                let (category, title) = get_category_and_title(key);
                let entry = to_print.entry(category).or_insert(Vec::new());
                entry.push(format!("{} {}", title, val));
            }

            for (key, val) in &self.memory_counters {
                if *val == 0 {
                    continue;
                }
                let (category, title) = get_category_and_title(key);
                assert!(category.len() > 0);
                let entry = to_print.entry(category).or_insert(Vec::new());
                let kb = (*val as f64) / 1024.0;
                let s = if kb < 1024.0 {
                    format!("{:.1} KB", kb)
                } else {
                    let mib = kb / 1024.0;
                    if mib < 1024.0 {
                        format!("{:.1} MiB", mib)
                    } else {
                        format!("{:.1} GiB", mib / 1024.0)
                    }
                };
                entry.push(format!("{} {}", title, s));
            }

            for (key, sum) in &self.int_distribution_sums {
                let count = self.int_distribution_counts.get(key).unwrap();
                if *count == 0 {
                    continue;
                }
                let min = self.int_distribution_mins.get(key).unwrap();
                let max = self.int_distribution_maxs.get(key).unwrap();
                let (category, title) = get_category_and_title(key);
                let entry = to_print.entry(category).or_insert(Vec::new());
                let avg = *sum as f64 / *count as f64;
                entry.push(format!(
                    "{} {:.3} avg [ range \" {} \" - \" {} \"]",
                    title, avg, min, max
                ));
            }

            for (key, (num, denom)) in &self.percentages {
                if *denom == 0 {
                    continue;
                }
                let (category, title) = get_category_and_title(key);
                let entry = to_print.entry(category).or_insert(Vec::new());
                entry.push(format!(
                    "{} {:.3}%% ({} / {})",
                    title,
                    100.0 * (*num as f64) / (*denom as f64),
                    num,
                    denom
                ));
            }

            for (key, (num, denom)) in &self.ratios {
                if *denom == 0 {
                    continue;
                }
                let (category, title) = get_category_and_title(key);
                let entry = to_print.entry(category).or_insert(Vec::new());
                entry.push(format!(
                    "{} {:.3} ({} / {})",
                    title,
                    (*num as f64) / (*denom as f64),
                    num,
                    denom
                ));
            }

            to_print
        }

        pub fn clear(&mut self) {
            self.counters.clear();
            self.memory_counters.clear();
            self.int_distribution_sums.clear();
            self.int_distribution_counts.clear();
            self.int_distribution_mins.clear();
            self.int_distribution_maxs.clear();
            self.float_distribution_sums.clear();
            self.float_distribution_counts.clear();
            self.float_distribution_mins.clear();
            self.float_distribution_maxs.clear();
            self.percentages.clear();
            self.ratios.clear();
        }
    }

    impl Display for StatsAccumulator {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            let category_map = self.get_category_map();
            let mut keys = category_map.keys().map(|s| s.clone()).collect::<Vec<_>>();
            keys.sort();
            let mut to_print = Vec::new();
            for category in keys.iter() {
                let items = category_map.get(category.as_str()).unwrap();
                let mut items = items.clone();
                items.sort();
                to_print.push((category.clone(), items));
            }
            write!(f, "  Statistics:\n")?;
            for (_j, (category, items)) in to_print.iter().enumerate() {
                write!(f, "    {}\n", category)?;
                for (_i, item) in items.iter().enumerate() {
                    write!(f, "      {}", item)?;
                    if _j != to_print.len() - 1 || _i != items.len() - 1 {
                        write!(f, "\n")?;
                    }
                }
            }
            Ok(())
        }
    }
}

#[cfg(not(feature = "stats"))]
mod _impl {}

#[allow(unused_imports)]
pub use _impl::*;
