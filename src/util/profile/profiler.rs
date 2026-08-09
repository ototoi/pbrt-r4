#[cfg(feature = "profile")]
mod detail {
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::RwLock;
    use std::thread::ThreadId;
    use std::time::Duration;

    use crate::util::base::ProfileCategory;

    type ThreadStateMap = HashMap<ThreadId, u64>;
    static THREAD_STATE_MAP: LazyLock<RwLock<ThreadStateMap>> =
        LazyLock::new(|| RwLock::new(ThreadStateMap::new()));

    pub fn set_profiler_state(state_bit: u64) {
        let tid = std::thread::current().id();
        let mut map = THREAD_STATE_MAP.write().unwrap();
        map.insert(tid, state_bit);
    }

    #[derive(Debug, Clone, Copy)]
    enum SamplerMessage {
        Stop,
    }

    type Sender = std::sync::mpsc::Sender<SamplerMessage>;

    type StateCountMap = HashMap<u64, u64>;
    fn sample_state_count(state_count: &mut StateCountMap) {
        let state_map = THREAD_STATE_MAP.read().unwrap();
        for (_, state) in state_map.iter() {
            let state = *state;
            if state == 0 {
                continue;
            }
            let count = state_count.entry(state).or_insert(0);
            *count += 1;
        }
    }

    struct ProfileSampler {
        handles: Vec<(Sender, std::thread::JoinHandle<()>)>,
        state_count: Arc<RwLock<StateCountMap>>,
        timer: std::time::Instant,
        durations: Vec<std::time::Duration>,
    }

    impl ProfileSampler {
        pub fn new() -> Self {
            ProfileSampler {
                handles: Vec::new(),
                state_count: Arc::new(RwLock::new(StateCountMap::new())),
                timer: std::time::Instant::now(),
                durations: Vec::new(),
            }
        }

        pub fn start_sample(&mut self, interval: std::time::Duration) {
            if self.handles.len() > 0 {
                return;
            }
            let state_count = self.state_count.clone();
            let (tx, rx) = std::sync::mpsc::channel::<SamplerMessage>();
            let handle = std::thread::spawn(move || loop {
                match rx.try_recv() {
                    Ok(SamplerMessage::Stop) => {
                        break;
                    }
                    _ => {}
                }
                {
                    let mut state_count = state_count.write().unwrap();
                    sample_state_count(&mut state_count);
                }
                std::thread::sleep(interval);
            });
            self.handles.push((tx, handle));
            self.timer = std::time::Instant::now();
        }

        pub fn stop_sample(&mut self) {
            for (tx, _) in self.handles.iter() {
                tx.send(SamplerMessage::Stop).unwrap();
            }
            for (_, handle) in self.handles.drain(..) {
                handle.join().unwrap();
            }
            self.handles.clear();
            self.durations.push(self.timer.elapsed());
        }

        pub fn clear(&mut self) {
            self.stop_sample();
            let mut state_count = self.state_count.write().unwrap();
            state_count.clear();
        }

        fn make_report(&self) -> Option<[(String, Vec<(u32, String, f32, Duration)>); 2]> {
            const NUM_PROF_CATEGORIES: usize = ProfileCategory::NumProfCategories.0 as usize;
            if self.durations.len() == 0 {
                return None;
            }

            let elapsed = *self.durations.last().unwrap();

            let mut overall_count = 0;
            let state_count = self.state_count.read().unwrap();
            for (_state, count) in state_count.iter() {
                overall_count += count;
            }
            let mut flat_results: HashMap<String, u64> = HashMap::new();
            let mut hierarchical_results: HashMap<String, u64> = HashMap::new();
            for (state, count) in state_count.iter() {
                let state = *state;
                let count = *count;
                if count == 0 {
                    continue;
                }
                let mut s = String::new();
                for b in 0..NUM_PROF_CATEGORIES {
                    let bit = 1u64 << b;
                    if (state & bit) != 0 {
                        if s.len() > 0 {
                            // contribute to the parents...
                            let entry = hierarchical_results.entry(s.clone()).or_insert(0);
                            *entry += count;
                            s.push_str("/");
                        }
                        let category = ProfileCategory(b as u32);
                        s += &format!("{}", category);
                    }
                }
                let entry = hierarchical_results.entry(s.clone()).or_insert(0);
                *entry += count;

                let name_index = log2int_(state);
                assert!(name_index < NUM_PROF_CATEGORIES as u32);
                let name = format!("{}", ProfileCategory(name_index as u32));
                let entry = flat_results.entry(name.clone()).or_insert(0);
                *entry += count;
            }

            let mut report = Vec::new();
            {
                // Sort the flattened ones by time, longest to shortest.
                let mut flat_vec = Vec::new();
                for (name, count) in hierarchical_results.iter() {
                    flat_vec.push((name.clone(), *count));
                }
                flat_vec.sort_by(|a, b| a.0.cmp(&b.0)); //sort by name
                for (name, count) in flat_vec.iter() {
                    let count = *count;
                    let pct = count as f32 / overall_count as f32;
                    let mut indent = 0;
                    let slash_index = name.rfind('/');
                    if slash_index.is_some() {
                        indent += name.matches("/").count();
                    }
                    let offset = if let Some(index) = slash_index {
                        index + 1
                    } else {
                        0
                    };
                    let to_print = name[offset..].to_string();
                    let consumed = Duration::from_secs_f32(elapsed.as_secs_f32() * pct);
                    report.push((indent as u32, to_print, pct, consumed));
                }
            }
            let mut report_flatten = Vec::new();
            {
                // Sort the flattened ones by time, longest to shortest.
                let mut flat_vec = Vec::new();
                for (name, count) in flat_results.iter() {
                    flat_vec.push((name.clone(), *count));
                }
                flat_vec.sort_by(|a, b| b.1.cmp(&a.1));
                for (name, count) in flat_vec.iter() {
                    let count = *count;
                    let pct = count as f32 / overall_count as f32;
                    let indent = 0;
                    let consumed = Duration::from_secs_f32(elapsed.as_secs_f32() * pct);
                    report_flatten.push((indent, name.clone(), pct, consumed));
                }
            }
            return Some([
                ("Profile".to_string(), report),
                ("Profile (flattened)".to_string(), report_flatten),
            ]);
        }

        pub fn report(&self, writer: &mut dyn Write) {
            if let Some(reports) = self.make_report() {
                for (title, report) in reports.iter() {
                    let r = &report;
                    let mut dest = "".to_string();
                    dest += &format!("  {}:\n", title);
                    for (indent, name, pct, consumed) in r.iter() {
                        let indent = (2 * (2 + *indent)) as usize;
                        dest += &format!(
                            "{:indent$}{} {:.1}% {:.6}s\n",
                            "",
                            name,
                            pct * 100.0,
                            consumed.as_secs_f32(),
                            indent = indent
                        );
                    }
                    writer.write_all(dest.as_bytes()).unwrap();
                }
            }
        }
    }

    impl Drop for ProfileSampler {
        fn drop(&mut self) {
            self.stop_sample();
        }
    }

    static PROFILE_SAMPLER: LazyLock<RwLock<ProfileSampler>> =
        LazyLock::new(|| RwLock::new(ProfileSampler::new()));

    //-----------------------------------------------------------------------
    pub fn start_profiler() {
        clear_profiler();
        let interval = std::time::Duration::from_millis(1000 / 100); // 100 Hz sampling
        let mut sampler = PROFILE_SAMPLER.write().unwrap();
        sampler.start_sample(interval);
    }

    pub fn stop_profiler() {
        let mut sampler = PROFILE_SAMPLER.write().unwrap();
        sampler.stop_sample();
    }

    pub fn clear_profiler() {
        THREAD_STATE_MAP.write().unwrap().clear();
        let mut sampler = PROFILE_SAMPLER.write().unwrap();
        sampler.clear();
    }

    pub fn init_profiler() {
        //
    }
    pub fn cleanup_profiler() {
        //
    }

    pub fn suspend_profiler() {
        //
    }

    pub fn resume_profiler() {
        //
    }

    pub fn print_profiler() {
        report_profiler_results();
    }

    pub fn report_profiler_results() {
        let sampler = PROFILE_SAMPLER.read().unwrap();
        sampler.report(&mut std::io::stdout());
    }
}

#[cfg(not(feature = "profile"))]
mod detail {
    pub fn start_profiler() {}
    pub fn stop_profiler() {}
    pub fn clear_profiler() {}
    pub fn init_profiler() {
        log::warn!("Profiling is not enabled.");
    }
    pub fn cleanup_profiler() {}
    pub fn suspend_profiler() {}
    pub fn resume_profiler() {}
    pub fn print_profiler() {}
    pub fn report_profiler_results() {}
}

pub use detail::*;
