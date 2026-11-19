use std::cmp::Ordering;
use std::sync::Mutex;
use std::time::Duration;

use crate::options::PbrtOptions;

pub struct StageTotal {
    pub label: &'static str,
    pub calls: u64,
    pub total_ns: f64,
}

static PHASES: Mutex<Vec<(String, Duration)>> = Mutex::new(Vec::new());

pub fn enabled() -> bool {
    PbrtOptions::get().gpu_profile
}

pub fn record_phase(name: &str, elapsed: Duration) {
    if !enabled() {
        return;
    }
    PHASES.lock().unwrap().push((name.to_string(), elapsed));
}

pub fn take_phases() -> Vec<(String, Duration)> {
    std::mem::take(&mut *PHASES.lock().unwrap())
}

pub fn format_report(
    adapter: &str,
    spp: u32,
    phases: &[(String, Duration)],
    stages: &[StageTotal],
) -> String {
    let mut out = format!("GPU profile (adapter: {adapter}, {spp} spp)\n");
    out.push_str("render loop includes per-sample profiling sync; measure total time without --gpu-profile\n");
    if !phases.is_empty() {
        out.push_str("phases:\n");
        for (name, elapsed) in phases {
            out.push_str(&format!(
                "  {name:<24} {:.1} ms\n",
                elapsed.as_secs_f64() * 1000.0
            ));
        }
    }
    if stages.is_empty() {
        out.push_str("GPU timestamps unavailable\n");
        return out;
    }
    out.push_str("stages:\n");
    let total_ns: f64 = stages.iter().map(|s| s.total_ns).sum();
    let mut rows: Vec<&StageTotal> = stages.iter().collect();
    rows.sort_by(|a, b| {
        b.total_ns
            .partial_cmp(&a.total_ns)
            .unwrap_or(Ordering::Equal)
    });
    for s in rows {
        let total_ms = s.total_ns / 1e6;
        let ms_per_sample = if spp > 0 {
            total_ms / f64::from(spp)
        } else {
            0.0
        };
        let percent = if total_ns > 0.0 {
            s.total_ns / total_ns * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "  {:<28} calls {:>4}  total ms {:8.1}  ms/sample {:8.3}  % {:5.1}\n",
            s.label, s.calls, total_ms, ms_per_sample, percent
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_lists_phases_and_stages_with_per_sample_time() {
        let phases = vec![("flatten".to_string(), Duration::from_millis(210))];
        let stages = vec![
            StageTotal {
                label: "pbrt-r4 shade surface",
                calls: 32,
                total_ns: 8.0e6,
            },
            StageTotal {
                label: "pbrt-r4 accumulate",
                calls: 16,
                total_ns: 2.0e6,
            },
        ];
        let report = format_report("Test Adapter", 16, &phases, &stages);
        assert!(report.contains("GPU profile (adapter: Test Adapter, 16 spp)"));
        assert!(report.contains("flatten"));
        assert!(report.contains("210.0"));
        // stage rows sorted by total, descending; ms/sample = total_ms / spp
        let shade = report.find("pbrt-r4 shade surface").unwrap();
        let accum = report.find("pbrt-r4 accumulate").unwrap();
        assert!(shade < accum);
        assert!(report.contains("0.500")); // 8.0 ms / 16 spp
        assert!(report.contains("80.0")); // percent of GPU total
    }

    #[test]
    fn report_without_stages_says_timestamps_unavailable() {
        let report = format_report("A", 1, &[], &[]);
        assert!(report.contains("GPU timestamps unavailable"));
    }
}
