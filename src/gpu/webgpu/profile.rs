use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use crate::options::PbrtOptions;
use crate::util::error::PbrtError;

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
            let ms = elapsed.as_secs_f64() * 1000.0;
            let mut line = format!("  {name:<24} {ms:.1} ms");
            if name == "render loop" && spp > 0 {
                line.push_str(&format!("  ({:.1} / sample)", ms / f64::from(spp)));
            }
            out.push_str(&line);
            out.push('\n');
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

const QUERY_SET_CAPACITY: u32 = wgpu::QUERY_SET_MAX_QUERIES;

struct ProfilerSlot {
    query_set: wgpu::QuerySet,
    next_index: u32,
    labels: Vec<&'static str>,
}

struct ProfilerState {
    slots: Vec<ProfilerSlot>,
    active: usize,
    totals: BTreeMap<&'static str, (u64, f64)>,
}

pub struct GpuProfiler {
    device: wgpu::Device,
    state: Mutex<ProfilerState>,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            state: Mutex::new(ProfilerState {
                slots: Vec::new(),
                active: 0,
                totals: BTreeMap::new(),
            }),
        }
    }

    pub fn reserve(&self, label: &'static str) -> (wgpu::QuerySet, u32) {
        let mut state = self.state.lock().unwrap();
        if state
            .slots
            .get(state.active)
            .is_some_and(|slot| slot.next_index + 2 > QUERY_SET_CAPACITY)
        {
            state.active += 1;
        }
        if state.active == state.slots.len() {
            state.slots.push(ProfilerSlot {
                query_set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("pbrt-r4 GPU profiler query set"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERY_SET_CAPACITY,
                }),
                next_index: 0,
                labels: Vec::new(),
            });
        }
        let active = state.active;
        let slot = &mut state.slots[active];
        let begin = slot.next_index;
        slot.next_index += 2;
        slot.labels.push(label);
        (slot.query_set.clone(), begin)
    }

    pub fn collect(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), PbrtError> {
        let period_ns = queue.get_timestamp_period();
        let mut state = self.state.lock().unwrap();
        for index in 0..state.slots.len() {
            let slot = &state.slots[index];
            if slot.next_index == 0 {
                continue;
            }
            let used = slot.next_index;
            let labels = slot.labels.clone();
            let query_set = slot.query_set.clone();
            let timestamps = read_timestamps(device, queue, &query_set, used)?;
            accumulate(&mut state.totals, &labels, &timestamps, period_ns);
            let slot = &mut state.slots[index];
            slot.next_index = 0;
            slot.labels.clear();
        }
        state.active = 0;
        Ok(())
    }

    pub fn stage_totals(&self) -> Vec<StageTotal> {
        self.state
            .lock()
            .unwrap()
            .totals
            .iter()
            .map(|(label, (calls, total_ns))| StageTotal {
                label: *label,
                calls: *calls,
                total_ns: *total_ns,
            })
            .collect()
    }
}

fn accumulate(
    totals: &mut BTreeMap<&'static str, (u64, f64)>,
    labels: &[&'static str],
    timestamps: &[u64],
    period_ns: f32,
) {
    for (label, pair) in labels.iter().zip(timestamps.chunks_exact(2)) {
        let (begin, end) = (pair[0], pair[1]);
        if end < begin {
            continue;
        }
        let entry = totals.entry(*label).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += (end - begin) as f64 * f64::from(period_ns);
    }
}

fn read_timestamps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    query_set: &wgpu::QuerySet,
    count: u32,
) -> Result<Vec<u64>, PbrtError> {
    let size = u64::from(count) * u64::from(wgpu::QUERY_SIZE);
    let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pbrt-r4 GPU profiler resolve buffer"),
        size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pbrt-r4 GPU profiler timestamps"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pbrt-r4 GPU profiler resolve encoder"),
    });
    encoder.resolve_query_set(query_set, 0..count, &resolve_buffer, 0);
    encoder.copy_buffer_to_buffer(&resolve_buffer, 0, &readback_buffer, 0, size);
    queue.submit(Some(encoder.finish()));
    let slice = readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| {
            PbrtError::error(&format!("WebGPU GPU profiler polling failed: {error}"))
        })?;
    receiver
        .recv()
        .map_err(|error| {
            PbrtError::error(&format!("WebGPU GPU profiler callback failed: {error}"))
        })?
        .map_err(|error| {
            PbrtError::error(&format!("WebGPU GPU profiler mapping failed: {error}"))
        })?;
    let mapped = slice.get_mapped_range().map_err(|error| {
        PbrtError::error(&format!(
            "WebGPU GPU profiler readback access failed: {error}"
        ))
    })?;
    let timestamps = bytemuck::try_cast_slice::<u8, u64>(&mapped)
        .map_err(|_| PbrtError::error("WebGPU GPU profiler readback was not u64-aligned."))?
        .to_vec();
    drop(mapped);
    readback_buffer.unmap();
    Ok(timestamps)
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

    #[test]
    fn accumulate_sums_by_label() {
        let mut totals = BTreeMap::new();
        accumulate(&mut totals, &["a", "b", "a"], &[0, 10, 10, 30, 30, 35], 2.0);
        assert_eq!(totals["a"], (2, 30.0)); // (10 + 5) * 2.0
        assert_eq!(totals["b"], (1, 40.0));
    }

    #[test]
    fn accumulate_skips_inverted_pairs() {
        let mut totals = BTreeMap::new();
        accumulate(&mut totals, &["a"], &[10, 5], 1.0);
        assert!(totals.get("a").is_none());
    }
}
