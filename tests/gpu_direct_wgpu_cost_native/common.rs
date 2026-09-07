use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::{
    Backend, Backends, Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, InstanceFlags, Limits, MapMode, MemoryHints, PollType, PowerPreference,
    Queue, RequestAdapterOptions, Trace,
};

pub(crate) const WARMUP_SAMPLES: usize = 1;
pub(crate) const MEASURED_SAMPLES: usize = 5;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

#[derive(Debug, Default)]
pub(crate) struct Measurements {
    samples: Vec<BTreeMap<String, f64>>,
}

impl Measurements {
    pub(crate) fn push(&mut self, phases: BTreeMap<String, f64>) {
        assert!(
            phases
                .values()
                .all(|value| value.is_finite() && *value >= 0.0),
            "all timing samples must be finite and non-negative"
        );
        self.samples.push(phases);
    }

    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }

    pub(crate) fn to_json(&self) -> Value {
        let mut phase_names = self
            .samples
            .iter()
            .flat_map(|sample| sample.keys().cloned())
            .collect::<Vec<_>>();
        phase_names.sort();
        phase_names.dedup();

        let summary = phase_names
            .into_iter()
            .map(|phase| {
                let values = self
                    .samples
                    .iter()
                    .filter_map(|sample| sample.get(&phase).copied())
                    .collect::<Vec<_>>();
                (phase, summarize(&values))
            })
            .collect::<serde_json::Map<_, _>>();

        json!({
            "count": self.samples.len(),
            "samples_us": self.samples,
            "summary_us": summary,
        })
    }
}

fn summarize(values: &[f64]) -> Value {
    assert!(!values.is_empty(), "summary requires at least one sample");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len().is_multiple_of(2) {
        let high = sorted.len() / 2;
        (sorted[high - 1] + sorted[high]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    json!({
        "count": sorted.len(),
        "min": sorted[0],
        "median": median,
        "max": sorted[sorted.len() - 1],
    })
}

pub(crate) fn summarize_u64(values: &[u64]) -> Value {
    let values = values.iter().map(|value| *value as f64).collect::<Vec<_>>();
    summarize(&values)
}

pub(crate) fn ratio_summary(
    runengpu: &Measurements,
    direct: &Measurements,
    phases: &[&str],
) -> Value {
    let ratios = phases
        .iter()
        .filter_map(|phase| {
            let runengpu_values = runengpu
                .samples
                .iter()
                .filter_map(|sample| sample.get(*phase).copied())
                .collect::<Vec<_>>();
            let direct_values = direct
                .samples
                .iter()
                .filter_map(|sample| sample.get(*phase).copied())
                .collect::<Vec<_>>();
            if runengpu_values.len() != direct_values.len()
                || runengpu_values.is_empty()
                || direct_values.contains(&0.0)
            {
                return None;
            }
            let values = runengpu_values
                .iter()
                .zip(&direct_values)
                .map(|(runengpu_value, direct_value)| runengpu_value / direct_value)
                .collect::<Vec<_>>();
            Some(((*phase).to_owned(), summarize(&values)))
        })
        .collect::<serde_json::Map<_, _>>();
    Value::Object(ratios)
}

pub(crate) struct DirectWgpuContext {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    pub(crate) timestamp_supported: bool,
    pub(crate) enabled_features: Features,
    pub(crate) setup_us: f64,
}

impl DirectWgpuContext {
    pub(crate) fn request(label: &str) -> Self {
        Self::request_with_features(label, Features::empty())
    }

    pub(crate) fn request_timestamp(label: &str) -> Self {
        Self::request_with_features(label, Features::TIMESTAMP_QUERY)
    }

    fn request_with_features(label: &str, required_features: Features) -> Self {
        let start = Instant::now();
        let mut descriptor = InstanceDescriptor::new_without_display_handle_from_env();
        descriptor
            .flags
            .insert(InstanceFlags::VALIDATION_INDIRECT_CALL);
        descriptor.backends = Backends::VULKAN;
        let instance = Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::None,
            force_fallback_adapter: true,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("direct-WGPU comparison requires the same forced Vulkan fallback adapter");
        let adapter_info = adapter.get_info();
        assert_eq!(
            adapter_info.backend,
            Backend::Vulkan,
            "direct comparison must execute through Vulkan"
        );
        let adapter_features = adapter.features();
        let timestamp_supported = adapter_features.contains(Features::TIMESTAMP_QUERY);
        assert!(
            adapter_features.contains(required_features),
            "direct-WGPU comparison adapter must support every explicitly requested feature"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some(label),
            required_features,
            required_limits: Limits::defaults(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
        }))
        .expect("direct-WGPU comparison device request must succeed");
        let setup_us = micros(start.elapsed());
        Self {
            device,
            queue,
            adapter_info,
            timestamp_supported,
            enabled_features: required_features,
            setup_us,
        }
    }

    pub(crate) fn facts_json(&self) -> Value {
        json!({
            "backend": format!("{:?}", self.adapter_info.backend),
            "device_type": format!("{:?}", self.adapter_info.device_type),
            "name": &self.adapter_info.name,
            "driver": &self.adapter_info.driver,
            "driver_info": &self.adapter_info.driver_info,
            "forced_fallback_adapter": true,
            "timestamp_query_supported": self.timestamp_supported,
            "timestamp_period_ns": self.queue.get_timestamp_period(),
            "enabled_features": format!("{:?}", self.enabled_features),
            "device_request_limits": "wgpu::Limits::defaults()",
            "memory_hints": "Performance",
            "trace": "Off",
        })
    }
}

struct CallbackState {
    ready: AtomicBool,
    error: Mutex<Option<String>>,
}

impl CallbackState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            ready: AtomicBool::new(false),
            error: Mutex::new(None),
        })
    }
}

pub(crate) struct DirectSubmissionResult {
    pub(crate) mapped: Vec<Vec<u8>>,
    pub(crate) readback_registration_us: f64,
    pub(crate) submit_call_us: f64,
    pub(crate) completion_readback_us: f64,
}

impl DirectSubmissionResult {
    pub(crate) fn boundary_submit_us(&self) -> f64 {
        self.readback_registration_us + self.submit_call_us
    }
}

pub(crate) fn submit_and_map(
    context: &DirectWgpuContext,
    command_buffer: wgpu::CommandBuffer,
    readbacks: &[&wgpu::Buffer],
) -> DirectSubmissionResult {
    let registration_start = Instant::now();
    let states = readbacks
        .iter()
        .map(|buffer| {
            let state = CallbackState::new();
            let callback = Arc::clone(&state);
            command_buffer.map_buffer_on_submit(buffer, MapMode::Read, .., move |result| {
                if let Err(error) = result {
                    *callback.error.lock().unwrap() = Some(error.to_string());
                }
                callback.ready.store(true, Ordering::Release);
            });
            state
        })
        .collect::<Vec<_>>();
    let completed = Arc::new(AtomicBool::new(false));
    let completed_callback = Arc::clone(&completed);
    command_buffer.on_submitted_work_done(move || {
        completed_callback.store(true, Ordering::Release);
    });
    let readback_registration_us = micros(registration_start.elapsed());

    let submit_start = Instant::now();
    context.queue.submit([command_buffer]);
    let submit_call_us = micros(submit_start.elapsed());
    let completion_start = Instant::now();
    let deadline = Instant::now() + COMPLETION_TIMEOUT;
    loop {
        context
            .device
            .poll(PollType::Poll)
            .expect("direct-WGPU progress poll must succeed");
        let mapped = states
            .iter()
            .all(|state| state.ready.load(Ordering::Acquire));
        if completed.load(Ordering::Acquire) && mapped {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "direct-WGPU submission/readback timed out"
        );
        std::thread::yield_now();
    }

    for state in &states {
        if let Some(error) = state.error.lock().unwrap().take() {
            panic!("direct-WGPU readback mapping failed: {error}");
        }
    }
    let mapped = readbacks
        .iter()
        .map(|buffer| {
            let view = buffer
                .slice(..)
                .get_mapped_range()
                .expect("completed direct-WGPU readback must expose a mapped range");
            let bytes = view.to_vec();
            drop(view);
            buffer.unmap();
            bytes
        })
        .collect();
    let completion_readback_us = micros(completion_start.elapsed());

    DirectSubmissionResult {
        mapped,
        readback_registration_us,
        submit_call_us,
        completion_readback_us,
    }
}

pub(crate) fn padded_bytes_per_row(logical_bytes_per_row: u32) -> u32 {
    logical_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

pub(crate) fn tightly_pack_texture_rows(
    mapped: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    physical_bytes_per_row: u32,
) -> Vec<u8> {
    let logical_row = usize::try_from(width * bytes_per_pixel).unwrap();
    let physical_row = usize::try_from(physical_bytes_per_row).unwrap();
    let mut result = Vec::with_capacity(usize::try_from(width * height * bytes_per_pixel).unwrap());
    for row in 0..usize::try_from(height).unwrap() {
        let start = row * physical_row;
        result.extend_from_slice(&mapped[start..start + logical_row]);
    }
    result
}
