//! RunenGPU #172: streaming adaptive background foreground-mask proof.
//!
//! Usage:
//!   cargo run --example gpu_streaming_background -- --mode serial --frames 64
//!   cargo run --example gpu_streaming_background -- --mode batch --batch 4
//!   cargo run --example gpu_streaming_background -- --mode cpu
//!   cargo run --example gpu_streaming_background -- --source stdin --mode serial
//!
//! stdin wire format: repeated LE u64 capture_unix_us, LE u32 q (2 or 4),
//! then width*height raw grayscale bytes. EOF between packets ends input.
//! The independent capture timestamp is reported, not assumed synchronized
//! with the local monotonic clock. --source generated is an offline fixture,
//! *not* evidence of unpredictably arriving camera frames.
//!
//! All GPU operations use public RunenGPU. No new execution authority is created.

#[path = "gpu_streaming_background/frame.rs"]
mod frame;
#[path = "gpu_streaming_background/gpu.rs"]
mod gpu;

use frame::{ExpectedFrame, Frame, Reference};
use gpu::{BatchError, GpuWorkload};
use runen_gpu::{GpuSubmissionRejectionKind, GpuSubmissionStatus, GpuWorkSubmissionError};
use serde_json::json;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};
use std::{env, fs, thread};

const SOURCE_QUEUE_BOUND: usize = 8;
const MAX_PENDING_RESULTS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Cpu,
    Serial,
    Batch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Generated,
    Stdin,
}

#[derive(Clone)]
struct Config {
    width: u32,
    height: u32,
    max_frames: usize,
    mode: Mode,
    source: Source,
    batch_size: usize,
    output_dir: Option<PathBuf>,
    producer_signals: bool,
    batch_wait_ms: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 256,
            height: 144,
            max_frames: 64,
            mode: Mode::Serial,
            source: Source::Generated,
            batch_size: 4,
            output_dir: None,
            producer_signals: false,
            batch_wait_ms: None,
        }
    }
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let args = env::args().skip(1).collect::<Vec<_>>();
        if args
            .iter()
            .any(|value| matches!(value.as_str(), "--help" | "-h"))
        {
            println!(
                "Options: --mode cpu|serial|batch --source generated|stdin \\n                 --width N --height N --frames N --batch 1|2|4|8 --batch-wait-ms N --output-dir PATH \\n                 [--producer-signals (stdin serial only)]"
            );
            std::process::exit(0);
        }
        let mut index = 0;
        while index < args.len() {
            let key = &args[index];
            if key == "--producer-signals" {
                config.producer_signals = true;
                index += 1;
                continue;
            }
            let val = args
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {key}"))?;
            match key.as_str() {
                "--mode" => {
                    config.mode = match val.as_str() {
                        "cpu" => Mode::Cpu,
                        "serial" => Mode::Serial,
                        "batch" => Mode::Batch,
                        _ => return Err(format!("unknown mode {val}")),
                    }
                }
                "--source" => {
                    config.source = match val.as_str() {
                        "stdin" => Source::Stdin,
                        "generated" => Source::Generated,
                        _ => return Err(format!("unknown source {val}")),
                    }
                }
                "--width" => config.width = val.parse().map_err(|_| "invalid width")?,
                "--height" => config.height = val.parse().map_err(|_| "invalid height")?,
                "--frames" => config.max_frames = val.parse().map_err(|_| "invalid frame count")?,
                "--batch" => config.batch_size = val.parse().map_err(|_| "invalid batch size")?,
                "--batch-wait-ms" => {
                    config.batch_wait_ms = Some(val.parse().map_err(|_| "invalid batch wait")?)
                }
                "--output-dir" => config.output_dir = Some(PathBuf::from(val)),
                _ => return Err(format!("unrecognized flag {key}")),
            }
            index += 2;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or("pixel count overflow")?;
        if pixels == 0 || self.max_frames == 0 || !matches!(self.batch_size, 1 | 2 | 4 | 8) {
            return Err("width/height/frames must be positive and batch one of 1,2,4,8".into());
        }
        if self.producer_signals && (self.source != Source::Stdin || self.mode != Mode::Serial) {
            return Err("producer signals require --source stdin --mode serial".into());
        }
        if self.mode == Mode::Batch && self.source == Source::Stdin && self.batch_wait_ms.is_none()
        {
            return Err("stdin batch mode requires an explicit --batch-wait-ms".into());
        }
        if self.batch_wait_ms.is_some() && self.mode != Mode::Batch {
            return Err("batch wait only applies to --mode batch".into());
        }
        if u64::from(pixels) * 4 > 64 * 1024 * 1024 {
            return Err("fixture exceeds the default 64 MiB upload/readback budget".into());
        }
        Ok(())
    }

    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    fn group(&self) -> usize {
        if self.mode == Mode::Batch {
            self.batch_size
        } else {
            1
        }
    }
}

fn producer(config: Config) -> (Receiver<Result<Frame, String>>, thread::JoinHandle<()>) {
    let (sender, receiver) = mpsc::sync_channel(SOURCE_QUEUE_BOUND);
    let handle = thread::spawn(move || {
        if config.source == Source::Generated {
            for id in 1..=config.max_frames {
                if sender
                    .send(Ok(frame::generated(id, config.width, config.height)))
                    .is_err()
                {
                    return;
                }
            }
        } else {
            let input = io::stdin();
            let mut lock = input.lock();
            for id in 1..=config.max_frames {
                match frame::read_stdin_frame(&mut lock, id, config.pixels()) {
                    Ok(Some(value)) => {
                        if sender.send(Ok(value)).is_err() {
                            return;
                        }
                    }
                    Ok(None) => return,
                    Err(e) => {
                        let _ = sender.send(Err(format!("stdin frame {id}: {e}")));
                        return;
                    }
                }
            }
        }
    });
    (receiver, handle)
}

fn checked_mask_png_free_pgm(
    path: &Path,
    width: u32,
    height: u32,
    mask: &[u32],
) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    write!(file, "P5\n{width} {height}\n255\n").map_err(|e| e.to_string())?;
    let pixels = mask
        .iter()
        .copied()
        .map(u8::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    file.write_all(&pixels).map_err(|e| e.to_string())
}

/// Only callers that have already established their CPU/reference comparison
/// can publish a result. Artifact I/O must not move the verification timestamp.
fn publish_verified(
    config: &Config,
    start: Instant,
    item: &ExpectedFrame,
    accepted_at: Option<Instant>,
    terminal_at: Option<Instant>,
    verified: Instant,
    mask: &[u32],
    kind: &str,
) -> Result<u128, String> {
    if let Some(root) = &config.output_dir {
        fs::create_dir_all(root).map_err(|e| e.to_string())?;
        let path = root.join(format!("mask_{:06}.pgm", item.frame.id));
        checked_mask_png_free_pgm(&path, config.width, config.height, mask)?;
    }
    println!(
        "{}",
        json!({
            "schema_version": 1,
            "kind": kind,
            "source": match config.source { Source::Stdin => "external_stdin", Source::Generated => "generated_offline_control" },
            "frame_id": item.frame.id,
            "capture_unix_us_unverified_clock": item.frame.external_capture_us,
            "received_us": item.frame.receipt.duration_since(start).as_micros(),
            "accepted_us": accepted_at.map(|x| x.duration_since(start).as_micros()),
            "terminal_us": terminal_at.map(|x| x.duration_since(start).as_micros()),
            "verified_us": verified.duration_since(start).as_micros(),
            "receipt_to_verified_us": verified.duration_since(item.frame.receipt).as_micros(),
            "exact_cpu_match": true,
        })
    );
    Ok(verified.duration_since(item.frame.receipt).as_micros())
}

/// Nearest-rank percentiles: observable receipt-to-verified values, no interpolation.
fn latency_summary(samples: &[u128]) -> serde_json::Value {
    if samples.is_empty() {
        return json!(null);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let at = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
    json!({
        "n": sorted.len(),
        "p50_us": at(50),
        "p95_us": at(95),
        "p99_us": at(99),
        "min_us": sorted[0],
        "max_us": sorted[sorted.len() - 1],
    })
}

fn run_cpu(
    config: &Config,
    receiver: Receiver<Result<Frame, String>>,
    start: Instant,
) -> Result<usize, String> {
    let mut reference = Reference::new(config.pixels());
    let mut count = 0;
    let mut latencies = Vec::new();
    let mut reference_cpu_us = 0_u128;
    for message in receiver {
        let begin = Instant::now();
        let item = reference.accept(message?)?;
        reference_cpu_us += begin.elapsed().as_micros();
        // The CPU recurrence itself is the control; no GPU result is implied.
        let verified = Instant::now();
        latencies.push(publish_verified(
            config,
            start,
            &item,
            None,
            None,
            verified,
            &item.mask,
            "cpu_reference",
        )?);
        count += 1;
    }
    if count == 0 {
        return Err("no input frames in CPU control".into());
    }
    println!(
        "{}",
        json!({
            "kind": "run_complete",
            "mode": "cpu",
            "source": match config.source { Source::Stdin => "external_stdin", Source::Generated => "generated_offline_control" },
            "frames_exactly_verified": count,
            "elapsed_us": start.elapsed().as_micros(),
            "reference_cpu_us": reference_cpu_us,
            "receipt_to_verified": latency_summary(&latencies),
            "benchmark_scope": "diagnostic_only",
        })
    );
    Ok(count)
}

struct Pending {
    batch: gpu::Batch,
    frames: Vec<ExpectedFrame>,
    accepted_at: Instant,
    terminal_at: Option<Instant>,
}

fn cap_backpressure(error: &GpuWorkSubmissionError) -> bool {
    match error {
        GpuWorkSubmissionError::SubmissionRejected(rejection) => matches!(
            rejection.kind(),
            GpuSubmissionRejectionKind::InFlightCapacityExceeded
                | GpuSubmissionRejectionKind::UploadBytesInFlightExceeded
                | GpuSubmissionRejectionKind::ReadbackBytesInFlightExceeded
                | GpuSubmissionRejectionKind::PendingReadbacksExceeded
        ),
        _ => false,
    }
}

fn run_gpu(
    config: &Config,
    receiver: Receiver<Result<Frame, String>>,
    start: Instant,
    mut gpu: GpuWorkload,
) -> Result<usize, String> {
    let mut oracle = Reference::new(config.pixels());
    let mut backlog: VecDeque<ExpectedFrame> = VecDeque::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut last_retained: Option<runen_gpu::GpuSubmission> = None;
    let mut producer_done = false;
    let mut verified = 0usize;
    let mut verified_latencies = Vec::<u128>::new();
    let mut reference_cpu_us = 0_u128;
    let mut progress_cpu_us = 0_u128;
    let mut prepare_and_submit_cpu_us = 0_u128;
    let mut capacity_rejections = 0_u64;
    let mut peak_application_backlog = 0_usize;
    let mut peak_pending_results = 0_usize;
    let mut accepted_batches = 0_usize;
    let mut received_after_prior_accept = 0_usize;
    let mut received_while_prior_still_accepted = 0_usize;
    let mut last_acceptance: Option<Instant> = None;
    let deadline = Instant::now() + Duration::from_secs(120);

    loop {
        let progress_start = Instant::now();
        gpu.context.progress();
        progress_cpu_us += progress_start.elapsed().as_micros();
        while backlog.len() < SOURCE_QUEUE_BOUND {
            match receiver.try_recv() {
                Ok(Ok(frame)) => {
                    // This second observation is intentional: receipt may have happened
                    // before polling, so both temporal facts are retained separately.
                    if last_acceptance.is_some_and(|t| frame.receipt > t) {
                        received_after_prior_accept += 1;
                        if last_retained.as_ref().is_some_and(|submission| {
                            matches!(submission.status(), GpuSubmissionStatus::Accepted)
                        }) {
                            received_while_prior_still_accepted += 1;
                        }
                    }
                    let begin = Instant::now();
                    let expected = oracle.accept(frame)?;
                    reference_cpu_us += begin.elapsed().as_micros();
                    backlog.push_back(expected);
                    peak_application_backlog = peak_application_backlog.max(backlog.len());
                }
                Ok(Err(e)) => return Err(e),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    producer_done = true;
                    break;
                }
            }
        }

        let mut index = 0usize;
        while index < pending.len() {
            if pending[index].terminal_at.is_none() {
                match pending[index].batch.submission.status() {
                    GpuSubmissionStatus::Accepted => {}
                    GpuSubmissionStatus::Completed => {
                        pending[index].terminal_at = Some(Instant::now())
                    }
                    GpuSubmissionStatus::Failed(e) => return Err(format!("GPU job failed: {e:?}")),
                }
            }
            let ready = GpuWorkload::ready_masks(&pending[index].batch)?;
            if let Some(masks) = ready {
                let finished = pending.remove(index);
                if masks.len() != finished.frames.len() {
                    return Err("lost or duplicated mask results".into());
                }
                // Compare every complete mask before any artifact I/O. Otherwise
                // earlier file writes bias later frames' verification timestamps.
                let mut verification_times = Vec::with_capacity(masks.len());
                for (item, mask) in finished.frames.iter().zip(&masks) {
                    if mask != &item.mask {
                        return Err(format!("GPU/CPU mismatch on frame {}", item.frame.id));
                    }
                    verification_times.push(Instant::now());
                }
                for ((item, mask), verified_at) in
                    finished.frames.iter().zip(&masks).zip(verification_times)
                {
                    verified_latencies.push(publish_verified(
                        config,
                        start,
                        item,
                        Some(finished.accepted_at),
                        finished.terminal_at,
                        verified_at,
                        mask,
                        "gpu_verified",
                    )?);
                    verified += 1;
                }
            } else {
                index += 1;
            }
        }

        if let Some(previous) = &last_retained {
            match previous.status() {
                GpuSubmissionStatus::Accepted => {}
                GpuSubmissionStatus::Completed => last_retained = None,
                GpuSubmissionStatus::Failed(e) => {
                    return Err(format!("retained work failed: {e:?}"));
                }
            }
        }

        // A live producer need not send another K-1 frames or close stdin.
        // The caller owns the maximum delay; no inferred frame-rate policy.
        let wait_expired = config.batch_wait_ms.is_some_and(|millis| {
            backlog
                .front()
                .is_some_and(|item| item.frame.receipt.elapsed() >= Duration::from_millis(millis))
        });
        let eligible = backlog.len() >= config.group()
            || (producer_done && !backlog.is_empty())
            || wait_expired;
        if eligible && last_retained.is_none() && pending.len() < MAX_PENDING_RESULTS {
            let size = backlog.len().min(config.group());
            let inputs = backlog
                .iter()
                .take(size)
                .map(|item| item.frame.clone())
                .collect::<Vec<_>>();
            let submit_started = Instant::now();
            match gpu.submit(&inputs) {
                Ok(batch) => {
                    let accepted_at = Instant::now();
                    prepare_and_submit_cpu_us +=
                        accepted_at.duration_since(submit_started).as_micros();
                    accepted_batches += 1;
                    last_acceptance = Some(accepted_at);
                    last_retained = Some(batch.submission.clone());
                    if config.producer_signals {
                        eprintln!("RUNEN_GPU_172_ACCEPTED {}", inputs[0].id);
                    }
                    let frames = (0..size)
                        .map(|_| backlog.pop_front().expect("checked backlog"))
                        .collect();
                    pending.push(Pending {
                        batch,
                        frames,
                        accepted_at,
                        terminal_at: None,
                    });
                    peak_pending_results = peak_pending_results.max(pending.len());
                }
                Err(BatchError::Submission(error)) if cap_backpressure(&error) => {
                    capacity_rejections += 1;
                    prepare_and_submit_cpu_us += submit_started.elapsed().as_micros();
                    // Capacity conflicts are transient observations. Re-author from current
                    // state on a future nonblocking progress point, never cache a rejected ticket.
                }
                Err(error) => return Err(format!("frame submission failed: {error}")),
            }
        }
        if producer_done && backlog.is_empty() && pending.is_empty() && last_retained.is_none() {
            break;
        }
        if Instant::now() >= deadline {
            return Err("proof timed out".into());
        }
        thread::yield_now();
    }
    if verified == 0 {
        return Err("no frames verified".into());
    }
    let actual = gpu.completed_background()?;
    if actual != oracle.background() {
        return Err("final persistent GPU background differs from CPU".into());
    }
    println!(
        "{}",
        json!({
            "kind": "run_complete",
            "source": match config.source { Source::Stdin => "external_stdin", Source::Generated => "generated_offline_control" },
            "mode": match config.mode { Mode::Serial => "serial", Mode::Batch => "batch", Mode::Cpu => "cpu" },
            "frames_exactly_verified": verified,
            "final_background_exact_cpu_match": true,
            "elapsed_us": Instant::now().duration_since(start).as_micros(),
            "adapter_backend": format!("{:?}", gpu.context.adapter_facts().backend()),
            "adapter_fallback": format!("{:?}", gpu.context.adapter_facts().fallback()),
            "adapter_name": gpu.context.adapter_facts().diagnostic_name(),
            "adapter_vendor": gpu.context.adapter_facts().vendor(),
            "adapter_device": gpu.context.adapter_facts().device(),
            "adapter_driver": gpu.context.adapter_facts().driver(),
            "adapter_driver_info": gpu.context.adapter_facts().driver_info(),
            "dimensions": [config.width, config.height],
            "accepted_batches": accepted_batches,
            "peak_application_backlog": peak_application_backlog,
            "peak_pending_results": peak_pending_results,
            "pending_input_channel_capacity": SOURCE_QUEUE_BOUND,
            "capacity_rejections": capacity_rejections,
            "received_after_prior_accept": received_after_prior_accept,
            "received_while_prior_still_accepted_at_poll": received_while_prior_still_accepted,
            "reference_cpu_us": reference_cpu_us,
            "progress_cpu_us": progress_cpu_us,
            "prepare_and_submit_cpu_us": prepare_and_submit_cpu_us,
            "receipt_to_verified": latency_summary(&verified_latencies),
            "benchmark_scope": "diagnostic_only",
        })
    );
    Ok(verified)
}

fn run(config: Config) -> Result<usize, String> {
    config.validate()?;
    // A producer must not be started before the GPU context is ready: otherwise
    // the initial frames can be buffered during adapter/pipeline initialization.
    let workload = if config.mode == Mode::Cpu {
        None
    } else {
        Some(GpuWorkload::new(config.width, config.height)?)
    };
    let start = Instant::now();
    let (receiver, worker) = producer(config.clone());
    if config.producer_signals {
        eprintln!("RUNEN_GPU_172_READY");
    }
    let result = if let Some(gpu) = workload {
        run_gpu(&config, receiver, start, gpu)
    } else {
        run_cpu(&config, receiver, start)
    };
    // Dropping the receiver on failure releases a backpressured producer.
    let joined = worker.join();
    match (result, joined) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(_)) => Err("input producer thread panicked".into()),
        (Ok(count), Ok(())) => Ok(count),
    }
}

fn main() {
    let result = Config::parse().and_then(run);
    if let Err(error) = result {
        eprintln!("streaming background example failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runen_gpu::{GpuRetainedSubmissionRejectionCause, GpuWorkGraphCause};

    #[test]
    fn cpu_oracle_is_exact_for_generated_sequence() {
        let config = Config {
            width: 32,
            height: 32,
            max_frames: 4,
            mode: Mode::Cpu,
            ..Config::default()
        };
        assert_eq!(run(config).unwrap(), 4);
    }

    #[test]
    fn nearest_rank_latencies_are_exact_for_small_samples() {
        let report = latency_summary(&[40, 10, 30, 20]);
        assert_eq!(report["p50_us"], 20);
        assert_eq!(report["p95_us"], 40);
        assert_eq!(report["p99_us"], 40);
        assert!(latency_summary(&[]).is_null());
    }

    #[test]
    fn producer_signals_are_serial_stdin_only() {
        let invalid = Config {
            producer_signals: true,
            ..Config::default()
        };
        assert!(invalid.validate().is_err());
        let valid = Config {
            source: Source::Stdin,
            mode: Mode::Serial,
            ..invalid
        };
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn live_batch_requires_operator_owned_wait_bound() {
        let source = Config {
            mode: Mode::Batch,
            source: Source::Stdin,
            ..Config::default()
        };
        assert!(source.validate().is_err());
        let bounded = Config {
            batch_wait_ms: Some(10),
            ..source
        };
        assert!(bounded.validate().is_ok());
    }

    #[test]
    #[ignore = "requires declared real Vulkan fallback adapter; invoked by Conformance CI"]
    fn native_serial_and_batch_match_full_cpu_oracle() {
        for mode in [Mode::Serial, Mode::Batch] {
            let config = Config {
                width: 64,
                height: 64,
                max_frames: 8,
                batch_size: 4,
                mode,
                ..Config::default()
            };
            assert_eq!(run(config).unwrap(), 8);
        }
    }

    #[test]
    #[ignore = "requires declared real Vulkan fallback adapter; invoked by Conformance CI"]
    fn native_dependent_write_reports_first_real_g3_or_g5_boundary() {
        let mut gpu = GpuWorkload::new(64, 64).unwrap();
        let first = frame::generated(1, 64, 64);
        let next = frame::generated(2, 64, 64);
        let accepted = gpu.submit(&[first]).unwrap();
        assert!(matches!(
            accepted.submission.status(),
            GpuSubmissionStatus::Accepted
        ));
        let attempted = gpu.submit(&[next.clone()]);
        match attempted {
            Err(BatchError::Submission(GpuWorkSubmissionError::GraphPreparation(error))) => {
                assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
                assert_eq!(error.resource(), Some(gpu.background_identity()));
                eprintln!(
                    "RUNEN_GPU_172_OBSERVED_PENDING_BOUNDARY=G3:{:?}",
                    error.cause()
                );
            }
            Err(BatchError::Submission(GpuWorkSubmissionError::SubmissionRejected(reason))) => {
                assert_eq!(
                    reason.kind(),
                    GpuSubmissionRejectionKind::RetainedContinuityChanged
                );
                let evidence = reason
                    .retained_continuity_evidence()
                    .expect("typed G5 record");
                assert_eq!(
                    evidence.cause(),
                    GpuRetainedSubmissionRejectionCause::InFlightParticipation
                );
                assert_eq!(evidence.resource(), gpu.background_identity());
                assert_eq!(evidence.affinity(), gpu.context.affinity());
                eprintln!(
                    "RUNEN_GPU_172_OBSERVED_PENDING_BOUNDARY=G5:{:?}",
                    evidence.cause()
                );
            }
            Err(other) => panic!("unexpected boundary: {other:?}"),
            Ok(_) => panic!("next pending dependent retained update must not already be admitted"),
        }

        let deadline = Instant::now() + Duration::from_secs(30);
        while !matches!(accepted.submission.status(), GpuSubmissionStatus::Completed) {
            gpu.context.progress();
            if Instant::now() >= deadline {
                panic!("first update did not complete");
            }
            thread::yield_now();
        }
        // Terminal GPU work does not imply its readback map callback fired.
        while GpuWorkload::ready_masks(&accepted).unwrap().is_none() {
            gpu.context.progress();
            assert!(Instant::now() < deadline);
            thread::yield_now();
        }
        let second = gpu.submit(&[next]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while GpuWorkload::ready_masks(&second).unwrap().is_none() {
            gpu.context.progress();
            assert!(Instant::now() < deadline);
            thread::yield_now();
        }
    }
}
