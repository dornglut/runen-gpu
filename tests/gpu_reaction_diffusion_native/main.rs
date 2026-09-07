//! Native conformance harness for the representative reaction-diffusion workload.
//!
//! The public-API authoring surface reviewed for G6-E01 lives in `workload.rs`.

mod workload;

use runen_gpu::*;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use workload::*;

const MANIFEST_SCHEMA_VERSION: u32 = 1;
const SURFACE_WIDTH: u32 = 64;
const SURFACE_HEIGHT: u32 = 64;
const SURFACE_FORMAT_CANDIDATES: [GpuTextureFormat; 4] = [
    GpuTextureFormat::Bgra8UnormSrgb,
    GpuTextureFormat::Rgba8UnormSrgb,
    GpuTextureFormat::Bgra8Unorm,
    GpuTextureFormat::Rgba8Unorm,
];

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn assert_prepared_graph_evidence(graph: &GpuPreparedWorkGraph) {
    assert!(
        graph.initialization().len() >= 3,
        "prepared graph must retain initialization evidence for both ping-pong states and parameters"
    );
    assert!(
        !graph.dependencies().is_empty(),
        "compute/render/readback ordering must be inferred from accepted resource hazards"
    );
    assert!(
        graph
            .dependencies()
            .iter()
            .flat_map(|dependency| dependency.reasons())
            .all(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. })),
        "reaction-diffusion proof must not rely on duplicate manual ordering"
    );
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::Copy,
    ] {
        assert!(
            graph.requirements().get(feature).is_some(),
            "prepared graph must mechanically require {feature:?}"
        );
    }
}

fn progress_to_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    ids: &[GpuReadbackId],
) -> Vec<GpuReadbackBytes> {
    let handles = ids
        .iter()
        .map(|id| {
            submission
                .readback(*id)
                .expect("readback must remain observable")
                .clone()
        })
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let mut all_ready = true;
        for handle in &handles {
            match handle.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("reaction-diffusion readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("reaction-diffusion submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(
            Instant::now() < deadline,
            "reaction-diffusion submission/readbacks timed out"
        );
        std::thread::yield_now();
    }
    handles
        .into_iter()
        .map(|handle| match handle.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal readback must be ready, got {other:?}"),
        })
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        digest ^= u64::from(*byte);
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    digest
}

fn validate_frame(bytes: &GpuReadbackBytes, envelope: Envelope) -> u64 {
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(envelope.width * envelope.height * 4).unwrap()
    );
    let mut pixel_chunks = bytes.as_bytes().chunks_exact(4);
    assert!(pixel_chunks.remainder().is_empty());
    let pixels = pixel_chunks
        .by_ref()
        .map(|pixel| <[u8; 4]>::try_from(pixel).unwrap())
        .collect::<Vec<_>>();
    assert!(pixels.iter().all(|pixel| pixel[3] == 255));
    let first = pixels[0];
    assert!(
        pixels.iter().any(|pixel| *pixel != first),
        "reaction-diffusion frame must contain spatially varying output"
    );
    fnv1a64(bytes.as_bytes())
}

fn artifact_root() -> PathBuf {
    std::env::var_os("RUNEN_GPU_PROOF_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/runengpu-proof-artifacts"))
        .join("reaction-diffusion")
}

fn write_png(path: &Path, bytes: &GpuReadbackBytes, envelope: Envelope) {
    image::save_buffer_with_format(
        path,
        bytes.as_bytes(),
        envelope.width,
        envelope.height,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .unwrap();
}

fn native_offscreen_context() -> GpuContext {
    let requirements = GpuCapabilityProfile::ComputeBaseline
        .requirements()
        .merge(&GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("reaction-diffusion offscreen sequence proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn native_reaction_diffusion_retains_bounded_png_sequences_and_manifest() {
    let context = native_offscreen_context();
    let sources = admitted_sources();
    let root = artifact_root();
    std::fs::create_dir_all(&root).unwrap();

    let mut jobs = Vec::new();
    for (envelope_index, envelope) in ENVELOPES.into_iter().enumerate() {
        let (graph_label, fragment, ids) = offscreen_work(&sources, envelope);
        let (submission, submission_path) = if envelope_index == 0 {
            let graph = GpuPreparedWorkGraph::prepare(label(&graph_label), [fragment]).unwrap();
            assert_prepared_graph_evidence(&graph);
            let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
            (
                context.submit_prepared(prepared).unwrap(),
                "advanced-inspected",
            )
        } else {
            (
                pollster::block_on(context.submit_work(&graph_label, [fragment])).unwrap(),
                "ordinary",
            )
        };
        let frames = progress_to_readbacks(&context, &submission, &ids);
        assert_eq!(frames.len(), usize::try_from(envelope.frames).unwrap());

        let job_dir = root.join(envelope.name);
        std::fs::create_dir_all(&job_dir).unwrap();
        let mut frame_records = Vec::new();
        for (frame_index, bytes) in frames.iter().enumerate() {
            let hash = validate_frame(bytes, envelope);
            let file_name = format!("frame_{frame_index:03}.png");
            let path = job_dir.join(&file_name);
            write_png(&path, bytes, envelope);
            assert!(path.metadata().unwrap().len() > 0);
            frame_records.push(json!({
                "logical_frame": frame_index,
                "png": format!("{}/{}", envelope.name, file_name),
                "fnv1a64": format!("{hash:016x}"),
            }));
        }
        assert_ne!(
            frame_records.first().unwrap()["fnv1a64"],
            frame_records.last().unwrap()["fnv1a64"],
            "bounded sequence must visibly evolve"
        );
        jobs.push(json!({
            "name": envelope.name,
            "width": envelope.width,
            "height": envelope.height,
            "logical_frames": envelope.frames,
            "iterations_per_frame": envelope.iterations_per_frame,
            "total_compute_iterations": envelope.frames * envelope.iterations_per_frame,
            "workgroup": [WORKGROUP, WORKGROUP, 1],
            "submission_path": submission_path,
            "frames": frame_records,
        }));
    }

    let enabled_features = context
        .device_facts()
        .enabled_features()
        .map(|feature| format!("{feature:?}"))
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload": "runengpu-reaction-diffusion",
        "model": "Gray-Scott",
        "seed": "A=1,B=0 with deterministic centered square A=0,B=1",
        "boundary": "toroidal-wrap",
        "parameters": {
            "dt": 1.0,
            "feed": 0.055,
            "kill": 0.062,
            "diffusion_a": 1.0,
            "diffusion_b": 0.5,
        },
        "backend": format!("{:?}", context.adapter_facts().backend()),
        "fallback": format!("{:?}", context.adapter_facts().fallback()),
        "enabled_capabilities": enabled_features,
        "programs": {
            "compute": {
                "source_key": sources.compute.identity().key().as_str(),
                "source_revision": sources.compute.identity().revision().get(),
                "canonical_wgsl_digest": sources.compute.digest().to_string(),
            },
            "render": {
                "source_key": sources.render.identity().key().as_str(),
                "source_revision": sources.render.identity().revision().get(),
                "canonical_wgsl_digest": sources.render.digest().to_string(),
            },
        },
        "jobs": jobs,
    });
    let manifest_path = root.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert!(manifest_path.metadata().unwrap().len() > 0);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}

fn native_surface_context(window: Arc<Window>) -> (GpuContext, GpuSurfaceHandle, GpuTextureFormat) {
    let requirements = GpuCapabilityProfile::ComputeBaseline
        .requirements()
        .merge(&GpuCapabilityProfile::DesktopPresentationBaseline.requirements())
        .unwrap();
    let mut descriptor = GpuContextDescriptor::new(requirements);
    for format in SURFACE_FORMAT_CANDIDATES {
        descriptor = descriptor.require_format_role(format, GpuFormatRole::ColorAttachment);
    }
    let descriptor = descriptor
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("reaction-diffusion surface proof");
    let (context, surface) =
        pollster::block_on(GpuContext::request_for_surface(descriptor, window))
            .expect("Xvfb/Lavapipe must admit reaction-diffusion presentation");
    let capabilities = context.surface_capabilities(surface).unwrap();
    assert!(capabilities.supports_usage(GpuTextureUsage::ColorAttachment));
    let format = capabilities
        .formats()
        .iter()
        .copied()
        .find(|format| SURFACE_FORMAT_CANDIDATES.contains(format) && format.is_srgb())
        .or_else(|| {
            capabilities
                .formats()
                .iter()
                .copied()
                .find(|format| SURFACE_FORMAT_CANDIDATES.contains(format))
        })
        .expect("surface must expose an admitted SDR color-attachment format");
    let present_mode = capabilities
        .present_modes()
        .iter()
        .copied()
        .find(|mode| *mode == GpuSurfacePresentMode::Fifo)
        .or_else(|| capabilities.present_modes().first().copied())
        .unwrap();
    let alpha_mode = capabilities
        .alpha_modes()
        .iter()
        .copied()
        .find(|mode| *mode == GpuSurfaceAlphaMode::Opaque)
        .or_else(|| capabilities.alpha_modes().first().copied())
        .unwrap();
    let surface = context
        .configure_surface(
            surface,
            GpuSurfaceConfiguration::new(
                SURFACE_WIDTH,
                SURFACE_HEIGHT,
                format,
                [GpuTextureUsage::ColorAttachment],
                present_mode,
                alpha_mode,
                2,
                [],
            )
            .unwrap(),
        )
        .unwrap();
    (context, surface, format)
}

fn progress_to_completion(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("reaction-diffusion surface submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(Instant::now() < deadline, "surface proof timed out");
        std::thread::yield_now();
    }
}

fn run_surface_proof(window: Arc<Window>) {
    let (context, surface, format) = native_surface_context(window);
    let image = context.acquire_surface_image(surface).unwrap();
    let sources = admitted_sources();
    let (graph_label, fragment) = surface_work(&sources, &image, format);
    let graph = GpuPreparedWorkGraph::prepare(label(&graph_label), [fragment]).unwrap();
    assert_prepared_graph_evidence(&graph);
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    progress_to_completion(&context, &submission);

    let next = context
        .acquire_surface_image(surface)
        .expect("successful reaction-diffusion Present must release the surface image lease");
    assert_ne!(next.lease_id(), image.lease_id());
    next.abandon();
    context.detach_surface(surface).unwrap();
    drop(image);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}

#[derive(Default)]
struct ReactionDiffusionSurfaceProof {
    ran: bool,
}

impl ApplicationHandler for ReactionDiffusionSurfaceProof {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ran {
            return;
        }
        self.ran = true;
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("RunenGPU reaction diffusion surface proof")
                        .with_inner_size(PhysicalSize::new(SURFACE_WIDTH, SURFACE_HEIGHT)),
                )
                .unwrap(),
        );
        run_surface_proof(window);
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        _event: WindowEvent,
    ) {
    }
}

#[test]
#[ignore = "requires Xvfb plus a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn native_reaction_diffusion_replays_same_compute_to_render_path_to_surface_present() {
    let mut event_loop_builder = EventLoop::builder();
    #[cfg(target_os = "linux")]
    event_loop_builder.with_x11().with_any_thread(true);
    let event_loop = event_loop_builder.build().unwrap();
    let mut proof = ReactionDiffusionSurfaceProof::default();
    event_loop.run_app(&mut proof).unwrap();
    assert!(proof.ran);
}
