use runen_gpu::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn configured_surface(window: Arc<Window>) -> (GpuContext, GpuSurfaceHandle, GpuTextureFormat) {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::DesktopPresentationBaseline.requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("G7A2 native surface Present proof");
    let (context, surface) =
        pollster::block_on(GpuContext::request_for_surface(descriptor, window))
            .expect("native surface conformance must admit the Xvfb/Lavapipe presentation target");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback,
        "native surface conformance must execute through the explicitly required fallback path"
    );

    let capabilities = context.surface_capabilities(surface).unwrap();
    assert!(
        capabilities.supports_usage(GpuTextureUsage::ColorAttachment),
        "presentation surface must admit color-attachment usage"
    );
    let format = *capabilities
        .formats()
        .first()
        .expect("presentation surface must publish at least one normalized color format");
    let present_mode = capabilities
        .present_modes()
        .iter()
        .copied()
        .find(|mode| *mode == GpuSurfacePresentMode::Fifo)
        .or_else(|| capabilities.present_modes().first().copied())
        .expect("presentation surface must publish at least one present mode");
    let alpha_mode = capabilities
        .alpha_modes()
        .iter()
        .copied()
        .find(|mode| *mode == GpuSurfaceAlphaMode::Opaque)
        .or_else(|| capabilities.alpha_modes().first().copied())
        .expect("presentation surface must publish at least one alpha mode");
    let configuration = GpuSurfaceConfiguration::new(
        WIDTH,
        HEIGHT,
        format,
        [GpuTextureUsage::ColorAttachment],
        present_mode,
        alpha_mode,
        2,
        [],
    )
    .unwrap();
    let configured = context.configure_surface(surface, configuration).unwrap();
    (context, configured, format)
}

fn clear_and_present_graph(image: &GpuAcquiredSurfaceImage) -> GpuPreparedWorkGraph {
    let texture = image.texture().clone();
    let view = image.default_view().clone();
    let attachment = GpuRenderColorAttachment::new(
        view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.125, 0.25, 0.5, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new(
        [attachment],
        None,
        std::iter::empty::<GpuRenderDraw>(),
        None,
    )
    .unwrap();
    let present =
        GpuPresentOperation::new(view.clone().into(), view.descriptor().subresources()).unwrap();

    let name = "native surface clear Present";
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.into()).unwrap();
    builder
        .add_node(
            label("native surface clear"),
            GpuWorkOperation::Render(render),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("native surface clear"),
        )
        .unwrap();
    builder
        .add_node(
            label("native surface Present"),
            GpuWorkOperation::Present(present),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("native surface Present"),
        )
        .unwrap();

    GpuPreparedWorkGraph::prepare(
        label("native surface clear Present graph"),
        [builder.finish().unwrap()],
    )
    .unwrap()
}

fn progress_to_completion(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("native G7A2 surface submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "native G7A2 surface submission did not terminalize"
        );
        std::thread::yield_now();
    }
}

fn run_surface_proof(window: Arc<Window>) {
    let (context, surface, format) = configured_surface(window);
    let image = context
        .acquire_surface_image(surface)
        .expect("configured surface must produce one acquired image");
    assert_eq!(image.surface_id(), surface.id());
    assert_eq!(image.surface_generation(), surface.generation());
    assert_eq!(image.affinity(), context.affinity());
    assert_eq!(image.texture().descriptor().format(), format);
    assert_eq!(
        image.texture().descriptor().common().ownership(),
        GpuResourceOwnership::SurfaceAcquired
    );

    let graph = clear_and_present_graph(&image);
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    progress_to_completion(&context, &submission);

    let next = context.acquire_surface_image(surface).expect(
        "successful Present must release physical surface authority even while the old logical image token still exists",
    );
    assert_ne!(next.lease_id(), image.lease_id());
    let retired_graph = clear_and_present_graph(&next);

    context
        .detach_surface(surface)
        .expect("surface retirement must release an active acquired-image lease");
    assert!(matches!(
        context.surface_capabilities(surface),
        Err(error) if error.category() == GpuSurfaceErrorCategory::UnknownSurface
    ));
    assert!(matches!(
        context.acquire_surface_image(surface),
        Err(error) if error.category() == GpuSurfaceAcquireErrorCategory::UnknownSurface
    ));
    assert!(matches!(
        context.detach_surface(surface),
        Err(error) if error.category() == GpuSurfaceErrorCategory::UnknownSurface
    ));

    let retired_prepare = pollster::block_on(context.prepare_submission(retired_graph))
        .expect_err("logical resources from a retired surface must not enter G5 preparation");
    assert_eq!(
        retired_prepare.kind(),
        GpuSubmissionPreparationErrorKind::SurfaceLease
    );
    assert_eq!(
        retired_prepare
            .surface_error()
            .expect("surface-lease rejection must preserve structured G7 evidence")
            .category(),
        GpuSurfaceLeaseErrorCategory::UnknownSurface
    );

    next.abandon();
    drop(image);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}

#[derive(Default)]
struct NativeSurfaceProof {
    ran: bool,
}

impl ApplicationHandler for NativeSurfaceProof {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ran {
            return;
        }
        self.ran = true;
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("RunenGPU G7A2 native surface proof")
                        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
                )
                .expect("Xvfb must provide one native window"),
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

fn main() {
    let mut event_loop_builder = EventLoop::builder();
    #[cfg(target_os = "linux")]
    event_loop_builder.with_x11().with_any_thread(true);
    let event_loop = event_loop_builder
        .build()
        .expect("native conformance environment must create a winit event loop");
    let mut proof = NativeSurfaceProof::default();
    event_loop
        .run_app(&mut proof)
        .expect("native surface proof event loop must exit cleanly");
    assert!(proof.ran, "native surface proof must execute after resume");
}
