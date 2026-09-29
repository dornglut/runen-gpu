use runen_gpu::*;
use std::cell::RefCell;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const MAX_EVENT_LOOP_TICKS: usize = 5_000;
const MAX_SUBMISSION_TICKS: usize = 2_000;

pub(super) const DISPOSITION_PREREQUISITE_ESTABLISHED: u32 = 1;
pub(super) const DISPOSITION_SURFACE_PATH_UNQUALIFIED: u32 = 2;
pub(super) const DISPOSITION_CHARACTERIZATION_UNQUALIFIED: u32 = 3;
pub(super) const DISPOSITION_BACKEND_CAPABILITY_INCONSISTENT: u32 = 4;

const BIT_PUBLIC_SURFACE: u32 = 1 << 0;
const BIT_DIRECT_CENSUS: u32 = 1 << 1;
const BIT_DISPLAY_P3_ADVERTISED: u32 = 1 << 2;
const BIT_DISPLAY_P3_EXECUTED: u32 = 1 << 3;
const BIT_RGBA16FLOAT_ADVERTISED: u32 = 1 << 4;
const BIT_EXTENDED_PAIR_EXECUTED: u32 = 1 << 5;
const BIT_PUBLIC_DISPLAY_P3_EXECUTED: u32 = 1 << 6;

const FORMAT_UNKNOWN: u32 = 0;
const FORMAT_RGBA8_UNORM: u32 = 1;
const FORMAT_BGRA8_UNORM: u32 = 2;
const FORMAT_RGBA16_FLOAT: u32 = 3;

const COLOR_SPACE_SRGB: u32 = 1 << 0;
const COLOR_SPACE_DISPLAY_P3: u32 = 1 << 1;
const COLOR_SPACE_EXTENDED_SRGB: u32 = 1 << 2;
const COLOR_SPACE_EXTENDED_DISPLAY_P3: u32 = 1 << 3;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SurfaceEvidence {
    pub disposition: u32,
    pub bits: u32,
    pub public_format: u32,
    pub direct_format: u32,
    pub advertised_color_spaces: u32,
    pub rgba16float_color_spaces: u32,
    pub diagnostic_stage: u32,
}

#[derive(Clone, Copy)]
enum SurfaceProbeEvent {
    Exit,
}

#[derive(Default)]
struct SurfaceProbeApp {
    created: bool,
}

thread_local! {
    static WINDOWS: RefCell<Option<(Arc<Window>, Arc<Window>)>> = const { RefCell::new(None) };
    static EVENT_LOOP_EXITED: RefCell<bool> = const { RefCell::new(false) };
}

impl ApplicationHandler<SurfaceProbeEvent> for SurfaceProbeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.created {
            return;
        }
        self.created = true;

        let first = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("RunenGPU public browser surface proof")
                        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
                        .with_append(true),
                )
                .expect("browser surface proof must create public RunenGPU canvas"),
        );
        let second = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("RunenGPU direct WGPU DisplayP3 probe")
                        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
                        .with_append(true),
                )
                .expect("browser surface proof must create direct WGPU canvas"),
        );
        WINDOWS.with(|slot| *slot.borrow_mut() = Some((first, second)));
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: SurfaceProbeEvent) {
        match event {
            SurfaceProbeEvent::Exit => {
                EVENT_LOOP_EXITED.with(|slot| *slot.borrow_mut() = true);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        _event: WindowEvent,
    ) {
    }
}

fn start_event_loop() -> EventLoopProxy<SurfaceProbeEvent> {
    WINDOWS.with(|slot| *slot.borrow_mut() = None);
    EVENT_LOOP_EXITED.with(|slot| *slot.borrow_mut() = false);
    let event_loop = EventLoop::<SurfaceProbeEvent>::with_user_event()
        .build()
        .expect("browser surface proof must create one winit event loop");
    let proxy = event_loop.create_proxy();
    event_loop.spawn_app(SurfaceProbeApp::default());
    proxy
}

async fn wait_for_windows() -> Option<(Arc<Window>, Arc<Window>)> {
    for _ in 0..MAX_EVENT_LOOP_TICKS {
        if let Some(windows) = WINDOWS.with(|slot| slot.borrow().clone()) {
            return Some(windows);
        }
        super::browser_yield().await;
    }
    None
}

async fn stop_event_loop(proxy: EventLoopProxy<SurfaceProbeEvent>) {
    let _ = proxy.send_event(SurfaceProbeEvent::Exit);
    for _ in 0..MAX_EVENT_LOOP_TICKS {
        if EVENT_LOOP_EXITED.with(|slot| *slot.borrow()) {
            WINDOWS.with(|slot| *slot.borrow_mut() = None);
            return;
        }
        super::browser_yield().await;
    }
    panic!("browser surface proof event loop did not exit within its bounded tick budget");
}

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).expect("surface proof labels are valid")
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
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

    let mut builder = GpuWorkFragmentBuilder::new(
        label("browser surface proof"),
        provenance("browser surface proof"),
    );
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.into()).unwrap();
    builder
        .add_node(
            label("browser surface clear"),
            GpuWorkOperation::Render(render),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("browser surface clear"),
        )
        .unwrap();
    builder
        .add_node(
            label("browser surface present"),
            GpuWorkOperation::Present(present),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("browser surface present"),
        )
        .unwrap();

    GpuPreparedWorkGraph::prepare(
        label("browser surface clear present graph"),
        [builder.finish().unwrap()],
    )
    .unwrap()
}

async fn terminalize(context: &GpuContext, submission: &GpuSubmission) -> Result<(), u32> {
    for _ in 0..MAX_SUBMISSION_TICKS {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return Ok(()),
            GpuSubmissionStatus::Failed(_) => return Err(114u32),
            GpuSubmissionStatus::Accepted => {}
        }
        super::browser_yield().await;
    }
    Err(119u32)
}

fn normalized_format_code(format: GpuTextureFormat) -> u32 {
    match format {
        GpuTextureFormat::Rgba8Unorm => FORMAT_RGBA8_UNORM,
        GpuTextureFormat::Bgra8Unorm => FORMAT_BGRA8_UNORM,
        _ => FORMAT_UNKNOWN,
    }
}

async fn run_public_surface(window: Arc<Window>) -> Result<u32, u32> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::DesktopPresentationBaseline.requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("browser surface capability evidence");

    let (context, surface) = GpuContext::request_for_surface(descriptor, window)
        .await
        .map_err(|_| 101u32)?;
    if context.adapter_facts().backend() != GpuBackendFamily::BrowserWebGpu {
        return Err(102u32);
    }

    let capabilities = context.surface_capabilities(surface).map_err(|_| 103u32)?;
    if !capabilities.supports_usage(GpuTextureUsage::ColorAttachment) {
        return Err(104u32);
    }
    let format = capabilities
        .format_capabilities()
        .iter()
        .find(|entry| {
            matches!(
                entry.format(),
                GpuTextureFormat::Rgba8Unorm | GpuTextureFormat::Bgra8Unorm
            ) && entry.supports_color_space(GpuSurfaceColorSpace::Srgb)
                && entry.supports_color_space(GpuSurfaceColorSpace::DisplayP3)
        })
        .ok_or(105u32)?
        .format();
    let present_mode = capabilities
        .present_modes()
        .iter()
        .copied()
        .find(|mode| *mode == GpuSurfacePresentMode::Fifo)
        .ok_or(106u32)?;
    let alpha_mode = capabilities
        .alpha_modes()
        .iter()
        .copied()
        .find(|mode| *mode == GpuSurfaceAlphaMode::Opaque)
        .ok_or(107u32)?;

    let mut configured = surface;
    for (index, color_space) in [GpuSurfaceColorSpace::Srgb, GpuSurfaceColorSpace::DisplayP3]
        .into_iter()
        .enumerate()
    {
        let stage = (index as u32) * 10;
        let configuration = GpuSurfaceConfiguration::new(
            WIDTH,
            HEIGHT,
            format,
            color_space,
            [GpuTextureUsage::ColorAttachment],
            present_mode,
            alpha_mode,
            2,
            [],
        )
        .map_err(|_| 108u32 + stage)?;
        configured = context
            .configure_surface(configured, configuration)
            .map_err(|_| 109u32 + stage)?;
        let image = context
            .acquire_surface_image(configured)
            .map_err(|_| 110u32 + stage)?;
        if image.texture().descriptor().common().ownership()
            != GpuResourceOwnership::SurfaceAcquired
        {
            return Err(111u32 + stage);
        }
        let graph = clear_and_present_graph(&image);
        let prepared = context
            .prepare_submission(graph)
            .await
            .map_err(|_| 112u32 + stage)?;
        let submission = context
            .submit_prepared(prepared)
            .map_err(|_| 113u32 + stage)?;
        terminalize(&context, &submission)
            .await
            .map_err(|failure| failure + stage)?;

        let next = context
            .acquire_surface_image(configured)
            .map_err(|_| 115u32 + stage)?;
        if next.lease_id() == image.lease_id() {
            return Err(116u32 + stage);
        }
        next.abandon();
        drop(image);
    }
    context.detach_surface(configured).map_err(|_| 117u32)?;

    let stats = context.execution_stats();
    if stats.prepared_submissions() != 0
        || stats.in_flight_submissions() != 0
        || stats.upload_bytes_in_flight() != 0
        || stats.readback_bytes_in_flight() != 0
        || stats.pending_readbacks() != 0
    {
        return Err(118u32);
    }

    Ok(normalized_format_code(format))
}

fn wgpu_format_code(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => FORMAT_RGBA8_UNORM,
        wgpu::TextureFormat::Bgra8Unorm => FORMAT_BGRA8_UNORM,
        wgpu::TextureFormat::Rgba16Float => FORMAT_RGBA16_FLOAT,
        _ => FORMAT_UNKNOWN,
    }
}

fn color_space_mask(spaces: wgpu::SurfaceColorSpaces) -> u32 {
    let mut mask = 0;
    if spaces.contains(wgpu::SurfaceColorSpaces::SRGB) {
        mask |= COLOR_SPACE_SRGB;
    }
    if spaces.contains(wgpu::SurfaceColorSpaces::DISPLAY_P3) {
        mask |= COLOR_SPACE_DISPLAY_P3;
    }
    if spaces.contains(wgpu::SurfaceColorSpaces::EXTENDED_SRGB) {
        mask |= COLOR_SPACE_EXTENDED_SRGB;
    }
    if spaces.contains(wgpu::SurfaceColorSpaces::EXTENDED_DISPLAY_P3) {
        mask |= COLOR_SPACE_EXTENDED_DISPLAY_P3;
    }
    mask
}

fn clear_direct_surface(device: &wgpu::Device, queue: &wgpu::Queue, frame: wgpu::SurfaceTexture) {
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RunenGPU direct DisplayP3 clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.125,
                        g: 0.25,
                        b: 0.5,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    queue.submit([encoder.finish()]);
    queue.present(frame);
}

async fn run_direct_wgpu(window: Arc<Window>) -> Result<SurfaceEvidence, u32> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let surface = instance
        .create_surface(window)
        .map_err(|_| DISPOSITION_CHARACTERIZATION_UNQUALIFIED)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|_| DISPOSITION_CHARACTERIZATION_UNQUALIFIED)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|_| DISPOSITION_CHARACTERIZATION_UNQUALIFIED)?;

    let caps = surface.get_capabilities(&adapter);
    if caps.format_capabilities.is_empty() {
        return Err(DISPOSITION_CHARACTERIZATION_UNQUALIFIED);
    }

    let rgba16float_entry = caps
        .format_capabilities
        .iter()
        .find(|entry| entry.format == wgpu::TextureFormat::Rgba16Float);

    let selected = [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
    ]
    .into_iter()
    .find_map(|format| {
        caps.format_capabilities.iter().find(|entry| {
            entry.format == format
                && entry
                    .color_spaces
                    .contains(wgpu::SurfaceColorSpaces::DISPLAY_P3)
        })
    })
    .or_else(|| {
        caps.format_capabilities.iter().find(|entry| {
            matches!(
                entry.format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            )
        })
    });

    let mut evidence = SurfaceEvidence {
        disposition: DISPOSITION_BACKEND_CAPABILITY_INCONSISTENT,
        bits: BIT_DIRECT_CENSUS,
        public_format: FORMAT_UNKNOWN,
        direct_format: selected.map_or(FORMAT_UNKNOWN, |entry| wgpu_format_code(entry.format)),
        advertised_color_spaces: selected.map_or(0, |entry| color_space_mask(entry.color_spaces)),
        rgba16float_color_spaces: rgba16float_entry
            .map(|entry| color_space_mask(entry.color_spaces))
            .unwrap_or(0),
        diagnostic_stage: 0,
    };
    if rgba16float_entry.is_some() {
        evidence.bits |= BIT_RGBA16FLOAT_ADVERTISED;
    }
    let Some(selected) = selected else {
        return Ok(evidence);
    };
    if !selected
        .color_spaces
        .contains(wgpu::SurfaceColorSpaces::DISPLAY_P3)
    {
        return Ok(evidence);
    }
    evidence.bits |= BIT_DISPLAY_P3_ADVERTISED;

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: selected.format,
        color_space: wgpu::SurfaceColorSpace::DisplayP3,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    let acquire = || match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Ok(frame),
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Outdated
        | wgpu::CurrentSurfaceTexture::Lost
        | wgpu::CurrentSurfaceTexture::Validation => {
            Err(DISPOSITION_BACKEND_CAPABILITY_INCONSISTENT)
        }
    };

    let Ok(first) = acquire() else {
        return Ok(evidence);
    };
    clear_direct_surface(&device, &queue, first);
    let Ok(second) = acquire() else {
        return Ok(evidence);
    };
    drop(second);

    evidence.bits |= BIT_DISPLAY_P3_EXECUTED;
    Ok(evidence)
}

pub(super) async fn run() -> SurfaceEvidence {
    let proxy = start_event_loop();
    let Some((public_window, direct_window)) = wait_for_windows().await else {
        stop_event_loop(proxy).await;
        return SurfaceEvidence {
            disposition: DISPOSITION_SURFACE_PATH_UNQUALIFIED,
            ..SurfaceEvidence::default()
        };
    };

    let public_format = match run_public_surface(public_window).await {
        Ok(format) => format,
        Err(stage) => {
            stop_event_loop(proxy).await;
            return SurfaceEvidence {
                disposition: DISPOSITION_SURFACE_PATH_UNQUALIFIED,
                diagnostic_stage: stage,
                ..SurfaceEvidence::default()
            };
        }
    };

    let mut evidence = match run_direct_wgpu(direct_window).await {
        Ok(value) => value,
        Err(disposition) => {
            stop_event_loop(proxy).await;
            return SurfaceEvidence {
                disposition,
                bits: BIT_PUBLIC_SURFACE | BIT_PUBLIC_DISPLAY_P3_EXECUTED,
                public_format,
                ..SurfaceEvidence::default()
            };
        }
    };
    evidence.bits |= BIT_PUBLIC_SURFACE | BIT_PUBLIC_DISPLAY_P3_EXECUTED;
    evidence.public_format = public_format;

    if evidence.bits
        & (BIT_PUBLIC_SURFACE
            | BIT_DIRECT_CENSUS
            | BIT_DISPLAY_P3_ADVERTISED
            | BIT_DISPLAY_P3_EXECUTED
            | BIT_PUBLIC_DISPLAY_P3_EXECUTED)
        == (BIT_PUBLIC_SURFACE
            | BIT_DIRECT_CENSUS
            | BIT_DISPLAY_P3_ADVERTISED
            | BIT_DISPLAY_P3_EXECUTED
            | BIT_PUBLIC_DISPLAY_P3_EXECUTED)
    {
        evidence.disposition = DISPOSITION_PREREQUISITE_ESTABLISHED;
    }

    stop_event_loop(proxy).await;
    evidence
}
