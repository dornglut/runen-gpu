//! Representative reaction-diffusion workload used for G6-E01 API review.
//!
//! Keep the actual RunenGPU authoring calls visible here. Native adapter setup,
//! proof assertions, artifact generation, and polling belong in `main.rs`.

use bytemuck::{Pod, Zeroable};
use runen_gpu::*;

const COMPUTE_SOURCE_KEY: &str = "proof.reaction-diffusion.compute";
const RENDER_SOURCE_KEY: &str = "proof.reaction-diffusion.render";
const SOURCE_REVISION: u64 = 1;
pub(crate) const WORKGROUP: u32 = 8;
const FRAME_COUNT: u32 = 8;

const REACTION_DIFFUSION_COMPUTE_WGSL: &str = include_str!("compute.wgsl");
const REACTION_DIFFUSION_RENDER_WGSL: &str = include_str!("render.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ReactionCell {
    a: f32,
    b: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ReactionParams {
    width: u32,
    height: u32,
    dt: f32,
    feed: f32,
    kill: f32,
    diffusion_a: f32,
    diffusion_b: f32,
    _pad: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Envelope {
    pub(crate) name: &'static str,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) iterations_per_frame: u32,
}

pub(crate) const ENVELOPES: [Envelope; 2] = [
    Envelope {
        name: "64x64-4-iterations",
        width: 64,
        height: 64,
        frames: FRAME_COUNT,
        iterations_per_frame: 4,
    },
    Envelope {
        name: "128x128-8-iterations",
        width: 128,
        height: 128,
        frames: FRAME_COUNT,
        iterations_per_frame: 8,
    },
];

#[derive(Clone)]
pub(crate) struct ProgramSources {
    pub(crate) compute: GpuAdmittedProgramSource,
    pub(crate) render: GpuAdmittedProgramSource,
}

fn reaction_params(width: u32, height: u32) -> ReactionParams {
    ReactionParams {
        width,
        height,
        dt: 1.0,
        feed: 0.055,
        kill: 0.062,
        diffusion_a: 1.0,
        diffusion_b: 0.5,
        _pad: 0.0,
    }
}

fn fixed_seed(width: u32, height: u32) -> Vec<ReactionCell> {
    let mut cells = vec![ReactionCell { a: 1.0, b: 0.0 }; usize::try_from(width * height).unwrap()];
    let half_w = width / 2;
    let half_h = height / 2;
    let radius = (width.min(height) / 10).max(2);
    for y in (half_h - radius)..(half_h + radius) {
        for x in (half_w - radius)..(half_w + radius) {
            let index = usize::try_from(y * width + x).unwrap();
            cells[index] = ReactionCell { a: 0.0, b: 1.0 };
        }
    }
    cells
}

pub(crate) fn admitted_sources() -> ProgramSources {
    let [compute, render] = admit_static_wgsl_sources([
        (
            COMPUTE_SOURCE_KEY,
            SOURCE_REVISION,
            REACTION_DIFFUSION_COMPUTE_WGSL,
        ),
        (
            RENDER_SOURCE_KEY,
            SOURCE_REVISION,
            REACTION_DIFFUSION_RENDER_WGSL,
        ),
    ])
    .unwrap();
    ProgramSources { compute, render }
}

fn compute_pipeline(source: &GpuAdmittedProgramSource) -> GpuComputePipelineDescriptor {
    GpuComputePipelineDescriptor::ordinary(source.clone(), "cs_main").unwrap()
}

fn render_pipeline(
    source: &GpuAdmittedProgramSource,
    format: GpuTextureFormat,
) -> GpuRenderPipelineDescriptor {
    GpuRenderPipelineDescriptor::ordinary_color(source.clone(), "vs_main", "fs_main", format)
        .unwrap()
}

fn buffer(resources: &mut GpuResourceScope, name: &str, byte_len: u64) -> GpuBufferHandle {
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                byte_len,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn compute_operation(
    pipeline: &GpuComputePipelineDescriptor,
    input: &GpuBufferHandle,
    output: &GpuBufferHandle,
    params: &GpuBufferHandle,
    width: u32,
    height: u32,
) -> GpuComputeOperation {
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, input),
            GpuRuntimeBindingValue::whole_buffer(0, 1, output),
            GpuRuntimeBindingValue::whole_buffer(0, 2, params),
        ])
        .unwrap();
    let dispatch = GpuDispatchIntent::direct(GpuDispatchSize::new(
        width.div_ceil(WORKGROUP),
        height.div_ceil(WORKGROUP),
        1,
    ));
    GpuComputeOperation::new(pipeline.clone(), bindings, dispatch).unwrap()
}

fn render_operation(
    pipeline: &GpuRenderPipelineDescriptor,
    state: &GpuBufferHandle,
    params: &GpuBufferHandle,
    view: &GpuTextureViewHandle,
) -> GpuRenderOperation {
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, state),
            GpuRuntimeBindingValue::whole_buffer(0, 1, params),
        ])
        .unwrap();
    GpuRenderOperation::ordinary_color_full_target_direct(
        pipeline,
        bindings,
        view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        GpuDrawRange::new(0, 3).unwrap(),
        GpuDrawRange::new(0, 1).unwrap(),
    )
    .unwrap()
}

fn state_resources(
    resources: &mut GpuResourceScope,
    envelope: Envelope,
) -> (
    GpuBufferHandle,
    GpuBufferHandle,
    GpuBufferHandle,
    PreparedGpuData<TransferData>,
    PreparedGpuData<TransferData>,
) {
    let seed_values = fixed_seed(envelope.width, envelope.height);
    let seed = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "reaction diffusion seed",
        &seed_values,
    )
    .unwrap();
    let params_value = reaction_params(envelope.width, envelope.height);
    let params = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "reaction diffusion parameters",
        &[params_value],
    )
    .unwrap();
    let state_bytes = seed.layout().byte_len();
    let params_bytes = params.layout().byte_len();
    (
        buffer(
            resources,
            &format!("{} state a", envelope.name),
            state_bytes,
        ),
        buffer(
            resources,
            &format!("{} state b", envelope.name),
            state_bytes,
        ),
        buffer(
            resources,
            &format!("{} params", envelope.name),
            params_bytes,
        ),
        seed,
        params,
    )
}

fn offscreen_target(
    resources: &mut GpuResourceScope,
    envelope: Envelope,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = format!("{} offscreen target", envelope.name);
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                &name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                envelope.width,
                envelope.height,
                GpuTextureFormat::Rgba8Unorm,
                [
                    GpuTextureUsage::ColorAttachment,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(
                format!("{} offscreen target view", envelope.name),
                &texture,
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn add_initialization(
    builder: &mut GpuWorkFragmentBuilder,
    state_a: &GpuBufferHandle,
    state_b: &GpuBufferHandle,
    params_buffer: &GpuBufferHandle,
    seed: PreparedGpuData<TransferData>,
    params: PreparedGpuData<TransferData>,
) -> Result<(), GpuWorkAuthoringError> {
    builder.operation(
        "reaction diffusion initialize state a",
        GpuUploadOperation::whole_buffer(state_a, seed.clone()).unwrap(),
    )?;
    builder.operation(
        "reaction diffusion initialize state b",
        GpuUploadOperation::whole_buffer(state_b, seed).unwrap(),
    )?;
    builder.operation(
        "reaction diffusion upload parameters",
        GpuUploadOperation::whole_buffer(params_buffer, params).unwrap(),
    )?;
    Ok(())
}

pub(crate) fn offscreen_work(
    sources: &ProgramSources,
    envelope: Envelope,
) -> (String, GpuWorkFragment, Vec<GpuReadbackId>) {
    assert!(envelope.width >= 32 && envelope.height >= 32);
    assert!(envelope.frames >= 8);
    assert!(envelope.iterations_per_frame > 0);

    let mut resources = GpuResourceScope::new();
    let (state_a, state_b, params_buffer, seed, params) = state_resources(&mut resources, envelope);
    let (texture, view) = offscreen_target(&mut resources, envelope);
    let compute = compute_pipeline(&sources.compute);
    let render = render_pipeline(&sources.render, GpuTextureFormat::Rgba8Unorm);

    let name = format!("{} reaction diffusion sequence", envelope.name);
    let graph_label = format!("{} reaction diffusion graph", envelope.name);
    let mut readbacks = Vec::with_capacity(usize::try_from(envelope.frames).unwrap());
    let fragment = GpuWorkFragment::build(&name, |builder| {
        add_initialization(builder, &state_a, &state_b, &params_buffer, seed, params)?;

        let mut current_is_a = true;
        for frame in 0..envelope.frames {
            for iteration in 0..envelope.iterations_per_frame {
                let (input, output) = if current_is_a {
                    (&state_a, &state_b)
                } else {
                    (&state_b, &state_a)
                };
                let operation = compute_operation(
                    &compute,
                    input,
                    output,
                    &params_buffer,
                    envelope.width,
                    envelope.height,
                );
                builder.operation(
                    format!(
                        "{} frame {frame:03} iteration {iteration:03}",
                        envelope.name
                    ),
                    operation,
                )?;
                current_is_a = !current_is_a;
            }

            let state = if current_is_a { &state_a } else { &state_b };
            builder.operation(
                format!("{} render frame {frame:03}", envelope.name),
                render_operation(&render, state, &params_buffer, &view),
            )?;
            let region = GpuTextureCopyRegion::whole_base_mip(&texture).unwrap();
            let readback = GpuReadbackOperation::ordinary(region.into()).unwrap();
            let readback_id = readback.id();
            builder.operation(
                format!("{} readback frame {frame:03}", envelope.name),
                readback,
            )?;
            readbacks.push(readback_id);
        }
        Ok(())
    })
    .unwrap();

    (graph_label, fragment, readbacks)
}

pub(crate) fn surface_work(
    sources: &ProgramSources,
    image: &GpuAcquiredSurfaceImage,
    format: GpuTextureFormat,
) -> (String, GpuWorkFragment) {
    let envelope = ENVELOPES[0];
    let mut resources = GpuResourceScope::new();
    let (state_a, state_b, params_buffer, seed, params) = state_resources(&mut resources, envelope);
    let compute = compute_pipeline(&sources.compute);
    let render = render_pipeline(&sources.render, format);
    let view = image.default_view().clone();

    let fragment = GpuWorkFragment::build("reaction diffusion surface replay", |builder| {
        add_initialization(builder, &state_a, &state_b, &params_buffer, seed, params)?;

        let mut current_is_a = true;
        for frame in 0..envelope.frames {
            for iteration in 0..envelope.iterations_per_frame {
                let (input, output) = if current_is_a {
                    (&state_a, &state_b)
                } else {
                    (&state_b, &state_a)
                };
                builder.operation(
                    format!("surface frame {frame:03} iteration {iteration:03}"),
                    compute_operation(
                        &compute,
                        input,
                        output,
                        &params_buffer,
                        envelope.width,
                        envelope.height,
                    ),
                )?;
                current_is_a = !current_is_a;
            }
        }

        let state = if current_is_a { &state_a } else { &state_b };
        builder.operation(
            "reaction diffusion surface render",
            render_operation(&render, state, &params_buffer, &view),
        )?;
        builder.operation(
            "reaction diffusion surface Present",
            GpuPresentOperation::whole_view(&view).unwrap(),
        )?;
        Ok(())
    })
    .unwrap();

    ("reaction diffusion surface graph".to_owned(), fragment)
}
