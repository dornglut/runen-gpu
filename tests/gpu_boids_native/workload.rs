//! Representative G6-I02 boids workload authored directly through RunenGPU.
//!
//! Keep workload semantics and ordinary RunenGPU authoring visible here. Native adapter setup,
//! proof assertions, polling, and artifact retention belong in `main.rs`.

use bytemuck::{Pod, Zeroable};
use runen_gpu::*;

const COMPUTE_SOURCE_KEY: &str = "proof.boids.compute";
const RENDER_SOURCE_KEY: &str = "proof.boids.render";
const SOURCE_REVISION: u64 = 1;
const COMPUTE_WGSL: &str = include_str!("compute.wgsl");
const RENDER_WGSL: &str = include_str!("render.wgsl");

pub(crate) const BOID_COUNT: u32 = 384;
pub(crate) const GRID_CELLS_X: u32 = 10;
pub(crate) const GRID_CELLS_Y: u32 = 10;
pub(crate) const GRID_CELL_COUNT: u32 = GRID_CELLS_X * GRID_CELLS_Y;
pub(crate) const FRAME_COUNT: u32 = 8;
pub(crate) const WIDTH: u32 = 192;
pub(crate) const HEIGHT: u32 = 192;
pub(crate) const WORKGROUP: u32 = 64;
pub(crate) const MAX_SPEED: f32 = 0.46;
pub(crate) const FIXED_DELTA_SECONDS: f32 = 1.0 / 60.0;

const MODE_CLEAR_COUNTS: u32 = 1;
const MODE_COUNT_CELLS: u32 = 2;
const MODE_SCAN_COUNTS: u32 = 3;
const MODE_RESET_CURSORS: u32 = 4;
const MODE_SCATTER_INDICES: u32 = 5;
const MODE_SIMULATE_GRID: u32 = 6;
const MODE_PUBLISH: u32 = 7;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct BoidAgent {
    pub(crate) position: [f32; 2],
    pub(crate) velocity: [f32; 2],
    pub(crate) visual_heading: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ComputeParams {
    frame_meta: [u32; 4],
    grid: [u32; 4],
    sim_a: [f32; 4],
    sim_b: [f32; 4],
    sim_c: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DrawParams {
    world_to_clip: [f32; 4],
    viewport: [f32; 4],
    visible_world: [f32; 4],
    sprite: [f32; 4],
}

#[derive(Clone)]
pub(crate) struct ProgramSources {
    pub(crate) compute: GpuAdmittedProgramSource,
    pub(crate) render: GpuAdmittedProgramSource,
}

#[derive(Clone)]
pub(crate) struct ProofResources {
    pub(crate) state_a: GpuBufferHandle,
    pub(crate) state_b: GpuBufferHandle,
    pub(crate) cell_counts: GpuBufferHandle,
    pub(crate) cell_offsets: GpuBufferHandle,
    pub(crate) cell_cursors: GpuBufferHandle,
    pub(crate) sorted_indices: GpuBufferHandle,
}

pub(crate) struct OffscreenWork {
    pub(crate) graph_label: String,
    pub(crate) fragment: GpuWorkFragment,
    pub(crate) frame_readbacks: Vec<GpuReadbackId>,
    pub(crate) state_readback: GpuReadbackId,
    pub(crate) counts_readback: GpuReadbackId,
    pub(crate) offsets_readback: GpuReadbackId,
    pub(crate) cursors_readback: GpuReadbackId,
    pub(crate) indices_readback: GpuReadbackId,
    pub(crate) resources: ProofResources,
}

pub(crate) fn admitted_sources() -> ProgramSources {
    let [compute, render] = admit_static_wgsl_sources([
        (COMPUTE_SOURCE_KEY, SOURCE_REVISION, COMPUTE_WGSL),
        (RENDER_SOURCE_KEY, SOURCE_REVISION, RENDER_WGSL),
    ])
    .unwrap();
    ProgramSources { compute, render }
}

fn fixed_seed() -> Vec<BoidAgent> {
    fn hash_u32(value: u32) -> u32 {
        let mut x = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277_803_737);
        (x >> 22) ^ x
    }

    fn rand01(seed: u32) -> f32 {
        (hash_u32(seed) & 0x00ff_ffff) as f32 / 16_777_215.0
    }

    (0..BOID_COUNT)
        .map(|index| {
            let i = index as f32;
            let count = BOID_COUNT as f32;
            let angle = i * 2.399_963_2;
            let radius = 0.15 + 0.34 * ((i + 0.5) / count).sqrt();
            let jitter = [
                (rand01(index.wrapping_mul(17).wrapping_add(3)) - 0.5) * 0.03,
                (rand01(index.wrapping_mul(29).wrapping_add(7)) - 0.5) * 0.03,
            ];
            let position = [
                (0.5 + angle.cos() * radius + jitter[0]).rem_euclid(1.0),
                (0.5 + angle.sin() * radius + jitter[1]).rem_euclid(1.0),
            ];
            let speed = 0.17 + 0.19 * rand01(index.wrapping_mul(41).wrapping_add(11));
            let heading = angle + std::f32::consts::FRAC_PI_2;
            let velocity = [heading.cos() * speed, heading.sin() * speed];
            let length = velocity[0].hypot(velocity[1]);
            let visual_heading = [velocity[0] / length, velocity[1] / length];
            BoidAgent {
                position,
                velocity,
                visual_heading,
            }
        })
        .collect()
}

fn compute_params(tick: u32, mode: u32) -> ComputeParams {
    ComputeParams {
        frame_meta: [tick, mode, BOID_COUNT, GRID_CELL_COUNT],
        grid: [GRID_CELLS_X, GRID_CELLS_Y, GRID_CELL_COUNT, 0],
        sim_a: [FIXED_DELTA_SECONDS, MAX_SPEED, 1.10, 0.10],
        sim_b: [0.035, 1.05, 0.72, 1.35],
        sim_c: [0.16, 0.12, 60.0, 0.0],
    }
}

fn draw_params() -> DrawParams {
    DrawParams {
        world_to_clip: [2.0, 2.0, -1.0, -1.0],
        viewport: [
            WIDTH as f32,
            HEIGHT as f32,
            1.0 / WIDTH as f32,
            1.0 / HEIGHT as f32,
        ],
        visible_world: [0.5, 0.5, 1.0, 1.0],
        sprite: [0.0084, 0.01575, 0.0, 0.0],
    }
}

fn prepared_buffer<T: Pod>(
    resources: &mut GpuResourceScope,
    name: &str,
    values: &[T],
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let prepared = PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, values).unwrap();
    let mut usages = usages.into_iter().collect::<Vec<_>>();
    if !usages.contains(&GpuBufferUsage::CopyDestination) {
        usages.push(GpuBufferUsage::CopyDestination);
    }
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                prepared.layout().byte_len(),
                usages,
                GpuBufferInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap()
}

fn proof_resources(
    resources: &mut GpuResourceScope,
) -> (ProofResources, GpuBufferHandle, GpuBufferHandle) {
    let seed = fixed_seed();
    let zeros_cells = vec![0_u32; usize::try_from(GRID_CELL_COUNT).unwrap()];
    let zeros_indices = vec![0_u32; usize::try_from(BOID_COUNT).unwrap()];
    let state_a = prepared_buffer(
        resources,
        "boids state a",
        &seed,
        [
            GpuBufferUsage::Storage,
            GpuBufferUsage::Vertex,
            GpuBufferUsage::CopySource,
        ],
    );
    let state_b = prepared_buffer(
        resources,
        "boids state b",
        &seed,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
    );
    let cell_counts = prepared_buffer(
        resources,
        "boids grid cell counts",
        &zeros_cells,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
    );
    let cell_offsets = prepared_buffer(
        resources,
        "boids grid cell offsets",
        &zeros_cells,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
    );
    let cell_cursors = prepared_buffer(
        resources,
        "boids grid scatter cursors",
        &zeros_cells,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
    );
    let sorted_indices = prepared_buffer(
        resources,
        "boids grid sorted indices",
        &zeros_indices,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
    );
    let initial_params = [compute_params(1, MODE_CLEAR_COUNTS)];
    let params = prepared_buffer(
        resources,
        "boids compute params",
        &initial_params,
        [GpuBufferUsage::Uniform],
    );
    let draw = prepared_buffer(
        resources,
        "boids draw params",
        &[draw_params()],
        [GpuBufferUsage::Uniform],
    );
    (
        ProofResources {
            state_a,
            state_b,
            cell_counts,
            cell_offsets,
            cell_cursors,
            sorted_indices,
        },
        params,
        draw,
    )
}

fn compute_pipeline(source: &GpuAdmittedProgramSource) -> GpuComputePipelineDescriptor {
    GpuComputePipelineDescriptor::ordinary(source.clone(), "cs_main").unwrap()
}

fn render_pipeline(source: &GpuAdmittedProgramSource) -> GpuRenderPipelineDescriptor {
    let vertex_entry = GpuEntryPointName::new("vs_main").unwrap();
    let fragment_entry = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source.clone(),
        [vertex_entry.clone(), fragment_entry.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let instance_layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        std::mem::size_of::<BoidAgent>() as u64,
        GpuVertexStepMode::Instance,
        [
            GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(1, 8, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(2, 16, GpuVertexFormat::Float32x2),
        ],
    )
    .unwrap();
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        GpuBlendMode::Alpha,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([instance_layout]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex_entry, Some(fragment_entry)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

fn compute_operation(
    pipeline: &GpuComputePipelineDescriptor,
    buffers: &ProofResources,
    params: &GpuBufferHandle,
    dispatch_x: u32,
) -> GpuComputeOperation {
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, params),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &buffers.state_a),
            GpuRuntimeBindingValue::whole_buffer(0, 2, &buffers.state_b),
            GpuRuntimeBindingValue::whole_buffer(0, 3, &buffers.cell_counts),
            GpuRuntimeBindingValue::whole_buffer(0, 4, &buffers.cell_offsets),
            GpuRuntimeBindingValue::whole_buffer(0, 5, &buffers.cell_cursors),
            GpuRuntimeBindingValue::whole_buffer(0, 6, &buffers.sorted_indices),
        ])
        .unwrap();
    GpuComputeOperation::new(
        pipeline.clone(),
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(dispatch_x, 1, 1)),
    )
    .unwrap()
}

fn render_operation(
    pipeline: &GpuRenderPipelineDescriptor,
    buffers: &ProofResources,
    draw_params: &GpuBufferHandle,
    view: &GpuTextureViewHandle,
) -> GpuRenderOperation {
    let bindings = pipeline
        .runtime_bindings([GpuRuntimeBindingValue::whole_buffer(0, 0, draw_params)])
        .unwrap();
    let vertex_binding = GpuVertexBufferBinding::new(
        0,
        &buffers.state_a,
        GpuBufferRange::whole(&buffers.state_a).unwrap(),
    )
    .unwrap();
    let draw = GpuRenderDraw::new(
        pipeline.clone(),
        bindings,
        [vertex_binding],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 6).unwrap(),
            GpuDrawRange::new(0, BOID_COUNT).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.02, 0.028, 0.04, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    GpuRenderOperation::new([attachment], None, [draw], None).unwrap()
}

fn offscreen_target(resources: &mut GpuResourceScope) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "boids offscreen target",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                WIDTH,
                HEIGHT,
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
            GpuTextureViewDescriptor::ordinary_full_owned("boids offscreen target view", &texture)
                .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn stage_dispatch(mode: u32) -> u32 {
    match mode {
        MODE_CLEAR_COUNTS | MODE_RESET_CURSORS => GRID_CELL_COUNT.div_ceil(WORKGROUP),
        MODE_SCAN_COUNTS => 1,
        MODE_COUNT_CELLS | MODE_SCATTER_INDICES | MODE_SIMULATE_GRID | MODE_PUBLISH => {
            BOID_COUNT.div_ceil(WORKGROUP)
        }
        _ => unreachable!("bounded boids proof uses only retained stages"),
    }
}

fn stage_name(mode: u32) -> &'static str {
    match mode {
        MODE_CLEAR_COUNTS => "clear counts",
        MODE_COUNT_CELLS => "count cells",
        MODE_SCAN_COUNTS => "scan counts",
        MODE_RESET_CURSORS => "reset cursors",
        MODE_SCATTER_INDICES => "scatter sorted indices",
        MODE_SIMULATE_GRID => "simulate neighbors",
        MODE_PUBLISH => "publish state",
        _ => unreachable!("bounded boids proof uses only retained stages"),
    }
}

fn add_stage(
    builder: &mut GpuWorkFragmentBuilder,
    pipeline: &GpuComputePipelineDescriptor,
    buffers: &ProofResources,
    params_buffer: &GpuBufferHandle,
    tick: u32,
    mode: u32,
) -> Result<(), GpuWorkAuthoringError> {
    let stage = stage_name(mode);
    let params = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        format!("boids tick {tick:03} {stage} params"),
        &[compute_params(tick, mode)],
    )
    .unwrap();
    builder.operation(
        format!("boids tick {tick:03} upload {stage} params"),
        GpuUploadOperation::whole_buffer(params_buffer, params).unwrap(),
    )?;
    builder.operation(
        format!("boids tick {tick:03} {stage}"),
        compute_operation(pipeline, buffers, params_buffer, stage_dispatch(mode)),
    )?;
    Ok(())
}

fn whole_buffer_readback(buffer: &GpuBufferHandle) -> GpuReadbackOperation {
    let region = GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    GpuReadbackOperation::ordinary(region.into()).unwrap()
}

pub(crate) fn offscreen_work(sources: &ProgramSources) -> OffscreenWork {
    let mut resources = GpuResourceScope::new();
    let (proof_resources, params_buffer, draw_params_buffer) = proof_resources(&mut resources);
    let (texture, view) = offscreen_target(&mut resources);
    let compute = compute_pipeline(&sources.compute);
    let render = render_pipeline(&sources.render);
    let mut frame_readbacks = Vec::with_capacity(usize::try_from(FRAME_COUNT).unwrap());
    let mut state_readback = None;
    let mut counts_readback = None;
    let mut offsets_readback = None;
    let mut cursors_readback = None;
    let mut indices_readback = None;

    let fragment = GpuWorkFragment::build("bounded offscreen boids", |builder| {
        for tick in 1..=FRAME_COUNT {
            for mode in [
                MODE_CLEAR_COUNTS,
                MODE_COUNT_CELLS,
                MODE_SCAN_COUNTS,
                MODE_RESET_CURSORS,
                MODE_SCATTER_INDICES,
                MODE_SIMULATE_GRID,
                MODE_PUBLISH,
            ] {
                add_stage(
                    builder,
                    &compute,
                    &proof_resources,
                    &params_buffer,
                    tick,
                    mode,
                )?;
            }

            builder.operation(
                format!("boids render frame {:03}", tick - 1),
                render_operation(&render, &proof_resources, &draw_params_buffer, &view),
            )?;
            let readback = GpuReadbackOperation::ordinary(
                GpuTextureCopyRegion::whole_base_mip(&texture)
                    .unwrap()
                    .into(),
            )
            .unwrap();
            frame_readbacks.push(readback.id());
            builder.operation(format!("boids readback frame {:03}", tick - 1), readback)?;
        }

        for (name, buffer, slot) in [
            ("state", &proof_resources.state_a, &mut state_readback),
            ("counts", &proof_resources.cell_counts, &mut counts_readback),
            (
                "offsets",
                &proof_resources.cell_offsets,
                &mut offsets_readback,
            ),
            (
                "cursors",
                &proof_resources.cell_cursors,
                &mut cursors_readback,
            ),
            (
                "indices",
                &proof_resources.sorted_indices,
                &mut indices_readback,
            ),
        ] {
            let readback = whole_buffer_readback(buffer);
            *slot = Some(readback.id());
            builder.operation(format!("boids final {name} readback"), readback)?;
        }
        Ok(())
    })
    .unwrap();

    OffscreenWork {
        graph_label: "bounded offscreen boids graph".to_owned(),
        fragment,
        frame_readbacks,
        state_readback: state_readback.unwrap(),
        counts_readback: counts_readback.unwrap(),
        offsets_readback: offsets_readback.unwrap(),
        cursors_readback: cursors_readback.unwrap(),
        indices_readback: indices_readback.unwrap(),
        resources: proof_resources,
    }
}
