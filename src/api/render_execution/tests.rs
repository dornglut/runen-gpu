//! Executable position-based vertex and bind-group limit checks.
//! These tests exercise the same draw validator used by context-bound preparation.

use crate::*;

const RENDER_WGSL: &str = r#"@group(2) @binding(0)
var<uniform> params: vec4<f32>;

@vertex
fn vs_main(@location(0) value: f32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(value * params.x, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}"#;

fn sparse_draw() -> GpuRenderDraw {
    let [source] =
        admit_static_wgsl_sources([("test.positional-render-slots", 1, RENDER_WGSL)]).unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();

    let vertex_layout = GpuVertexBufferLayoutDescriptor::new(
        2,
        4,
        GpuVertexStepMode::Vertex,
        [GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32)],
    )
    .unwrap();
    let color = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        None,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([vertex_layout]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    let pipeline = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap();

    let mut resources = GpuResourceScope::new();
    let uniform = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "positional-limit uniform",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                16,
                [GpuBufferUsage::Uniform],
                GpuBufferInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();
    let vertices = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "positional-limit vertices",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                16,
                [GpuBufferUsage::Vertex],
                GpuBufferInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();

    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [GpuRuntimeBindingValue::whole_buffer(2, 0, &uniform)],
    )
    .unwrap();
    let vertex_binding =
        GpuVertexBufferBinding::new(2, &vertices, GpuBufferRange::whole(&vertices).unwrap())
            .unwrap();
    GpuRenderDraw::new(
        pipeline,
        bindings,
        [vertex_binding],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 1).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, 1.0, 1.0, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, 1, 1).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap()
}

fn limits(vertex_slots: u32, group_slots: u32, combined_slots: u32) -> GpuLimits {
    GpuLimits::new(
        65_536,
        128 * 1024 * 1024,
        4,
        vertex_slots,
        8,
        2048,
        group_slots,
        combined_slots,
        0,
        0,
        65_535,
        128 * 1024 * 1024,
        2048,
        2048,
        256,
        16,
        2048,
    )
    .unwrap()
}

#[test]
fn sparse_vertex_slot_requires_its_highest_position_not_its_binding_count() {
    let draw = sparse_draw();
    assert_eq!(draw.vertex_buffers().len(), 1);
    assert_eq!(draw.vertex_buffers()[0].slot(), 2);
    let error = draw.validate_limits(limits(2, 3, 6)).unwrap_err();
    assert_eq!(error.cause(), GpuWorkOperationCause::InvalidDraw);
}

#[test]
fn combined_limit_counts_sparse_vertex_and_bind_group_positions() {
    let draw = sparse_draw();
    assert_eq!(draw.bindings().groups().len(), 1);
    assert!(draw.bindings().group(2).is_some());
    let error = draw.validate_limits(limits(3, 3, 5)).unwrap_err();
    assert_eq!(error.cause(), GpuWorkOperationCause::InvalidDraw);
}

#[test]
fn exact_positional_vertex_plus_group_limit_is_admitted() {
    let draw = sparse_draw();
    draw.validate_limits(limits(3, 3, 6)).unwrap();
}
