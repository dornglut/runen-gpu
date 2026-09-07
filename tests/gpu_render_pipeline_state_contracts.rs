use runen_gpu::{
    GpuBlendMode, GpuColorTargetStateDescriptor, GpuColorWriteMask, GpuCompareFunction,
    GpuDepthStencilStateDescriptor, GpuFragmentOutputStateDescriptor,
    GpuMultisampleStateDescriptor, GpuPrimitiveStateDescriptor, GpuProgramContractCause,
    GpuRenderPipelineStateDescriptor, GpuTextureFormat, GpuVertexInputStateDescriptor,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn vertex_input() -> GpuVertexInputStateDescriptor {
    GpuVertexInputStateDescriptor::new([]).expect("vertexless input state should be valid")
}

fn color_target(format: GpuTextureFormat) -> GpuColorTargetStateDescriptor {
    GpuColorTargetStateDescriptor::new(format, GpuBlendMode::Replace, GpuColorWriteMask::ALL)
        .expect("test color target should be valid")
}

fn depth_state() -> GpuDepthStencilStateDescriptor {
    GpuDepthStencilStateDescriptor::new(
        GpuTextureFormat::Depth32Float,
        true,
        GpuCompareFunction::LessEqual,
    )
    .expect("test depth state should be valid")
}

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn aggregate_render_state_retains_all_correctness_facts() {
    let fragment_output =
        GpuFragmentOutputStateDescriptor::new([color_target(GpuTextureFormat::Rgba8UnormSrgb)]);
    let state = GpuRenderPipelineStateDescriptor::new(
        vertex_input(),
        Some(fragment_output),
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_state()),
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    let equivalent = state.clone();

    assert_eq!(state, equivalent);
    assert_eq!(hash_of(&state), hash_of(&equivalent));
    assert_eq!(state.vertex_input().layouts().len(), 0);
    assert!(state.has_fragment_stage());
    assert!(state.has_color_targets());
    assert_eq!(state.fragment_output().unwrap().color_targets().len(), 1);
    assert_eq!(state.primitive(), GpuPrimitiveStateDescriptor::default());
    assert_eq!(state.depth_stencil(), Some(depth_state()));
    assert_eq!(
        state.multisample(),
        GpuMultisampleStateDescriptor::default()
    );
}

#[test]
fn aggregate_render_state_preserves_both_depth_only_forms() {
    let vertex_only = GpuRenderPipelineStateDescriptor::new(
        vertex_input(),
        None,
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_state()),
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    assert!(!vertex_only.has_fragment_stage());
    assert!(!vertex_only.has_color_targets());

    let fragment_depth_only = GpuRenderPipelineStateDescriptor::new(
        vertex_input(),
        Some(GpuFragmentOutputStateDescriptor::new([])),
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_state()),
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    assert!(fragment_depth_only.has_fragment_stage());
    assert!(!fragment_depth_only.has_color_targets());
}

#[test]
fn aggregate_render_state_rejects_missing_attachments() {
    for fragment_output in [None, Some(GpuFragmentOutputStateDescriptor::new([]))] {
        let error = GpuRenderPipelineStateDescriptor::new(
            vertex_input(),
            fragment_output,
            GpuPrimitiveStateDescriptor::default(),
            None,
            GpuMultisampleStateDescriptor::default(),
        )
        .expect_err("render state needs at least one color or depth attachment");

        assert_eq!(
            error.cause(),
            GpuProgramContractCause::RenderPipelineStateInvalid
        );
    }
}

#[test]
fn alpha_to_coverage_requires_the_first_blendable_alpha_target() {
    let alpha_to_coverage = GpuMultisampleStateDescriptor::new(4, 0b1111, true).unwrap();

    let no_fragment = GpuRenderPipelineStateDescriptor::new(
        vertex_input(),
        None,
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_state()),
        alpha_to_coverage,
    )
    .expect_err("alpha-to-coverage requires a fragment color target");
    assert_eq!(
        no_fragment.cause(),
        GpuProgramContractCause::RenderPipelineStateInvalid
    );

    let integer_first = GpuRenderPipelineStateDescriptor::new(
        vertex_input(),
        Some(GpuFragmentOutputStateDescriptor::new([
            color_target(GpuTextureFormat::R32Uint),
            color_target(GpuTextureFormat::Rgba8Unorm),
        ])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        alpha_to_coverage,
    )
    .expect_err("the first color target must expose blendable alpha");
    assert_eq!(
        integer_first.cause(),
        GpuProgramContractCause::RenderPipelineStateInvalid
    );

    let valid = GpuRenderPipelineStateDescriptor::new(
        vertex_input(),
        Some(GpuFragmentOutputStateDescriptor::new([
            color_target(GpuTextureFormat::Bgra8UnormSrgb),
            color_target(GpuTextureFormat::R32Uint),
        ])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        alpha_to_coverage,
    )
    .expect("the first color target supports alpha-to-coverage");
    assert!(valid.has_color_targets());
}
