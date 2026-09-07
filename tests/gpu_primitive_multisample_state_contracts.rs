use runen_gpu::{
    GpuCullMode, GpuFrontFace, GpuIndexFormat, GpuMultisampleStateDescriptor,
    GpuPrimitiveStateDescriptor, GpuPrimitiveTopology, GpuProgramContractCause,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn primitive_state_retains_normalized_correctness_facts() {
    let state = GpuPrimitiveStateDescriptor::new(
        GpuPrimitiveTopology::TriangleStrip,
        Some(GpuIndexFormat::Uint32),
        GpuFrontFace::Clockwise,
        GpuCullMode::Back,
    )
    .unwrap();
    let equivalent = state;

    assert_eq!(state, equivalent);
    assert_eq!(hash_of(&state), hash_of(&equivalent));
    assert_eq!(state.topology(), GpuPrimitiveTopology::TriangleStrip);
    assert_eq!(state.strip_index_format(), Some(GpuIndexFormat::Uint32));
    assert_eq!(state.front_face(), GpuFrontFace::Clockwise);
    assert_eq!(state.cull_mode(), GpuCullMode::Back);
}

#[test]
fn primitive_state_rejects_strip_index_format_for_list_topology() {
    let error = GpuPrimitiveStateDescriptor::new(
        GpuPrimitiveTopology::TriangleList,
        Some(GpuIndexFormat::Uint16),
        GpuFrontFace::CounterClockwise,
        GpuCullMode::None,
    )
    .expect_err("list topology cannot declare strip restart index state");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RenderPrimitiveStateInvalid
    );
}

#[test]
fn primitive_defaults_match_the_neutral_triangle_list_contract() {
    let state = GpuPrimitiveStateDescriptor::default();

    assert_eq!(state.topology(), GpuPrimitiveTopology::TriangleList);
    assert_eq!(state.strip_index_format(), None);
    assert_eq!(state.front_face(), GpuFrontFace::CounterClockwise);
    assert_eq!(state.cull_mode(), GpuCullMode::None);
}

#[test]
fn multisample_state_retains_count_mask_and_alpha_coverage() {
    let state = GpuMultisampleStateDescriptor::new(4, 0b1011, true).unwrap();
    let equivalent = state;

    assert_eq!(state, equivalent);
    assert_eq!(hash_of(&state), hash_of(&equivalent));
    assert_eq!(state.sample_count(), 4);
    assert_eq!(state.sample_mask(), 0b1011);
    assert!(state.alpha_to_coverage_enabled());
}

#[test]
fn multisample_state_rejects_invalid_counts_masks_and_alpha_coverage() {
    for count in [0, 3, 65] {
        let error = GpuMultisampleStateDescriptor::new(count, 0, false)
            .expect_err("invalid sample counts must be rejected");
        assert_eq!(
            error.cause(),
            GpuProgramContractCause::RenderMultisampleStateInvalid
        );
    }

    let mask_error = GpuMultisampleStateDescriptor::new(4, 1 << 4, false)
        .expect_err("mask bits outside the sample count must be rejected");
    assert_eq!(
        mask_error.cause(),
        GpuProgramContractCause::RenderMultisampleStateInvalid
    );

    let alpha_error = GpuMultisampleStateDescriptor::new(1, 1, true)
        .expect_err("alpha-to-coverage requires multisampling");
    assert_eq!(
        alpha_error.cause(),
        GpuProgramContractCause::RenderMultisampleStateInvalid
    );
}
