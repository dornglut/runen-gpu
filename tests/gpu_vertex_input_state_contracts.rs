use runen_gpu::{
    GpuEntryPointName, GpuProgramContractCause, GpuShaderIoScalarClass, GpuVertexAttribute,
    GpuVertexBufferLayoutDescriptor, GpuVertexFormat, GpuVertexInputStateDescriptor,
    GpuVertexStepMode,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn vertex_input_state_normalizes_layouts_and_derives_stage_io() {
    let position = GpuVertexBufferLayoutDescriptor::new(
        0,
        20,
        GpuVertexStepMode::Vertex,
        [
            GpuVertexAttribute::new(1, 12, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x3),
        ],
    )
    .unwrap();
    let instance = GpuVertexBufferLayoutDescriptor::new(
        2,
        4,
        GpuVertexStepMode::Instance,
        [GpuVertexAttribute::new(2, 0, GpuVertexFormat::Uint32)],
    )
    .unwrap();

    let state = GpuVertexInputStateDescriptor::new([instance.clone(), position.clone()]).unwrap();
    let equivalent = GpuVertexInputStateDescriptor::new([position, instance]).unwrap();
    assert_eq!(state, equivalent);
    assert_eq!(hash_of(&state), hash_of(&equivalent));
    assert_eq!(
        state
            .layouts()
            .map(|layout| layout.slot())
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert!(state.layout(2).is_some());

    let signature = state
        .expected_signature(GpuEntryPointName::new("vertex_main").unwrap())
        .unwrap();
    assert_eq!(
        signature
            .locations()
            .map(|location| (
                location.location(),
                location.value_type().scalar_class(),
                location.value_type().vector_width().get(),
            ))
            .collect::<Vec<_>>(),
        [
            (0, GpuShaderIoScalarClass::Float, 3),
            (1, GpuShaderIoScalarClass::Float, 2),
            (2, GpuShaderIoScalarClass::Uint, 1),
        ]
    );
}

#[test]
fn vertex_buffer_layout_rejects_invalid_stride_and_attribute_ranges() {
    for stride in [0, 6] {
        let error = GpuVertexBufferLayoutDescriptor::new(
            0,
            stride,
            GpuVertexStepMode::Vertex,
            [GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32)],
        )
        .expect_err("stride must be nonzero and four-byte aligned");
        assert_eq!(
            error.cause(),
            GpuProgramContractCause::VertexInputStateInvalid
        );
    }

    let error = GpuVertexBufferLayoutDescriptor::new(
        0,
        8,
        GpuVertexStepMode::Vertex,
        [GpuVertexAttribute::new(0, 4, GpuVertexFormat::Float32x2)],
    )
    .expect_err("attribute range must stay inside the stride");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::VertexInputStateInvalid
    );
}

#[test]
fn vertex_input_state_rejects_duplicate_slots_and_locations() {
    let first = GpuVertexBufferLayoutDescriptor::new(
        0,
        4,
        GpuVertexStepMode::Vertex,
        [GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32)],
    )
    .unwrap();
    let duplicate_slot = GpuVertexBufferLayoutDescriptor::new(
        0,
        4,
        GpuVertexStepMode::Instance,
        [GpuVertexAttribute::new(1, 0, GpuVertexFormat::Uint32)],
    )
    .unwrap();
    let error = GpuVertexInputStateDescriptor::new([first.clone(), duplicate_slot])
        .expect_err("vertex-buffer slots must be unique");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::VertexInputStateInvalid
    );

    let duplicate_location = GpuVertexBufferLayoutDescriptor::new(
        1,
        4,
        GpuVertexStepMode::Instance,
        [GpuVertexAttribute::new(0, 0, GpuVertexFormat::Uint32)],
    )
    .unwrap();
    let error = GpuVertexInputStateDescriptor::new([first, duplicate_location])
        .expect_err("shader locations must be unique across buffer layouts");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::VertexInputStateInvalid
    );
}

#[test]
fn empty_vertex_input_state_supports_vertexless_draws() {
    let state = GpuVertexInputStateDescriptor::new([]).unwrap();
    assert_eq!(state.layouts().len(), 0);
    assert_eq!(
        state
            .expected_signature(GpuEntryPointName::new("vertex_main").unwrap())
            .unwrap()
            .locations()
            .len(),
        0
    );
}
