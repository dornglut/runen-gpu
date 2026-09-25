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


#[test]
fn compact_vertex_formats_derive_exact_size_alignment_and_shader_io() {
    let cases = [
        (GpuVertexFormat::Uint8, 1, 1, GpuShaderIoScalarClass::Uint, 1),
        (GpuVertexFormat::Uint8x2, 2, 2, GpuShaderIoScalarClass::Uint, 2),
        (GpuVertexFormat::Uint8x4, 4, 4, GpuShaderIoScalarClass::Uint, 4),
        (GpuVertexFormat::Sint8, 1, 1, GpuShaderIoScalarClass::Sint, 1),
        (GpuVertexFormat::Sint8x2, 2, 2, GpuShaderIoScalarClass::Sint, 2),
        (GpuVertexFormat::Sint8x4, 4, 4, GpuShaderIoScalarClass::Sint, 4),
        (GpuVertexFormat::Unorm8, 1, 1, GpuShaderIoScalarClass::Float, 1),
        (GpuVertexFormat::Unorm8x2, 2, 2, GpuShaderIoScalarClass::Float, 2),
        (GpuVertexFormat::Unorm8x4, 4, 4, GpuShaderIoScalarClass::Float, 4),
        (GpuVertexFormat::Snorm8, 1, 1, GpuShaderIoScalarClass::Float, 1),
        (GpuVertexFormat::Snorm8x2, 2, 2, GpuShaderIoScalarClass::Float, 2),
        (GpuVertexFormat::Snorm8x4, 4, 4, GpuShaderIoScalarClass::Float, 4),
    ];
    assert_eq!(cases.len(), 12);
    for (format, size, alignment, class, width) in cases {
        assert_eq!(format.size_bytes(), size, "{format:?}");
        assert_eq!(format.attribute_alignment_bytes(), alignment, "{format:?}");
        let value_type = format.shader_io_type();
        assert_eq!(value_type.scalar_class(), class, "{format:?}");
        assert_eq!(value_type.vector_width().get(), width, "{format:?}");
    }

    for format in [
        GpuVertexFormat::Float32,
        GpuVertexFormat::Float32x2,
        GpuVertexFormat::Float32x3,
        GpuVertexFormat::Float32x4,
        GpuVertexFormat::Uint32,
        GpuVertexFormat::Uint32x2,
        GpuVertexFormat::Uint32x3,
        GpuVertexFormat::Uint32x4,
        GpuVertexFormat::Sint32,
        GpuVertexFormat::Sint32x2,
        GpuVertexFormat::Sint32x3,
        GpuVertexFormat::Sint32x4,
    ] {
        assert_eq!(format.attribute_alignment_bytes(), 4, "{format:?}");
    }
}

#[test]
fn compact_vertex_attribute_offsets_use_per_format_alignment_while_stride_stays_four_byte_aligned() {
    for (format, offset) in [
        (GpuVertexFormat::Uint8, 1),
        (GpuVertexFormat::Sint8, 1),
        (GpuVertexFormat::Unorm8, 1),
        (GpuVertexFormat::Snorm8, 1),
        (GpuVertexFormat::Uint8x2, 2),
        (GpuVertexFormat::Sint8x2, 2),
        (GpuVertexFormat::Unorm8x2, 2),
        (GpuVertexFormat::Snorm8x2, 2),
        (GpuVertexFormat::Uint8x4, 0),
        (GpuVertexFormat::Sint8x4, 0),
        (GpuVertexFormat::Unorm8x4, 0),
        (GpuVertexFormat::Snorm8x4, 0),
    ] {
        GpuVertexBufferLayoutDescriptor::new(
            0,
            4,
            GpuVertexStepMode::Vertex,
            [GpuVertexAttribute::new(0, offset, format)],
        )
        .unwrap_or_else(|error| panic!("{format:?} offset {offset} should be valid: {error:?}"));
    }

    for (format, offset) in [
        (GpuVertexFormat::Uint8x2, 1),
        (GpuVertexFormat::Sint8x2, 1),
        (GpuVertexFormat::Unorm8x2, 1),
        (GpuVertexFormat::Snorm8x2, 1),
        (GpuVertexFormat::Uint8x4, 2),
        (GpuVertexFormat::Sint8x4, 2),
        (GpuVertexFormat::Unorm8x4, 2),
        (GpuVertexFormat::Snorm8x4, 2),
    ] {
        assert!(
            GpuVertexBufferLayoutDescriptor::new(
                0,
                8,
                GpuVertexStepMode::Vertex,
                [GpuVertexAttribute::new(0, offset, format)],
            )
            .is_err(),
            "{format:?} offset {offset} must fail its format alignment"
        );
    }

    assert!(
        GpuVertexBufferLayoutDescriptor::new(
            0,
            2,
            GpuVertexStepMode::Vertex,
            [GpuVertexAttribute::new(0, 0, GpuVertexFormat::Uint8x2)],
        )
        .is_err(),
        "compact attributes do not relax the four-byte vertex stride alignment"
    );
}
