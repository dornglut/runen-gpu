use runen_gpu::{
    GpuEntryPointName, GpuExpectedFragmentOutputSignature, GpuExpectedVertexInputSignature,
    GpuProgramContractCause, GpuShaderIoLocation, GpuShaderIoScalarClass, GpuShaderIoValueType,
};

fn entry_point(name: &str) -> GpuEntryPointName {
    GpuEntryPointName::new(name).expect("test entry-point name should be valid")
}

fn io_type(scalar_class: GpuShaderIoScalarClass, vector_width: u8) -> GpuShaderIoValueType {
    GpuShaderIoValueType::try_new(scalar_class, vector_width)
        .expect("test shader IO type should be valid")
}

fn location(
    location: u32,
    scalar_class: GpuShaderIoScalarClass,
    vector_width: u8,
) -> GpuShaderIoLocation {
    GpuShaderIoLocation::new(location, io_type(scalar_class, vector_width))
}

#[test]
fn expected_vertex_input_signature_normalizes_location_order() {
    let expected = GpuExpectedVertexInputSignature::new(
        entry_point("vs_main"),
        [
            location(1, GpuShaderIoScalarClass::Uint, 1),
            location(0, GpuShaderIoScalarClass::Float, 3),
        ],
    )
    .expect("expected vertex signature should normalize");

    assert_eq!(expected.entry_point().as_str(), "vs_main");
    assert_eq!(
        expected
            .locations()
            .map(|location| location.location())
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn expected_fragment_output_signature_exposes_normalized_locations() {
    let expected = GpuExpectedFragmentOutputSignature::new(
        entry_point("fs_main"),
        [
            location(1, GpuShaderIoScalarClass::Uint, 1),
            location(0, GpuShaderIoScalarClass::Float, 4),
        ],
    )
    .expect("expected fragment signature should normalize");

    assert_eq!(expected.entry_point().as_str(), "fs_main");
    assert_eq!(
        expected
            .locations()
            .map(|location| location.location())
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn expected_stage_io_signatures_reject_duplicate_locations() {
    let error = GpuExpectedVertexInputSignature::new(
        entry_point("vs_main"),
        [
            location(2, GpuShaderIoScalarClass::Float, 2),
            location(2, GpuShaderIoScalarClass::Float, 2),
        ],
    )
    .expect_err("duplicate shader locations must be rejected");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::StageIoSignatureInvalid
    );
}

#[test]
fn stage_io_value_types_reject_invalid_vector_widths() {
    for vector_width in [0, 5] {
        let error = GpuShaderIoValueType::try_new(GpuShaderIoScalarClass::Float, vector_width)
            .expect_err("shader IO vector width must remain in one through four");
        assert_eq!(
            error.cause(),
            GpuProgramContractCause::StageIoSignatureInvalid
        );
    }
}
