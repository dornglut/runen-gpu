use runen_gpu::{
    GpuBlendSource, GpuEntryPointName, GpuExpectedFragmentOutputSignature,
    GpuExpectedVertexInputSignature, GpuFragmentOutputLocation, GpuProgramContractCause,
    GpuShaderIoLocation, GpuShaderIoScalarClass, GpuShaderIoValueType,
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

fn fragment_location(
    location: u32,
    blend_source: Option<GpuBlendSource>,
    scalar_class: GpuShaderIoScalarClass,
    vector_width: u8,
) -> GpuFragmentOutputLocation {
    GpuFragmentOutputLocation::new(location, blend_source, io_type(scalar_class, vector_width))
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
            fragment_location(1, None, GpuShaderIoScalarClass::Uint, 1),
            fragment_location(0, None, GpuShaderIoScalarClass::Float, 4),
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

    let error = GpuExpectedFragmentOutputSignature::new(
        entry_point("fs_main"),
        [
            fragment_location(2, None, GpuShaderIoScalarClass::Float, 2),
            fragment_location(2, None, GpuShaderIoScalarClass::Float, 2),
        ],
    )
    .expect_err("duplicate ordinary fragment locations must be rejected");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::StageIoSignatureInvalid
    );
}

#[test]
fn expected_fragment_output_signature_accepts_exact_dual_source_pair() {
    let expected = GpuExpectedFragmentOutputSignature::new(
        entry_point("fs_main"),
        [
            fragment_location(
                0,
                Some(GpuBlendSource::Secondary),
                GpuShaderIoScalarClass::Float,
                4,
            ),
            fragment_location(
                0,
                Some(GpuBlendSource::Primary),
                GpuShaderIoScalarClass::Float,
                4,
            ),
        ],
    )
    .expect("exact dual-source pair should normalize");

    assert_eq!(
        expected
            .locations()
            .map(|location| location.blend_source())
            .collect::<Vec<_>>(),
        [
            Some(GpuBlendSource::Primary),
            Some(GpuBlendSource::Secondary)
        ]
    );
}

#[test]
fn expected_fragment_output_signature_rejects_malformed_dual_source_shapes() {
    let float4 = io_type(GpuShaderIoScalarClass::Float, 4);
    let float3 = io_type(GpuShaderIoScalarClass::Float, 3);
    let cases = [
        vec![GpuFragmentOutputLocation::new(
            0,
            Some(GpuBlendSource::Primary),
            float4,
        )],
        vec![GpuFragmentOutputLocation::new(
            0,
            Some(GpuBlendSource::Secondary),
            float4,
        )],
        vec![
            GpuFragmentOutputLocation::new(1, Some(GpuBlendSource::Primary), float4),
            GpuFragmentOutputLocation::new(1, Some(GpuBlendSource::Secondary), float4),
        ],
        vec![
            GpuFragmentOutputLocation::new(0, Some(GpuBlendSource::Primary), float4),
            GpuFragmentOutputLocation::new(0, Some(GpuBlendSource::Secondary), float3),
        ],
        vec![
            GpuFragmentOutputLocation::new(0, Some(GpuBlendSource::Primary), float4),
            GpuFragmentOutputLocation::new(0, Some(GpuBlendSource::Primary), float4),
        ],
        vec![
            GpuFragmentOutputLocation::new(0, Some(GpuBlendSource::Primary), float4),
            GpuFragmentOutputLocation::new(0, Some(GpuBlendSource::Secondary), float4),
            GpuFragmentOutputLocation::new(1, None, float4),
        ],
    ];

    for locations in cases {
        let error = GpuExpectedFragmentOutputSignature::new(entry_point("fs_main"), locations)
            .expect_err("malformed dual-source fragment signatures must be rejected");
        assert_eq!(
            error.cause(),
            GpuProgramContractCause::StageIoSignatureInvalid
        );
    }
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
