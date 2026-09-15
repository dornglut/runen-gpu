use runen_gpu::{
    GpuBlendMode, GpuColorTargetStateDescriptor, GpuColorWriteMask, GpuEntryPointName,
    GpuFragmentOutputStateDescriptor, GpuMemoryIntent, GpuReconstruction, GpuResourceCommon,
    GpuResourceLabel, GpuResourceLifetime, GpuResourceProvenance, GpuShaderIoScalarClass,
    GpuTextureDescriptor, GpuTextureDimension, GpuTextureExtent, GpuTextureFormat,
    GpuTextureInitialization, GpuTextureUsage, GpuTextureUsages,
};

fn common(value: &str) -> GpuResourceCommon {
    let label = GpuResourceLabel::new(value).unwrap();
    GpuResourceCommon::owned(
        label.clone(),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        GpuResourceProvenance::new(label, None, None),
    )
    .unwrap()
}

#[test]
fn r32float_is_public_backend_neutral_and_four_bytes_per_texel() {
    let format = GpuTextureFormat::R32Float;

    assert_eq!(format.bytes_per_texel(), 4);
    assert!(!format.is_depth());
    assert!(!format.is_srgb());
}

#[test]
fn r32float_participates_in_public_texture_and_fragment_output_contracts() {
    let texture_label = GpuResourceLabel::new("R32Float public texture").unwrap();
    let extent = GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 16, 8, 1).unwrap();
    let descriptor = GpuTextureDescriptor::new(
        common("R32Float public texture"),
        GpuTextureDimension::D2,
        extent,
        1,
        1,
        GpuTextureFormat::R32Float,
        GpuTextureUsages::new(
            &texture_label,
            [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
        )
        .unwrap(),
        GpuTextureInitialization::Uninitialized,
    )
    .unwrap();
    assert_eq!(descriptor.format(), GpuTextureFormat::R32Float);
    assert_eq!(descriptor.extent(), extent);

    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::R32Float,
        GpuBlendMode::Replace,
        GpuColorWriteMask::RED,
    )
    .unwrap();
    let signature = GpuFragmentOutputStateDescriptor::new([target])
        .expected_signature(GpuEntryPointName::new("fragment_main").unwrap())
        .unwrap();
    let output = signature.locations().next().unwrap();
    assert_eq!(output.value_type().scalar_class(), GpuShaderIoScalarClass::Float);
    assert_eq!(output.value_type().vector_width().get(), 1);
}
