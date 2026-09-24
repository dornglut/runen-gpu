use runen_gpu::*;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: &str) -> GpuResourceCommon {
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn test_limits() -> GpuLimits {
    GpuLimits::new(
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        256 * 1024 * 1024,
        8192,
        2048,
        256,
        16,
        2048,
    )
    .unwrap()
}

fn admitted_storage_format(wgsl_format: &str) -> GpuTextureFormat {
    let source_text = format!(
        r#"
@group(0) @binding(0)
var image: texture_storage_2d<{wgsl_format}, write>;

@compute @workgroup_size(1)
fn inspect() {{
    let dimensions = textureDimensions(image);
}}
"#
    );
    let owner = GpuProgramSourceOwnerId::allocate().unwrap();
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new("r1.32bit.storage").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = registry
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new("r1-32bit-format-test", None).unwrap(),
        )
        .unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [GpuEntryPointName::new("inspect").unwrap()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    program
        .interface()
        .binding(GpuBindingKey::try_new(0, 0).unwrap())
        .unwrap()
        .kind()
        .storage_texture_format()
        .unwrap()
}

#[test]
fn baseline_32bit_formats_expose_exact_public_copy_block_sizes() {
    for (format, bytes) in [
        (GpuTextureFormat::R32Sint, 4),
        (GpuTextureFormat::Rg32Uint, 8),
        (GpuTextureFormat::Rg32Sint, 8),
        (GpuTextureFormat::Rg32Float, 8),
        (GpuTextureFormat::Rgba32Uint, 16),
        (GpuTextureFormat::Rgba32Sint, 16),
        (GpuTextureFormat::Rgba32Float, 16),
    ] {
        assert_eq!(format.block_dimensions(), (1, 1));
        assert_eq!(format.copy_block_size(GpuTextureAspect::All), Some(bytes));
        assert_eq!(format.copy_block_size(GpuTextureAspect::Color), Some(bytes));
        assert_eq!(format.copy_block_size(GpuTextureAspect::DepthOnly), None);
        assert!(!format.is_depth());
        assert!(!format.is_srgb());
    }
}

#[test]
fn structural_normalization_preserves_backend_role_facts() {
    let supplied = GpuTextureFormatCapabilities {
        sampled: true,
        filterable: false,
        storage_read: false,
        storage_write: true,
        color_attachment: false,
        depth_stencil: false,
        copy_source: true,
        copy_destination: false,
        block_dimensions: Some((99, 77)),
        block_copy_size: Some(123),
    };
    let capabilities = GpuCapabilities::from_normalized_facts(
        [],
        test_limits(),
        [(GpuTextureFormat::Rgba32Float, supplied)],
    );
    let normalized = capabilities.format(GpuTextureFormat::Rgba32Float).unwrap();

    assert_eq!(normalized.block_dimensions, Some((1, 1)));
    assert_eq!(normalized.block_copy_size, Some(16));
    assert_eq!(normalized.sampled, supplied.sampled);
    assert_eq!(normalized.filterable, supplied.filterable);
    assert_eq!(normalized.storage_read, supplied.storage_read);
    assert_eq!(normalized.storage_write, supplied.storage_write);
    assert_eq!(normalized.color_attachment, supplied.color_attachment);
    assert_eq!(normalized.depth_stencil, supplied.depth_stencil);
    assert_eq!(normalized.copy_source, supplied.copy_source);
    assert_eq!(normalized.copy_destination, supplied.copy_destination);
}

#[test]
fn canonical_wgsl_normalizes_the_core_32bit_storage_family() {
    for (wgsl_format, normalized) in [
        ("r32uint", GpuTextureFormat::R32Uint),
        ("r32sint", GpuTextureFormat::R32Sint),
        ("r32float", GpuTextureFormat::R32Float),
        ("rg32uint", GpuTextureFormat::Rg32Uint),
        ("rg32sint", GpuTextureFormat::Rg32Sint),
        ("rg32float", GpuTextureFormat::Rg32Float),
        ("rgba32uint", GpuTextureFormat::Rgba32Uint),
        ("rgba32sint", GpuTextureFormat::Rgba32Sint),
        ("rgba32float", GpuTextureFormat::Rgba32Float),
    ] {
        assert_eq!(admitted_storage_format(wgsl_format), normalized);
    }
}

#[test]
fn fragment_output_shape_and_integer_blend_rules_follow_format_semantics() {
    for (format, class, width, alpha) in [
        (
            GpuTextureFormat::R32Sint,
            GpuShaderIoScalarClass::Sint,
            1,
            false,
        ),
        (
            GpuTextureFormat::Rg32Float,
            GpuShaderIoScalarClass::Float,
            2,
            false,
        ),
        (
            GpuTextureFormat::Rgba32Uint,
            GpuShaderIoScalarClass::Uint,
            4,
            true,
        ),
        (
            GpuTextureFormat::Rgba32Sint,
            GpuShaderIoScalarClass::Sint,
            4,
            true,
        ),
        (
            GpuTextureFormat::Rgba32Float,
            GpuShaderIoScalarClass::Float,
            4,
            true,
        ),
    ] {
        let target = GpuColorTargetStateDescriptor::new(
            format,
            GpuBlendMode::Replace,
            GpuColorWriteMask::ALL,
        )
        .unwrap();
        let signature = GpuFragmentOutputStateDescriptor::new([target])
            .expected_signature(GpuEntryPointName::new("fragment_main").unwrap())
            .unwrap();
        let output = signature.locations().next().unwrap();
        assert_eq!(output.value_type().scalar_class(), class);
        assert_eq!(output.value_type().vector_width().get(), width);
        assert_eq!(target.has_blendable_alpha_channel(), alpha);
    }

    for format in [
        GpuTextureFormat::R32Uint,
        GpuTextureFormat::R32Sint,
        GpuTextureFormat::Rgba32Uint,
        GpuTextureFormat::Rgba32Sint,
    ] {
        assert!(
            GpuColorTargetStateDescriptor::new(
                format,
                GpuBlendMode::Alpha,
                GpuColorWriteMask::ALL,
            )
            .is_err()
        );
        assert!(
            GpuColorTargetStateDescriptor::new(
                format,
                GpuBlendMode::Replace,
                GpuColorWriteMask::ALL,
            )
            .is_ok()
        );
    }
}

#[test]
fn prepared_texture_data_uses_4_8_and_16_byte_block_rows_through_public_descriptors() {
    const WIDTH: u32 = 3;
    const HEIGHT: u32 = 2;

    for (name, format, bytes_per_texel) in [
        ("r32sint", GpuTextureFormat::R32Sint, 4_u32),
        ("rg32float", GpuTextureFormat::Rg32Float, 8_u32),
        ("rgba32sint", GpuTextureFormat::Rgba32Sint, 16_u32),
    ] {
        let resource_label = label(name);
        let extent =
            GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                .unwrap();
        let bytes_per_row = WIDTH * bytes_per_texel;
        let byte_len = (bytes_per_row * HEIGHT) as usize;
        let data = PreparedGpuData::<TransferData>::from_pod_transfer(
            format!("{name} bytes"),
            &vec![0_u8; byte_len],
            provenance(&format!("{name} bytes")),
        )
        .unwrap();
        let prepared =
            GpuPreparedTextureData::new(&resource_label, data, format, extent, bytes_per_row, 0)
                .unwrap();
        assert_eq!(prepared.bytes_per_row(), bytes_per_row);
        assert_eq!(prepared.rows_per_image(), 0);
        assert_eq!(prepared.data().layout().byte_len(), byte_len as u64);

        let descriptor = GpuTextureDescriptor::new(
            common(name),
            GpuTextureDimension::D2,
            extent,
            1,
            1,
            format,
            GpuTextureUsages::new(&resource_label, [GpuTextureUsage::CopyDestination]).unwrap(),
            GpuTextureInitialization::Prepared(prepared),
        )
        .unwrap();
        assert_eq!(descriptor.format(), format);
        assert_eq!(descriptor.extent(), extent);
    }
}
