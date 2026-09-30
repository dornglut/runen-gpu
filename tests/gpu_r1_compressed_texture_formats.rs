use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

#[derive(Clone, Copy)]
struct CompressedCase {
    format: GpuTextureFormat,
    block_bytes: u32,
    srgb: bool,
}

const BC_CASES: [CompressedCase; 14] = [
    CompressedCase {
        format: GpuTextureFormat::Bc1RgbaUnorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc1RgbaUnormSrgb,
        block_bytes: 8,
        srgb: true,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc2RgbaUnorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc2RgbaUnormSrgb,
        block_bytes: 16,
        srgb: true,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc3RgbaUnorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc3RgbaUnormSrgb,
        block_bytes: 16,
        srgb: true,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc4RUnorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc4RSnorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc5RgUnorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc5RgSnorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc6hRgbUfloat,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc6hRgbFloat,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc7RgbaUnorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Bc7RgbaUnormSrgb,
        block_bytes: 16,
        srgb: true,
    },
];

const ETC_CASES: [CompressedCase; 10] = [
    CompressedCase {
        format: GpuTextureFormat::Etc2Rgb8Unorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Etc2Rgb8UnormSrgb,
        block_bytes: 8,
        srgb: true,
    },
    CompressedCase {
        format: GpuTextureFormat::Etc2Rgb8A1Unorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Etc2Rgb8A1UnormSrgb,
        block_bytes: 8,
        srgb: true,
    },
    CompressedCase {
        format: GpuTextureFormat::Etc2Rgba8Unorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::Etc2Rgba8UnormSrgb,
        block_bytes: 16,
        srgb: true,
    },
    CompressedCase {
        format: GpuTextureFormat::EacR11Unorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::EacR11Snorm,
        block_bytes: 8,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::EacRg11Unorm,
        block_bytes: 16,
        srgb: false,
    },
    CompressedCase {
        format: GpuTextureFormat::EacRg11Snorm,
        block_bytes: 16,
        srgb: false,
    },
];

#[derive(Clone, Copy)]
struct AstcCase {
    format: GpuTextureFormat,
    block_width: u32,
    block_height: u32,
    srgb: bool,
}

const ASTC_CASES: [AstcCase; 28] = [
    AstcCase {
        format: GpuTextureFormat::Astc4x4Unorm,
        block_width: 4,
        block_height: 4,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc4x4UnormSrgb,
        block_width: 4,
        block_height: 4,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc5x4Unorm,
        block_width: 5,
        block_height: 4,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc5x4UnormSrgb,
        block_width: 5,
        block_height: 4,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc5x5Unorm,
        block_width: 5,
        block_height: 5,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc5x5UnormSrgb,
        block_width: 5,
        block_height: 5,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc6x5Unorm,
        block_width: 6,
        block_height: 5,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc6x5UnormSrgb,
        block_width: 6,
        block_height: 5,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc6x6Unorm,
        block_width: 6,
        block_height: 6,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc6x6UnormSrgb,
        block_width: 6,
        block_height: 6,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc8x5Unorm,
        block_width: 8,
        block_height: 5,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc8x5UnormSrgb,
        block_width: 8,
        block_height: 5,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc8x6Unorm,
        block_width: 8,
        block_height: 6,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc8x6UnormSrgb,
        block_width: 8,
        block_height: 6,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc8x8Unorm,
        block_width: 8,
        block_height: 8,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc8x8UnormSrgb,
        block_width: 8,
        block_height: 8,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x5Unorm,
        block_width: 10,
        block_height: 5,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x5UnormSrgb,
        block_width: 10,
        block_height: 5,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x6Unorm,
        block_width: 10,
        block_height: 6,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x6UnormSrgb,
        block_width: 10,
        block_height: 6,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x8Unorm,
        block_width: 10,
        block_height: 8,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x8UnormSrgb,
        block_width: 10,
        block_height: 8,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x10Unorm,
        block_width: 10,
        block_height: 10,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc10x10UnormSrgb,
        block_width: 10,
        block_height: 10,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc12x10Unorm,
        block_width: 12,
        block_height: 10,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc12x10UnormSrgb,
        block_width: 12,
        block_height: 10,
        srgb: true,
    },
    AstcCase {
        format: GpuTextureFormat::Astc12x12Unorm,
        block_width: 12,
        block_height: 12,
        srgb: false,
    },
    AstcCase {
        format: GpuTextureFormat::Astc12x12UnormSrgb,
        block_width: 12,
        block_height: 12,
        srgb: true,
    },
];

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    let value = value.as_ref();
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: impl AsRef<str>) -> GpuResourceCommon {
    let value = value.as_ref();
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn copy_requirements() -> GpuCapabilityRequirements {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    requirements
}

fn etc_execution_requirements() -> GpuCapabilityRequirements {
    let mut requirements = copy_requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::RenderPipeline,
        ))
        .unwrap();
    requirements
}

fn require_bc_roles(mut descriptor: GpuContextDescriptor) -> GpuContextDescriptor {
    for case in BC_CASES {
        for role in [
            GpuFormatRole::Sampled,
            GpuFormatRole::CopySource,
            GpuFormatRole::CopyDestination,
        ] {
            descriptor = descriptor.require_format_role(case.format, role);
        }
    }
    descriptor
}

fn require_etc_roles(mut descriptor: GpuContextDescriptor) -> GpuContextDescriptor {
    for case in ETC_CASES {
        for role in [
            GpuFormatRole::Sampled,
            GpuFormatRole::CopySource,
            GpuFormatRole::CopyDestination,
        ] {
            descriptor = descriptor.require_format_role(case.format, role);
        }
    }
    descriptor = descriptor
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment);
    descriptor =
        descriptor.require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource);
    descriptor
}

fn require_astc_roles(mut descriptor: GpuContextDescriptor) -> GpuContextDescriptor {
    for case in ASTC_CASES {
        for role in [
            GpuFormatRole::Sampled,
            GpuFormatRole::Filterable,
            GpuFormatRole::CopySource,
            GpuFormatRole::CopyDestination,
        ] {
            descriptor = descriptor.require_format_role(case.format, role);
        }
    }
    descriptor
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
}

fn descriptor(
    dimension: GpuTextureDimension,
    width: u32,
    height: u32,
    depth_or_layers: u32,
    sample_count: u32,
    format: GpuTextureFormat,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> Result<GpuTextureDescriptor, GpuResourceDescriptorError> {
    let resource_label = label("BC descriptor");
    GpuTextureDescriptor::new(
        common("BC descriptor"),
        dimension,
        GpuTextureExtent::new(&resource_label, dimension, width, height, depth_or_layers).unwrap(),
        1,
        sample_count,
        format,
        GpuTextureUsages::new(&resource_label, usages).unwrap(),
        GpuTextureInitialization::Uninitialized,
    )
}

fn buffer(
    scope: &mut GpuResourceScope,
    name: &str,
    size: u64,
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let resource_label = label(name);
    scope
        .buffer(
            GpuBufferDescriptor::new(
                common(name),
                size,
                GpuBufferUsages::new(&resource_label, usages).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn bc_public_format_census_has_exact_block_semantics() {
    assert_eq!(BC_CASES.len(), 14);
    for case in BC_CASES {
        assert_eq!(case.format.block_dimensions(), (4, 4), "{:?}", case.format);
        assert_eq!(
            case.format.copy_block_size(GpuTextureAspect::Color),
            Some(case.block_bytes),
            "{:?}",
            case.format
        );
        assert_eq!(
            case.format.copy_block_size(GpuTextureAspect::All),
            Some(case.block_bytes),
            "{:?}",
            case.format
        );
        assert_eq!(
            case.format.copy_block_size(GpuTextureAspect::DepthOnly),
            None,
            "{:?}",
            case.format
        );
        assert_eq!(case.format.is_srgb(), case.srgb, "{:?}", case.format);
        assert!(!case.format.is_depth(), "{:?}", case.format);
        assert!(!case.format.is_stencil(), "{:?}", case.format);
    }
}

#[test]
fn bc_resource_contract_is_d2_single_sample_block_aligned_and_non_render_storage() {
    for case in BC_CASES {
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                8,
                12,
                1,
                1,
                case.format,
                [GpuTextureUsage::Sampled, GpuTextureUsage::CopySource],
            )
            .is_ok(),
            "{:?}",
            case.format
        );
        for (width, height) in [(6, 8), (8, 6)] {
            assert!(
                descriptor(
                    GpuTextureDimension::D2,
                    width,
                    height,
                    1,
                    1,
                    case.format,
                    [GpuTextureUsage::CopySource],
                )
                .is_err(),
                "{:?} {width}x{height}",
                case.format
            );
        }
        assert!(
            descriptor(
                GpuTextureDimension::D1,
                8,
                1,
                1,
                1,
                case.format,
                [GpuTextureUsage::CopySource],
            )
            .is_err(),
            "{:?} D1",
            case.format
        );
        assert!(
            descriptor(
                GpuTextureDimension::D3,
                8,
                8,
                4,
                1,
                case.format,
                [GpuTextureUsage::CopySource],
            )
            .is_err(),
            "{:?} D3",
            case.format
        );
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                8,
                8,
                1,
                4,
                case.format,
                [GpuTextureUsage::CopySource],
            )
            .is_err(),
            "{:?} multisampled",
            case.format
        );
        for usage in [
            GpuTextureUsage::StorageRead,
            GpuTextureUsage::StorageWrite,
            GpuTextureUsage::ColorAttachment,
            GpuTextureUsage::DepthStencilAttachment,
        ] {
            assert!(
                descriptor(GpuTextureDimension::D2, 8, 8, 1, 1, case.format, [usage]).is_err(),
                "{:?} {usage:?}",
                case.format
            );
        }
    }
}

fn copy_texture(
    scope: &mut GpuResourceScope,
    format: GpuTextureFormat,
    name: &str,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> GpuTextureHandle {
    let resource_label = label(name);
    scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 1).unwrap(),
                3,
                1,
                format,
                GpuTextureUsages::new(&resource_label, usages).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn bc_copy_regions_enforce_blocks_and_terminal_mips_use_physical_rounding() {
    let mut scope = GpuResourceScope::new();
    let texture = copy_texture(
        &mut scope,
        GpuTextureFormat::Bc1RgbaUnorm,
        "BC copy region",
        [
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ],
    );
    assert!(
        GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(8, 8, 1).unwrap(),
        )
        .is_ok()
    );
    for (origin, extent) in [
        (
            GpuTextureOrigin::new(2, 0, 0),
            GpuCopyExtent::new(4, 4, 1).unwrap(),
        ),
        (
            GpuTextureOrigin::new(0, 2, 0),
            GpuCopyExtent::new(4, 4, 1).unwrap(),
        ),
        (
            GpuTextureOrigin::new(0, 0, 0),
            GpuCopyExtent::new(2, 4, 1).unwrap(),
        ),
        (
            GpuTextureOrigin::new(0, 0, 0),
            GpuCopyExtent::new(4, 2, 1).unwrap(),
        ),
    ] {
        assert!(
            GpuTextureCopyRegion::new(&texture, 0, origin, GpuTextureAspect::Color, extent,)
                .is_err()
        );
    }
    assert!(
        GpuTextureCopyRegion::new(
            &texture,
            2,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(4, 4, 1).unwrap(),
        )
        .is_ok(),
        "logical 2x2 terminal mip must expose one physical 4x4 block"
    );
    assert!(
        GpuTextureCopyRegion::new(
            &texture,
            2,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(8, 4, 1).unwrap(),
        )
        .is_err()
    );
}

#[test]
fn bc_buffer_copy_offsets_follow_copy_block_bytes() {
    for case in [
        CompressedCase {
            format: GpuTextureFormat::Bc1RgbaUnorm,
            block_bytes: 8,
            srgb: false,
        },
        CompressedCase {
            format: GpuTextureFormat::Bc2RgbaUnorm,
            block_bytes: 16,
            srgb: false,
        },
    ] {
        let mut scope = GpuResourceScope::new();
        let texture = copy_texture(
            &mut scope,
            case.format,
            "BC buffer destination",
            [GpuTextureUsage::CopyDestination],
        );
        let buffer = buffer(
            &mut scope,
            "BC upload buffer",
            128,
            [GpuBufferUsage::CopySource],
        );
        let region = GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(4, 4, 1).unwrap(),
        )
        .unwrap();
        let invalid = GpuBufferTextureLayout::new(
            &buffer,
            u64::from(case.block_bytes / 2),
            case.block_bytes,
            0,
        )
        .unwrap();
        assert!(GpuCopyOperation::buffer_to_texture(invalid, region.clone()).is_err());
        let valid =
            GpuBufferTextureLayout::new(&buffer, u64::from(case.block_bytes), case.block_bytes, 0)
                .unwrap();
        assert!(GpuCopyOperation::buffer_to_texture(valid, region).is_ok());
    }
}

#[test]
fn bc_copy_compatibility_is_exact_or_matching_srgb_pair() {
    let mut scope = GpuResourceScope::new();
    let source = copy_texture(
        &mut scope,
        GpuTextureFormat::Bc1RgbaUnorm,
        "BC linear source",
        [GpuTextureUsage::CopySource],
    );
    let paired = copy_texture(
        &mut scope,
        GpuTextureFormat::Bc1RgbaUnormSrgb,
        "BC sRGB destination",
        [GpuTextureUsage::CopyDestination],
    );
    let incompatible = copy_texture(
        &mut scope,
        GpuTextureFormat::Bc2RgbaUnorm,
        "BC2 destination",
        [GpuTextureUsage::CopyDestination],
    );
    let extent = GpuCopyExtent::new(8, 8, 1).unwrap();
    let region = |texture: &GpuTextureHandle| {
        GpuTextureCopyRegion::new(
            texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            extent,
        )
        .unwrap()
    };
    assert!(GpuCopyOperation::texture_to_texture(region(&source), region(&paired)).is_ok());
    assert!(GpuCopyOperation::texture_to_texture(region(&source), region(&incompatible)).is_err());
}

fn expected_bytes(case_index: usize, block_bytes: u32, block_count: u32, salt: u8) -> Vec<u8> {
    (0..block_bytes * block_count)
        .map(|offset| {
            let value = (usize::from(salt) + case_index * 17 + offset as usize * 13) % 251;
            u8::try_from(value + 1).unwrap()
        })
        .collect()
}

fn runtime_texture(
    scope: &mut GpuResourceScope,
    case: CompressedCase,
    name: &str,
) -> GpuTextureHandle {
    let resource_label = label(name);
    scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 2).unwrap(),
                3,
                1,
                case.format,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::Sampled,
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn runtime_graph(
    case_index: usize,
    case: CompressedCase,
) -> (GpuPreparedWorkGraph, [(GpuReadbackId, Vec<u8>); 2]) {
    let mut scope = GpuResourceScope::new();
    let source = runtime_texture(
        &mut scope,
        case,
        &format!("{:?} compressed source", case.format),
    );
    let destination = runtime_texture(
        &mut scope,
        case,
        &format!("{:?} compressed destination", case.format),
    );
    let base_extent = GpuCopyExtent::new(8, 8, 2).unwrap();
    let terminal_extent = GpuCopyExtent::new(4, 4, 2).unwrap();
    let region = |texture: &GpuTextureHandle, mip_level, extent| {
        GpuTextureCopyRegion::new(
            texture,
            mip_level,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            extent,
        )
        .unwrap()
    };
    let source_base = region(&source, 0, base_extent);
    let destination_base = region(&destination, 0, base_extent);
    let source_terminal = region(&source, 2, terminal_extent);
    let destination_terminal = region(&destination, 2, terminal_extent);
    let base_bytes = expected_bytes(case_index, case.block_bytes, 8, 11);
    let terminal_bytes = expected_bytes(case_index, case.block_bytes, 2, 97);
    let base_upload = GpuUploadOperation::new(
        source_base.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "compressed base upload",
            base_bytes.as_slice(),
            provenance("compressed base upload"),
        )
        .unwrap(),
    )
    .unwrap();
    let terminal_upload = GpuUploadOperation::new(
        source_terminal.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "compressed terminal upload",
            terminal_bytes.as_slice(),
            provenance("compressed terminal upload"),
        )
        .unwrap(),
    )
    .unwrap();
    let base_copy =
        GpuCopyOperation::texture_to_texture(source_base, destination_base.clone()).unwrap();
    let terminal_copy =
        GpuCopyOperation::texture_to_texture(source_terminal, destination_terminal.clone())
            .unwrap();
    let base_id = GpuReadbackId::allocate().unwrap();
    let terminal_id = GpuReadbackId::allocate().unwrap();
    let base_readback = GpuReadbackOperation::new(destination_base.into(), base_id).unwrap();
    let terminal_readback =
        GpuReadbackOperation::new(destination_terminal.into(), terminal_id).unwrap();
    let name = format!("{:?} compressed runtime proof", case.format);
    let fragment = GpuWorkFragment::build(&name, |builder| {
        builder.operation("upload compressed base mip", base_upload)?;
        builder.operation("copy compressed base mip", base_copy)?;
        builder.operation("read compressed base mip", base_readback)?;
        builder.operation("upload compressed terminal mip", terminal_upload)?;
        builder.operation("copy compressed terminal mip", terminal_copy)?;
        builder.operation("read compressed terminal mip", terminal_readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{name} graph")), [fragment]).unwrap(),
        [(base_id, base_bytes), (terminal_id, terminal_bytes)],
    )
}

#[test]
fn bc_terminal_physical_block_initializes_logical_terminal_mip_for_following_work() {
    let (graph, readbacks) = runtime_graph(0, BC_CASES[0]);
    assert_eq!(graph.nodes().len(), 6);
    assert_eq!(readbacks.len(), 2);
}

#[test]
fn etc2_eac_public_contract_and_terminal_blocks() {
    assert_eq!(ETC_CASES.len(), 10);
    for (index, case) in ETC_CASES.into_iter().enumerate() {
        assert_eq!(case.format.block_dimensions(), (4, 4));
        assert_eq!(
            case.format.copy_block_size(GpuTextureAspect::Color),
            Some(case.block_bytes)
        );
        assert_eq!(case.format.is_srgb(), case.srgb);
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                8,
                8,
                2,
                1,
                case.format,
                [
                    GpuTextureUsage::Sampled,
                    GpuTextureUsage::CopySource,
                    GpuTextureUsage::CopyDestination
                ]
            )
            .is_ok()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D1,
                8,
                1,
                1,
                1,
                case.format,
                [GpuTextureUsage::Sampled]
            )
            .is_err()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D3,
                8,
                8,
                4,
                1,
                case.format,
                [GpuTextureUsage::Sampled]
            )
            .is_err()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                8,
                8,
                1,
                4,
                case.format,
                [GpuTextureUsage::Sampled]
            )
            .is_err()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                6,
                8,
                1,
                1,
                case.format,
                [GpuTextureUsage::Sampled]
            )
            .is_err()
        );
        for usage in [
            GpuTextureUsage::StorageRead,
            GpuTextureUsage::StorageWrite,
            GpuTextureUsage::ColorAttachment,
            GpuTextureUsage::DepthStencilAttachment,
        ] {
            assert!(descriptor(GpuTextureDimension::D2, 8, 8, 1, 1, case.format, [usage]).is_err());
        }
        let (graph, readbacks) = runtime_graph(index, case);
        assert_eq!(graph.nodes().len(), 6);
        assert_eq!(readbacks.len(), 2);
        let mut scope = GpuResourceScope::new();
        let texture = copy_texture(
            &mut scope,
            case.format,
            "ETC terminal",
            [GpuTextureUsage::CopySource],
        );
        assert!(
            GpuTextureCopyRegion::new(
                &texture,
                2,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                GpuCopyExtent::new(4, 4, 1).unwrap()
            )
            .is_ok()
        );
        assert!(
            GpuTextureCopyRegion::new(
                &texture,
                2,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                GpuCopyExtent::new(8, 4, 1).unwrap()
            )
            .is_err()
        );
    }
}

#[test]
fn etc2_copy_compatibility_accepts_only_exact_or_matching_srgb_pair() {
    let mut scope = GpuResourceScope::new();
    let source = copy_texture(
        &mut scope,
        GpuTextureFormat::Etc2Rgb8Unorm,
        "ETC linear source",
        [GpuTextureUsage::CopySource],
    );
    let paired = copy_texture(
        &mut scope,
        GpuTextureFormat::Etc2Rgb8UnormSrgb,
        "ETC sRGB destination",
        [GpuTextureUsage::CopyDestination],
    );
    let unrelated = copy_texture(
        &mut scope,
        GpuTextureFormat::Etc2Rgb8A1Unorm,
        "ETC punchthrough destination",
        [GpuTextureUsage::CopyDestination],
    );
    let region = |texture: &GpuTextureHandle| {
        GpuTextureCopyRegion::new(
            texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(4, 4, 1).unwrap(),
        )
        .unwrap()
    };
    assert!(GpuCopyOperation::texture_to_texture(region(&source), region(&paired)).is_ok());
    assert!(GpuCopyOperation::texture_to_texture(region(&source), region(&unrelated)).is_err());
}

const ETC_SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
struct VertexOutput { @builtin(position) position: vec4<f32> };
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VertexOutput;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return textureLoad(source, vec2<i32>(0, 0), 0);
}
"#;

async fn sampled_render_oracle(
    context: &GpuContext,
    format: GpuTextureFormat,
    width: u32,
    height: u32,
    block: &[u8],
) -> [u8; 4] {
    let mut scope = GpuResourceScope::new();
    let source_name = format!("{format:?} decoded source");
    let source = scope
        .texture(
            GpuTextureDescriptor::new(
                common(&source_name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &label(&source_name),
                    GpuTextureDimension::D2,
                    width,
                    height,
                    1,
                )
                .unwrap(),
                1,
                1,
                format,
                GpuTextureUsages::new(
                    &label(&source_name),
                    [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let source_view = scope
        .texture_view(
            GpuTextureViewDescriptor::new(
                common("ETC decoded source view"),
                &source,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&source).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let target = scope
        .texture(
            GpuTextureDescriptor::new(
                common("ETC decoded target"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &label("ETC decoded target"),
                    GpuTextureDimension::D2,
                    1,
                    1,
                    1,
                )
                .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &label("ETC decoded target"),
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::CopySource,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let target_view = scope
        .texture_view(
            GpuTextureViewDescriptor::new(
                common("ETC decoded target view"),
                &target,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&target).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let upload_region = GpuTextureCopyRegion::new(
        &source,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(width, height, 1).unwrap(),
    )
    .unwrap();
    let upload = GpuUploadOperation::new(
        upload_region.into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "ETC decoded block",
            block,
            provenance("ETC decoded block"),
        )
        .unwrap(),
    )
    .unwrap();
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new("r1.etc2.decoded.render").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source_code = registry
        .admit_wgsl(
            identity,
            ETC_SAMPLE_WGSL,
            GpuProgramSourceProvenance::new("ETC2 decoded render oracle", None).unwrap(),
        )
        .unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source_code,
        [vertex.clone(), fragment.clone()],
        [
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
        ],
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([
            GpuColorTargetStateDescriptor::new(
                GpuTextureFormat::Rgba8Unorm,
                None,
                GpuColorWriteMask::ALL,
            )
            .unwrap(),
        ])),
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
    let binding = GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(source_view),
        )],
    )
    .unwrap();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), [binding]).unwrap();
    let draw = GpuRenderDraw::new(
        pipeline,
        bindings,
        [],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 3).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, 1.0, 1.0, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, 1, 1).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        target_view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new([attachment], None, [draw], None).unwrap();
    let id = GpuReadbackId::allocate().unwrap();
    let read_region = GpuTextureCopyRegion::new(
        &target,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let readback = GpuReadbackOperation::new(read_region.into(), id).unwrap();
    let name = format!("{format:?} ETC decoded render");
    let fragment = GpuWorkFragment::build(&name, |builder| {
        builder.operation("upload validated ETC block", upload)?;
        builder.operation("sample ETC block into RGBA8", render)?;
        builder.operation("read decoded RGBA8 pixel", readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(label(&name), [fragment]).unwrap();
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(context, &submission, id, &name).await;
    bytes
        .as_bytes()
        .try_into()
        .expect("decoded RGBA8 readback is one pixel")
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EtcDecodedMismatch {
    pub(crate) case: u32,
    pub(crate) actual: [u8; 4],
    pub(crate) expected: [u8; 4],
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_no_decoded_mismatch(mismatch: Option<EtcDecodedMismatch>) {
    if let Some(mismatch) = mismatch {
        panic!(
            "ETC decoded case {} differs: actual={:?}, expected={:?}",
            mismatch.case, mismatch.actual, mismatch.expected
        );
    }
}

async fn run_etc_decoded_oracles(context: &GpuContext) -> Option<EtcDecodedMismatch> {
    // Khronos Data Format 1.4, ETC2/EAC block tables: RGB selector 00 adds +2;
    // The EAC selector contributes +4 to the expanded 11-bit base value.
    let rgb = [0xff, 0, 0, 0, 0, 0, 0, 0];
    let alpha = [0x80, 0x1d, 0x92, 0x49, 0x24, 0x92, 0x49, 0x24];
    let mut rgba = alpha.to_vec();
    rgba.extend_from_slice(&rgb);
    let mut rg = alpha.to_vec();
    rg.extend_from_slice(&alpha);
    let signed = [0x40, 0x1d, 0x92, 0x49, 0x24, 0x92, 0x49, 0x24];
    let mut signed_rg = signed.to_vec();
    signed_rg.extend_from_slice(&signed);
    let cases = [
        (
            GpuTextureFormat::Etc2Rgb8Unorm,
            rgb.as_slice(),
            [255, 2, 2, 255],
        ),
        (
            GpuTextureFormat::Etc2Rgb8UnormSrgb,
            rgb.as_slice(),
            [255, 0, 0, 255],
        ),
        (
            GpuTextureFormat::Etc2Rgba8Unorm,
            rgba.as_slice(),
            [255, 2, 2, 128],
        ),
        (
            GpuTextureFormat::Etc2Rgba8UnormSrgb,
            rgba.as_slice(),
            [255, 0, 0, 128],
        ),
        (
            GpuTextureFormat::EacR11Unorm,
            alpha.as_slice(),
            [128, 0, 0, 255],
        ),
        (
            GpuTextureFormat::EacRg11Unorm,
            rg.as_slice(),
            [128, 128, 0, 255],
        ),
        (
            GpuTextureFormat::EacR11Snorm,
            signed.as_slice(),
            [129, 0, 0, 255],
        ),
        (
            GpuTextureFormat::EacRg11Snorm,
            signed_rg.as_slice(),
            [129, 129, 0, 255],
        ),
    ];
    for (index, (format, block, expected)) in cases.into_iter().enumerate() {
        let actual = sampled_render_oracle(context, format, 4, 4, block).await;
        if actual != expected {
            return Some(EtcDecodedMismatch {
                case: u32::try_from(index + 1).unwrap(),
                actual,
                expected,
            });
        }
    }
    None
}

async fn run_suite(context: &GpuContext, cases: &[CompressedCase]) -> u32 {
    let full_mask = (1_u32 << cases.len()) - 1;
    let mut mask = 0_u32;
    for (case_index, &case) in cases.iter().enumerate() {
        let facts = context
            .adapter_facts()
            .supported()
            .format(case.format)
            .expect("compressed format must remain in the normalized census");
        assert!(facts.sampled, "{:?}", case.format);
        assert!(facts.copy_source, "{:?}", case.format);
        assert!(facts.copy_destination, "{:?}", case.format);
        assert!(facts.filterable, "{:?}", case.format);
        assert!(
            !facts.storage_read && !facts.storage_write,
            "{:?}",
            case.format
        );
        assert!(
            !facts.color_attachment && !facts.depth_stencil,
            "{:?}",
            case.format
        );
        let (graph, readbacks) = runtime_graph(case_index, case);
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        for (id, expected) in readbacks {
            let bytes = readback_wait::wait_for_readback(
                context,
                &submission,
                id,
                format!("{:?} BC", case.format),
            )
            .await;
            assert_eq!(bytes.as_bytes(), expected.as_slice(), "{:?}", case.format);
            assert_eq!(bytes.texture_format(), Some(case.format));
        }
        println!(
            "{:?} compressed: EXERCISED (two array layers, 8x8 multi-block + terminal 2x2 logical mips through 4x4 physical blocks)",
            case.format
        );
        mask |= 1 << case_index;
    }
    assert_eq!(mask, full_mask);
    mask
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_bc() -> u32 {
    let census = GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser BC census"),
    )
    .await
    .expect("actual-browser Conformance must provide WebGPU");
    let mut supported_count = 0;
    for case in BC_CASES {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .expect("BC format must be represented even when its feature is absent");
        let portable = facts.sampled && facts.copy_source && facts.copy_destination;
        let any_portable = facts.sampled || facts.copy_source || facts.copy_destination;
        assert!(
            !facts.storage_read
                && !facts.storage_write
                && !facts.color_attachment
                && !facts.depth_stencil,
            "{:?} browser BC facts must retain the portable role clamp",
            case.format
        );
        if portable {
            assert!(
                facts.filterable,
                "{:?} supported browser BC must preserve guaranteed filtering",
                case.format
            );
            supported_count += 1;
        } else {
            assert!(
                !any_portable && !facts.filterable,
                "{:?} browser BC family must be wholly available or wholly absent",
                case.format
            );
        }
    }
    if supported_count == 0 {
        return 0;
    }
    assert_eq!(
        supported_count,
        BC_CASES.len(),
        "browser BC support must not expose a partial family"
    );
    let context = GpuContext::request(require_bc_roles(
        GpuContextDescriptor::new(copy_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser BC proof"),
    ))
    .await
    .expect("advertised browser BC roles must admit a device with private BC feature enabled");
    run_suite(&context, &BC_CASES).await
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_etc2(
    correlated: &GpuAdapterFacts,
) -> (u32, Option<EtcDecodedMismatch>) {
    let census = GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser ETC2 census"),
    )
    .await
    .expect("actual-browser Conformance must provide WebGPU");
    assert_eq!(
        census.adapter_facts(),
        correlated,
        "ETC2 census must use the correlated browser adapter"
    );
    for case in ETC_CASES {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .expect("ETC2/EAC format must remain in normalized census");
        assert!(
            facts.sampled && facts.copy_source && facts.copy_destination && facts.filterable,
            "{:?} browser ETC2 feature is advertised but public roles are absent",
            case.format
        );
        assert!(
            !facts.storage_read
                && !facts.storage_write
                && !facts.color_attachment
                && !facts.depth_stencil
        );
    }
    let context = GpuContext::request(require_etc_roles(
        GpuContextDescriptor::new(etc_execution_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser ETC2 execution"),
    ))
    .await
    .expect("advertised browser ETC2 roles must admit a feature-enabled device");
    assert_eq!(
        context.adapter_facts(),
        correlated,
        "ETC2 execution must use the correlated browser adapter"
    );
    let mask = run_suite(&context, &ETC_CASES).await;
    (mask, run_etc_decoded_oracles(&context).await)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)] // Called when this retained test module is included by Metal qualification.
pub(crate) async fn run_etc2_on_adapter(
    backend: GpuBackendFamily,
    correlated: &GpuAdapterFacts,
    direct_feature: bool,
) -> u32 {
    let census = GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_allowed_backends([backend])
            .with_label("R1 ETC2 correlated census"),
    )
    .await
    .expect("qualified adapter must remain available");
    assert_eq!(census.adapter_facts(), correlated);
    for case in ETC_CASES {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .unwrap();
        let portable =
            facts.sampled && facts.copy_source && facts.copy_destination && facts.filterable;
        assert_eq!(
            portable, direct_feature,
            "{:?} normalized roles must match direct feature",
            case.format
        );
        assert!(
            !facts.storage_read
                && !facts.storage_write
                && !facts.color_attachment
                && !facts.depth_stencil
        );
    }
    let descriptor = require_etc_roles(
        GpuContextDescriptor::new(etc_execution_requirements())
            .with_allowed_backends([backend])
            .with_label("R1 ETC2 correlated execution"),
    );
    if !direct_feature {
        let error = GpuContext::request(descriptor)
            .await
            .expect_err("absent ETC2 feature must reject required public roles");
        assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        return 0;
    }
    let context = GpuContext::request(descriptor)
        .await
        .expect("advertised ETC2 feature must admit public format roles");
    assert_eq!(context.adapter_facts(), correlated);
    let mask = run_suite(&context, &ETC_CASES).await;
    assert_no_decoded_mismatch(run_etc_decoded_oracles(&context).await);
    mask
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn etc2_native_unsupported_roles_are_typed() {
    let census = pollster::block_on(GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("R1 native ETC2 census"),
    ))
    .expect("retained Vulkan software adapter must be available");
    let all_supported = ETC_CASES.iter().all(|case| {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .unwrap();
        facts.sampled && facts.copy_source && facts.copy_destination
    });
    if all_supported {
        let context = pollster::block_on(GpuContext::request(require_etc_roles(
            GpuContextDescriptor::new(etc_execution_requirements())
                .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
                .with_allowed_backends([GpuBackendFamily::Vulkan])
                .with_label("R1 native ETC2 execution"),
        )))
        .expect("advertised native ETC2 roles must admit a device");
        assert_eq!(context.adapter_facts(), census.adapter_facts());
        assert_eq!(
            pollster::block_on(run_suite(&context, &ETC_CASES)),
            (1 << ETC_CASES.len()) - 1
        );
        assert_no_decoded_mismatch(pollster::block_on(run_etc_decoded_oracles(&context)));
    } else {
        for case in ETC_CASES {
            let facts = census
                .adapter_facts()
                .supported()
                .format(case.format)
                .unwrap();
            assert!(
                !facts.sampled
                    && !facts.copy_source
                    && !facts.copy_destination
                    && !facts.filterable,
                "ETC2/EAC must be wholly absent when the feature is absent"
            );
        }
        assert!(
            pollster::block_on(GpuContext::request(require_etc_roles(
                GpuContextDescriptor::new(etc_execution_requirements())
                    .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
                    .with_allowed_backends([GpuBackendFamily::Vulkan])
                    .with_label("R1 native ETC2 rejected roles"),
            )))
            .is_err()
        );
    }
}

#[test]
fn astc_ldr_public_contract_has_exact_variable_block_semantics() {
    assert_eq!(ASTC_CASES.len(), 28);
    for case in ASTC_CASES {
        assert_eq!(
            case.format.block_dimensions(),
            (case.block_width, case.block_height)
        );
        assert_eq!(
            case.format.copy_block_size(GpuTextureAspect::Color),
            Some(16)
        );
        assert_eq!(case.format.is_srgb(), case.srgb);
        assert!(!case.format.is_depth());
        assert!(!case.format.is_stencil());
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                case.block_width * 2,
                case.block_height * 2,
                2,
                1,
                case.format,
                [
                    GpuTextureUsage::Sampled,
                    GpuTextureUsage::CopySource,
                    GpuTextureUsage::CopyDestination,
                ],
            )
            .is_ok()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D1,
                case.block_width,
                1,
                1,
                1,
                case.format,
                [GpuTextureUsage::Sampled],
            )
            .is_err()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D3,
                case.block_width,
                case.block_height,
                2,
                1,
                case.format,
                [GpuTextureUsage::Sampled],
            )
            .is_err()
        );
        assert!(
            descriptor(
                GpuTextureDimension::D2,
                case.block_width,
                case.block_height,
                1,
                4,
                case.format,
                [GpuTextureUsage::Sampled],
            )
            .is_err()
        );
        for usage in [
            GpuTextureUsage::StorageRead,
            GpuTextureUsage::StorageWrite,
            GpuTextureUsage::ColorAttachment,
            GpuTextureUsage::DepthStencilAttachment,
        ] {
            assert!(
                descriptor(
                    GpuTextureDimension::D2,
                    case.block_width,
                    case.block_height,
                    1,
                    1,
                    case.format,
                    [usage],
                )
                .is_err()
            );
        }
    }
}

fn astc_runtime_texture(
    scope: &mut GpuResourceScope,
    case: AstcCase,
    name: &str,
) -> GpuTextureHandle {
    let resource_label = label(name);
    scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &resource_label,
                    GpuTextureDimension::D2,
                    case.block_width * 2,
                    case.block_height * 2,
                    2,
                )
                .unwrap(),
                3,
                1,
                case.format,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::Sampled,
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn astc_runtime_graph(
    case_index: usize,
    case: AstcCase,
) -> (GpuPreparedWorkGraph, [(GpuReadbackId, Vec<u8>); 2]) {
    let mut scope = GpuResourceScope::new();
    let source = astc_runtime_texture(&mut scope, case, &format!("{:?} ASTC source", case.format));
    let destination = astc_runtime_texture(
        &mut scope,
        case,
        &format!("{:?} ASTC destination", case.format),
    );
    let base_extent = GpuCopyExtent::new(case.block_width * 2, case.block_height * 2, 2).unwrap();
    let terminal_extent = GpuCopyExtent::new(case.block_width, case.block_height, 2).unwrap();
    let region = |texture: &GpuTextureHandle, mip_level, extent| {
        GpuTextureCopyRegion::new(
            texture,
            mip_level,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            extent,
        )
        .unwrap()
    };
    let source_base = region(&source, 0, base_extent);
    let destination_base = region(&destination, 0, base_extent);
    let source_terminal = region(&source, 2, terminal_extent);
    let destination_terminal = region(&destination, 2, terminal_extent);
    let base_bytes = expected_bytes(case_index, 16, 8, 31);
    let terminal_bytes = expected_bytes(case_index, 16, 2, 149);
    let base_upload = GpuUploadOperation::new(
        source_base.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "ASTC base upload",
            base_bytes.as_slice(),
            provenance("ASTC base upload"),
        )
        .unwrap(),
    )
    .unwrap();
    let terminal_upload = GpuUploadOperation::new(
        source_terminal.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "ASTC terminal upload",
            terminal_bytes.as_slice(),
            provenance("ASTC terminal upload"),
        )
        .unwrap(),
    )
    .unwrap();
    let base_copy =
        GpuCopyOperation::texture_to_texture(source_base, destination_base.clone()).unwrap();
    let terminal_copy =
        GpuCopyOperation::texture_to_texture(source_terminal, destination_terminal.clone())
            .unwrap();
    let base_id = GpuReadbackId::allocate().unwrap();
    let terminal_id = GpuReadbackId::allocate().unwrap();
    let base_readback = GpuReadbackOperation::new(destination_base.into(), base_id).unwrap();
    let terminal_readback =
        GpuReadbackOperation::new(destination_terminal.into(), terminal_id).unwrap();
    let name = format!("{:?} ASTC runtime proof", case.format);
    let fragment = GpuWorkFragment::build(&name, |builder| {
        builder.operation("upload ASTC base mip", base_upload)?;
        builder.operation("copy ASTC base mip", base_copy)?;
        builder.operation("read ASTC base mip", base_readback)?;
        builder.operation("upload ASTC terminal mip", terminal_upload)?;
        builder.operation("copy ASTC terminal mip", terminal_copy)?;
        builder.operation("read ASTC terminal mip", terminal_readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{name} graph")), [fragment]).unwrap(),
        [(base_id, base_bytes), (terminal_id, terminal_bytes)],
    )
}

fn astc_void_extent_red() -> [u8; 16] {
    [
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff,
        0xff,
    ]
}

fn astc_void_extent_mid_gray() -> [u8; 16] {
    [
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff,
        0xff,
    ]
}

async fn run_astc_decoded_oracles(context: &GpuContext) {
    let red = astc_void_extent_red();
    for case in [ASTC_CASES[0], ASTC_CASES[12], ASTC_CASES[26]] {
        let actual = sampled_render_oracle(
            context,
            case.format,
            case.block_width,
            case.block_height,
            &red,
        )
        .await;
        assert_eq!(actual, [255, 0, 0, 255], "{:?}", case.format);
    }

    // ASTC LDR void-extent constants store UNORM16 channel values. Using the same
    // mid-gray payload proves that the linear and sRGB public formats do not collapse
    // to one sampled interpretation.
    let gray = astc_void_extent_mid_gray();
    let linear = sampled_render_oracle(context, ASTC_CASES[0].format, 4, 4, &gray).await;
    let srgb = sampled_render_oracle(context, ASTC_CASES[1].format, 4, 4, &gray).await;
    assert_eq!(linear, [128, 128, 128, 255]);
    assert_eq!(srgb, [55, 55, 55, 255]);
}

async fn run_astc_suite(context: &GpuContext) -> u32 {
    let full_mask = (1_u32 << ASTC_CASES.len()) - 1;
    let mut mask = 0_u32;
    for (case_index, case) in ASTC_CASES.into_iter().enumerate() {
        let facts = context
            .adapter_facts()
            .supported()
            .format(case.format)
            .expect("ASTC format must remain in normalized census");
        assert!(
            facts.sampled && facts.copy_source && facts.copy_destination && facts.filterable,
            "{:?}",
            case.format
        );
        assert!(
            !facts.storage_read
                && !facts.storage_write
                && !facts.color_attachment
                && !facts.blendable
                && !facts.depth_stencil
        );
        let (graph, readbacks) = astc_runtime_graph(case_index, case);
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        for (id, expected) in readbacks {
            let bytes = readback_wait::wait_for_readback(
                context,
                &submission,
                id,
                format!("{:?} ASTC", case.format),
            )
            .await;
            assert_eq!(bytes.as_bytes(), expected.as_slice(), "{:?}", case.format);
            assert_eq!(bytes.texture_format(), Some(case.format));
        }
        mask |= 1 << case_index;
    }
    assert_eq!(mask, full_mask);
    run_astc_decoded_oracles(context).await;
    mask
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_astc(correlated: &GpuAdapterFacts) -> u32 {
    let census = GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser ASTC census"),
    )
    .await
    .expect("actual-browser Conformance must provide WebGPU");
    assert_eq!(census.adapter_facts(), correlated);
    for case in ASTC_CASES {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .unwrap();
        assert!(facts.sampled && facts.copy_source && facts.copy_destination && facts.filterable);
    }
    let context = GpuContext::request(require_astc_roles(
        GpuContextDescriptor::new(etc_execution_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser ASTC execution"),
    ))
    .await
    .expect("advertised browser ASTC roles must admit a feature-enabled device");
    assert_eq!(context.adapter_facts(), correlated);
    run_astc_suite(&context).await
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub(crate) async fn run_astc_on_adapter(
    backend: GpuBackendFamily,
    correlated: &GpuAdapterFacts,
    direct_feature: bool,
) -> u32 {
    let census = GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_allowed_backends([backend])
            .with_label("R1 ASTC correlated census"),
    )
    .await
    .expect("qualified adapter must remain available");
    assert_eq!(census.adapter_facts(), correlated);
    for case in ASTC_CASES {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .unwrap();
        let portable =
            facts.sampled && facts.copy_source && facts.copy_destination && facts.filterable;
        assert_eq!(portable, direct_feature, "{:?}", case.format);
    }
    let descriptor = require_astc_roles(
        GpuContextDescriptor::new(etc_execution_requirements())
            .with_allowed_backends([backend])
            .with_label("R1 ASTC correlated execution"),
    );
    if !direct_feature {
        let error = GpuContext::request(descriptor)
            .await
            .expect_err("absent ASTC feature must reject required public roles");
        assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        return 0;
    }
    let context = GpuContext::request(descriptor)
        .await
        .expect("advertised ASTC feature must admit public format roles");
    assert_eq!(context.adapter_facts(), correlated);
    run_astc_suite(&context).await
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn astc_native_support_or_typed_absence_is_backend_proven() {
    let census = pollster::block_on(GpuContext::request(
        GpuContextDescriptor::new(copy_requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("R1 native ASTC census"),
    ))
    .expect("retained Vulkan software adapter must be available");
    let direct_feature = ASTC_CASES.iter().all(|case| {
        let facts = census
            .adapter_facts()
            .supported()
            .format(case.format)
            .unwrap();
        facts.sampled && facts.copy_source && facts.copy_destination && facts.filterable
    });
    let mask = pollster::block_on(run_astc_on_adapter(
        GpuBackendFamily::Vulkan,
        census.adapter_facts(),
        direct_feature,
    ));
    assert_eq!(mask, if direct_feature { (1_u32 << 28) - 1 } else { 0 });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn bc_native_family_and_terminal_mips_are_backend_proven() {
    let context = pollster::block_on(GpuContext::request(require_bc_roles(
        GpuContextDescriptor::new(copy_requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("R1 native BC proof"),
    )))
    .expect("retained Vulkan/Lavapipe Conformance must support the portable BC family");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    let mask = pollster::block_on(run_suite(&context, &BC_CASES));
    assert_eq!(mask, (1_u32 << BC_CASES.len()) - 1);
}
