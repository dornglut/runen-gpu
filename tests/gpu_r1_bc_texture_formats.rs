use runen_gpu::*;

#[derive(Clone, Copy)]
struct BcCase {
    format: GpuTextureFormat,
    block_bytes: u32,
    srgb: bool,
}

const BC_CASES: [BcCase; 14] = [
    BcCase { format: GpuTextureFormat::Bc1RgbaUnorm, block_bytes: 8, srgb: false },
    BcCase { format: GpuTextureFormat::Bc1RgbaUnormSrgb, block_bytes: 8, srgb: true },
    BcCase { format: GpuTextureFormat::Bc2RgbaUnorm, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc2RgbaUnormSrgb, block_bytes: 16, srgb: true },
    BcCase { format: GpuTextureFormat::Bc3RgbaUnorm, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc3RgbaUnormSrgb, block_bytes: 16, srgb: true },
    BcCase { format: GpuTextureFormat::Bc4RUnorm, block_bytes: 8, srgb: false },
    BcCase { format: GpuTextureFormat::Bc4RSnorm, block_bytes: 8, srgb: false },
    BcCase { format: GpuTextureFormat::Bc5RgUnorm, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc5RgSnorm, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc6hRgbUfloat, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc6hRgbFloat, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc7RgbaUnorm, block_bytes: 16, srgb: false },
    BcCase { format: GpuTextureFormat::Bc7RgbaUnormSrgb, block_bytes: 16, srgb: true },
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
        .insert(GpuCapabilityRequirement::Required(GpuCapabilityFeature::Copy))
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

fn descriptor(
    dimension: GpuTextureDimension,
    width: u32,
    height: u32,
    depth_or_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
    format: GpuTextureFormat,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> Result<GpuTextureDescriptor, GpuResourceDescriptorError> {
    let resource_label = label("BC descriptor");
    GpuTextureDescriptor::new(
        common("BC descriptor"),
        dimension,
        GpuTextureExtent::new(
            &resource_label,
            dimension,
            width,
            height,
            depth_or_layers,
        )
        .unwrap(),
        mip_level_count,
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
                descriptor(
                    GpuTextureDimension::D2,
                    8,
                    8,
                    1,
                    1,
                    1,
                    case.format,
                    [usage],
                )
                .is_err(),
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
        [GpuTextureUsage::CopySource, GpuTextureUsage::CopyDestination],
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
        (GpuTextureOrigin::new(2, 0, 0), GpuCopyExtent::new(4, 4, 1).unwrap()),
        (GpuTextureOrigin::new(0, 2, 0), GpuCopyExtent::new(4, 4, 1).unwrap()),
        (GpuTextureOrigin::new(0, 0, 0), GpuCopyExtent::new(2, 4, 1).unwrap()),
        (GpuTextureOrigin::new(0, 0, 0), GpuCopyExtent::new(4, 2, 1).unwrap()),
    ] {
        assert!(
            GpuTextureCopyRegion::new(
                &texture,
                0,
                origin,
                GpuTextureAspect::Color,
                extent,
            )
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
        BcCase { format: GpuTextureFormat::Bc1RgbaUnorm, block_bytes: 8, srgb: false },
        BcCase { format: GpuTextureFormat::Bc2RgbaUnorm, block_bytes: 16, srgb: false },
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
        let valid = GpuBufferTextureLayout::new(
            &buffer,
            u64::from(case.block_bytes),
            case.block_bytes,
            0,
        )
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
    case: BcCase,
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
    case: BcCase,
) -> (GpuPreparedWorkGraph, [(GpuReadbackId, Vec<u8>); 2]) {
    let mut scope = GpuResourceScope::new();
    let source = runtime_texture(&mut scope, case, &format!("{:?} BC source", case.format));
    let destination =
        runtime_texture(&mut scope, case, &format!("{:?} BC destination", case.format));
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
            "BC base upload",
            base_bytes.as_slice(),
            provenance("BC base upload"),
        )
        .unwrap(),
    )
    .unwrap();
    let terminal_upload = GpuUploadOperation::new(
        source_terminal.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "BC terminal upload",
            terminal_bytes.as_slice(),
            provenance("BC terminal upload"),
        )
        .unwrap(),
    )
    .unwrap();
    let base_copy =
        GpuCopyOperation::texture_to_texture(source_base, destination_base.clone()).unwrap();
    let terminal_copy =
        GpuCopyOperation::texture_to_texture(source_terminal, destination_terminal.clone()).unwrap();
    let base_id = GpuReadbackId::allocate().unwrap();
    let terminal_id = GpuReadbackId::allocate().unwrap();
    let base_readback = GpuReadbackOperation::new(destination_base.into(), base_id).unwrap();
    let terminal_readback =
        GpuReadbackOperation::new(destination_terminal.into(), terminal_id).unwrap();
    let name = format!("{:?} BC runtime proof", case.format);
    let fragment = GpuWorkFragment::build(&name, |builder| {
        builder.operation("upload BC base mip", base_upload)?;
        builder.operation("copy BC base mip", base_copy)?;
        builder.operation("read BC base mip", base_readback)?;
        builder.operation("upload BC terminal mip", terminal_upload)?;
        builder.operation("copy BC terminal mip", terminal_copy)?;
        builder.operation("read BC terminal mip", terminal_readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{name} graph")), [fragment]).unwrap(),
        [(base_id, base_bytes), (terminal_id, terminal_bytes)],
    )
}

#[cfg(target_arch = "wasm32")]
struct YieldOnce(bool);

#[cfg(target_arch = "wasm32")]
impl std::future::Future for YieldOnce {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        _context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            std::task::Poll::Pending
        }
    }
}

async fn progress_yield() {
    #[cfg(target_arch = "wasm32")]
    YieldOnce(false).await;
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::yield_now();
}

async fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    id: GpuReadbackId,
    format: GpuTextureFormat,
) -> GpuReadbackBytes {
    const MAX_PROGRESS_TICKS: usize = 4_000;
    let readback = submission.readback(id).unwrap().clone();
    for _ in 0..MAX_PROGRESS_TICKS {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes)
                if matches!(submission.status(), GpuSubmissionStatus::Completed) =>
            {
                return bytes;
            }
            GpuReadbackStatus::Ready(_) | GpuReadbackStatus::Pending => {}
            GpuReadbackStatus::Failed(error) => {
                panic!("{format:?} BC readback failed: {error:?}")
            }
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{format:?} BC submission failed: {error:?}");
        }
        progress_yield().await;
    }
    panic!("{format:?} BC proof exceeded its bounded progress budget")
}

async fn run_suite(context: &GpuContext) -> u32 {
    let full_mask = (1_u32 << BC_CASES.len()) - 1;
    let mut mask = 0_u32;
    for (case_index, case) in BC_CASES.into_iter().enumerate() {
        let facts = context
            .adapter_facts()
            .supported()
            .format(case.format)
            .expect("BC format must remain in the normalized census");
        assert!(facts.sampled, "{:?}", case.format);
        assert!(facts.copy_source, "{:?}", case.format);
        assert!(facts.copy_destination, "{:?}", case.format);
        assert!(!facts.filterable, "{:?}", case.format);
        assert!(!facts.storage_read && !facts.storage_write, "{:?}", case.format);
        assert!(!facts.color_attachment && !facts.depth_stencil, "{:?}", case.format);
        let (graph, readbacks) = runtime_graph(case_index, case);
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        for (id, expected) in readbacks {
            let bytes = wait_for_readback(context, &submission, id, case.format).await;
            assert_eq!(bytes.as_bytes(), expected.as_slice(), "{:?}", case.format);
            assert_eq!(bytes.texture_format(), Some(case.format));
        }
        println!(
            "{:?} BC: EXERCISED (two array layers, 8x8 multi-block + terminal 2x2 logical mips through 4x4 physical blocks)",
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
            !facts.filterable
                && !facts.storage_read
                && !facts.storage_write
                && !facts.color_attachment
                && !facts.depth_stencil,
            "{:?} browser BC facts must retain the portable role clamp",
            case.format
        );
        if portable {
            supported_count += 1;
        } else {
            assert!(
                !any_portable,
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
    run_suite(&context).await
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
    let mask = pollster::block_on(run_suite(&context));
    assert_eq!(mask, (1_u32 << BC_CASES.len()) - 1);
}
