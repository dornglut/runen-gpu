#[cfg(target_arch = "wasm32")]
#[path = "gpu_offscreen_indexed_native.rs"]
mod retained_offscreen_indexed;
#[cfg(target_arch = "wasm32")]
#[path = "gpu_prefix_scan_native.rs"]
mod retained_prefix_scan;

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::{retained_offscreen_indexed, retained_prefix_scan};
    use runen_gpu::*;
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    thread_local! {
        static BROWSER_PROOF: RefCell<Option<Pin<Box<dyn Future<Output = ()>>>>> = RefCell::new(None);
        static RGBA16_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static RGBA8_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static R8_NEW_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static RG8_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static R16_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static RG16_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static DEPTH_SAMPLED_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static DEPTH_ATTACHMENT_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static DEPTH_COPY_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static DEPTH_LINEAR_EXERCISED_MASK: RefCell<u32> = RefCell::new(0);
        static STENCIL8_EXERCISED: RefCell<u32> = RefCell::new(0);
        static DEPTH24PLUS_STENCIL8_EXERCISED: RefCell<u32> = RefCell::new(0);
        static DEPTH24PLUS_STENCIL8_SAMPLED_EXERCISED: RefCell<u32> = RefCell::new(0);
        static DEPTH32FLOAT_STENCIL8_EXERCISED: RefCell<u32> = RefCell::new(0);
        static DEPTH32FLOAT_STENCIL8_SAMPLED_EXERCISED: RefCell<u32> = RefCell::new(0);
    }

    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                Poll::Pending
            }
        }
    }

    async fn browser_yield() {
        YieldOnce(false).await;
    }

    async fn wait_for_terminal_readbacks(
        context: &GpuContext,
        submission: &GpuSubmission,
        ids: &[GpuReadbackId],
    ) -> Vec<GpuReadbackBytes> {
        const MAX_PROGRESS_TICKS: usize = 2_000;

        let readbacks = ids
            .iter()
            .map(|id| {
                submission
                    .readback(*id)
                    .expect("retained browser readback must remain observable")
                    .clone()
            })
            .collect::<Vec<_>>();

        for _ in 0..MAX_PROGRESS_TICKS {
            context.progress();

            let mut ready = Vec::with_capacity(readbacks.len());
            let mut all_ready = true;
            for readback in &readbacks {
                match readback.status() {
                    GpuReadbackStatus::Ready(bytes) => ready.push(bytes),
                    GpuReadbackStatus::Failed(failure) => {
                        panic!("RunenGPU browser readback failed: {failure:?}")
                    }
                    GpuReadbackStatus::Pending => all_ready = false,
                }
            }

            match submission.status() {
                GpuSubmissionStatus::Failed(failure) => {
                    panic!("RunenGPU browser submission failed: {failure:?}")
                }
                GpuSubmissionStatus::Completed if all_ready => return ready,
                GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
            }

            browser_yield().await;
        }

        panic!("RunenGPU browser proof exceeded its bounded progress budget")
    }

    fn assert_execution_drained(context: &GpuContext) {
        let stats = context.execution_stats();
        assert_eq!(stats.prepared_submissions(), 0);
        assert_eq!(stats.in_flight_submissions(), 0);
        assert_eq!(stats.upload_bytes_in_flight(), 0);
        assert_eq!(stats.readback_bytes_in_flight(), 0);
        assert_eq!(stats.pending_readbacks(), 0);
    }

    async fn browser_compute_context() -> GpuContext {
        let mut requirements = GpuCapabilityRequirements::new();
        for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
            requirements
                .insert(GpuCapabilityRequirement::Required(feature))
                .unwrap();
        }
        let descriptor = GpuContextDescriptor::new(requirements)
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("retained prefix-scan browser conformance");
        let context = GpuContext::request(descriptor)
            .await
            .expect("declared browser-conformance environment must provide WebGPU");
        assert_eq!(
            context.adapter_facts().backend(),
            GpuBackendFamily::BrowserWebGpu,
            "browser compute evidence must execute through BrowserWebGpu"
        );
        context
    }

    fn retained_prefix_scan_graph_label(mode: retained_prefix_scan::ScanMode) -> GpuResourceLabel {
        let mode = match mode {
            retained_prefix_scan::ScanMode::Exclusive => "exclusive",
            retained_prefix_scan::ScanMode::Inclusive => "inclusive",
        };
        GpuResourceLabel::new(format!("prefix scan {mode} prepared graph")).unwrap()
    }

    async fn run_browser_prefix_scan_mode(
        context: &GpuContext,
        sources: &retained_prefix_scan::ProgramSources,
        mode: retained_prefix_scan::ScanMode,
    ) {
        let (fragment, output_id, total_id) = retained_prefix_scan::author_scan(sources, mode);
        let graph =
            GpuPreparedWorkGraph::prepare(retained_prefix_scan_graph_label(mode), [fragment])
                .unwrap();

        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks =
            wait_for_terminal_readbacks(context, &submission, &[output_id, total_id]).await;
        assert_eq!(readbacks.len(), 2);
        let output = retained_prefix_scan::decode_u32(&readbacks[0]);
        let total = retained_prefix_scan::decode_u32(&readbacks[1]);
        retained_prefix_scan::assert_exact_output(mode, &output, &total);
    }

    async fn run_browser_prefix_scan() {
        let context = browser_compute_context().await;
        let sources = retained_prefix_scan::admitted_sources();
        run_browser_prefix_scan_mode(
            &context,
            &sources,
            retained_prefix_scan::ScanMode::Exclusive,
        )
        .await;
        run_browser_prefix_scan_mode(
            &context,
            &sources,
            retained_prefix_scan::ScanMode::Inclusive,
        )
        .await;
        assert_execution_drained(&context);
    }

    async fn browser_offscreen_context() -> GpuContext {
        let descriptor = GpuContextDescriptor::new(
            GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements(),
        )
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
        .with_label("retained indexed-offscreen browser conformance");
        let context = GpuContext::request(descriptor)
            .await
            .expect("declared browser-conformance environment must provide offscreen WebGPU");
        assert_eq!(
            context.adapter_facts().backend(),
            GpuBackendFamily::BrowserWebGpu,
            "browser offscreen evidence must execute through BrowserWebGpu"
        );
        context
    }

    async fn run_browser_offscreen_indexed() {
        let context = browser_offscreen_context().await;
        let (graph, readback_id) = retained_offscreen_indexed::render_graph();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks = wait_for_terminal_readbacks(&context, &submission, &[readback_id]).await;
        assert_eq!(readbacks.len(), 1);
        retained_offscreen_indexed::assert_known_pattern(&readbacks[0]);
        assert_execution_drained(&context);
    }

    fn format_label(name: &str) -> GpuResourceLabel {
        GpuResourceLabel::new(name).unwrap()
    }

    fn format_provenance(name: &str) -> GpuResourceProvenance {
        GpuResourceProvenance::new(format_label(name), None, None)
    }

    fn format_texture_common(name: &str) -> GpuResourceCommon {
        GpuResourceCommon::owned(
            format_label(name),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            format_provenance(name),
        )
        .unwrap()
    }

    fn add_format_operation(
        builder: &mut GpuWorkFragmentBuilder,
        name: &str,
        operation: GpuWorkOperation,
    ) {
        builder
            .add_node(
                format_label(name),
                operation,
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                format_provenance(name),
            )
            .unwrap();
    }

    async fn run_browser_format_copy(
        formats: &[GpuTextureFormat],
        widths: &[u32],
        bytes_per_texel: u32,
        family: &str,
    ) -> u32 {
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Copy,
            ))
            .unwrap();
        let census = GpuContext::request(
            GpuContextDescriptor::new(requirements.clone())
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label(format!("{family} browser copy format census")),
        )
        .await
        .expect("declared browser-conformance environment must provide WebGPU copy");
        assert_eq!(
            census.adapter_facts().backend(),
            GpuBackendFamily::BrowserWebGpu
        );
        let mut exercised_mask = 0_u32;
        for (index, format) in formats.iter().copied().enumerate() {
            let facts = census
                .adapter_facts()
                .supported()
                .format(format)
                .expect("browser proof format must be enumerated");
            if !facts.copy_source || !facts.copy_destination {
                continue;
            }
            let context = GpuContext::request(
                GpuContextDescriptor::new(requirements.clone())
                    .require_format_role(format, GpuFormatRole::CopySource)
                    .require_format_role(format, GpuFormatRole::CopyDestination)
                    .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                    .with_label(format!("{family} browser format copy proof")),
            )
            .await
            .expect("observed browser copy roles must admit the selected format");
            let admitted = context.adapter_facts().supported().format(format).unwrap();
            assert!(admitted.copy_source && admitted.copy_destination);
            for width in widths.iter().copied() {
                let height = 2;
                let name = format!("{family} browser {format:?} {width}x{height}");
                let expected = (0..width * height * bytes_per_texel)
                    .map(|byte| (byte % 251) as u8)
                    .collect::<Vec<_>>();
                let mut allocator = GpuWorkResourceIdAllocator::new();
                let mut texture = |suffix: &str| {
                    let texture_name = format!("{name} {suffix}");
                    let texture_label = format_label(&texture_name);
                    let extent = GpuTextureExtent::new(
                        &texture_label,
                        GpuTextureDimension::D2,
                        width,
                        height,
                        1,
                    )
                    .unwrap();
                    allocator
                        .allocate_texture_handle(
                            GpuTextureDescriptor::new(
                                format_texture_common(&texture_name),
                                GpuTextureDimension::D2,
                                extent,
                                1,
                                1,
                                format,
                                GpuTextureUsages::new(
                                    &texture_label,
                                    [
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
                };
                let source = texture("source");
                let destination = texture("destination");
                let realized_source = context.realize_texture(&source).unwrap();
                let realized_destination = context.realize_texture(&destination).unwrap();
                let source_region = GpuTextureCopyRegion::new(
                    &source,
                    0,
                    GpuTextureOrigin::new(0, 0, 0),
                    GpuTextureAspect::Color,
                    GpuCopyExtent::new(width, height, 1).unwrap(),
                )
                .unwrap();
                let destination_region = GpuTextureCopyRegion::new(
                    &destination,
                    0,
                    GpuTextureOrigin::new(0, 0, 0),
                    GpuTextureAspect::Color,
                    GpuCopyExtent::new(width, height, 1).unwrap(),
                )
                .unwrap();
                let upload = GpuUploadOperation::new(
                    source_region.clone().into(),
                    PreparedGpuData::<TransferData>::from_pod_transfer(
                        &name,
                        expected.as_slice(),
                        format_provenance(&name),
                    )
                    .unwrap(),
                )
                .unwrap();
                let copy =
                    GpuCopyOperation::texture_to_texture(source_region, destination_region.clone())
                        .unwrap();
                let readback_id = GpuReadbackId::allocate().unwrap();
                let readback =
                    GpuReadbackOperation::new(destination_region.into(), readback_id).unwrap();
                let mut builder =
                    GpuWorkFragmentBuilder::new(format_label(&name), format_provenance(&name));
                builder.declare_resource(source.into()).unwrap();
                builder.declare_resource(destination.into()).unwrap();
                add_format_operation(
                    &mut builder,
                    &format!("upload {name}"),
                    GpuWorkOperation::Upload(upload),
                );
                add_format_operation(
                    &mut builder,
                    &format!("copy {name}"),
                    GpuWorkOperation::Copy(copy),
                );
                add_format_operation(
                    &mut builder,
                    &format!("readback {name}"),
                    GpuWorkOperation::Readback(readback),
                );
                let graph =
                    GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
                        .unwrap();
                let prepared = context.prepare_submission(graph).await.unwrap();
                let submission = context.submit_prepared(prepared).unwrap();
                let readbacks =
                    wait_for_terminal_readbacks(&context, &submission, &[readback_id]).await;
                assert_eq!(readbacks.len(), 1);
                assert_eq!(
                    readbacks[0].as_bytes(),
                    expected.as_slice(),
                    "{format:?} {width}px browser content"
                );
                assert_eq!(readbacks[0].layout().byte_len(), expected.len() as u64);
                assert_eq!(readbacks[0].texture_format(), Some(format));
                assert_execution_drained(&context);
                drop(realized_destination);
                drop(realized_source);
            }
            exercised_mask |= 1 << index;
        }
        exercised_mask
    }

    async fn run_browser_rgba16_copy() {
        let mask = run_browser_format_copy(
            &[
                GpuTextureFormat::Rgba16Uint,
                GpuTextureFormat::Rgba16Sint,
                GpuTextureFormat::Rgba16Float,
            ],
            &[31, 32],
            8,
            "RGBA16",
        )
        .await;
        RGBA16_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = mask);
    }

    async fn run_browser_rgba8_copy() {
        let mask = run_browser_format_copy(
            &[
                GpuTextureFormat::Rgba8Snorm,
                GpuTextureFormat::Rgba8Uint,
                GpuTextureFormat::Rgba8Sint,
            ],
            &[63, 64],
            4,
            "RGBA8",
        )
        .await;
        RGBA8_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = mask);
    }

    async fn run_browser_r8_new_copy() {
        let mask = run_browser_format_copy(
            &[
                GpuTextureFormat::R8Snorm,
                GpuTextureFormat::R8Uint,
                GpuTextureFormat::R8Sint,
            ],
            &[255, 256],
            1,
            "R8-new",
        )
        .await;
        R8_NEW_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = mask);
    }

    async fn run_browser_rg8_copy() {
        let mask = run_browser_format_copy(
            &[
                GpuTextureFormat::Rg8Unorm,
                GpuTextureFormat::Rg8Snorm,
                GpuTextureFormat::Rg8Uint,
                GpuTextureFormat::Rg8Sint,
            ],
            &[127, 128],
            2,
            "RG8",
        )
        .await;
        RG8_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = mask);
    }

    async fn run_browser_r16_copy() {
        let mask = run_browser_format_copy(
            &[
                GpuTextureFormat::R16Uint,
                GpuTextureFormat::R16Sint,
                GpuTextureFormat::R16Float,
            ],
            &[127, 128],
            2,
            "R16",
        )
        .await;
        R16_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = mask);
    }

    async fn run_browser_rg16_copy() {
        let mask = run_browser_format_copy(
            &[
                GpuTextureFormat::Rg16Uint,
                GpuTextureFormat::Rg16Sint,
                GpuTextureFormat::Rg16Float,
            ],
            &[63, 64],
            4,
            "RG16",
        )
        .await;
        RG16_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = mask);
    }

    fn depth_requirements() -> GpuCapabilityRequirements {
        let mut requirements = GpuCapabilityRequirements::new();
        for feature in [
            GpuCapabilityFeature::RenderPipeline,
            GpuCapabilityFeature::DepthAttachment,
            GpuCapabilityFeature::Copy,
        ] {
            requirements
                .insert(GpuCapabilityRequirement::Required(feature))
                .unwrap();
        }
        requirements
    }

    fn browser_depth_texture(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        format: GpuTextureFormat,
        usages: impl IntoIterator<Item = GpuTextureUsage>,
        initialization: GpuTextureInitialization,
        width: u32,
        height: u32,
    ) -> GpuTextureHandle {
        let resource_label = format_label(name);
        allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    format_texture_common(name),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(
                        &resource_label,
                        GpuTextureDimension::D2,
                        width,
                        height,
                        1,
                    )
                    .unwrap(),
                    1,
                    1,
                    format,
                    GpuTextureUsages::new(&resource_label, usages).unwrap(),
                    initialization,
                )
                .unwrap(),
            )
            .unwrap()
    }

    async fn run_browser_supported_depth_usages(
        format: GpuTextureFormat,
        sampled: bool,
        depth_stencil: bool,
        copy_source: bool,
        copy_destination: bool,
    ) {
        let mut descriptor = GpuContextDescriptor::new(depth_requirements())
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label(format!("{format:?} browser supported-usage realization"));
        let mut usages = Vec::new();

        if sampled {
            descriptor = descriptor.require_format_role(format, GpuFormatRole::Sampled);
            usages.push(GpuTextureUsage::Sampled);
        }
        if depth_stencil {
            descriptor = descriptor.require_format_role(format, GpuFormatRole::DepthStencil);
            usages.push(GpuTextureUsage::DepthStencilAttachment);
        }
        if copy_source {
            descriptor = descriptor.require_format_role(format, GpuFormatRole::CopySource);
            usages.push(GpuTextureUsage::CopySource);
        }
        if copy_destination {
            descriptor = descriptor.require_format_role(format, GpuFormatRole::CopyDestination);
            usages.push(GpuTextureUsage::CopyDestination);
        }

        if usages.is_empty() {
            return;
        }

        let context = GpuContext::request(descriptor)
            .await
            .expect("advertised browser depth-format roles must admit the selected format");
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let name = format!("{format:?} browser supported-usage realization");
        let texture = browser_depth_texture(
            &mut allocator,
            &name,
            format,
            usages,
            GpuTextureInitialization::Uninitialized,
            32,
            16,
        );
        let _realized = context.realize_texture(&texture).unwrap();
    }

    fn browser_depth_view(
        allocator: &mut GpuWorkResourceIdAllocator,
        texture: &GpuTextureHandle,
        name: &str,
    ) -> GpuTextureViewHandle {
        let range = GpuTextureSubresourceRange::new(
            texture.descriptor().common().label(),
            0,
            1,
            0,
            1,
            GpuTextureAspect::DepthOnly,
        )
        .unwrap();
        allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    format_texture_common(name),
                    texture,
                    None,
                    GpuTextureViewDimension::D2,
                    range,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn browser_depth_clear(view: GpuTextureViewHandle) -> GpuRenderOperation {
        let attachment = GpuRenderDepthStencilAttachment::new(
            view,
            Some(
                GpuDepthAttachmentState::new(
                    GpuDepthStencilAccess::ReadWrite,
                    GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.5).unwrap()),
                    GpuAttachmentStore::Store,
                )
                .unwrap(),
            ),
            None,
        )
        .unwrap();
        GpuRenderOperation::new([], Some(attachment), [], None).unwrap()
    }

    async fn run_browser_depth_clear(format: GpuTextureFormat) {
        let context = GpuContext::request(
            GpuContextDescriptor::new(depth_requirements())
                .require_format_role(format, GpuFormatRole::DepthStencil)
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label(format!("{format:?} browser depth clear proof")),
        )
        .await
        .expect("observed browser depth role must admit the selected format");
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let name = format!("{format:?} browser depth clear");
        let texture = browser_depth_texture(
            &mut allocator,
            &name,
            format,
            [GpuTextureUsage::DepthStencilAttachment],
            GpuTextureInitialization::Uninitialized,
            32,
            16,
        );
        let view = browser_depth_view(&mut allocator, &texture, &format!("{name} view"));
        let realized = context.realize_texture(&texture).unwrap();
        let _realized_view = context.realize_texture_view(&view, &realized).unwrap();

        let mut builder =
            GpuWorkFragmentBuilder::new(format_label(&name), format_provenance(&name));
        builder.declare_resource(texture.into()).unwrap();
        builder.declare_resource(view.clone().into()).unwrap();
        add_format_operation(
            &mut builder,
            &format!("{name} render"),
            GpuWorkOperation::Render(browser_depth_clear(view)),
        );
        let graph = GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
            .unwrap();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks = wait_for_terminal_readbacks(&context, &submission, &[]).await;
        assert!(readbacks.is_empty());
        assert_execution_drained(&context);
    }

    async fn run_browser_depth_texture_copy(format: GpuTextureFormat) {
        let context = GpuContext::request(
            GpuContextDescriptor::new(depth_requirements())
                .require_format_role(format, GpuFormatRole::CopySource)
                .require_format_role(format, GpuFormatRole::CopyDestination)
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label(format!("{format:?} browser depth texture copy proof")),
        )
        .await
        .expect("observed browser copy roles must admit the selected depth format");
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let name = format!("{format:?} browser depth texture copy");
        let source = browser_depth_texture(
            &mut allocator,
            &format!("{name} source"),
            format,
            [GpuTextureUsage::CopySource],
            GpuTextureInitialization::Zeroed,
            32,
            16,
        );
        let destination = browser_depth_texture(
            &mut allocator,
            &format!("{name} destination"),
            format,
            [GpuTextureUsage::CopyDestination],
            GpuTextureInitialization::Uninitialized,
            32,
            16,
        );
        context.realize_texture(&source).unwrap();
        context.realize_texture(&destination).unwrap();

        let extent = GpuCopyExtent::new(32, 16, 1).unwrap();
        let source_region = GpuTextureCopyRegion::new(
            &source,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            extent,
        )
        .unwrap();
        let destination_region = GpuTextureCopyRegion::new(
            &destination,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            extent,
        )
        .unwrap();
        let copy = GpuCopyOperation::texture_to_texture(source_region, destination_region).unwrap();

        let mut builder =
            GpuWorkFragmentBuilder::new(format_label(&name), format_provenance(&name));
        builder.declare_resource(source.into()).unwrap();
        builder.declare_resource(destination.into()).unwrap();
        add_format_operation(
            &mut builder,
            &format!("{name} copy"),
            GpuWorkOperation::Copy(copy),
        );
        let graph = GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
            .unwrap();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks = wait_for_terminal_readbacks(&context, &submission, &[]).await;
        assert!(readbacks.is_empty());
        assert_execution_drained(&context);
    }

    async fn run_browser_depth16_linear_roundtrip(width: u32) {
        let format = GpuTextureFormat::Depth16Unorm;
        let context = GpuContext::request(
            GpuContextDescriptor::new(depth_requirements())
                .require_format_role(format, GpuFormatRole::CopySource)
                .require_format_role(format, GpuFormatRole::CopyDestination)
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label("Depth16Unorm browser linear copy proof"),
        )
        .await
        .expect("observed browser Depth16 copy roles must admit the selected format");
        let height = 2;
        let name = format!("Depth16Unorm browser {width}x{height}");
        let expected = (0..width * height * 2)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let source = browser_depth_texture(
            &mut allocator,
            &format!("{name} source"),
            format,
            [
                GpuTextureUsage::CopySource,
                GpuTextureUsage::CopyDestination,
            ],
            GpuTextureInitialization::Uninitialized,
            width,
            height,
        );
        let destination = browser_depth_texture(
            &mut allocator,
            &format!("{name} destination"),
            format,
            [
                GpuTextureUsage::CopySource,
                GpuTextureUsage::CopyDestination,
            ],
            GpuTextureInitialization::Uninitialized,
            width,
            height,
        );
        let extent = GpuCopyExtent::new(width, height, 1).unwrap();
        let source_region = GpuTextureCopyRegion::new(
            &source,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            extent,
        )
        .unwrap();
        let destination_region = GpuTextureCopyRegion::new(
            &destination,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            extent,
        )
        .unwrap();
        let upload = GpuUploadOperation::new(
            source_region.clone().into(),
            PreparedGpuData::<TransferData>::from_pod_transfer(
                &name,
                expected.as_slice(),
                format_provenance(&name),
            )
            .unwrap(),
        )
        .unwrap();
        let copy = GpuCopyOperation::texture_to_texture(source_region, destination_region.clone())
            .unwrap();
        let readback_id = GpuReadbackId::allocate().unwrap();
        let readback = GpuReadbackOperation::new(destination_region.into(), readback_id).unwrap();
        let mut builder =
            GpuWorkFragmentBuilder::new(format_label(&name), format_provenance(&name));
        builder.declare_resource(source.into()).unwrap();
        builder.declare_resource(destination.into()).unwrap();
        add_format_operation(
            &mut builder,
            &format!("{name} upload"),
            GpuWorkOperation::Upload(upload),
        );
        add_format_operation(
            &mut builder,
            &format!("{name} copy"),
            GpuWorkOperation::Copy(copy),
        );
        add_format_operation(
            &mut builder,
            &format!("{name} readback"),
            GpuWorkOperation::Readback(readback),
        );
        let graph = GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
            .unwrap();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks = wait_for_terminal_readbacks(&context, &submission, &[readback_id]).await;
        assert_eq!(readbacks.len(), 1);
        assert_eq!(readbacks[0].as_bytes(), expected.as_slice());
        assert_eq!(readbacks[0].layout().byte_len(), expected.len() as u64);
        assert_eq!(readbacks[0].texture_format(), Some(format));
        assert_execution_drained(&context);
    }

    const STENCIL8_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[index], 0.0, 1.0);
}
"#;

    fn browser_stencil8_texture(
        allocator: &mut GpuWorkResourceIdAllocator,
        width: u32,
    ) -> (GpuTextureHandle, GpuTextureViewHandle) {
        let resource_label = format_label("browser Stencil8 target");
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    GpuResourceCommon::owned(
                        resource_label.clone(),
                        GpuResourceLifetime::Transient,
                        GpuMemoryIntent::Device,
                        GpuReconstruction::SourceBacked,
                        format_provenance("browser Stencil8 target"),
                    )
                    .unwrap(),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, width, 2, 1)
                        .unwrap(),
                    1,
                    1,
                    GpuTextureFormat::Stencil8,
                    GpuTextureUsages::new(
                        &resource_label,
                        [
                            GpuTextureUsage::DepthStencilAttachment,
                            GpuTextureUsage::CopySource,
                            GpuTextureUsage::CopyDestination,
                        ],
                    )
                    .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let range = GpuTextureSubresourceRange::new(
            texture.descriptor().common().label(),
            0,
            1,
            0,
            1,
            GpuTextureAspect::StencilOnly,
        )
        .unwrap();
        let view = allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    GpuResourceCommon::owned(
                        format_label("browser Stencil8 target view"),
                        GpuResourceLifetime::Transient,
                        GpuMemoryIntent::Device,
                        GpuReconstruction::SourceBacked,
                        format_provenance("browser Stencil8 target view"),
                    )
                    .unwrap(),
                    &texture,
                    None,
                    GpuTextureViewDimension::D2,
                    range,
                )
                .unwrap(),
            )
            .unwrap();
        (texture, view)
    }

    fn browser_stencil_face(
        compare: GpuCompareFunction,
        pass_op: GpuStencilOperation,
    ) -> GpuStencilFaceStateDescriptor {
        GpuStencilFaceStateDescriptor::new(
            compare,
            GpuStencilOperation::Keep,
            GpuStencilOperation::Keep,
            pass_op,
        )
    }

    fn browser_stencil_pipeline(write: bool) -> GpuRenderPipelineDescriptor {
        let vertex = GpuEntryPointName::new("vs_main").unwrap();
        let identity = GpuProgramSourceIdentity::new(
            GpuProgramSourceOwnerId::allocate().unwrap(),
            GpuProgramSourceKey::new(if write {
                "proof.browser.stencil8.write"
            } else {
                "proof.browser.stencil8.read-only"
            })
            .unwrap(),
            GpuProgramSourceRevision::try_from_raw(1).unwrap(),
        );
        let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
        let source = sources
            .admit_wgsl(
                identity,
                STENCIL8_WGSL,
                GpuProgramSourceProvenance::new("browser Stencil8 proof", None).unwrap(),
            )
            .unwrap();
        let program = GpuProgramDescriptor::new(
            source,
            [vertex.clone()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )
        .unwrap();
        let face = if write {
            browser_stencil_face(GpuCompareFunction::Always, GpuStencilOperation::Replace)
        } else {
            browser_stencil_face(GpuCompareFunction::Equal, GpuStencilOperation::Keep)
        };
        let stencil =
            GpuStencilStateDescriptor::new(face, face, u32::MAX, if write { 0xff } else { 0 });
        let depth_stencil =
            GpuDepthStencilStateDescriptor::new(GpuTextureFormat::Stencil8, None, Some(stencil))
                .unwrap();
        let state = GpuRenderPipelineStateDescriptor::new(
            GpuVertexInputStateDescriptor::new([]).unwrap(),
            None,
            GpuPrimitiveStateDescriptor::default(),
            Some(depth_stencil),
            GpuMultisampleStateDescriptor::default(),
        )
        .unwrap();
        GpuRenderPipelineDescriptor::new(
            program,
            GpuRenderEntryPoints::new(vertex, None),
            state,
            GpuPipelineConfiguration::default(),
        )
        .unwrap()
    }

    fn browser_stencil_draw(
        pipeline: GpuRenderPipelineDescriptor,
        width: u32,
        reference: u32,
    ) -> GpuRenderDraw {
        let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
        GpuRenderDraw::new(
            pipeline,
            bindings,
            [],
            None,
            GpuDrawIntent::direct(
                GpuDrawRange::new(0, 3).unwrap(),
                GpuDrawRange::new(0, 1).unwrap(),
            ),
            GpuViewport::new(0.0, 0.0, width as f32, 2.0, 0.0, 1.0).unwrap(),
            GpuScissorRect::new(0, 0, width, 2).unwrap(),
            GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
            reference,
        )
        .unwrap()
    }

    async fn run_browser_stencil8_width(context: &GpuContext, width: u32) {
        const REFERENCE: u32 = 91;

        let mut allocator = GpuWorkResourceIdAllocator::new();
        let (texture, view) = browser_stencil8_texture(&mut allocator, width);

        let write_state = GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadWrite,
            GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(0).unwrap()),
            GpuAttachmentStore::Store,
        )
        .unwrap();
        let write_attachment =
            GpuRenderDepthStencilAttachment::new(view.clone(), None, Some(write_state)).unwrap();
        let write_render = GpuRenderOperation::new(
            [],
            Some(write_attachment),
            [browser_stencil_draw(
                browser_stencil_pipeline(true),
                width,
                REFERENCE,
            )],
            None,
        )
        .unwrap();

        let read_state = GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadOnly,
            GpuStencilAttachmentLoad::Load,
            GpuAttachmentStore::Store,
        )
        .unwrap();
        let read_attachment =
            GpuRenderDepthStencilAttachment::new(view.clone(), None, Some(read_state)).unwrap();
        let read_render = GpuRenderOperation::new(
            [],
            Some(read_attachment),
            [browser_stencil_draw(
                browser_stencil_pipeline(false),
                width,
                REFERENCE,
            )],
            None,
        )
        .unwrap();

        let region = GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::StencilOnly,
            GpuCopyExtent::new(width, 2, 1).unwrap(),
        )
        .unwrap();
        let readback_id = GpuReadbackId::allocate().unwrap();
        let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();

        let name = format!("browser Stencil8 {width}x2");
        let mut builder =
            GpuWorkFragmentBuilder::new(format_label(&name), format_provenance(&name));
        builder.declare_resource(texture.into()).unwrap();
        builder.declare_resource(view.into()).unwrap();
        for (node, operation) in [
            (
                "browser Stencil8 write",
                GpuWorkOperation::Render(write_render),
            ),
            (
                "browser Stencil8 read-only test",
                GpuWorkOperation::Render(read_render),
            ),
            (
                "browser Stencil8 readback",
                GpuWorkOperation::Readback(readback),
            ),
        ] {
            add_format_operation(&mut builder, node, operation);
        }
        let graph = GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
            .unwrap();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks = wait_for_terminal_readbacks(context, &submission, &[readback_id]).await;
        assert_eq!(readbacks.len(), 1);
        assert_eq!(
            readbacks[0].texture_format(),
            Some(GpuTextureFormat::Stencil8)
        );
        assert_eq!(
            readbacks[0].as_bytes().len(),
            usize::try_from(width * 2).unwrap()
        );
        assert!(
            readbacks[0]
                .as_bytes()
                .iter()
                .all(|byte| *byte == REFERENCE as u8),
            "browser Stencil8 Replace must write the dynamic reference across the full target"
        );
        assert_execution_drained(context);
    }

    async fn run_browser_stencil8() {
        let census = GpuContext::request(
            GpuContextDescriptor::new(depth_requirements())
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label("browser Stencil8 census"),
        )
        .await
        .expect("declared browser-conformance environment must provide WebGPU");
        let facts = census
            .adapter_facts()
            .supported()
            .format(GpuTextureFormat::Stencil8)
            .expect("Stencil8 must be present in the normalized format census");

        if !(facts.depth_stencil && facts.copy_source && facts.copy_destination) {
            STENCIL8_EXERCISED.with(|slot| *slot.borrow_mut() = 0);
            return;
        }

        let context = GpuContext::request(
            GpuContextDescriptor::new(depth_requirements())
                .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::DepthStencil)
                .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::CopySource)
                .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::CopyDestination)
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label("browser Stencil8 execution proof"),
        )
        .await
        .expect("advertised browser Stencil8 attachment/copy roles must admit a context");

        for width in [255, 256] {
            run_browser_stencil8_width(&context, width).await;
        }
        STENCIL8_EXERCISED.with(|slot| *slot.borrow_mut() = 1);
    }

    const COMBINED_DEPTH_STENCIL_SAMPLED_WGSL: &str = r#"
@group(0) @binding(0)
var depth_texture: texture_depth_2d;

@group(0) @binding(1)
var stencil_texture: texture_2d<u32>;

var<workgroup> sampled_sink: u32;

@compute @workgroup_size(1)
fn cs_main() {
    let depth_value = textureLoad(depth_texture, vec2<i32>(0, 0), 0);
    let stencil_value = textureLoad(stencil_texture, vec2<i32>(0, 0), 0).x;
    sampled_sink = select(0u, stencil_value, depth_value >= 0.0);
}
"#;

    fn browser_combined_requirements() -> GpuCapabilityRequirements {
        let mut requirements = depth_requirements();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();
        requirements
    }

    fn browser_combined_texture(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        width: u32,
        format: GpuTextureFormat,
        usages: impl IntoIterator<Item = GpuTextureUsage>,
    ) -> (GpuTextureHandle, GpuTextureViewHandle) {
        let resource_label = format_label(name);
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    format_texture_common(name),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, width, 2, 1)
                        .unwrap(),
                    1,
                    1,
                    format,
                    GpuTextureUsages::new(&resource_label, usages).unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let range = GpuTextureSubresourceRange::new(
            texture.descriptor().common().label(),
            0,
            1,
            0,
            1,
            GpuTextureAspect::All,
        )
        .unwrap();
        let view = allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    format_texture_common(&format!("{name} view")),
                    &texture,
                    None,
                    GpuTextureViewDimension::D2,
                    range,
                )
                .unwrap(),
            )
            .unwrap();
        (texture, view)
    }

    fn browser_combined_aspect_view(
        allocator: &mut GpuWorkResourceIdAllocator,
        texture: &GpuTextureHandle,
        name: &str,
        aspect: GpuTextureAspect,
    ) -> GpuTextureViewHandle {
        let range = GpuTextureSubresourceRange::new(
            texture.descriptor().common().label(),
            0,
            1,
            0,
            1,
            aspect,
        )
        .unwrap();
        allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    format_texture_common(name),
                    texture,
                    None,
                    GpuTextureViewDimension::D2,
                    range,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn browser_combined_sampled_pipeline(
        key: &str,
        proof_label: &str,
    ) -> GpuComputePipelineDescriptor {
        let entry = GpuEntryPointName::new("cs_main").unwrap();
        let identity = GpuProgramSourceIdentity::new(
            GpuProgramSourceOwnerId::allocate().unwrap(),
            GpuProgramSourceKey::new(key).unwrap(),
            GpuProgramSourceRevision::try_from_raw(1).unwrap(),
        );
        let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
        let source = sources
            .admit_wgsl(
                identity,
                COMBINED_DEPTH_STENCIL_SAMPLED_WGSL,
                GpuProgramSourceProvenance::new(
                    format!("browser {proof_label} sampled proof"),
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let program = GpuProgramDescriptor::new(
            source,
            [entry.clone()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )
        .unwrap();
        GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default())
            .unwrap()
    }

    fn browser_combined_sampled_binding(
        binding: u32,
        view: &GpuTextureViewHandle,
    ) -> GpuRuntimeBindingValue {
        GpuRuntimeBindingValue::new(
            GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
            [GpuRuntimeBindingResource::TextureView(
                GpuRuntimeTextureViewBinding::new(view.clone()),
            )],
        )
        .unwrap()
    }

    fn browser_combined_sampled_operation(
        depth_view: &GpuTextureViewHandle,
        stencil_view: &GpuTextureViewHandle,
        key: &str,
        proof_label: &str,
    ) -> GpuComputeOperation {
        let pipeline = browser_combined_sampled_pipeline(key, proof_label);
        let bindings = pipeline
            .runtime_bindings([
                browser_combined_sampled_binding(0, depth_view),
                browser_combined_sampled_binding(1, stencil_view),
            ])
            .unwrap();
        GpuComputeOperation::new(
            pipeline,
            bindings,
            GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
        )
        .unwrap()
    }

    fn browser_combined_pipeline(
        key: &str,
        proof_label: &str,
        format: GpuTextureFormat,
        depth_write: bool,
        depth_compare: GpuCompareFunction,
    ) -> GpuRenderPipelineDescriptor {
        let vertex = GpuEntryPointName::new("vs_main").unwrap();
        let identity = GpuProgramSourceIdentity::new(
            GpuProgramSourceOwnerId::allocate().unwrap(),
            GpuProgramSourceKey::new(key).unwrap(),
            GpuProgramSourceRevision::try_from_raw(1).unwrap(),
        );
        let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
        let source = sources
            .admit_wgsl(
                identity,
                STENCIL8_WGSL,
                GpuProgramSourceProvenance::new(format!("browser {proof_label} proof"), None)
                    .unwrap(),
            )
            .unwrap();
        let program = GpuProgramDescriptor::new(
            source,
            [vertex.clone()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )
        .unwrap();
        let face = browser_stencil_face(GpuCompareFunction::Always, GpuStencilOperation::Replace);
        let stencil = GpuStencilStateDescriptor::new(face, face, u32::MAX, 0xff);
        let depth_stencil = GpuDepthStencilStateDescriptor::new(
            format,
            Some(GpuDepthStateDescriptor::new(depth_write, depth_compare)),
            Some(stencil),
        )
        .unwrap();
        let state = GpuRenderPipelineStateDescriptor::new(
            GpuVertexInputStateDescriptor::new([]).unwrap(),
            None,
            GpuPrimitiveStateDescriptor::default(),
            Some(depth_stencil),
            GpuMultisampleStateDescriptor::default(),
        )
        .unwrap();
        GpuRenderPipelineDescriptor::new(
            program,
            GpuRenderEntryPoints::new(vertex, None),
            state,
            GpuPipelineConfiguration::default(),
        )
        .unwrap()
    }

    fn browser_combined_seed_attachment(
        view: GpuTextureViewHandle,
    ) -> GpuRenderDepthStencilAttachment {
        let depth = GpuDepthAttachmentState::new(
            GpuDepthStencilAccess::ReadWrite,
            GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(1.0).unwrap()),
            GpuAttachmentStore::Store,
        )
        .unwrap();
        let stencil = GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadWrite,
            GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(0).unwrap()),
            GpuAttachmentStore::Store,
        )
        .unwrap();
        GpuRenderDepthStencilAttachment::new(view, Some(depth), Some(stencil)).unwrap()
    }

    fn browser_combined_mixed_attachment(
        view: GpuTextureViewHandle,
    ) -> GpuRenderDepthStencilAttachment {
        let depth = GpuDepthAttachmentState::new(
            GpuDepthStencilAccess::ReadOnly,
            GpuDepthAttachmentLoad::Load,
            GpuAttachmentStore::Store,
        )
        .unwrap();
        let stencil = GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadWrite,
            GpuStencilAttachmentLoad::Load,
            GpuAttachmentStore::Store,
        )
        .unwrap();
        GpuRenderDepthStencilAttachment::new(view, Some(depth), Some(stencil)).unwrap()
    }

    fn browser_combined_region(
        texture: &GpuTextureHandle,
        width: u32,
        aspect: GpuTextureAspect,
    ) -> GpuTextureCopyRegion {
        GpuTextureCopyRegion::new(
            texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            aspect,
            GpuCopyExtent::new(width, 2, 1).unwrap(),
        )
        .unwrap()
    }

    async fn run_browser_combined_width(
        context: &GpuContext,
        width: u32,
        sampled: bool,
        format: GpuTextureFormat,
        proof_slug: &str,
        proof_label: &str,
    ) {
        const FIRST_REFERENCE: u32 = 91;
        const MIXED_REFERENCE: u32 = 123;
        const COPIED_DEPTH_REFERENCE: u32 = 177;

        let mut allocator = GpuWorkResourceIdAllocator::new();
        let source_name = format!("browser {proof_label} source");
        let (source, source_view) = browser_combined_texture(
            &mut allocator,
            &source_name,
            width,
            format,
            [
                GpuTextureUsage::DepthStencilAttachment,
                GpuTextureUsage::CopySource,
            ],
        );
        let mut destination_usages = vec![
            GpuTextureUsage::DepthStencilAttachment,
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ];
        if sampled {
            destination_usages.push(GpuTextureUsage::Sampled);
        }
        let destination_name = format!("browser {proof_label} destination");
        let (destination, destination_view) = browser_combined_texture(
            &mut allocator,
            &destination_name,
            width,
            format,
            destination_usages,
        );
        let sampled_views = sampled.then(|| {
            (
                browser_combined_aspect_view(
                    &mut allocator,
                    &destination,
                    &format!("browser {proof_label} sampled depth view"),
                    GpuTextureAspect::DepthOnly,
                ),
                browser_combined_aspect_view(
                    &mut allocator,
                    &destination,
                    &format!("browser {proof_label} sampled stencil view"),
                    GpuTextureAspect::StencilOnly,
                ),
            )
        });
        let sampled_key = format!("proof.browser.{proof_slug}.sampled");
        let sampled_compute = sampled_views.as_ref().map(|(depth, stencil)| {
            browser_combined_sampled_operation(depth, stencil, &sampled_key, proof_label)
        });

        let seed_key = format!("proof.browser.{proof_slug}.seed");
        let mixed_key = format!("proof.browser.{proof_slug}.mixed");
        let copied_depth_key = format!("proof.browser.{proof_slug}.copied-depth");

        let seed = GpuRenderOperation::new(
            [],
            Some(browser_combined_seed_attachment(source_view.clone())),
            [browser_stencil_draw(
                browser_combined_pipeline(
                    &seed_key,
                    proof_label,
                    format,
                    true,
                    GpuCompareFunction::Always,
                ),
                width,
                FIRST_REFERENCE,
            )],
            None,
        )
        .unwrap();
        let mixed = GpuRenderOperation::new(
            [],
            Some(browser_combined_mixed_attachment(source_view.clone())),
            [browser_stencil_draw(
                browser_combined_pipeline(
                    &mixed_key,
                    proof_label,
                    format,
                    false,
                    GpuCompareFunction::Equal,
                ),
                width,
                MIXED_REFERENCE,
            )],
            None,
        )
        .unwrap();

        let extent = GpuCopyExtent::new(width, 2, 1).unwrap();
        let source_all = GpuTextureCopyRegion::new(
            &source,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::All,
            extent,
        )
        .unwrap();
        let destination_all = GpuTextureCopyRegion::new(
            &destination,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::All,
            extent,
        )
        .unwrap();
        let copy = GpuCopyOperation::texture_to_texture(source_all, destination_all).unwrap();

        let copied_id = GpuReadbackId::allocate().unwrap();
        let copied_readback = GpuReadbackOperation::new(
            browser_combined_region(&destination, width, GpuTextureAspect::StencilOnly).into(),
            copied_id,
        )
        .unwrap();

        let copied_depth_id = format
            .copy_block_size(GpuTextureAspect::DepthOnly)
            .map(|_| GpuReadbackId::allocate().unwrap());
        let copied_depth_readback = copied_depth_id.map(|id| {
            GpuReadbackOperation::new(
                browser_combined_region(&destination, width, GpuTextureAspect::DepthOnly).into(),
                id,
            )
            .unwrap()
        });

        let copied_depth_gate = GpuRenderOperation::new(
            [],
            Some(browser_combined_mixed_attachment(destination_view.clone())),
            [browser_stencil_draw(
                browser_combined_pipeline(
                    &copied_depth_key,
                    proof_label,
                    format,
                    false,
                    GpuCompareFunction::Equal,
                ),
                width,
                COPIED_DEPTH_REFERENCE,
            )],
            None,
        )
        .unwrap();

        let final_id = GpuReadbackId::allocate().unwrap();
        let final_readback = GpuReadbackOperation::new(
            browser_combined_region(&destination, width, GpuTextureAspect::StencilOnly).into(),
            final_id,
        )
        .unwrap();

        let name = format!("browser {proof_label} {width}x2");
        let mut builder =
            GpuWorkFragmentBuilder::new(format_label(&name), format_provenance(&name));
        for resource in [
            source.clone().into(),
            source_view.into(),
            destination.clone().into(),
            destination_view.into(),
        ] {
            builder.declare_resource(resource).unwrap();
        }
        if let Some((depth_view, stencil_view)) = &sampled_views {
            builder.declare_resource(depth_view.clone().into()).unwrap();
            builder
                .declare_resource(stencil_view.clone().into())
                .unwrap();
        }
        for (node, operation) in [
            (
                "browser combined seed depth + stencil",
                GpuWorkOperation::Render(seed),
            ),
            (
                "browser combined mixed depth-read-only + stencil-write",
                GpuWorkOperation::Render(mixed),
            ),
            (
                "browser combined all-aspect texture copy",
                GpuWorkOperation::Copy(copy),
            ),
            (
                "browser combined copied stencil snapshot",
                GpuWorkOperation::Readback(copied_readback),
            ),
            (
                "browser combined copied depth gate",
                GpuWorkOperation::Render(copied_depth_gate),
            ),
        ] {
            builder
                .add_node(
                    format_label(node),
                    operation,
                    [],
                    GpuCapabilityRequirements::new(),
                    GpuExecutionPreference::Automatic,
                    format_provenance(node),
                )
                .unwrap();
        }
        if let Some(readback) = copied_depth_readback {
            builder
                .add_node(
                    format_label("browser combined copied depth snapshot"),
                    GpuWorkOperation::Readback(readback),
                    [],
                    GpuCapabilityRequirements::new(),
                    GpuExecutionPreference::Automatic,
                    format_provenance("browser combined copied depth snapshot"),
                )
                .unwrap();
        }
        if let Some(compute) = sampled_compute {
            builder
                .add_node(
                    format_label("browser combined aspect-specific sampled dispatch"),
                    GpuWorkOperation::Compute(compute),
                    [],
                    GpuCapabilityRequirements::new(),
                    GpuExecutionPreference::Automatic,
                    format_provenance("browser combined aspect-specific sampled dispatch"),
                )
                .unwrap();
        }
        builder
            .add_node(
                format_label("browser combined final stencil readback"),
                GpuWorkOperation::Readback(final_readback),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::Automatic,
                format_provenance("browser combined final stencil readback"),
            )
            .unwrap();

        let graph = GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
            .unwrap();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let mut readback_ids = vec![copied_id];
        if let Some(id) = copied_depth_id {
            readback_ids.push(id);
        }
        readback_ids.push(final_id);
        let readbacks = wait_for_terminal_readbacks(context, &submission, &readback_ids).await;
        assert_eq!(readbacks.len(), readback_ids.len());
        assert_eq!(readbacks[0].texture_format(), Some(format));
        assert_eq!(
            readbacks[0].as_bytes().len(),
            usize::try_from(width * 2).unwrap()
        );
        assert!(
            readbacks[0]
                .as_bytes()
                .iter()
                .all(|byte| *byte == MIXED_REFERENCE as u8),
            "browser all-aspect copy must preserve the mixed-pass stencil result"
        );
        let final_index = if copied_depth_id.is_some() { 2 } else { 1 };
        if copied_depth_id.is_some() {
            assert_eq!(readbacks[1].texture_format(), Some(format));
            assert_eq!(
                readbacks[1].as_bytes().len(),
                usize::try_from(width * 2 * 4).unwrap()
            );
            let expected_depth = 0.0_f32.to_le_bytes();
            assert!(
                readbacks[1]
                    .as_bytes()
                    .chunks_exact(4)
                    .all(|bytes| bytes == expected_depth),
                "browser DepthOnly linear readback must preserve the exact copied 32-bit depth result"
            );
        }
        assert!(
            readbacks[final_index]
                .as_bytes()
                .iter()
                .all(|byte| *byte == COPIED_DEPTH_REFERENCE as u8),
            "browser copied depth must pass Equal and gate final stencil Replace"
        );
        assert_execution_drained(context);
    }

    async fn run_browser_combined_format(
        format: GpuTextureFormat,
        proof_slug: &str,
        proof_label: &str,
    ) -> (u32, u32) {
        let census = GpuContext::request(
            GpuContextDescriptor::new(browser_combined_requirements())
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label(format!("browser {proof_label} census")),
        )
        .await
        .expect("declared browser-conformance environment must provide WebGPU");
        let facts = census
            .adapter_facts()
            .supported()
            .format(format)
            .expect("combined format must be present in the normalized format census");

        if !(facts.depth_stencil && facts.copy_source && facts.copy_destination) {
            return (0, 0);
        }

        let mut descriptor = GpuContextDescriptor::new(browser_combined_requirements())
            .require_format_role(format, GpuFormatRole::DepthStencil)
            .require_format_role(format, GpuFormatRole::CopySource)
            .require_format_role(format, GpuFormatRole::CopyDestination);
        if facts.sampled {
            descriptor = descriptor.require_format_role(format, GpuFormatRole::Sampled);
        }
        let context = GpuContext::request(
            descriptor
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label(format!("browser {proof_label} execution proof")),
        )
        .await
        .expect("advertised browser combined roles must admit a context");

        for width in [255, 256] {
            run_browser_combined_width(
                &context,
                width,
                facts.sampled,
                format,
                proof_slug,
                proof_label,
            )
            .await;
        }
        (1, u32::from(facts.sampled))
    }

    async fn run_browser_depth24plus_stencil8() {
        let (exercised, sampled) = run_browser_combined_format(
            GpuTextureFormat::Depth24PlusStencil8,
            "depth24plus-stencil8",
            "Depth24PlusStencil8",
        )
        .await;
        DEPTH24PLUS_STENCIL8_EXERCISED.with(|slot| *slot.borrow_mut() = exercised);
        DEPTH24PLUS_STENCIL8_SAMPLED_EXERCISED.with(|slot| *slot.borrow_mut() = sampled);
    }

    async fn run_browser_depth32float_stencil8() {
        let (exercised, sampled) = run_browser_combined_format(
            GpuTextureFormat::Depth32FloatStencil8,
            "depth32float-stencil8",
            "Depth32FloatStencil8",
        )
        .await;
        DEPTH32FLOAT_STENCIL8_EXERCISED.with(|slot| *slot.borrow_mut() = exercised);
        DEPTH32FLOAT_STENCIL8_SAMPLED_EXERCISED.with(|slot| *slot.borrow_mut() = sampled);
    }

    async fn run_browser_depth_formats() {
        let census = GpuContext::request(
            GpuContextDescriptor::new(depth_requirements())
                .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
                .with_label("baseline depth browser format census"),
        )
        .await
        .expect("declared browser-conformance environment must provide depth rendering");
        let formats = [
            GpuTextureFormat::Depth16Unorm,
            GpuTextureFormat::Depth24Plus,
        ];
        let mut sampled_mask = 0_u32;
        let mut attachment_mask = 0_u32;
        let mut copy_mask = 0_u32;
        let mut linear_mask = 0_u32;

        for (index, format) in formats.into_iter().enumerate() {
            let facts = census
                .adapter_facts()
                .supported()
                .format(format)
                .expect("baseline depth format must be enumerated");
            run_browser_supported_depth_usages(
                format,
                facts.sampled,
                facts.depth_stencil,
                facts.copy_source,
                facts.copy_destination,
            )
            .await;
            if facts.sampled {
                sampled_mask |= 1 << index;
            }

            if facts.depth_stencil {
                run_browser_depth_clear(format).await;
                attachment_mask |= 1 << index;
            }

            if facts.copy_source && facts.copy_destination {
                run_browser_depth_texture_copy(format).await;
                copy_mask |= 1 << index;
                if format == GpuTextureFormat::Depth16Unorm {
                    for width in [127, 128] {
                        run_browser_depth16_linear_roundtrip(width).await;
                    }
                    linear_mask |= 1 << index;
                }
            }
        }

        DEPTH_SAMPLED_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = sampled_mask);
        DEPTH_ATTACHMENT_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = attachment_mask);
        DEPTH_COPY_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = copy_mask);
        DEPTH_LINEAR_EXERCISED_MASK.with(|slot| *slot.borrow_mut() = linear_mask);
    }

    async fn run_browser_webgpu_conformance() {
        run_browser_prefix_scan().await;
        run_browser_offscreen_indexed().await;
        run_browser_rgba16_copy().await;
        run_browser_rgba8_copy().await;
        run_browser_r8_new_copy().await;
        run_browser_rg8_copy().await;
        run_browser_r16_copy().await;
        run_browser_rg16_copy().await;
        run_browser_depth_formats().await;
        run_browser_stencil8().await;
        run_browser_depth24plus_stencil8().await;
        run_browser_depth32float_stencil8().await;
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_start() {
        BROWSER_PROOF.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "RunenGPU browser proof already started");
            *slot = Some(Box::pin(run_browser_webgpu_conformance()));
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_poll() -> u32 {
        BROWSER_PROOF.with(|slot| {
            let mut slot = slot.borrow_mut();
            let Some(future) = slot.as_mut() else {
                return 2;
            };
            let waker = Waker::noop();
            let mut context = Context::from_waker(waker);
            let status = match future.as_mut().poll(&mut context) {
                Poll::Pending => 0,
                Poll::Ready(()) => 1,
            };
            if status == 1 {
                slot.take();
            }
            status
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_rgba16_exercised_mask() -> u32 {
        RGBA16_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_rgba8_exercised_mask() -> u32 {
        RGBA8_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_r8_new_exercised_mask() -> u32 {
        R8_NEW_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_rg8_exercised_mask() -> u32 {
        RG8_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_r16_exercised_mask() -> u32 {
        R16_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_rg16_exercised_mask() -> u32 {
        RG16_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth_sampled_exercised_mask() -> u32 {
        DEPTH_SAMPLED_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth_attachment_exercised_mask() -> u32 {
        DEPTH_ATTACHMENT_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth_copy_exercised_mask() -> u32 {
        DEPTH_COPY_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth_linear_exercised_mask() -> u32 {
        DEPTH_LINEAR_EXERCISED_MASK.with(|mask| *mask.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_stencil8_exercised() -> u32 {
        STENCIL8_EXERCISED.with(|value| *value.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth24plus_stencil8_exercised() -> u32 {
        DEPTH24PLUS_STENCIL8_EXERCISED.with(|value| *value.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth24plus_stencil8_sampled_exercised() -> u32 {
        DEPTH24PLUS_STENCIL8_SAMPLED_EXERCISED.with(|value| *value.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth32float_stencil8_exercised() -> u32 {
        DEPTH32FLOAT_STENCIL8_EXERCISED.with(|value| *value.borrow())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_depth32float_stencil8_sampled_exercised() -> u32 {
        DEPTH32FLOAT_STENCIL8_SAMPLED_EXERCISED.with(|value| *value.borrow())
    }
}

fn main() {}
