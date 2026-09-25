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
            GpuDepthStencilAccess::ReadWrite,
            GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.5).unwrap()),
            GpuAttachmentStore::Store,
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
        let graph =
            GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
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
        let graph =
            GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
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
            [GpuTextureUsage::CopySource, GpuTextureUsage::CopyDestination],
            GpuTextureInitialization::Uninitialized,
            width,
            height,
        );
        let destination = browser_depth_texture(
            &mut allocator,
            &format!("{name} destination"),
            format,
            [GpuTextureUsage::CopySource, GpuTextureUsage::CopyDestination],
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
        let copy =
            GpuCopyOperation::texture_to_texture(source_region, destination_region.clone()).unwrap();
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
        let graph =
            GpuPreparedWorkGraph::prepare(format_label(&name), [builder.finish().unwrap()])
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
}

fn main() {}
