use super::support::*;

fn summary_for(
    graph: &GpuPreparedWorkGraph,
    identity: GpuWorkResourceId,
) -> &GpuPreparedResourceInitialization {
    graph
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == identity)
        .expect("declared storage resource should have an initialization summary")
}

fn prepared_buffer(allocator: &mut GpuWorkResourceIdAllocator, name: &str) -> GpuBufferHandle {
    let data = PreparedGpuData::<TransferData>::from_pod_transfer(
        format!("{name} bytes"),
        &[7_u8; 64],
        provenance(&format!("{name} bytes")),
    )
    .unwrap();
    buffer(
        allocator,
        name,
        GpuBufferInitialization::Prepared(data),
        [GpuBufferUsage::CopyDestination],
    )
}

fn retained_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    initialization: GpuBufferInitialization,
) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(name),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                64,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                initialization,
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn initialization_explanations_distinguish_descriptor_input_and_generic_shader_write() {
    let mut allocator = allocator();
    let descriptor = buffer(
        &mut allocator,
        "descriptor guaranteed",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let input = buffer(
        &mut allocator,
        "declared input",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let input_range = GpuBufferRange::new(&input, 0, 16).unwrap();
    let input_coverage =
        GpuInitialCoverage::buffer(&input, [GpuBufferCoverage::dense(input_range)]).unwrap();
    let generic_write = GpuBufferRange::new(&input, 16, 16).unwrap();

    let mut fragment = builder("entry explanation fragment");
    fragment
        .declare_resource(GpuResourceRef::Buffer(descriptor.clone()))
        .unwrap();
    fragment
        .declare_resource(GpuResourceRef::Buffer(input.clone()))
        .unwrap();
    fragment
        .add_input(
            GpuWorkResourceInput::new(
                GpuResourceRef::Buffer(input.clone()),
                input_coverage.clone(),
                provenance("declared input evidence"),
            )
            .unwrap(),
        )
        .unwrap();
    add_compute(
        &mut fragment,
        "generic shader write",
        [buffer_access(
            &input,
            generic_write,
            GpuBufferAccessKind::StorageWrite,
        )],
    );

    let prepared = GpuPreparedWorkGraph::prepare(
        label("entry explanation graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();

    let descriptor_summary = summary_for(&prepared, descriptor.diagnostic_identity());
    assert_eq!(descriptor_summary.explanations().len(), 1);
    assert_eq!(
        descriptor_summary.explanations()[0].kind(),
        GpuInitializationExplanationKind::DescriptorGuaranteed
    );
    assert_eq!(
        descriptor_summary.explanations()[0]
            .coverage()
            .buffer_values()
            .unwrap(),
        &[GpuBufferCoverage::dense(
            GpuBufferRange::whole(&descriptor).unwrap()
        )]
    );

    let input_summary = summary_for(&prepared, input.diagnostic_identity());
    assert_eq!(input_summary.initial(), Some(&input_coverage));
    assert_eq!(input_summary.final_coverage(), Some(&input_coverage));
    assert_eq!(input_summary.explanations().len(), 1);
    assert_eq!(
        input_summary.explanations()[0].kind(),
        GpuInitializationExplanationKind::DeclaredInput
    );
    assert_eq!(input_summary.explanations()[0].coverage(), &input_coverage);
    assert!(
        input_summary
            .explanations()
            .iter()
            .all(|explanation| !matches!(
                explanation.kind(),
                GpuInitializationExplanationKind::OperationGuaranteed
            )),
        "generic shader writes must not manufacture initialization guarantees"
    );
}

#[test]
fn retained_seed_presence_replaces_descriptor_origin_and_retained_coverage_is_typed() {
    let mut allocator = allocator();
    let zeroed = retained_buffer(
        &mut allocator,
        "retained descriptor suppression",
        GpuBufferInitialization::Zeroed,
    );
    let mut fragment = builder("retained descriptor suppression");
    fragment
        .declare_resource(GpuResourceRef::Buffer(zeroed.clone()))
        .unwrap();
    let seed = GpuRetainedInitializationSeed::new(GpuResourceRef::Buffer(zeroed.clone()), None);
    let prepared = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("retained descriptor suppression graph"),
        [fragment.finish().unwrap()],
        &[seed],
    )
    .unwrap();
    let summary = summary_for(&prepared, zeroed.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert!(summary.final_coverage().is_none());
    assert!(summary.explanations().is_empty());

    let retained = retained_buffer(
        &mut allocator,
        "retained coverage explanation",
        GpuBufferInitialization::Uninitialized,
    );
    let retained_range = GpuBufferRange::new(&retained, 8, 16).unwrap();
    let retained_coverage =
        GpuInitialCoverage::buffer(&retained, [GpuBufferCoverage::dense(retained_range)]).unwrap();
    let mut fragment = builder("retained coverage explanation");
    fragment
        .declare_resource(GpuResourceRef::Buffer(retained.clone()))
        .unwrap();
    let seed = GpuRetainedInitializationSeed::new(
        GpuResourceRef::Buffer(retained.clone()),
        Some(retained_coverage.clone()),
    );
    let prepared = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("retained coverage explanation graph"),
        [fragment.finish().unwrap()],
        &[seed],
    )
    .unwrap();
    let summary = summary_for(&prepared, retained.diagnostic_identity());
    assert_eq!(summary.initial(), Some(&retained_coverage));
    assert_eq!(summary.explanations().len(), 1);
    assert_eq!(
        summary.explanations()[0].kind(),
        GpuInitializationExplanationKind::RetainedCoverage
    );
    assert_eq!(summary.explanations()[0].coverage(), &retained_coverage);
}

#[test]
fn prepared_materialization_and_exact_operation_guarantees_remain_distinct() {
    let mut allocator = allocator();
    let prepared_buffer = prepared_buffer(&mut allocator, "materialized then cleared");
    let whole = GpuBufferRange::whole(&prepared_buffer).unwrap();
    let clear =
        GpuClearOperation::buffer_zero(GpuBufferRegion::new(&prepared_buffer, whole).unwrap())
            .unwrap();

    let mut fragment = builder("materialization explanation");
    fragment
        .declare_resource(GpuResourceRef::Buffer(prepared_buffer.clone()))
        .unwrap();
    fragment
        .operation("exact clear after materialization", clear)
        .unwrap();

    let prepared = GpuPreparedWorkGraph::prepare(
        label("materialization explanation graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    assert_eq!(prepared.initial_content().len(), 1);
    let summary = summary_for(&prepared, prepared_buffer.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert_eq!(summary.explanations().len(), 2);
    assert_eq!(
        summary.explanations()[0].kind(),
        GpuInitializationExplanationKind::PreparedInitialContent
    );
    assert_eq!(
        summary.explanations()[1].kind(),
        GpuInitializationExplanationKind::OperationGuaranteed
    );
    for explanation in summary.explanations() {
        assert_eq!(
            explanation.coverage().buffer_values().unwrap(),
            &[GpuBufferCoverage::dense(whole)]
        );
    }
}

#[test]
fn attachment_discard_records_exact_loss_after_operation_guarantee() {
    let mut allocator = allocator();
    let texture = texture(
        &mut allocator,
        "discard explanation texture",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment],
    );
    let range = GpuTextureSubresourceRange::whole(&texture).unwrap();
    let view = texture_view(&mut allocator, &texture, "discard explanation view", range);
    let render = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Discard,
            None,
        )
        .unwrap()],
        None,
        [],
        None,
    )
    .unwrap();

    let mut fragment = builder("discard explanation fragment");
    fragment
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    fragment
        .declare_resource(GpuResourceRef::TextureView(view))
        .unwrap();
    fragment
        .operation("clear then discard attachment", render)
        .unwrap();

    let prepared = GpuPreparedWorkGraph::prepare(
        label("discard explanation graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    let summary = summary_for(&prepared, texture.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert!(summary.final_coverage().is_none());
    assert_eq!(summary.explanations().len(), 2);
    assert_eq!(
        summary.explanations()[0].kind(),
        GpuInitializationExplanationKind::OperationGuaranteed
    );
    assert_eq!(
        summary.explanations()[1].kind(),
        GpuInitializationExplanationKind::AttachmentDiscard
    );
    for explanation in summary.explanations() {
        assert_eq!(
            explanation.coverage().texture_subresource_values().unwrap(),
            &[range]
        );
    }
}
