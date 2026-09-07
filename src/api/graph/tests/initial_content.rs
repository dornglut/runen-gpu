use super::support::*;

fn prepared_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
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
        usages,
    )
}

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

#[test]
fn unused_prepared_resource_does_not_manufacture_initial_content() {
    let mut allocator = allocator();
    let prepared = prepared_buffer(
        &mut allocator,
        "unused prepared",
        [GpuBufferUsage::CopyDestination],
    );
    let unrelated = buffer(
        &mut allocator,
        "unrelated",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination],
    );
    let mut fragment = builder("unused prepared");
    for resource in [&prepared, &unrelated] {
        fragment
            .declare_resource(GpuResourceRef::Buffer(resource.clone()))
            .unwrap();
    }
    let unrelated_range = GpuBufferRange::whole(&unrelated).unwrap();
    fragment
        .add_node(
            label("clear unrelated"),
            GpuWorkOperation::Clear(
                GpuClearOperation::buffer_zero(
                    GpuBufferRegion::new(&unrelated, unrelated_range).unwrap(),
                )
                .unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("clear unrelated"),
        )
        .unwrap();

    let graph =
        GpuPreparedWorkGraph::prepare(label("unused prepared graph"), [fragment.finish().unwrap()])
            .unwrap();
    assert!(graph.initial_content().is_empty());
    let summary = summary_for(&graph, prepared.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert!(summary.final_coverage().is_none());
}

#[test]
fn caller_only_prepared_read_does_not_select_materialization() {
    let mut allocator = allocator();
    let prepared = prepared_buffer(
        &mut allocator,
        "caller only prepared",
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let mut fragment = builder("caller only prepared");
    fragment
        .declare_resource(GpuResourceRef::Buffer(prepared.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "caller only read",
        [buffer_access(
            &prepared,
            GpuBufferRange::whole(&prepared).unwrap(),
            GpuBufferAccessKind::StorageRead,
        )],
    );

    let error = GpuPreparedWorkGraph::prepare(
        label("caller only prepared graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(prepared.diagnostic_identity()));
}

#[test]
fn operation_derived_prepared_read_uses_one_planned_materialization_effect() {
    let mut allocator = allocator();
    let prepared = prepared_buffer(
        &mut allocator,
        "readback prepared",
        [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
    );
    let whole = GpuBufferRange::whole(&prepared).unwrap();
    let readback = GpuReadbackOperation::new(
        GpuBufferRegion::new(&prepared, whole).unwrap().into(),
        GpuReadbackId::allocate().unwrap(),
    )
    .unwrap();
    let mut fragment = builder("readback prepared");
    fragment
        .declare_resource(GpuResourceRef::Buffer(prepared.clone()))
        .unwrap();
    fragment
        .add_node(
            label("readback"),
            GpuWorkOperation::Readback(readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("readback"),
        )
        .unwrap();
    let fragment = fragment.finish().unwrap();

    let graph = GpuPreparedWorkGraph::prepare(label("readback prepared graph"), [fragment.clone()])
        .unwrap();
    let repeated =
        GpuPreparedWorkGraph::prepare(label("readback prepared graph"), [fragment]).unwrap();
    assert_eq!(graph.initial_content(), repeated.initial_content());
    assert_eq!(graph.initial_content().len(), 1);
    assert_eq!(
        graph.initial_content()[0].resource_identity(),
        prepared.diagnostic_identity()
    );
    assert_eq!(
        graph.requirements().get(GpuCapabilityFeature::Copy),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy
        ))
    );
    let summary = summary_for(&graph, prepared.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert_eq!(
        summary.final_coverage().unwrap().buffer_values().unwrap(),
        [GpuBufferCoverage::dense(whole)]
    );
}

#[test]
fn first_overwrite_does_not_dead_store_elide_prepared_materialization() {
    let mut allocator = allocator();
    let prepared = prepared_buffer(
        &mut allocator,
        "overwritten prepared",
        [GpuBufferUsage::CopyDestination],
    );
    let whole = GpuBufferRange::whole(&prepared).unwrap();
    let mut fragment = builder("overwritten prepared");
    fragment
        .declare_resource(GpuResourceRef::Buffer(prepared.clone()))
        .unwrap();
    fragment
        .add_node(
            label("overwrite whole buffer"),
            GpuWorkOperation::Clear(
                GpuClearOperation::buffer_zero(GpuBufferRegion::new(&prepared, whole).unwrap())
                    .unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("overwrite whole buffer"),
        )
        .unwrap();

    let graph = GpuPreparedWorkGraph::prepare(
        label("overwritten prepared graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    assert_eq!(graph.initial_content().len(), 1);
    assert_eq!(
        graph.initial_content()[0].resource_identity(),
        prepared.diagnostic_identity()
    );
    let summary = summary_for(&graph, prepared.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert_eq!(
        summary.final_coverage().unwrap().buffer_values().unwrap(),
        [GpuBufferCoverage::dense(whole)]
    );
}

#[test]
fn texture_view_operation_selects_parent_materialization_and_copy_capability() {
    let mut allocator = allocator();
    let prepared = texture(
        &mut allocator,
        "prepared render target",
        prepared_texture_initialization("prepared render target"),
        1,
        1,
        [
            GpuTextureUsage::ColorAttachment,
            GpuTextureUsage::CopyDestination,
        ],
    );
    let range = GpuTextureSubresourceRange::whole(&prepared).unwrap();
    let view = texture_view(
        &mut allocator,
        &prepared,
        "prepared render target view",
        range,
    );
    let attachment = GpuRenderColorAttachment::new(
        view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let mut fragment = builder("prepared view render");
    fragment
        .declare_resource(GpuResourceRef::Texture(prepared.clone()))
        .unwrap();
    fragment
        .declare_resource(GpuResourceRef::TextureView(view))
        .unwrap();
    fragment
        .add_node(
            label("clear through view"),
            GpuWorkOperation::Render(
                GpuRenderOperation::new([attachment], None, [], None).unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("clear through view"),
        )
        .unwrap();

    let graph = GpuPreparedWorkGraph::prepare(
        label("prepared view render graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    assert_eq!(graph.initial_content().len(), 1);
    assert_eq!(
        graph.initial_content()[0].resource_identity(),
        prepared.diagnostic_identity(),
        "view-derived work must normalize initial-content identity to parent storage"
    );
    assert_eq!(
        graph.requirements().get(GpuCapabilityFeature::Copy),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy
        )),
        "planned Prepared materialization must contribute the canonical Copy requirement"
    );
    assert_eq!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::RenderPipeline),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::RenderPipeline
        ))
    );
    let summary = summary_for(&graph, prepared.diagnostic_identity());
    assert!(summary.initial().is_none());
    assert_eq!(
        summary
            .final_coverage()
            .unwrap()
            .texture_subresource_values()
            .unwrap(),
        [range]
    );
}
