use super::support::*;

fn buffer_with_identity(
    identity: GpuWorkResourceId,
    name: &str,
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let resource_label = label(name);
    GpuBufferHandle::from_descriptor(
        identity,
        GpuBufferDescriptor::new(
            common(name),
            64,
            GpuBufferUsages::new(&resource_label, usages).unwrap(),
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap(),
    )
}

#[test]
fn failed_lexical_resource_transaction_rolls_back_and_preserves_next_node_identity() {
    let mut allocator = allocator();
    let retained = texture(
        &mut allocator,
        "retained texture identity",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::Sampled],
    );
    let source = buffer(
        &mut allocator,
        "tentative source",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
    );
    let conflicting_destination = buffer_with_identity(
        retained.diagnostic_identity(),
        "conflicting buffer destination",
        [GpuBufferUsage::CopyDestination],
    );
    assert_ne!(
        GpuResourceRef::Texture(retained.clone()),
        GpuResourceRef::Buffer(conflicting_destination.clone())
    );

    let mut fragment = builder("lexical rollback");
    fragment
        .declare_resource(GpuResourceRef::Texture(retained.clone()))
        .unwrap();

    let conflicting_copy = GpuCopyOperation::buffer_to_buffer(
        GpuBufferRegion::whole(&source).unwrap(),
        GpuBufferRegion::whole(&conflicting_destination).unwrap(),
    )
    .unwrap();
    let error = fragment
        .operation("conflicting copy", conflicting_copy)
        .unwrap_err();
    assert_eq!(error.cause(), GpuWorkAuthoringCause::InvalidResourceKind);

    // This succeeds only when the source inserted before the kind collision was rolled back.
    fragment
        .declare_resource(GpuResourceRef::Buffer(source.clone()))
        .unwrap();
    let continued = fragment
        .operation(
            "continued clear",
            GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&source).unwrap()).unwrap(),
        )
        .unwrap();
    assert_eq!(continued.diagnostic_local(), 1);

    let fragment = fragment.finish().unwrap();
    assert_eq!(fragment.nodes().len(), 1);
    assert_eq!(fragment.resources().len(), 2);
    assert!(
        fragment
            .resources()
            .contains(&GpuResourceRef::Texture(retained))
    );
    assert!(
        fragment
            .resources()
            .contains(&GpuResourceRef::Buffer(source))
    );
}

#[test]
fn successful_lexical_resource_transaction_reuses_same_kind_identity_without_replacement() {
    let mut allocator = allocator();
    let source = buffer(
        &mut allocator,
        "predeclared source",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopySource],
    );
    let source_alias = buffer_with_identity(
        source.diagnostic_identity(),
        "same identity source alias",
        [GpuBufferUsage::CopySource],
    );
    assert_eq!(
        GpuResourceRef::Buffer(source.clone()),
        GpuResourceRef::Buffer(source_alias.clone())
    );
    let destination = buffer(
        &mut allocator,
        "derived destination",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination],
    );

    let mut fragment = builder("lexical success");
    fragment
        .declare_resource(GpuResourceRef::Buffer(source.clone()))
        .unwrap();
    let copy = GpuCopyOperation::buffer_to_buffer(
        GpuBufferRegion::whole(&source_alias).unwrap(),
        GpuBufferRegion::whole(&destination).unwrap(),
    )
    .unwrap();
    let id = fragment.operation("copy", copy).unwrap();
    assert_eq!(id.diagnostic_local(), 1);

    let fragment = fragment.finish().unwrap();
    assert_eq!(fragment.nodes().len(), 1);
    assert_eq!(fragment.resources().len(), 2);
    let retained_source = fragment
        .resources()
        .iter()
        .find(|resource| resource.diagnostic_identity() == source.diagnostic_identity())
        .unwrap();
    assert!(matches!(
        retained_source,
        GpuResourceRef::Buffer(retained)
            if retained.descriptor().common().label().as_str() == "predeclared source"
    ));
    assert!(
        fragment
            .resources()
            .contains(&GpuResourceRef::Buffer(destination))
    );
}
