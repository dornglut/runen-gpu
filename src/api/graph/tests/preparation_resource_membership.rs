use super::support::*;

#[test]
fn prepared_graph_resource_membership_remains_fragment_local() {
    let mut allocator = allocator();
    let shared = buffer(
        &mut allocator,
        "shared",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let local = buffer(
        &mut allocator,
        "local",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );

    let mut declaring_fragment = builder("declaring fragment");
    declaring_fragment
        .declare_resource(GpuResourceRef::Buffer(shared.clone()))
        .unwrap();
    let declaring_fragment = declaring_fragment.finish().unwrap();

    let mut accessing_fragment = builder("accessing fragment");
    accessing_fragment
        .declare_resource(GpuResourceRef::Buffer(local.clone()))
        .unwrap();
    add_compute(
        &mut accessing_fragment,
        "read local",
        [buffer_access(
            &local,
            GpuBufferRange::whole(&local).unwrap(),
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let mut accessing_fragment = accessing_fragment.finish().unwrap();
    accessing_fragment.nodes[0].accesses = vec![buffer_access(
        &shared,
        GpuBufferRange::whole(&shared).unwrap(),
        GpuBufferAccessKind::StorageRead,
    )];

    let error = GpuPreparedWorkGraph::prepare(
        label("fragment-local resource membership"),
        [declaring_fragment, accessing_fragment],
    )
    .unwrap_err();

    assert_eq!(error.cause(), GpuWorkGraphCause::UnknownIdentity);
    assert_eq!(error.resource(), Some(shared.diagnostic_identity()));
}
