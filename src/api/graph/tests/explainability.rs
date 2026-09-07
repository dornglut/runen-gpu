use super::support::*;

#[test]
fn read_before_initialization_exposes_exact_typed_buffer_coverage() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "explainable uninitialized buffer",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let required_range = GpuBufferRange::new(&buffer, 8, 16).unwrap();

    let mut fragment = builder("explainable initialization fragment");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read exact uninitialized range",
        [buffer_access(
            &buffer,
            required_range,
            GpuBufferAccessKind::StorageRead,
        )],
    );

    let error = GpuPreparedWorkGraph::prepare(
        label("explainable initialization graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap_err();

    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(buffer.diagnostic_identity()));

    let required = error
        .required_initialization()
        .expect("read-before-initialization exposes typed required coverage");
    assert_eq!(required.resource(), &GpuResourceRef::Buffer(buffer.clone()));
    assert_eq!(
        required.buffer_values().unwrap(),
        &[GpuBufferCoverage::dense(required_range)]
    );
    assert!(required.texture_subresource_values().is_none());
    assert!(required.query_range_values().is_none());
}
