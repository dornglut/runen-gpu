use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;
#[path = "gpu_transient_attachment/mod.rs"]
mod retained_transient_attachment;

#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn transient_attachment_native_execution_is_backend_proven() {
    let context = pollster::block_on(GpuContext::request(
        retained_transient_attachment::descriptor(GpuBackendFamily::Vulkan)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require),
    ))
    .expect("native Conformance must provide the retained Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );

    let (graph, readback_id) = retained_transient_attachment::graph();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = pollster::block_on(readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_id,
        "native transient attachment resolve",
    ));
    retained_transient_attachment::assert_resolved(&bytes);
}
