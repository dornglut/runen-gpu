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

    let (depth_graph, depth_readback_id) = retained_transient_attachment::depth_graph();
    let depth_prepared = pollster::block_on(context.prepare_submission(depth_graph)).unwrap();
    let depth_submission = context.submit_prepared(depth_prepared).unwrap();
    let depth_bytes = pollster::block_on(readback_wait::wait_for_readback(
        &context,
        &depth_submission,
        depth_readback_id,
        "native transient depth observable color",
    ));
    retained_transient_attachment::assert_depth_color(&depth_bytes);

    if retained_transient_attachment::stencil_supported(&context) {
        let stencil_context = pollster::block_on(GpuContext::request(
            retained_transient_attachment::stencil_descriptor(GpuBackendFamily::Vulkan)
                .with_fallback_policy(GpuSoftwareFallbackPolicy::Require),
        ))
        .expect("advertised native Stencil8 depth/stencil role must admit a context");
        assert_eq!(
            stencil_context.adapter_facts(),
            context.adapter_facts(),
            "conditional transient Stencil8 proof must stay on the retained Vulkan adapter"
        );
        let (stencil_graph, stencil_readback_id) = retained_transient_attachment::stencil_graph();
        let stencil_prepared =
            pollster::block_on(stencil_context.prepare_submission(stencil_graph)).unwrap();
        let stencil_submission = stencil_context.submit_prepared(stencil_prepared).unwrap();
        let stencil_bytes = pollster::block_on(readback_wait::wait_for_readback(
            &stencil_context,
            &stencil_submission,
            stencil_readback_id,
            "native transient Stencil8 terminal color",
        ));
        retained_transient_attachment::assert_stencil_terminal(&stencil_bytes);
    } else {
        println!("transient Stencil8: UNSUPPORTED (normalized depth/stencil role absent)");
    }
}
