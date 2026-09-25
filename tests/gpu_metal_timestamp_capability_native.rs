use runen_gpu::{
    GpuBackendFamily, GpuCapabilityFeature, GpuCapabilityProfile, GpuCapabilityRequirement,
    GpuContext, GpuContextDescriptor, GpuPreferredFallback,
};

#[test]
#[ignore = "requires a real Apple Metal adapter; run for Metal timestamp capability acceptance"]
fn apple_metal_preferred_timestamp_request_disables_instrumentation_without_losing_context() {
    let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Preferred {
            feature: GpuCapabilityFeature::TimestampQuery,
            fallback: GpuPreferredFallback::DisableInstrumentation,
        })
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::Metal])
        .with_label("Apple Metal timestamp capability truth proof");

    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("Apple Metal context must remain creatable when timestamp instrumentation degrades");

    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Metal);
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::Compute)
    );
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::Copy)
    );
    assert!(
        !context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::TimestampQuery),
        "current Metal backend must not enable unproven timestamp instrumentation"
    );
    assert_eq!(
        context.timestamp_period_ns(),
        None,
        "a context that degraded timestamp instrumentation must not expose a timestamp scale"
    );

    context.progress();
}
