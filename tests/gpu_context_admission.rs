use runen_gpu::{
    GpuCapabilityFeature, GpuCapabilityProfile, GpuContext, GpuContextDescriptor,
    GpuContextRequestErrorCategory,
};
use std::collections::BTreeSet;

#[test]
fn headless_context_admission_reports_a_real_context_or_a_strict_environment_outcome() {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("engine headless context admission test");

    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => {
            let enabled = context
                .device_facts()
                .enabled_features()
                .collect::<BTreeSet<_>>();
            assert!(
                enabled.contains(&GpuCapabilityFeature::Compute)
                    && enabled.contains(&GpuCapabilityFeature::Copy)
            );
            assert_eq!(enabled.len(), 2, "no unrelated optional feature is enabled");
            assert_eq!(
                context.generation(),
                runen_gpu::GpuDeviceGeneration::first()
            );
            assert!(context.id().is_nonzero());
            assert!(
                context
                    .adapter_facts()
                    .supported()
                    .supports(GpuCapabilityFeature::Copy)
            );
            assert!(
                !context
                    .adapter_facts()
                    .supported()
                    .supports(GpuCapabilityFeature::Presentation),
                "headless admission must not claim surface presentation without surface evidence"
            );
            assert_eq!(
                context.adapter_facts(),
                context.admission_report().candidate().adapter()
            );
            assert_eq!(
                context.device_facts().workload_budget(),
                context.admission_report().candidate().workload_budget()
            );
            assert_eq!(
                context.device_facts().admission_contract(),
                context.admission_report().candidate().contract()
            );
            assert_eq!(
                context.device_facts().candidate_dispositions(),
                context.admission_report().candidate_dispositions()
            );
            assert_eq!(
                context.admission_report().selection_kind(),
                runen_gpu::GpuCandidateSelectionKind::DeterministicallyRanked
            );
            assert!(
                !context
                    .admission_report()
                    .candidate_dispositions()
                    .is_empty()
            );
            assert!(
                context
                    .admission_report()
                    .candidate_dispositions_are_canonically_ordered()
            );
            let selected = context.admission_report().candidate().adapter();
            let rank = context.admission_report().selection_evidence().rank();
            assert_eq!(rank.vendor(), selected.vendor());
            assert_eq!(rank.device(), selected.device());
            assert_eq!(
                rank.fallback_priority(),
                match selected.fallback() {
                    runen_gpu::GpuFallbackStatus::ConfirmedNotFallback => 0,
                    runen_gpu::GpuFallbackStatus::Unknown => 1,
                    runen_gpu::GpuFallbackStatus::ConfirmedFallback => 2,
                }
            );
        }
        Err(error) if accepts_environment_absence(error.category()) => {}
        Err(error) => panic!("unexpected native GPU context admission failure: {error}"),
    }
}

const fn accepts_environment_absence(category: GpuContextRequestErrorCategory) -> bool {
    matches!(category, GpuContextRequestErrorCategory::NoAdapterAvailable)
}

#[test]
fn strict_environment_contract_rejects_backend_and_device_failures() {
    assert!(accepts_environment_absence(
        GpuContextRequestErrorCategory::NoAdapterAvailable
    ));
    for category in [
        GpuContextRequestErrorCategory::NoAdmissibleCandidate,
        GpuContextRequestErrorCategory::AmbiguousAdapterSelection,
        GpuContextRequestErrorCategory::BackendAdapterRequestFailure,
        GpuContextRequestErrorCategory::BackendDeviceRequestFailure,
        GpuContextRequestErrorCategory::MandatoryFeatureMissing,
    ] {
        assert!(
            !accepts_environment_absence(category),
            "{category:?} must fail this environment-dependent proof"
        );
    }
}
