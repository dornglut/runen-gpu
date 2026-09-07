use super::admission::GpuAdmissionContract;
use super::descriptor::{GpuAdapterClass, GpuBackendFamily};
use super::selection::GpuCandidateDisposition;
use crate::{GpuCapabilities, GpuCapabilityFeature, GpuLimits};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSoftwareStatus {
    Software,
    Hardware,
    Unknown,
}

/// Observed adapter-selection-path evidence, separate from caller request policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuFallbackStatus {
    ConfirmedFallback,
    ConfirmedNotFallback,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDeviceRequestProfile {
    ModernPortable,
    Downlevel,
    BrowserWebGpu,
    DownlevelWebGl2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPortabilityClass {
    PortableBaseline,
    PortableWithDeclaredExtensions,
    BackendSpecialized,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPortabilityReason {
    DeclaredExtension(GpuCapabilityFeature),
    PreferredRequirementDegraded(GpuCapabilityFeature),
    BackendSpecialization(GpuBackendFamily),
    UnknownBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPortabilityEvidence {
    class: GpuPortabilityClass,
    reasons: BTreeSet<GpuPortabilityReason>,
}

impl GpuPortabilityEvidence {
    pub(crate) fn new(
        class: GpuPortabilityClass,
        reasons: impl IntoIterator<Item = GpuPortabilityReason>,
    ) -> Self {
        Self {
            class,
            reasons: reasons.into_iter().collect(),
        }
    }

    pub const fn class(&self) -> GpuPortabilityClass {
        self.class
    }

    pub fn reasons(&self) -> impl ExactSizeIterator<Item = GpuPortabilityReason> + '_ {
        self.reasons.iter().copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuAlignmentFacts {
    pub uniform_dynamic_offset: Option<u64>,
    pub storage_dynamic_offset: Option<u64>,
    pub copy_buffer_offset: Option<u64>,
    pub bytes_per_row: Option<u64>,
    pub query_resolve_destination: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuAdapterLimits {
    values: GpuLimits,
}

impl GpuAdapterLimits {
    pub const fn new(values: GpuLimits) -> Self {
        Self { values }
    }

    pub const fn values(&self) -> GpuLimits {
        self.values
    }
}

/// Facts reported by the created device, never caller workload caps or adapter maxima.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuDeviceLimits {
    values: GpuLimits,
    alignments: GpuAlignmentFacts,
}

impl GpuDeviceLimits {
    pub(crate) const fn new(values: GpuLimits, alignments: GpuAlignmentFacts) -> Self {
        Self { values, alignments }
    }

    pub const fn values(&self) -> GpuLimits {
        self.values
    }

    pub const fn alignments(&self) -> GpuAlignmentFacts {
        self.alignments
    }
}

/// RunenGPU policy applied to workloads; it is not an adapter or device fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkloadBudget {
    limits: GpuLimits,
    alignment_maximums: BTreeMap<super::descriptor::GpuAlignmentKind, u64>,
}

impl GpuWorkloadBudget {
    pub(crate) fn new(
        limits: GpuLimits,
        alignment_maximums: BTreeMap<super::descriptor::GpuAlignmentKind, u64>,
    ) -> Self {
        Self {
            limits,
            alignment_maximums,
        }
    }

    pub const fn limits(&self) -> GpuLimits {
        self.limits
    }

    pub fn alignment_maximums(
        &self,
    ) -> impl ExactSizeIterator<Item = (super::descriptor::GpuAlignmentKind, u64)> + '_ {
        self.alignment_maximums
            .iter()
            .map(|(kind, value)| (*kind, *value))
    }

    pub(crate) fn alignment_maximum(
        &self,
        kind: super::descriptor::GpuAlignmentKind,
    ) -> Option<u64> {
        self.alignment_maximums.get(&kind).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAdapterFacts {
    backend: GpuBackendFamily,
    class: GpuAdapterClass,
    software: GpuSoftwareStatus,
    fallback: GpuFallbackStatus,
    diagnostic_name: Option<String>,
    driver: Option<String>,
    driver_info: Option<String>,
    vendor: Option<u32>,
    device: Option<u32>,
    supported: GpuCapabilities,
    adapter_limits: GpuAdapterLimits,
    alignments: GpuAlignmentFacts,
    device_profile: GpuDeviceRequestProfile,
    device_profile_supported: bool,
}

impl GpuAdapterFacts {
    pub fn new(
        backend: GpuBackendFamily,
        class: GpuAdapterClass,
        software: GpuSoftwareStatus,
        fallback: GpuFallbackStatus,
        supported: GpuCapabilities,
        adapter_limits: GpuAdapterLimits,
        alignments: GpuAlignmentFacts,
    ) -> Self {
        Self {
            backend,
            class,
            software,
            fallback,
            diagnostic_name: None,
            driver: None,
            driver_info: None,
            vendor: None,
            device: None,
            supported,
            adapter_limits,
            alignments,
            device_profile: GpuDeviceRequestProfile::ModernPortable,
            device_profile_supported: true,
        }
    }

    pub(crate) fn with_diagnostics(
        mut self,
        name: String,
        vendor: u32,
        device: u32,
        driver: String,
        driver_info: String,
    ) -> Self {
        self.diagnostic_name = super::diagnostics::sanitized_diagnostic(name);
        self.driver = super::diagnostics::sanitized_diagnostic(driver);
        self.driver_info = super::diagnostics::sanitized_diagnostic(driver_info);
        self.vendor = Some(vendor);
        self.device = Some(device);
        self
    }

    pub(crate) fn with_device_profile(
        mut self,
        profile: GpuDeviceRequestProfile,
        supported: bool,
    ) -> Self {
        self.device_profile = profile;
        self.device_profile_supported = supported;
        self
    }

    pub const fn backend(&self) -> GpuBackendFamily {
        self.backend
    }

    pub const fn class(&self) -> GpuAdapterClass {
        self.class
    }

    pub const fn software(&self) -> GpuSoftwareStatus {
        self.software
    }

    pub const fn fallback(&self) -> GpuFallbackStatus {
        self.fallback
    }

    pub fn diagnostic_name(&self) -> Option<&str> {
        self.diagnostic_name.as_deref()
    }

    /// Sanitized backend diagnostic fact. It is not a rank, retry, or persistent identity.
    pub fn driver(&self) -> Option<&str> {
        self.driver.as_deref()
    }

    /// Sanitized backend diagnostic fact. It is not a rank, retry, or persistent identity.
    pub fn driver_info(&self) -> Option<&str> {
        self.driver_info.as_deref()
    }

    pub const fn vendor(&self) -> Option<u32> {
        self.vendor
    }

    pub const fn device(&self) -> Option<u32> {
        self.device
    }

    pub fn supported(&self) -> &GpuCapabilities {
        &self.supported
    }

    pub const fn adapter_limits(&self) -> GpuAdapterLimits {
        self.adapter_limits
    }

    pub const fn alignments(&self) -> GpuAlignmentFacts {
        self.alignments
    }

    pub const fn device_request_profile(&self) -> GpuDeviceRequestProfile {
        self.device_profile
    }

    pub const fn device_request_profile_supported(&self) -> bool {
        self.device_profile_supported
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuAdapterClass, GpuBackendFamily, GpuCapabilities, GpuLimits};

    fn test_limits() -> GpuLimits {
        GpuLimits::new(
            64 * 1024,
            128 * 1024 * 1024,
            1,
            8,
            16,
            8192,
            4,
            12,
            8,
            4,
            65_535,
        )
        .unwrap()
    }

    fn adapter() -> GpuAdapterFacts {
        let limits = test_limits();
        GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            GpuFallbackStatus::Unknown,
            GpuCapabilities::from_normalized_facts([], limits, []),
            GpuAdapterLimits::new(limits),
            GpuAlignmentFacts {
                uniform_dynamic_offset: Some(256),
                storage_dynamic_offset: Some(256),
                copy_buffer_offset: Some(4),
                bytes_per_row: Some(256),
                query_resolve_destination: Some(256),
            },
        )
    }

    #[test]
    fn adapter_driver_facts_are_sanitized_and_optional() {
        let facts = adapter().with_diagnostics(
            "adapter".to_owned(),
            1,
            2,
            "driver".to_owned(),
            "é".repeat(129),
        );
        assert_eq!(facts.driver(), Some("driver"));
        assert_eq!(facts.driver_info().map(str::len), Some(256));
        assert!(
            adapter()
                .with_diagnostics("adapter".to_owned(), 1, 2, " ".to_owned(), String::new())
                .driver()
                .is_none()
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAdmittedDeviceFacts {
    enabled: BTreeSet<GpuCapabilityFeature>,
    device_limits: GpuDeviceLimits,
    workload_budget: GpuWorkloadBudget,
    admission_contract: GpuAdmissionContract,
    candidate_dispositions: Vec<GpuCandidateDisposition>,
}

impl GpuAdmittedDeviceFacts {
    pub(crate) fn new(
        enabled: BTreeSet<GpuCapabilityFeature>,
        device_limits: GpuDeviceLimits,
        workload_budget: GpuWorkloadBudget,
        admission_contract: GpuAdmissionContract,
        candidate_dispositions: Vec<GpuCandidateDisposition>,
    ) -> Self {
        Self {
            enabled,
            device_limits,
            workload_budget,
            admission_contract,
            candidate_dispositions,
        }
    }

    pub fn enabled_features(&self) -> impl ExactSizeIterator<Item = GpuCapabilityFeature> + '_ {
        self.enabled.iter().copied()
    }

    pub fn is_enabled(&self, feature: GpuCapabilityFeature) -> bool {
        self.enabled.contains(&feature)
    }

    pub const fn device_limits(&self) -> GpuDeviceLimits {
        self.device_limits
    }

    pub fn workload_budget(&self) -> &GpuWorkloadBudget {
        &self.workload_budget
    }

    pub fn admission_contract(&self) -> &GpuAdmissionContract {
        &self.admission_contract
    }

    /// Immutable canonical admission evidence retained for future G4C realization.
    pub fn candidate_dispositions(&self) -> &[GpuCandidateDisposition] {
        &self.candidate_dispositions
    }
}
