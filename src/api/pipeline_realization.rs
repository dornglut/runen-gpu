use super::{
    GpuComputePipelineDescriptor, GpuContextAffinity, GpuRenderPipelineDescriptor,
    sanitized_diagnostic,
};
use core::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Stable semantic classes for G4C3 compute/render pipeline realization rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPipelineRealizationErrorCategory {
    ForeignContext,
    StaleDeviceGeneration,
    UnknownProgramOrLayoutRealization,
    PipelineDescriptorInvalid,
    EntryPointStageMismatch,
    ProgramInterfaceMismatch,
    PipelineStageIoMismatch,
    RequirementNotAdmitted,
    FormatOrAlignmentNotAdmitted,
    RegistryCapacityExceeded,
    CacheRejected,
    UnexpectedBackendPipelineValidationRejection,
    BackendResourceExhaustion,
    ContextOrDeviceUnavailableOrLost,
    ExecutionAuthorityViolation,
}

impl GpuPipelineRealizationErrorCategory {
    pub const fn correction(self) -> &'static str {
        match self {
            Self::ForeignContext => "use pipeline dependencies realized by this GPU context",
            Self::StaleDeviceGeneration => {
                "realize the pipeline again against the current GPU device generation"
            }
            Self::UnknownProgramOrLayoutRealization => {
                "use program and pipeline-layout realizations retained by this GPU context"
            }
            Self::PipelineDescriptorInvalid => {
                "use one complete accepted G4B compute or render pipeline descriptor"
            }
            Self::EntryPointStageMismatch => {
                "select entry points declared for the pipeline's required shader stages"
            }
            Self::ProgramInterfaceMismatch => {
                "make the descriptor program and pipeline layout agree with the accepted G4C2 realizations"
            }
            Self::PipelineStageIoMismatch => {
                "make explicit vertex/color pipeline state agree with the selected program entry-point signatures"
            }
            Self::RequirementNotAdmitted => {
                "admit every required pipeline capability when requesting the GPU context"
            }
            Self::FormatOrAlignmentNotAdmitted => {
                "use pipeline state within the admitted format, alignment, and device-limit facts"
            }
            Self::RegistryCapacityExceeded => {
                "release unused realized pipeline handles before creating more authoritative records"
            }
            Self::CacheRejected => {
                "discard the derived candidate and realize the pipeline ordinarily"
            }
            Self::UnexpectedBackendPipelineValidationRejection => {
                "inspect the bounded backend evidence and RunenGPU pipeline-realization invariant"
            }
            Self::BackendResourceExhaustion => {
                "reduce backend pipeline/resource pressure without treating registry count as GPU memory"
            }
            Self::ContextOrDeviceUnavailableOrLost => {
                "stop using this context and let the owning product choose recovery"
            }
            Self::ExecutionAuthorityViolation => {
                "use only the audited lexical current-render execution terminal"
            }
        }
    }
}

/// Structured failure from deterministic G4C3 compatibility, authoritative lookup, or backend
/// pipeline publication. Backend text is bounded diagnostic evidence only and never identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPipelineRealizationError {
    pub(crate) category: GpuPipelineRealizationErrorCategory,
    pub(crate) request: Option<Box<str>>,
    pub(crate) detail: Option<Box<str>>,
    pub(crate) secondary_detail: Option<Box<str>>,
    pub(crate) expected_affinity: Option<GpuContextAffinity>,
    pub(crate) observed_affinity: Option<GpuContextAffinity>,
    pub(crate) retained_records: Option<usize>,
    pub(crate) max_records: Option<NonZeroUsize>,
}

impl GpuPipelineRealizationError {
    pub(crate) fn new(
        category: GpuPipelineRealizationErrorCategory,
        request: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            category,
            request: sanitized_diagnostic(request.into()).map(String::into_boxed_str),
            detail: sanitized_diagnostic(detail.into()).map(String::into_boxed_str),
            secondary_detail: None,
            expected_affinity: None,
            observed_affinity: None,
            retained_records: None,
            max_records: None,
        }
    }

    pub(crate) fn affinity(
        category: GpuPipelineRealizationErrorCategory,
        request: impl Into<String>,
        expected: GpuContextAffinity,
        observed: GpuContextAffinity,
    ) -> Self {
        let mut error = Self::new(
            category,
            request,
            "realized pipeline dependency affinity does not match",
        );
        error.expected_affinity = Some(expected);
        error.observed_affinity = Some(observed);
        error
    }

    pub(crate) fn capacity(
        request: impl Into<String>,
        retained_records: usize,
        max_records: NonZeroUsize,
    ) -> Self {
        let mut error = Self::new(
            GpuPipelineRealizationErrorCategory::RegistryCapacityExceeded,
            request,
            "the pipeline realization registry is occupied by live ready or in-flight records",
        );
        error.retained_records = Some(retained_records);
        error.max_records = Some(max_records);
        error
    }

    pub(crate) fn with_secondary_detail(mut self, detail: impl Into<String>) -> Self {
        self.secondary_detail = sanitized_diagnostic(detail.into()).map(String::into_boxed_str);
        self
    }

    pub const fn category(&self) -> GpuPipelineRealizationErrorCategory {
        self.category
    }

    pub fn request(&self) -> Option<&str> {
        self.request.as_deref()
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub fn secondary_detail(&self) -> Option<&str> {
        self.secondary_detail.as_deref()
    }

    pub const fn expected_affinity(&self) -> Option<GpuContextAffinity> {
        self.expected_affinity
    }

    pub const fn observed_affinity(&self) -> Option<GpuContextAffinity> {
        self.observed_affinity
    }

    pub const fn retained_records(&self) -> Option<usize> {
        self.retained_records
    }

    pub const fn max_records(&self) -> Option<NonZeroUsize> {
        self.max_records
    }
}

impl fmt::Display for GpuPipelineRealizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU pipeline realization rejected ({:?}): {}",
            self.category,
            self.category.correction(),
        )?;
        if let Some(request) = self.request() {
            write!(formatter, " [request: {request}]")?;
        }
        if let Some(detail) = self.detail() {
            write!(formatter, " [detail: {detail}]")?;
        }
        if let Some(secondary_detail) = self.secondary_detail() {
            write!(formatter, " [secondary detail: {secondary_detail}]")?;
        }
        if let (Some(retained), Some(maximum)) = (self.retained_records(), self.max_records()) {
            write!(formatter, " [records: {retained}/{}]", maximum.get())?;
        }
        Ok(())
    }
}

impl std::error::Error for GpuPipelineRealizationError {}

/// Opaque context/device-generation-bound realization of one complete compute pipeline.
#[derive(Clone)]
pub struct GpuRealizedComputePipeline {
    pub(crate) record: Arc<crate::backend::ComputePipelineRealizationRecord>,
}

impl GpuRealizedComputePipeline {
    pub(crate) fn from_record(
        record: Arc<crate::backend::ComputePipelineRealizationRecord>,
    ) -> Self {
        Self { record }
    }

    pub fn affinity(&self) -> GpuContextAffinity {
        self.record.affinity()
    }

    pub fn descriptor(&self) -> &GpuComputePipelineDescriptor {
        self.record.descriptor()
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
}

impl fmt::Debug for GpuRealizedComputePipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuRealizedComputePipeline")
            .field("affinity", &self.affinity())
            .field("entry_point", self.descriptor().entry_point())
            .finish_non_exhaustive()
    }
}

/// Opaque context/device-generation-bound realization of one complete render pipeline.
#[derive(Clone)]
pub struct GpuRealizedRenderPipeline {
    pub(crate) record: Arc<crate::backend::RenderPipelineRealizationRecord>,
}

impl GpuRealizedRenderPipeline {
    pub(crate) fn from_record(
        record: Arc<crate::backend::RenderPipelineRealizationRecord>,
    ) -> Self {
        Self { record }
    }

    pub fn affinity(&self) -> GpuContextAffinity {
        self.record.affinity()
    }

    pub fn descriptor(&self) -> &GpuRenderPipelineDescriptor {
        self.record.descriptor()
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
}

impl fmt::Debug for GpuRealizedRenderPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuRealizedRenderPipeline")
            .field("affinity", &self.affinity())
            .field("entry_points", self.descriptor().entry_points())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContextId, GpuDeviceGeneration};
    use core::num::NonZeroU64;

    #[test]
    fn every_pipeline_failure_class_has_actionable_correction() {
        let categories = [
            GpuPipelineRealizationErrorCategory::ForeignContext,
            GpuPipelineRealizationErrorCategory::StaleDeviceGeneration,
            GpuPipelineRealizationErrorCategory::UnknownProgramOrLayoutRealization,
            GpuPipelineRealizationErrorCategory::PipelineDescriptorInvalid,
            GpuPipelineRealizationErrorCategory::EntryPointStageMismatch,
            GpuPipelineRealizationErrorCategory::ProgramInterfaceMismatch,
            GpuPipelineRealizationErrorCategory::PipelineStageIoMismatch,
            GpuPipelineRealizationErrorCategory::RequirementNotAdmitted,
            GpuPipelineRealizationErrorCategory::FormatOrAlignmentNotAdmitted,
            GpuPipelineRealizationErrorCategory::RegistryCapacityExceeded,
            GpuPipelineRealizationErrorCategory::CacheRejected,
            GpuPipelineRealizationErrorCategory::UnexpectedBackendPipelineValidationRejection,
            GpuPipelineRealizationErrorCategory::BackendResourceExhaustion,
            GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            GpuPipelineRealizationErrorCategory::ExecutionAuthorityViolation,
        ];
        for category in categories {
            assert!(!category.correction().trim().is_empty(), "{category:?}");
        }
    }

    #[test]
    fn pipeline_error_keeps_capacity_pressure_structured() {
        let error = GpuPipelineRealizationError::capacity(
            "representative pipeline",
            7,
            NonZeroUsize::new(7).unwrap(),
        );
        assert_eq!(
            error.category(),
            GpuPipelineRealizationErrorCategory::RegistryCapacityExceeded
        );
        assert_eq!(error.retained_records(), Some(7));
        assert_eq!(error.max_records().map(NonZeroUsize::get), Some(7));
        assert!(error.to_string().contains("records: 7/7"));
    }

    #[test]
    fn pipeline_cache_rejection_remains_a_typed_category() {
        let error = GpuPipelineRealizationError::new(
            GpuPipelineRealizationErrorCategory::CacheRejected,
            "representative cached pipeline",
            "derived cache candidate was rejected",
        );
        assert_eq!(
            error.category(),
            GpuPipelineRealizationErrorCategory::CacheRejected
        );
    }

    #[test]
    fn stale_pipeline_generation_keeps_expected_and_observed_affinity_typed() {
        let context = GpuContextId::test_value(NonZeroU64::new(9).unwrap());
        let stale_generation = GpuDeviceGeneration::first();
        let current_generation = stale_generation.next().unwrap();
        let current = GpuContextAffinity::test_value(context, current_generation);
        let stale = GpuContextAffinity::test_value(context, stale_generation);
        let error = GpuPipelineRealizationError::affinity(
            GpuPipelineRealizationErrorCategory::StaleDeviceGeneration,
            "representative stale pipeline",
            current,
            stale,
        );

        assert_eq!(
            error.category(),
            GpuPipelineRealizationErrorCategory::StaleDeviceGeneration
        );
        assert_eq!(error.expected_affinity(), Some(current));
        assert_eq!(error.observed_affinity(), Some(stale));
        assert_eq!(
            error
                .expected_affinity()
                .map(|affinity| affinity.generation()),
            Some(current_generation)
        );
        assert_eq!(
            error
                .observed_affinity()
                .map(|affinity| affinity.generation()),
            Some(stale_generation)
        );
    }
}
