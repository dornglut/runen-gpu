//! Backend-neutral contracts for context-bound resource realization.

use super::{
    GpuBindGroupLayoutDescriptor, GpuBufferDescriptor, GpuContextAffinity,
    GpuPipelineLayoutDescriptor, GpuProgramDescriptor, GpuQuerySetDescriptor, GpuSamplerDescriptor,
    GpuTextureDescriptor, GpuTextureViewDescriptor, GpuWorkResourceId, sanitized_diagnostic,
};
use core::fmt;
use core::num::NonZeroUsize;
use std::sync::Arc;

/// Default maximum number of authoritative resource-realization records retained by one context.
///
/// This bounds process-local lookup authority, not GPU memory, residency, or hardware capacity.
/// The value leaves room for ordinary long-running renderer and compute workloads while keeping
/// accidental unbounded identity growth observable. General-purpose hosts can provide an explicit
/// [`GpuResourceRealizationPolicy`] when requesting a context.
pub const DEFAULT_MAX_RESOURCE_REALIZATION_RECORDS: NonZeroUsize =
    NonZeroUsize::new(16_384).expect("the default realization-record bound is nonzero");

/// Default maximum number of authoritative program and binding realization records retained by
/// one context/device generation.
///
/// The bound is shared by ready and in-flight program, bind-group-layout, pipeline-layout, and
/// bind-group records. It is lookup authority only; it is neither a GPU-memory budget nor G5
/// completion/retirement evidence.
const DEFAULT_MAX_PROGRAM_BINDING_REALIZATION_RECORDS: NonZeroUsize =
    NonZeroUsize::new(16_384).expect("the default realization-record bound is nonzero");

/// Narrow operational policy for one context's resource-realization authority.
///
/// This policy does not participate in adapter selection, device admission, retry identity, or
/// [`super::GpuWorkloadBudget`]. A record count must not be interpreted as byte or residency cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuResourceRealizationPolicy {
    max_records: NonZeroUsize,
}

impl GpuResourceRealizationPolicy {
    /// Creates an explicit total bound shared by buffers, textures, texture views, samplers, and
    /// query sets. No per-kind quota is implied.
    pub const fn new(max_records: NonZeroUsize) -> Self {
        Self { max_records }
    }

    pub const fn max_records(self) -> NonZeroUsize {
        self.max_records
    }
}

impl Default for GpuResourceRealizationPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_RESOURCE_REALIZATION_RECORDS)
    }
}

/// Narrow operational policy for one context's G4C2 program and binding realization authority.
///
/// One nonzero total bound is deliberately shared across every G4C2 realization kind. This keeps
/// in-flight reservations visible to capacity accounting and prevents per-kind quotas from
/// silently changing the authority contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuProgramBindingRealizationPolicy {
    max_records: NonZeroUsize,
}

impl GpuProgramBindingRealizationPolicy {
    pub const fn new(max_records: NonZeroUsize) -> Self {
        Self { max_records }
    }

    pub const fn max_records(self) -> NonZeroUsize {
        self.max_records
    }
}

impl Default for GpuProgramBindingRealizationPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_PROGRAM_BINDING_REALIZATION_RECORDS)
    }
}

/// Complete realization policy composition selected when admitting a context.
///
/// This is the only explicit context-construction override path. It composes the existing G4C1
/// resource-record bound with the G4C2 program/binding bound without changing adapter admission,
/// device selection, or retry identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuRealizationPolicies {
    resource: GpuResourceRealizationPolicy,
    program_binding: GpuProgramBindingRealizationPolicy,
}

impl GpuRealizationPolicies {
    pub const fn new(
        resource: GpuResourceRealizationPolicy,
        program_binding: GpuProgramBindingRealizationPolicy,
    ) -> Self {
        Self {
            resource,
            program_binding,
        }
    }

    pub const fn resource(self) -> GpuResourceRealizationPolicy {
        self.resource
    }

    pub const fn program_binding(self) -> GpuProgramBindingRealizationPolicy {
        self.program_binding
    }
}

impl Default for GpuRealizationPolicies {
    fn default() -> Self {
        Self::new(
            GpuResourceRealizationPolicy::default(),
            GpuProgramBindingRealizationPolicy::default(),
        )
    }
}

/// Point-in-time lookup-authority counts for one context/device generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuResourceRealizationStats {
    retained_records: usize,
    max_records: NonZeroUsize,
    buffers: usize,
    textures: usize,
    texture_views: usize,
    samplers: usize,
    query_sets: usize,
}

impl GpuResourceRealizationStats {
    pub(crate) const fn new(
        max_records: NonZeroUsize,
        buffers: usize,
        textures: usize,
        texture_views: usize,
        samplers: usize,
        query_sets: usize,
    ) -> Self {
        Self {
            retained_records: buffers + textures + texture_views + samplers + query_sets,
            max_records,
            buffers,
            textures,
            texture_views,
            samplers,
            query_sets,
        }
    }

    pub const fn retained_records(self) -> usize {
        self.retained_records
    }

    pub const fn max_records(self) -> NonZeroUsize {
        self.max_records
    }

    pub const fn buffers(self) -> usize {
        self.buffers
    }

    pub const fn textures(self) -> usize {
        self.textures
    }

    pub const fn texture_views(self) -> usize {
        self.texture_views
    }

    pub const fn samplers(self) -> usize {
        self.samplers
    }

    pub const fn query_sets(self) -> usize {
        self.query_sets
    }
}

/// Point-in-time lookup-authority counts for one context/device generation's G4C2 records.
///
/// `retained_records` includes ready records and active single-flight reservations. Per-kind
/// observations describe ready records only and are never independent quotas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuProgramBindingRealizationStats {
    retained_records: usize,
    in_flight_records: usize,
    max_records: NonZeroUsize,
    programs: usize,
    bind_group_layouts: usize,
    pipeline_layouts: usize,
    bind_groups: usize,
}

impl GpuProgramBindingRealizationStats {
    pub(crate) const fn new(
        max_records: NonZeroUsize,
        in_flight_records: usize,
        programs: usize,
        bind_group_layouts: usize,
        pipeline_layouts: usize,
        bind_groups: usize,
    ) -> Self {
        Self {
            retained_records: in_flight_records
                + programs
                + bind_group_layouts
                + pipeline_layouts
                + bind_groups,
            in_flight_records,
            max_records,
            programs,
            bind_group_layouts,
            pipeline_layouts,
            bind_groups,
        }
    }

    pub const fn retained_records(self) -> usize {
        self.retained_records
    }

    pub const fn in_flight_records(self) -> usize {
        self.in_flight_records
    }

    pub const fn max_records(self) -> NonZeroUsize {
        self.max_records
    }

    pub const fn programs(self) -> usize {
        self.programs
    }

    pub const fn bind_group_layouts(self) -> usize {
        self.bind_group_layouts
    }

    pub const fn pipeline_layouts(self) -> usize {
        self.pipeline_layouts
    }

    pub const fn bind_groups(self) -> usize {
        self.bind_groups
    }
}

/// Stable semantic classes for resource-realization rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuResourceRealizationErrorCategory {
    ForeignContext,
    StaleDeviceGeneration,
    UnknownLogicalResource,
    DescriptorChangedForIdentity,
    ResourceKindMismatch,
    RequirementNotAdmitted,
    FormatOrAlignmentNotAdmitted,
    ImportGenerationMismatch,
    ImportSourceUnavailable,
    RegistryCapacityExceeded,
    CacheRejected,
    UnexpectedBackendValidationRejection,
    BackendResourceExhaustion,
    ContextOrDeviceUnavailableOrLost,
    ExecutionAuthorityViolation,
}

impl GpuResourceRealizationErrorCategory {
    const fn correction(self) -> &'static str {
        match self {
            Self::ForeignContext => "use a realization retained by this GPU context",
            Self::StaleDeviceGeneration => {
                "realize the resource again against the current device generation"
            }
            Self::UnknownLogicalResource => {
                "realize the exact declared logical resource before its dependent resource"
            }
            Self::DescriptorChangedForIdentity => {
                "allocate a new logical identity for the changed resource descriptor"
            }
            Self::ResourceKindMismatch => {
                "use one logical identity with exactly one typed resource family"
            }
            Self::RequirementNotAdmitted => {
                "request the required capability when admitting the GPU context"
            }
            Self::FormatOrAlignmentNotAdmitted => {
                "use format, usage, size, and alignment facts admitted by the device"
            }
            Self::ImportGenerationMismatch => {
                "use a concrete import source from the admitted source generation"
            }
            Self::ImportSourceUnavailable => {
                "provide an accepted concrete import source or use owned resource semantics"
            }
            Self::RegistryCapacityExceeded => {
                "release unused realizations or request a larger explicit record policy"
            }
            Self::CacheRejected => "discard the derived candidate and realize ordinarily",
            Self::UnexpectedBackendValidationRejection => {
                "inspect the bounded backend evidence and RunenGPU admission invariant"
            }
            Self::BackendResourceExhaustion => {
                "reduce backend resource pressure without treating record count as GPU memory"
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

/// Structured failure from deterministic admission, authoritative lookup, or backend creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuResourceRealizationError {
    category: GpuResourceRealizationErrorCategory,
    resource: Option<GpuWorkResourceId>,
    detail: Option<String>,
    expected_affinity: Option<GpuContextAffinity>,
    observed_affinity: Option<GpuContextAffinity>,
    retained_records: Option<usize>,
    max_records: Option<NonZeroUsize>,
}

impl GpuResourceRealizationError {
    pub(crate) fn new(
        category: GpuResourceRealizationErrorCategory,
        resource: Option<GpuWorkResourceId>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            category,
            resource,
            detail: sanitized_diagnostic(detail.into()),
            expected_affinity: None,
            observed_affinity: None,
            retained_records: None,
            max_records: None,
        }
    }

    pub(crate) fn affinity(
        category: GpuResourceRealizationErrorCategory,
        resource: Option<GpuWorkResourceId>,
        expected: GpuContextAffinity,
        observed: GpuContextAffinity,
    ) -> Self {
        let mut error = Self::new(category, resource, "realized input affinity does not match");
        error.expected_affinity = Some(expected);
        error.observed_affinity = Some(observed);
        error
    }

    pub(crate) fn capacity(
        resource: GpuWorkResourceId,
        retained_records: usize,
        max_records: NonZeroUsize,
    ) -> Self {
        let mut error = Self::new(
            GpuResourceRealizationErrorCategory::RegistryCapacityExceeded,
            Some(resource),
            "the total authoritative realization-record bound is occupied by live records",
        );
        error.retained_records = Some(retained_records);
        error.max_records = Some(max_records);
        error
    }

    pub const fn category(&self) -> GpuResourceRealizationErrorCategory {
        self.category
    }

    pub const fn resource(&self) -> Option<GpuWorkResourceId> {
        self.resource
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
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

impl fmt::Display for GpuResourceRealizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU resource realization failed ({:?})",
            self.category
        )?;
        if let Some(resource) = self.resource {
            write!(formatter, " for resource {resource}")?;
        }
        if let Some(detail) = &self.detail {
            write!(formatter, ": {detail}")?;
        }
        if let (Some(retained), Some(maximum)) = (self.retained_records, self.max_records) {
            write!(formatter, "; records: {retained}/{}", maximum.get())?;
        }
        write!(formatter, "; correction: {}", self.category.correction())
    }
}

impl std::error::Error for GpuResourceRealizationError {}

/// Stable semantic classes for G4C2 program, layout, and bind-group realization rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuProgramBindingRealizationErrorCategory {
    ForeignContext,
    StaleDeviceGeneration,
    UnknownAdmittedSource,
    SourceRevisionConflict,
    NagaDependencyBaselineMismatch,
    WgslParseOrValidationFailed,
    ShaderValidationPathMismatch,
    ProgramInterfaceMismatch,
    ObservedStageIoInvalid,
    UnknownLayout,
    LayoutDescriptorInvalid,
    RuntimeBindingIncompatible,
    BindingValueMismatch,
    RegistryCapacityExceeded,
    CacheRejected,
    UnexpectedBackendProgramOrBindingValidationRejection,
    BackendResourceExhaustion,
    ContextOrDeviceUnavailableOrLost,
    ExecutionAuthorityViolation,
}

impl GpuProgramBindingRealizationErrorCategory {
    const fn correction(self) -> &'static str {
        match self {
            Self::ForeignContext => "use a realization retained by this GPU context",
            Self::StaleDeviceGeneration => {
                "realize the program, layout, or binding again against the current device generation"
            }
            Self::UnknownAdmittedSource => {
                "use a program descriptor retaining one admitted canonical WGSL source"
            }
            Self::SourceRevisionConflict => {
                "admit changed canonical WGSL under a distinct nonzero source revision"
            }
            Self::NagaDependencyBaselineMismatch => {
                "use the accepted direct Naga dependency baseline for this realization profile"
            }
            Self::WgslParseOrValidationFailed => {
                "make the exact admitted canonical WGSL valid for the accepted Naga profile"
            }
            Self::ShaderValidationPathMismatch => {
                "inspect the bounded WGPU evidence because accepted Naga and WGPU paths disagree"
            }
            Self::ProgramInterfaceMismatch => {
                "make explicit program declarations agree with normalized WGSL evidence"
            }
            Self::ObservedStageIoInvalid => {
                "use supported, unambiguous vertex-input and fragment-output WGSL signatures"
            }
            Self::UnknownLayout => {
                "realize the exact descriptor-owned bind-group layout through this context"
            }
            Self::LayoutDescriptorInvalid => {
                "use an admitted typed layout descriptor compatible with this context"
            }
            Self::RuntimeBindingIncompatible => {
                "provide runtime resources compatible with the exact typed bind-group layout"
            }
            Self::BindingValueMismatch => {
                "provide one complete typed runtime value for each declared binding"
            }
            Self::RegistryCapacityExceeded => {
                "release unused realization handles or request a larger explicit record policy"
            }
            Self::CacheRejected => "discard the derived candidate and realize ordinarily",
            Self::UnexpectedBackendProgramOrBindingValidationRejection => {
                "inspect the bounded backend evidence and RunenGPU realization invariant"
            }
            Self::BackendResourceExhaustion => {
                "reduce backend resource pressure without treating record count as GPU memory"
            }
            Self::ContextOrDeviceUnavailableOrLost => {
                "stop using this context and let the owning product choose recovery"
            }
            Self::ExecutionAuthorityViolation => {
                "use only an audited lexical current-render execution terminal"
            }
        }
    }
}

/// Structured failure from G4C2 deterministic admission, single-flight lookup, or backend
/// publication. Backend diagnostic text is bounded evidence only and never cache identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProgramBindingRealizationError {
    category: GpuProgramBindingRealizationErrorCategory,
    // These diagnostics are bounded and immutable after construction. Store their existing
    // allocations as boxed slices so the public structured error stays small enough to carry
    // through the async G4C2 realization result without imposing a large `Result` error value on
    // every private owner helper.
    request: Option<Box<str>>,
    detail: Option<Box<str>>,
    secondary_detail: Option<Box<str>>,
    expected_affinity: Option<GpuContextAffinity>,
    observed_affinity: Option<GpuContextAffinity>,
    retained_records: Option<usize>,
    max_records: Option<NonZeroUsize>,
}

impl GpuProgramBindingRealizationError {
    pub(crate) fn new(
        category: GpuProgramBindingRealizationErrorCategory,
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
        category: GpuProgramBindingRealizationErrorCategory,
        request: impl Into<String>,
        expected: GpuContextAffinity,
        observed: GpuContextAffinity,
    ) -> Self {
        let mut error = Self::new(category, request, "realized input affinity does not match");
        error.expected_affinity = Some(expected);
        error.observed_affinity = Some(observed);
        error
    }

    /// Retains bounded evidence from lower-precedence backend observations for one attempt.
    ///
    /// The primary category and detail remain the deterministic outcome. Secondary evidence is
    /// diagnostic only: it cannot become lookup identity or a second failure authority.
    pub(crate) fn with_secondary_detail(mut self, detail: impl Into<String>) -> Self {
        self.secondary_detail = sanitized_diagnostic(detail.into()).map(String::into_boxed_str);
        self
    }

    pub(crate) fn capacity(
        request: impl Into<String>,
        retained_records: usize,
        max_records: NonZeroUsize,
    ) -> Self {
        let mut error = Self::new(
            GpuProgramBindingRealizationErrorCategory::RegistryCapacityExceeded,
            request,
            "the total authoritative program/binding realization-record bound is occupied by live records",
        );
        error.retained_records = Some(retained_records);
        error.max_records = Some(max_records);
        error
    }

    pub const fn category(&self) -> GpuProgramBindingRealizationErrorCategory {
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

impl fmt::Display for GpuProgramBindingRealizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU program/binding realization rejected ({:?}): {}",
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

impl std::error::Error for GpuProgramBindingRealizationError {}

macro_rules! realized_handle {
    ($name:ident, $record:ty, $descriptor:ty) => {
        #[derive(Clone)]
        pub struct $name {
            pub(crate) record: Arc<$record>,
        }

        impl $name {
            pub(crate) fn from_record(record: Arc<$record>) -> Self {
                Self { record }
            }

            pub fn affinity(&self) -> GpuContextAffinity {
                self.record.affinity()
            }

            pub fn logical_identity(&self) -> GpuWorkResourceId {
                self.record.logical_identity()
            }

            pub fn descriptor(&self) -> &$descriptor {
                self.record.descriptor()
            }

            pub fn is_same_record(&self, other: &Self) -> bool {
                Arc::ptr_eq(&self.record, &other.record)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("affinity", &self.affinity())
                    .field("logical_identity", &self.logical_identity())
                    .finish_non_exhaustive()
            }
        }
    };
}

realized_handle!(
    GpuRealizedBuffer,
    crate::backend::BufferRealizationRecord,
    GpuBufferDescriptor
);
realized_handle!(
    GpuRealizedTexture,
    crate::backend::TextureRealizationRecord,
    GpuTextureDescriptor
);
realized_handle!(
    GpuRealizedTextureView,
    crate::backend::TextureViewRealizationRecord,
    GpuTextureViewDescriptor
);
realized_handle!(
    GpuRealizedSampler,
    crate::backend::SamplerRealizationRecord,
    GpuSamplerDescriptor
);
realized_handle!(
    GpuRealizedQuerySet,
    crate::backend::QuerySetRealizationRecord,
    GpuQuerySetDescriptor
);

impl GpuRealizedTextureView {
    pub fn parent_texture_identity(&self) -> GpuWorkResourceId {
        self.record.parent_texture_identity()
    }
}

/// Opaque context/device-generation-bound realization of one admitted canonical WGSL program.
#[derive(Clone)]
pub struct GpuRealizedProgram {
    pub(crate) record: Arc<crate::backend::ProgramRealizationRecord>,
}

impl GpuRealizedProgram {
    pub(crate) fn from_record(record: Arc<crate::backend::ProgramRealizationRecord>) -> Self {
        Self { record }
    }

    pub fn affinity(&self) -> GpuContextAffinity {
        self.record.affinity()
    }

    pub fn descriptor(&self) -> &GpuProgramDescriptor {
        self.record.descriptor()
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
}

impl fmt::Debug for GpuRealizedProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuRealizedProgram")
            .field("affinity", &self.affinity())
            .field("source", self.descriptor().source().identity())
            .finish_non_exhaustive()
    }
}

/// Opaque context/device-generation-bound realization of one typed bind-group layout.
#[derive(Clone)]
pub struct GpuRealizedBindGroupLayout {
    pub(crate) record: Arc<crate::backend::BindGroupLayoutRealizationRecord>,
}

impl GpuRealizedBindGroupLayout {
    pub(crate) fn from_record(
        record: Arc<crate::backend::BindGroupLayoutRealizationRecord>,
    ) -> Self {
        Self { record }
    }

    pub fn affinity(&self) -> GpuContextAffinity {
        self.record.affinity()
    }

    pub fn descriptor(&self) -> &GpuBindGroupLayoutDescriptor {
        self.record.descriptor()
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
}

impl fmt::Debug for GpuRealizedBindGroupLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuRealizedBindGroupLayout")
            .field("affinity", &self.affinity())
            .field("group", &self.descriptor().group())
            .finish_non_exhaustive()
    }
}

/// Opaque context/device-generation-bound realization of one typed pipeline layout.
#[derive(Clone)]
pub struct GpuRealizedPipelineLayout {
    pub(crate) record: Arc<crate::backend::PipelineLayoutRealizationRecord>,
}

impl GpuRealizedPipelineLayout {
    pub(crate) fn from_record(
        record: Arc<crate::backend::PipelineLayoutRealizationRecord>,
    ) -> Self {
        Self { record }
    }

    pub fn affinity(&self) -> GpuContextAffinity {
        self.record.affinity()
    }

    pub fn descriptor(&self) -> &GpuPipelineLayoutDescriptor {
        self.record.descriptor()
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
}

impl fmt::Debug for GpuRealizedPipelineLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuRealizedPipelineLayout")
            .field("affinity", &self.affinity())
            .field("groups", &self.descriptor().groups().len())
            .finish_non_exhaustive()
    }
}

/// Opaque context/device-generation-bound realization of one typed bind group.
#[derive(Clone)]
pub struct GpuRealizedBindGroup {
    pub(crate) record: Arc<crate::backend::BindGroupRealizationRecord>,
}

impl GpuRealizedBindGroup {
    pub(crate) fn from_record(record: Arc<crate::backend::BindGroupRealizationRecord>) -> Self {
        Self { record }
    }

    pub fn affinity(&self) -> GpuContextAffinity {
        self.record.affinity()
    }

    pub fn layout_descriptor(&self) -> &GpuBindGroupLayoutDescriptor {
        self.record.layout_descriptor()
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.record, &other.record)
    }
}

impl fmt::Debug for GpuRealizedBindGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuRealizedBindGroup")
            .field("affinity", &self.affinity())
            .field("group", &self.layout_descriptor().group())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_explicit_policies_are_nonzero_total_record_bounds() {
        assert_eq!(
            GpuResourceRealizationPolicy::default().max_records(),
            DEFAULT_MAX_RESOURCE_REALIZATION_RECORDS
        );
        let explicit = GpuResourceRealizationPolicy::new(NonZeroUsize::new(7).unwrap());
        assert_eq!(explicit.max_records().get(), 7);
    }

    #[test]
    fn capacity_error_keeps_record_pressure_structured() {
        let resource = {
            let mut allocator = super::super::GpuWorkResourceIdAllocator::new();
            allocator.allocate().unwrap()
        };
        let error =
            GpuResourceRealizationError::capacity(resource, 3, NonZeroUsize::new(3).unwrap());

        assert_eq!(
            error.category(),
            GpuResourceRealizationErrorCategory::RegistryCapacityExceeded
        );
        assert_eq!(error.retained_records(), Some(3));
        assert_eq!(error.max_records().map(NonZeroUsize::get), Some(3));
        assert!(error.to_string().contains("records: 3/3"));
    }

    #[test]
    fn realized_handles_remain_clone_only_and_expose_no_raw_backend_contract() {
        let source = std::fs::read_to_string(file!()).expect("realization source must be readable");
        for name in [
            "GpuRealizedBuffer",
            "GpuRealizedTexture",
            "GpuRealizedTextureView",
            "GpuRealizedSampler",
            "GpuRealizedQuerySet",
        ] {
            assert!(source.contains(&format!("{name},")));
        }
        assert!(!source.contains(&["impl Copy", " for GpuRealized"].concat()));
        assert!(!source.contains(&["pub fn as", "_raw"].concat()));
        assert!(!source.contains(&["impl Deref", " for GpuRealized"].concat()));
        assert!(!source.contains(&["impl AsRef", "<"].concat()));
    }
}
