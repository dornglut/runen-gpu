use core::fmt;

use super::{
    GpuCapabilityFeature, GpuPreparedWorkNodeId, GpuResourceProvenance, GpuWorkNodeId,
    GpuWorkResourceId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAccessCause {
    ZeroRange,
    ArithmeticOverflow,
    OutOfBounds,
    InvalidDescriptorUsage,
    InvalidTextureAspect,
    InvalidViewIntersection,
    InvalidD3Interpretation,
    ParentLeaseMismatch,
    WrongResourceKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuAccessError {
    Invalid {
        operation: &'static str,
        label: String,
        resource: Option<GpuWorkResourceId>,
        cause: GpuAccessCause,
        correction: &'static str,
    },
}

impl GpuAccessError {
    pub(crate) fn invalid(
        operation: &'static str,
        label: impl Into<String>,
        resource: Option<GpuWorkResourceId>,
        cause: GpuAccessCause,
        correction: &'static str,
    ) -> Self {
        Self::Invalid {
            operation,
            label: label.into(),
            resource,
            cause,
            correction,
        }
    }

    pub const fn cause(&self) -> GpuAccessCause {
        match self {
            Self::Invalid { cause, .. } => *cause,
        }
    }

    pub const fn resource(&self) -> Option<GpuWorkResourceId> {
        match self {
            Self::Invalid { resource, .. } => *resource,
        }
    }
}

impl fmt::Display for GpuAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            resource,
            cause,
            correction,
        } = self;
        write!(f, "cannot {operation} '{label}'")?;
        if let Some(resource) = resource {
            write!(f, " for resource {resource}")?;
        }
        write!(f, ": {cause:?}; correction: {correction}")
    }
}

impl std::error::Error for GpuAccessError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWorkOperationCause {
    ZeroDispatch,
    ZeroDrawCount,
    InvalidDraw,
    InvalidCopyRegion,
    InvalidCopyLayout,
    NonFiniteClearValue,
    OutOfRangeClearValue,
    InvalidAttachment,
    InvalidMultisampleResolve,
    InvalidBufferZero,
    InvalidQueryRange,
    InvalidQueryResolution,
    QueryDestinationOverflow,
    QueryDestinationOutOfBounds,
    ZeroWork,
    OperationAccessContradiction,
    MechanicalCapabilityContradiction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuWorkOperationError {
    Invalid {
        operation: &'static str,
        label: String,
        resource: Option<GpuWorkResourceId>,
        cause: GpuWorkOperationCause,
        correction: &'static str,
        source: Option<Box<GpuAccessError>>,
    },
}

impl GpuWorkOperationError {
    pub(crate) fn invalid(
        operation: &'static str,
        label: impl Into<String>,
        resource: Option<GpuWorkResourceId>,
        cause: GpuWorkOperationCause,
        correction: &'static str,
    ) -> Self {
        Self::Invalid {
            operation,
            label: label.into(),
            resource,
            cause,
            correction,
            source: None,
        }
    }

    pub(crate) fn from_access(
        operation: &'static str,
        label: impl Into<String>,
        cause: GpuWorkOperationCause,
        correction: &'static str,
        source: GpuAccessError,
    ) -> Self {
        Self::Invalid {
            operation,
            label: label.into(),
            resource: source.resource(),
            cause,
            correction,
            source: Some(Box::new(source)),
        }
    }

    pub const fn cause(&self) -> GpuWorkOperationCause {
        match self {
            Self::Invalid { cause, .. } => *cause,
        }
    }

    pub const fn resource(&self) -> Option<GpuWorkResourceId> {
        match self {
            Self::Invalid { resource, .. } => *resource,
        }
    }
}

impl fmt::Display for GpuWorkOperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            resource,
            cause,
            correction,
            ..
        } = self;
        write!(f, "cannot {operation} '{label}'")?;
        if let Some(resource) = resource {
            write!(f, " for resource {resource}")?;
        }
        write!(f, ": {cause:?}; correction: {correction}")
    }
}

impl std::error::Error for GpuWorkOperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid { source, .. } => source
                .as_deref()
                .map(|source| source as &(dyn std::error::Error + 'static)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWorkAuthoringCause {
    InvalidLabel,
    InvalidCoverage,
    InvalidResourceKind,
    DuplicateResource,
    DuplicateInput,
    DuplicateImport,
    DuplicateOutput,
    DuplicateExportKey,
    UnknownIdentity,
    ForeignIdentity,
    IdentityExhausted,
    InvalidExplicitOrder,
    DuplicateExplicitOrder,
    IncompatibleSameNodeAccess,
    OperationAccessContradiction,
    MechanicalCapabilityContradiction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuWorkAuthoringErrorSource {
    Access(GpuAccessError),
    Operation(GpuWorkOperationError),
    Capability(GpuCapabilityRequirementError),
    Descriptor(GpuResourceDescriptorError),
}

impl fmt::Display for GpuWorkAuthoringErrorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Access(source) => source.fmt(f),
            Self::Operation(source) => source.fmt(f),
            Self::Capability(source) => source.fmt(f),
            Self::Descriptor(source) => source.fmt(f),
        }
    }
}

impl std::error::Error for GpuWorkAuthoringErrorSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Access(source) => Some(source),
            Self::Operation(source) => Some(source),
            Self::Capability(source) => Some(source),
            Self::Descriptor(source) => Some(source),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkAuthoringError {
    details: Box<GpuWorkAuthoringErrorDetails>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuWorkAuthoringErrorDetails {
    operation: &'static str,
    fragment_label: Option<String>,
    node_label: Option<String>,
    node: Option<GpuWorkNodeId>,
    resource: Option<GpuWorkResourceId>,
    cause: GpuWorkAuthoringCause,
    correction: &'static str,
    provenance: Option<GpuResourceProvenance>,
    source: Option<Box<GpuWorkAuthoringErrorSource>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuWorkAuthoringErrorContext {
    fragment_label: Option<String>,
    node_label: Option<String>,
    node: Option<GpuWorkNodeId>,
    resource: Option<GpuWorkResourceId>,
    provenance: Option<GpuResourceProvenance>,
}

impl GpuWorkAuthoringErrorContext {
    pub(crate) const fn new(
        fragment_label: Option<String>,
        node_label: Option<String>,
        node: Option<GpuWorkNodeId>,
        resource: Option<GpuWorkResourceId>,
        provenance: Option<GpuResourceProvenance>,
    ) -> Self {
        Self {
            fragment_label,
            node_label,
            node,
            resource,
            provenance,
        }
    }
}

impl GpuWorkAuthoringError {
    pub(crate) fn invalid(
        operation: &'static str,
        context: GpuWorkAuthoringErrorContext,
        cause: GpuWorkAuthoringCause,
        correction: &'static str,
    ) -> Self {
        Self {
            details: Box::new(GpuWorkAuthoringErrorDetails {
                operation,
                fragment_label: context.fragment_label,
                node_label: context.node_label,
                node: context.node,
                resource: context.resource,
                cause,
                correction,
                provenance: context.provenance,
                source: None,
            }),
        }
    }

    pub(crate) fn with_source(
        operation: &'static str,
        context: GpuWorkAuthoringErrorContext,
        cause: GpuWorkAuthoringCause,
        correction: &'static str,
        source: GpuWorkAuthoringErrorSource,
    ) -> Self {
        let mut error = Self::invalid(operation, context, cause, correction);
        error.details.source = Some(Box::new(source));
        error
    }

    pub const fn cause(&self) -> GpuWorkAuthoringCause {
        self.details.cause
    }

    pub const fn resource(&self) -> Option<GpuWorkResourceId> {
        self.details.resource
    }

    pub fn node(&self) -> Option<&GpuWorkNodeId> {
        self.details.node.as_ref()
    }
}

impl fmt::Display for GpuWorkAuthoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let details = &self.details;
        write!(f, "cannot {}", details.operation)?;
        if let Some(fragment) = &details.fragment_label {
            write!(f, " in fragment '{fragment}'")?;
        }
        if let Some(node_label) = &details.node_label {
            write!(f, " at node '{node_label}'")?;
        }
        if let Some(node) = &details.node {
            write!(f, " ({node})")?;
        }
        if let Some(resource) = details.resource {
            write!(f, " for resource {resource}")?;
        }
        if let Some(provenance) = &details.provenance {
            write!(f, " from '{}'", provenance.producer().as_str())?;
        }
        write!(
            f,
            ": {:?}; correction: {}",
            details.cause, details.correction
        )
    }
}

impl std::error::Error for GpuWorkAuthoringError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.details
            .source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

impl From<GpuAccessError> for GpuWorkAuthoringError {
    fn from(source: GpuAccessError) -> Self {
        let resource = source.resource();
        Self::with_source(
            "author checked GPU access",
            GpuWorkAuthoringErrorContext::new(None, None, None, resource, None),
            GpuWorkAuthoringCause::InvalidCoverage,
            "provide a checked range bounded by the same typed resource",
            GpuWorkAuthoringErrorSource::Access(source),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWorkGraphCause {
    ReadBeforeInitialization,
    IncompatibleSameNodeAccess,
    OperationAccessContradiction,
    MechanicalCapabilityContradiction,
    MissingCrossFragmentCausality,
    AmbiguousWriter,
    DuplicateExportKey,
    ImportExportMismatch,
    UnknownIdentity,
    ForeignIdentity,
    RedundantExplicitDataOrder,
    ExplicitOrderConflict,
    Cycle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuWorkGraphErrorSource {
    Authoring(GpuWorkAuthoringError),
    Operation(GpuWorkOperationError),
    Capability(GpuCapabilityRequirementError),
}

impl fmt::Display for GpuWorkGraphErrorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authoring(source) => source.fmt(f),
            Self::Operation(source) => source.fmt(f),
            Self::Capability(source) => source.fmt(f),
        }
    }
}

impl std::error::Error for GpuWorkGraphErrorSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authoring(source) => Some(source),
            Self::Operation(source) => Some(source),
            Self::Capability(source) => Some(source),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkGraphError {
    details: Box<GpuWorkGraphErrorDetails>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuWorkGraphErrorDetails {
    operation: &'static str,
    graph_label: String,
    fragment_label: Option<String>,
    node_label: Option<String>,
    node: Option<GpuPreparedWorkNodeId>,
    resource: Option<GpuWorkResourceId>,
    region: Option<String>,
    required_initialization: Option<super::GpuInitialCoverage>,
    cause: GpuWorkGraphCause,
    correction: &'static str,
    provenance: Option<GpuResourceProvenance>,
    source: Option<Box<GpuWorkGraphErrorSource>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuWorkGraphErrorContext {
    graph_label: String,
    fragment_label: Option<String>,
    node_label: Option<String>,
    node: Option<GpuPreparedWorkNodeId>,
    resource: Option<GpuWorkResourceId>,
    region: Option<String>,
    required_initialization: Option<super::GpuInitialCoverage>,
    provenance: Option<GpuResourceProvenance>,
}

impl GpuWorkGraphErrorContext {
    pub(crate) fn new(
        graph_label: impl Into<String>,
        fragment_label: Option<String>,
        node_label: Option<String>,
        node: Option<GpuPreparedWorkNodeId>,
        resource: Option<GpuWorkResourceId>,
        region: Option<String>,
        provenance: Option<GpuResourceProvenance>,
    ) -> Self {
        Self {
            graph_label: graph_label.into(),
            fragment_label,
            node_label,
            node,
            resource,
            region,
            required_initialization: None,
            provenance,
        }
    }

    pub(crate) fn with_required_initialization(
        mut self,
        required_initialization: super::GpuInitialCoverage,
    ) -> Self {
        self.required_initialization = Some(required_initialization);
        self
    }
}

impl GpuWorkGraphError {
    pub(crate) fn invalid(
        operation: &'static str,
        context: GpuWorkGraphErrorContext,
        cause: GpuWorkGraphCause,
        correction: &'static str,
    ) -> Self {
        Self {
            details: Box::new(GpuWorkGraphErrorDetails {
                operation,
                graph_label: context.graph_label,
                fragment_label: context.fragment_label,
                node_label: context.node_label,
                node: context.node,
                resource: context.resource,
                region: context.region,
                required_initialization: context.required_initialization,
                cause,
                correction,
                provenance: context.provenance,
                source: None,
            }),
        }
    }

    pub(crate) fn with_source(
        operation: &'static str,
        context: GpuWorkGraphErrorContext,
        cause: GpuWorkGraphCause,
        correction: &'static str,
        source: GpuWorkGraphErrorSource,
    ) -> Self {
        let mut error = Self::invalid(operation, context, cause, correction);
        error.details.source = Some(Box::new(source));
        error
    }

    pub const fn cause(&self) -> GpuWorkGraphCause {
        self.details.cause
    }

    pub const fn node(&self) -> Option<GpuPreparedWorkNodeId> {
        self.details.node
    }

    pub const fn resource(&self) -> Option<GpuWorkResourceId> {
        self.details.resource
    }

    /// Exact normalized storage coverage required by a failed initialization read, when applicable.
    ///
    /// This is canonical graph-initialization evidence, not a display string and not a second
    /// initialization model. Other graph failures return `None`.
    pub fn required_initialization(&self) -> Option<&super::GpuInitialCoverage> {
        self.details.required_initialization.as_ref()
    }
}

impl fmt::Display for GpuWorkGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let details = &self.details;
        write!(
            f,
            "cannot {} graph '{}'",
            details.operation, details.graph_label
        )?;
        if let Some(fragment) = &details.fragment_label {
            write!(f, " in fragment '{fragment}'")?;
        }
        if let Some(node_label) = &details.node_label {
            write!(f, " at node '{node_label}'")?;
        }
        if let Some(node) = details.node {
            write!(f, " ({node})")?;
        }
        if let Some(resource) = details.resource {
            write!(f, " for resource {resource}")?;
        }
        if let Some(region) = &details.region {
            write!(f, " over {region}")?;
        }
        if let Some(provenance) = &details.provenance {
            write!(f, " from '{}'", provenance.producer().as_str())?;
        }
        write!(
            f,
            ": {:?}; correction: {}",
            details.cause, details.correction
        )
    }
}

impl std::error::Error for GpuWorkGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.details
            .source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuCapabilityRequirementCause {
    ConflictingStrength,
    AmbiguousPreferredFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuCapabilityRequirementError {
    Invalid {
        operation: &'static str,
        label: String,
        cause: GpuCapabilityRequirementCause,
        correction: &'static str,
    },
}

impl fmt::Display for GpuCapabilityRequirementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            cause,
            correction,
        } = self;
        write!(
            f,
            "cannot {operation} '{label}': {cause:?}; correction: {correction}"
        )
    }
}

impl std::error::Error for GpuCapabilityRequirementError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuCapabilityAdmissionCause {
    RequiredUnavailable,
    RequiredNotEnabled,
    DisabledEnabled,
    EnabledUnavailable,
    InvalidLimit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuCapabilityAdmissionError {
    Rejected {
        operation: &'static str,
        label: String,
        cause: GpuCapabilityAdmissionCause,
        feature: Option<GpuCapabilityFeature>,
        correction: &'static str,
    },
}

impl GpuCapabilityAdmissionError {
    pub const fn cause(&self) -> GpuCapabilityAdmissionCause {
        match self {
            Self::Rejected { cause, .. } => *cause,
        }
    }

    /// Exact normalized capability feature implicated by this admission rejection, when the
    /// rejection is feature-specific. Limit-domain rejections return `None`.
    pub const fn feature(&self) -> Option<GpuCapabilityFeature> {
        match self {
            Self::Rejected { feature, .. } => *feature,
        }
    }
}

impl fmt::Display for GpuCapabilityAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Rejected {
            operation,
            label,
            cause,
            correction,
            ..
        } = self;
        write!(
            f,
            "cannot {operation} '{label}': {cause:?}; correction: {correction}"
        )
    }
}

impl std::error::Error for GpuCapabilityAdmissionError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuResourceDescriptorCause {
    EmptyLabel,
    ZeroSize,
    ArithmeticOverflow,
    EmptyUsage,
    InvalidOwnership,
    InvalidReconstruction,
    InvalidMemoryIntent,
    InvalidInitialization,
    InitializationLengthMismatch,
    InvalidExtent,
    InvalidMipCount,
    InvalidSampleCount,
    InvalidFormatUsage,
    InvalidAspect,
    InvalidRowLayout,
    InsufficientTextureData,
    ParentLeaseMismatch,
    SubresourceOutOfBounds,
    IncompatibleViewFormat,
    IncompatibleViewDimension,
    InvalidLodRange,
    InvalidQueryCount,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuResourceDescriptorError {
    Invalid {
        operation: &'static str,
        label: String,
        cause: GpuResourceDescriptorCause,
        correction: &'static str,
    },
}

impl GpuResourceDescriptorError {
    pub(crate) fn invalid(
        operation: &'static str,
        label: impl Into<String>,
        cause: GpuResourceDescriptorCause,
        correction: &'static str,
    ) -> Self {
        Self::Invalid {
            operation,
            label: label.into(),
            cause,
            correction,
        }
    }

    pub fn cause(&self) -> GpuResourceDescriptorCause {
        match self {
            Self::Invalid { cause, .. } => *cause,
        }
    }
}

impl fmt::Display for GpuResourceDescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            cause,
            correction,
        } = self;
        write!(
            f,
            "cannot {operation} '{label}': {cause:?}; correction: {correction}"
        )
    }
}

impl std::error::Error for GpuResourceDescriptorError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHandleCause {
    WrongKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuHandleError {
    Invalid {
        operation: &'static str,
        label: String,
        cause: GpuHandleCause,
        correction: &'static str,
    },
}

impl fmt::Display for GpuHandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            cause,
            correction,
        } = self;
        write!(
            f,
            "cannot {operation} '{label}': {cause:?}; correction: {correction}"
        )
    }
}

impl std::error::Error for GpuHandleError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDataPreparationCause {
    ZeroLength,
    InvalidAlignment,
    InvalidStride,
    InvalidElementCount,
    ArithmeticOverflow,
    LengthMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuDataPreparationError {
    Invalid {
        operation: &'static str,
        label: String,
        cause: GpuDataPreparationCause,
        correction: &'static str,
    },
}

impl GpuDataPreparationError {
    pub(crate) fn invalid(
        operation: &'static str,
        label: impl Into<String>,
        cause: GpuDataPreparationCause,
        correction: &'static str,
    ) -> Self {
        Self::Invalid {
            operation,
            label: label.into(),
            cause,
            correction,
        }
    }
}

impl fmt::Display for GpuDataPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            cause,
            correction,
        } = self;
        write!(
            f,
            "cannot {operation} '{label}': {cause:?}; correction: {correction}"
        )
    }
}

impl std::error::Error for GpuDataPreparationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuReadbackDecodeCause {
    InvalidLength,
    InvalidFormat,
    DecoderRejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuReadbackDecodeError {
    Invalid {
        operation: &'static str,
        label: String,
        cause: GpuReadbackDecodeCause,
        correction: &'static str,
    },
}

impl fmt::Display for GpuReadbackDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Invalid {
            operation,
            label,
            cause,
            correction,
        } = self;
        write!(
            f,
            "cannot {operation} '{label}': {cause:?}; correction: {correction}"
        )
    }
}

impl std::error::Error for GpuReadbackDecodeError {}
