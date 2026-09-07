use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuProgramContractCause {
    InvalidEntryPointName,
    EntryPointMissing,
    DuplicateEntryPoint,
    CanonicalWgslInvalid,
    InvalidBindingKey,
    EmptyStageVisibility,
    BindingDeclarationInvalid,
    BindingRefinementInvalid,
    DuplicateBindingKey,
    ProgramInterfaceMismatch,
    BindGroupLayoutInvalid,
    DuplicateBindGroupLayout,
    RuntimeBindingIncompatible,
    StageIoSignatureInvalid,
    PipelineStageIoMismatch,
    InvalidSpecializationKey,
    InvalidSpecializationValue,
    DuplicateSpecializationKey,
    SpecializationUnknownMissingOrTypeMismatch,
    SpecializationRequirementConflict,
    SpecializationOverridesUnsupported,
    PipelineDescriptorInvalid,
    VertexInputStateInvalid,
    RenderAttachmentStateInvalid,
    RenderPrimitiveStateInvalid,
    RenderMultisampleStateInvalid,
    RenderPipelineStateInvalid,
    InvalidDiagnosticMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProgramContractError {
    operation: &'static str,
    label: String,
    cause: GpuProgramContractCause,
    detail: Option<String>,
    correction: &'static str,
}

impl GpuProgramContractError {
    pub(crate) fn invalid(
        operation: &'static str,
        label: impl Into<String>,
        cause: GpuProgramContractCause,
        correction: &'static str,
    ) -> Self {
        Self {
            operation,
            label: label.into(),
            cause,
            detail: None,
            correction,
        }
    }

    pub(crate) fn invalid_with_detail(
        operation: &'static str,
        label: impl Into<String>,
        cause: GpuProgramContractCause,
        detail: impl Into<String>,
        correction: &'static str,
    ) -> Self {
        Self {
            operation,
            label: label.into(),
            cause,
            detail: Some(detail.into()),
            correction,
        }
    }

    pub const fn cause(&self) -> GpuProgramContractCause {
        self.cause
    }

    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub const fn correction(&self) -> &'static str {
        self.correction
    }
}

impl fmt::Display for GpuProgramContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot {} '{}': {:?}",
            self.operation, self.label, self.cause
        )?;
        if let Some(detail) = self.detail.as_deref() {
            write!(formatter, "; detail: {detail}")?;
        }
        write!(formatter, "; correction: {}", self.correction)
    }
}

impl std::error::Error for GpuProgramContractError {}
