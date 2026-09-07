use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::key::GpuBindingKey;
use super::kind::GpuBindingKind;
use super::stage::GpuShaderStages;
use core::num::{NonZeroU32, NonZeroU64};

const MAX_DIAGNOSTIC_FIELD_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBindingProvenance {
    producer: String,
    detail: Option<String>,
}

impl GpuBindingProvenance {
    pub fn new(
        producer: impl Into<String>,
        detail: Option<String>,
    ) -> Result<Self, GpuProgramContractError> {
        let producer = producer.into();
        validate_diagnostic_text("binding provenance producer", &producer)?;
        if let Some(detail) = detail.as_deref() {
            validate_diagnostic_text("binding provenance detail", detail)?;
        }
        Ok(Self { producer, detail })
    }

    pub fn producer(&self) -> &str {
        self.producer.as_str()
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

#[derive(Debug, Clone)]
pub struct GpuBindingDeclaration {
    key: GpuBindingKey,
    visibility: GpuShaderStages,
    kind: GpuBindingKind,
    array_count: Option<NonZeroU32>,
    compiler_required_minimum_size: Option<NonZeroU64>,
    label: String,
    provenance: GpuBindingProvenance,
}

impl GpuBindingDeclaration {
    /// Constructs an explicit typed layout declaration.
    ///
    /// The `GpuBindingKind` minimum-size field is host/layout policy. Compiler-required
    /// shader minima are attached only by canonical-WGSL program admission.
    pub fn new(
        key: GpuBindingKey,
        visibility: GpuShaderStages,
        kind: GpuBindingKind,
        array_count: Option<NonZeroU32>,
        label: impl Into<String>,
        provenance: GpuBindingProvenance,
    ) -> Result<Self, GpuProgramContractError> {
        Self::from_program_analysis(key, visibility, kind, array_count, None, label, provenance)
    }

    pub(crate) fn from_program_analysis(
        key: GpuBindingKey,
        visibility: GpuShaderStages,
        kind: GpuBindingKind,
        array_count: Option<NonZeroU32>,
        compiler_required_minimum_size: Option<NonZeroU64>,
        label: impl Into<String>,
        provenance: GpuBindingProvenance,
    ) -> Result<Self, GpuProgramContractError> {
        let label = label.into();
        validate_diagnostic_text("binding label", &label)?;
        if let (Some(host), Some(compiler)) =
            (kind.minimum_buffer_size(), compiler_required_minimum_size)
        {
            if host < compiler {
                return Err(GpuProgramContractError::invalid(
                    "construct GPU binding declaration",
                    format!(
                        "binding {key}: host_minimum={} compiler_minimum={}",
                        host, compiler
                    ),
                    GpuProgramContractCause::BindingDeclarationInvalid,
                    "use a host/layout minimum at least as large as the compiler-required shader minimum",
                ));
            }
        }
        Ok(Self {
            key,
            visibility,
            kind,
            array_count,
            compiler_required_minimum_size,
            label,
            provenance,
        })
    }

    pub const fn key(&self) -> GpuBindingKey {
        self.key
    }

    pub const fn visibility(&self) -> GpuShaderStages {
        self.visibility
    }

    pub fn kind(&self) -> &GpuBindingKind {
        &self.kind
    }

    pub const fn array_count(&self) -> Option<NonZeroU32> {
        self.array_count
    }

    pub const fn compiler_required_minimum_size(&self) -> Option<NonZeroU64> {
        self.compiler_required_minimum_size
    }

    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    pub fn provenance(&self) -> &GpuBindingProvenance {
        &self.provenance
    }
}

impl PartialEq for GpuBindingDeclaration {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.visibility == other.visibility
            && self.kind == other.kind
            && self.array_count == other.array_count
            && self.compiler_required_minimum_size == other.compiler_required_minimum_size
    }
}

impl Eq for GpuBindingDeclaration {}

impl PartialOrd for GpuBindingDeclaration {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuBindingDeclaration {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (
            &self.key,
            &self.visibility,
            &self.kind,
            &self.array_count,
            &self.compiler_required_minimum_size,
        )
            .cmp(&(
                &other.key,
                &other.visibility,
                &other.kind,
                &other.array_count,
                &other.compiler_required_minimum_size,
            ))
    }
}

impl core::hash::Hash for GpuBindingDeclaration {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.key.hash(state);
        self.visibility.hash(state);
        self.kind.hash(state);
        self.array_count.hash(state);
        self.compiler_required_minimum_size.hash(state);
    }
}

fn validate_diagnostic_text(
    field: &'static str,
    value: &str,
) -> Result<(), GpuProgramContractError> {
    if value.is_empty()
        || value.len() > MAX_DIAGNOSTIC_FIELD_BYTES
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(GpuProgramContractError::invalid(
            "construct GPU binding declaration",
            field,
            GpuProgramContractCause::InvalidDiagnosticMetadata,
            "provide bounded non-empty diagnostic text without surrounding whitespace or control characters",
        ));
    }
    Ok(())
}
