use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::entry_point::GpuEntryPointName;
use super::{GpuBlendSource, GpuFragmentOutputLocation, GpuShaderIoLocation};
use super::builtin::{
    GpuFragmentOutputBuiltin, GpuVertexInputBuiltin, normalize_fragment_output_builtins,
    normalize_vertex_input_builtins,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GpuShaderIoSignature {
    locations: Vec<GpuShaderIoLocation>,
}

impl GpuShaderIoSignature {
    fn new(
        role: &'static str,
        locations: impl IntoIterator<Item = GpuShaderIoLocation>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut locations = locations.into_iter().collect::<Vec<_>>();
        locations.sort_by_key(|location| location.location());
        if let Some(duplicate) = locations
            .windows(2)
            .find(|pair| pair[0].location() == pair[1].location())
            .map(|pair| pair[0].location())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU shader-stage IO signature",
                format!("{role} location={duplicate}"),
                GpuProgramContractCause::StageIoSignatureInvalid,
                "provide each shader location exactly once",
            ));
        }
        Ok(Self { locations })
    }

    fn locations(&self) -> impl ExactSizeIterator<Item = &GpuShaderIoLocation> {
        self.locations.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GpuFragmentOutputSignature {
    locations: Vec<GpuFragmentOutputLocation>,
}

impl GpuFragmentOutputSignature {
    fn new(
        role: &'static str,
        locations: impl IntoIterator<Item = GpuFragmentOutputLocation>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut locations = locations.into_iter().collect::<Vec<_>>();
        locations.sort_by_key(|location| (location.location(), location.blend_source()));

        if locations.iter().any(|location| location.blend_source().is_some()) {
            let valid_pair = locations.len() == 2
                && locations[0].location() == 0
                && locations[1].location() == 0
                && locations[0].blend_source() == Some(GpuBlendSource::Primary)
                && locations[1].blend_source() == Some(GpuBlendSource::Secondary)
                && locations[0].value_type() == locations[1].value_type();
            if !valid_pair {
                return Err(GpuProgramContractError::invalid(
                    "construct GPU shader-stage IO signature",
                    format!("{role} locations={locations:?}"),
                    GpuProgramContractCause::StageIoSignatureInvalid,
                    "use exactly the location-0 Primary/Secondary pair with identical value types for dual-source fragment output",
                ));
            }
        } else if let Some(duplicate) = locations
            .windows(2)
            .find(|pair| pair[0].location() == pair[1].location())
            .map(|pair| pair[0].location())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU shader-stage IO signature",
                format!("{role} location={duplicate}"),
                GpuProgramContractCause::StageIoSignatureInvalid,
                "provide each ordinary fragment-output location exactly once",
            ));
        }

        Ok(Self { locations })
    }

    fn locations(&self) -> impl ExactSizeIterator<Item = &GpuFragmentOutputLocation> {
        self.locations.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuExpectedVertexInputSignature {
    entry_point: GpuEntryPointName,
    signature: GpuShaderIoSignature,
}

impl GpuExpectedVertexInputSignature {
    pub fn new(
        entry_point: GpuEntryPointName,
        locations: impl IntoIterator<Item = GpuShaderIoLocation>,
    ) -> Result<Self, GpuProgramContractError> {
        Ok(Self {
            entry_point,
            signature: GpuShaderIoSignature::new("expected vertex input", locations)?,
        })
    }

    pub fn entry_point(&self) -> &GpuEntryPointName {
        &self.entry_point
    }

    pub fn locations(&self) -> impl ExactSizeIterator<Item = &GpuShaderIoLocation> {
        self.signature.locations()
    }
}

/// Compiler-observed vertex-input locations for one admitted entry point.
/// Supported builtins are validated during canonical-WGSL admission but are not retained because
/// render-pipeline vertex-buffer state has no caller-authored builtin expectation to compare.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GpuObservedVertexInputSignature {
    entry_point: GpuEntryPointName,
    signature: GpuShaderIoSignature,
}

impl GpuObservedVertexInputSignature {
    pub(crate) fn new(
        entry_point: GpuEntryPointName,
        locations: impl IntoIterator<Item = GpuShaderIoLocation>,
        builtins: impl IntoIterator<Item = GpuVertexInputBuiltin>,
    ) -> Result<Self, GpuProgramContractError> {
        normalize_vertex_input_builtins(builtins)?;
        Ok(Self {
            entry_point,
            signature: GpuShaderIoSignature::new("observed vertex input", locations)?,
        })
    }

    pub(crate) fn entry_point(&self) -> &GpuEntryPointName {
        &self.entry_point
    }

    pub(crate) fn locations(&self) -> impl ExactSizeIterator<Item = &GpuShaderIoLocation> {
        self.signature.locations()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuExpectedFragmentOutputSignature {
    entry_point: GpuEntryPointName,
    signature: GpuFragmentOutputSignature,
}

impl GpuExpectedFragmentOutputSignature {
    pub fn new(
        entry_point: GpuEntryPointName,
        locations: impl IntoIterator<Item = GpuFragmentOutputLocation>,
    ) -> Result<Self, GpuProgramContractError> {
        Ok(Self {
            entry_point,
            signature: GpuFragmentOutputSignature::new("expected fragment output", locations)?,
        })
    }

    pub fn entry_point(&self) -> &GpuEntryPointName {
        &self.entry_point
    }

    pub fn locations(&self) -> impl ExactSizeIterator<Item = &GpuFragmentOutputLocation> {
        self.signature.locations()
    }
}

/// Compiler-observed fragment-output locations for one admitted entry point.
/// Supported builtins are validated during canonical-WGSL admission but are not retained because
/// render-pipeline fragment-target state has no caller-authored builtin expectation to compare.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GpuObservedFragmentOutputSignature {
    entry_point: GpuEntryPointName,
    signature: GpuFragmentOutputSignature,
}

impl GpuObservedFragmentOutputSignature {
    pub(crate) fn new(
        entry_point: GpuEntryPointName,
        locations: impl IntoIterator<Item = GpuFragmentOutputLocation>,
        builtins: impl IntoIterator<Item = GpuFragmentOutputBuiltin>,
    ) -> Result<Self, GpuProgramContractError> {
        normalize_fragment_output_builtins(builtins)?;
        Ok(Self {
            entry_point,
            signature: GpuFragmentOutputSignature::new("observed fragment output", locations)?,
        })
    }

    pub(crate) fn entry_point(&self) -> &GpuEntryPointName {
        &self.entry_point
    }

    pub(crate) fn locations(&self) -> impl ExactSizeIterator<Item = &GpuFragmentOutputLocation> {
        self.signature.locations()
    }
}
