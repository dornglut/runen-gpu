use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::{GpuSpecializationKey, GpuSpecializationValue, GpuSpecializationValueType};
use crate::{GpuCapabilityRequirement, GpuCapabilityRequirements};
use core::hash::Hash;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GpuSpecializationDeclaration {
    key: GpuSpecializationKey,
    value_type: GpuSpecializationValueType,
    default: Option<GpuSpecializationValue>,
    requirement_implications: GpuCapabilityRequirements,
}

impl GpuSpecializationDeclaration {
    pub fn new(
        key: GpuSpecializationKey,
        value_type: GpuSpecializationValueType,
        default: Option<GpuSpecializationValue>,
        requirement_implications: GpuCapabilityRequirements,
    ) -> Result<Self, GpuProgramContractError> {
        if default.is_some_and(|value| value.value_type() != value_type) {
            return Err(GpuProgramContractError::invalid(
                "construct GPU specialization declaration",
                key.to_string(),
                GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch,
                "make the default value type match the declared specialization type",
            ));
        }
        Ok(Self {
            key,
            value_type,
            default,
            requirement_implications,
        })
    }

    pub fn key(&self) -> &GpuSpecializationKey {
        &self.key
    }

    pub const fn value_type(&self) -> GpuSpecializationValueType {
        self.value_type
    }

    pub const fn default(&self) -> Option<GpuSpecializationValue> {
        self.default
    }

    pub fn requirement_implications(&self) -> &GpuCapabilityRequirements {
        &self.requirement_implications
    }
}

impl PartialEq for GpuSpecializationDeclaration {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.value_type == other.value_type
            && self.default == other.default
            && self.requirement_implications == other.requirement_implications
    }
}

impl Eq for GpuSpecializationDeclaration {}

impl Hash for GpuSpecializationDeclaration {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.key.hash(state);
        self.value_type.hash(state);
        self.default.hash(state);
        self.requirement_implications.iter().len().hash(state);
        for requirement in self.requirement_implications.iter() {
            hash_requirement(requirement, state);
        }
    }
}

fn hash_requirement<State: core::hash::Hasher>(
    requirement: GpuCapabilityRequirement,
    state: &mut State,
) {
    match requirement {
        GpuCapabilityRequirement::Required(feature) => {
            0u8.hash(state);
            feature.hash(state);
        }
        GpuCapabilityRequirement::Preferred { feature, fallback } => {
            1u8.hash(state);
            feature.hash(state);
            fallback.hash(state);
        }
        GpuCapabilityRequirement::Disabled(feature) => {
            2u8.hash(state);
            feature.hash(state);
        }
    }
}

#[derive(Debug)]
struct GpuSpecializationSchemaInner {
    declarations: Vec<GpuSpecializationDeclaration>,
    requirements: GpuCapabilityRequirements,
}

#[derive(Debug, Clone)]
pub struct GpuSpecializationSchema(Arc<GpuSpecializationSchemaInner>);

impl GpuSpecializationSchema {
    pub fn new(
        declarations: impl IntoIterator<Item = GpuSpecializationDeclaration>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut declarations = declarations.into_iter().collect::<Vec<_>>();
        declarations.sort_by(|left, right| left.key().cmp(right.key()));
        if let Some(duplicate) = declarations
            .windows(2)
            .find(|pair| pair[0].key() == pair[1].key())
            .map(|pair| pair[0].key())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU specialization schema",
                duplicate.to_string(),
                GpuProgramContractCause::DuplicateSpecializationKey,
                "declare each specialization key exactly once",
            ));
        }
        let mut requirements = GpuCapabilityRequirements::new();
        for declaration in &declarations {
            for requirement in declaration.requirement_implications().iter() {
                requirements.insert(requirement).map_err(|error| {
                    GpuProgramContractError::invalid(
                        "construct GPU specialization schema",
                        format!("{}: {error}", declaration.key()),
                        GpuProgramContractCause::SpecializationRequirementConflict,
                        "remove conflicting capability implications from the specialization schema",
                    )
                })?;
            }
        }
        Ok(Self(Arc::new(GpuSpecializationSchemaInner {
            declarations,
            requirements,
        })))
    }

    pub fn declarations(&self) -> impl ExactSizeIterator<Item = &GpuSpecializationDeclaration> {
        self.0.declarations.iter()
    }

    pub fn declaration(&self, key: &GpuSpecializationKey) -> Option<&GpuSpecializationDeclaration> {
        self.0
            .declarations
            .binary_search_by(|declaration| declaration.key().cmp(key))
            .ok()
            .map(|index| &self.0.declarations[index])
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        &self.0.requirements
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for GpuSpecializationSchema {
    fn eq(&self, other: &Self) -> bool {
        self.0.declarations == other.0.declarations
    }
}

impl Eq for GpuSpecializationSchema {}

impl Hash for GpuSpecializationSchema {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.declarations.hash(state);
    }
}
