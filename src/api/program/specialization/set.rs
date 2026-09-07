use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::{GpuSpecializationKey, GpuSpecializationSchema, GpuSpecializationValue};
use core::hash::Hash;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSpecializationEntry {
    key: GpuSpecializationKey,
    value: GpuSpecializationValue,
}

impl GpuSpecializationEntry {
    pub fn new(key: GpuSpecializationKey, value: GpuSpecializationValue) -> Self {
        Self { key, value }
    }

    pub fn key(&self) -> &GpuSpecializationKey {
        &self.key
    }

    pub const fn value(&self) -> GpuSpecializationValue {
        self.value
    }
}

#[derive(Debug)]
struct GpuSpecializationValueSetInner {
    schema: GpuSpecializationSchema,
    entries: Vec<GpuSpecializationEntry>,
}

#[derive(Debug, Clone)]
pub struct GpuSpecializationValueSet(Arc<GpuSpecializationValueSetInner>);

impl GpuSpecializationValueSet {
    pub fn new(
        schema: GpuSpecializationSchema,
        entries: impl IntoIterator<Item = GpuSpecializationEntry>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut supplied = entries.into_iter().collect::<Vec<_>>();
        supplied.sort_by(|left, right| left.key().cmp(right.key()));
        if let Some(duplicate) = supplied
            .windows(2)
            .find(|pair| pair[0].key() == pair[1].key())
            .map(|pair| pair[0].key())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU specialization value set",
                duplicate.to_string(),
                GpuProgramContractCause::DuplicateSpecializationKey,
                "provide each specialization value at most once",
            ));
        }

        for entry in &supplied {
            let Some(declaration) = schema.declaration(entry.key()) else {
                return Err(GpuProgramContractError::invalid(
                    "construct GPU specialization value set",
                    entry.key().to_string(),
                    GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch,
                    "remove the unknown value or declare its key in the specialization schema",
                ));
            };
            if entry.value().value_type() != declaration.value_type() {
                return Err(GpuProgramContractError::invalid(
                    "construct GPU specialization value set",
                    entry.key().to_string(),
                    GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch,
                    "make the supplied value type match the specialization schema",
                ));
            }
        }

        let mut normalized = Vec::with_capacity(schema.declarations().len());
        for declaration in schema.declarations() {
            match supplied.binary_search_by(|entry| entry.key().cmp(declaration.key())) {
                Ok(index) => normalized.push(supplied[index].clone()),
                Err(_) => {
                    let Some(default) = declaration.default() else {
                        return Err(GpuProgramContractError::invalid(
                            "construct GPU specialization value set",
                            declaration.key().to_string(),
                            GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch,
                            "supply the required specialization value or declare a default",
                        ));
                    };
                    normalized.push(GpuSpecializationEntry::new(
                        declaration.key().clone(),
                        default,
                    ));
                }
            }
        }

        Ok(Self(Arc::new(GpuSpecializationValueSetInner {
            schema,
            entries: normalized,
        })))
    }

    pub fn schema(&self) -> &GpuSpecializationSchema {
        &self.0.schema
    }

    pub fn entries(&self) -> impl ExactSizeIterator<Item = &GpuSpecializationEntry> {
        self.0.entries.iter()
    }

    pub fn requirements(&self) -> &crate::GpuCapabilityRequirements {
        self.0.schema.requirements()
    }

    pub fn value(&self, key: &GpuSpecializationKey) -> Option<GpuSpecializationValue> {
        self.0
            .entries
            .binary_search_by(|entry| entry.key().cmp(key))
            .ok()
            .map(|index| self.0.entries[index].value())
    }

    pub fn requires_override_support(&self) -> bool {
        self.0.entries.iter().any(|entry| {
            self.0
                .schema
                .declaration(entry.key())
                .is_some_and(|declaration| declaration.default() != Some(entry.value()))
        })
    }

    pub fn validate_override_support(
        &self,
        override_constants_supported: bool,
    ) -> Result<(), GpuProgramContractError> {
        if override_constants_supported || !self.requires_override_support() {
            return Ok(());
        }
        Err(GpuProgramContractError::invalid(
            "validate GPU specialization override support",
            "specialization value set",
            GpuProgramContractCause::SpecializationOverridesUnsupported,
            "select a WGSL/backend path that consumes override constants or use only declared defaults",
        ))
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for GpuSpecializationValueSet {
    fn eq(&self, other: &Self) -> bool {
        self.0.schema == other.0.schema && self.0.entries == other.0.entries
    }
}

impl Eq for GpuSpecializationValueSet {}

impl Hash for GpuSpecializationValueSet {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.schema.hash(state);
        self.0.entries.hash(state);
    }
}
