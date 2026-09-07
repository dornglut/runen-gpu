use runen_gpu::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements,
    GpuProgramContractCause, GpuSpecializationDeclaration, GpuSpecializationEntry,
    GpuSpecializationF32, GpuSpecializationKey, GpuSpecializationSchema, GpuSpecializationValue,
    GpuSpecializationValueSet, GpuSpecializationValueType,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn key(value: &str) -> GpuSpecializationKey {
    GpuSpecializationKey::new(value).expect("test specialization key should be valid")
}

fn declaration(
    key_value: &str,
    value_type: GpuSpecializationValueType,
    default: Option<GpuSpecializationValue>,
) -> GpuSpecializationDeclaration {
    GpuSpecializationDeclaration::new(
        key(key_value),
        value_type,
        default,
        GpuCapabilityRequirements::new(),
    )
    .expect("test specialization declaration should be valid")
}

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn specialization_keys_reject_invalid_source_identifiers() {
    for invalid in ["", "9iterations", "has-dash", " leading", "trailing "] {
        let error = GpuSpecializationKey::new(invalid)
            .expect_err("invalid source identifiers must be rejected");
        assert_eq!(
            error.cause(),
            GpuProgramContractCause::InvalidSpecializationKey
        );
    }
}

#[test]
fn specialization_f32_is_finite_and_normalizes_negative_zero() {
    let positive_zero = GpuSpecializationF32::try_new(0.0).unwrap();
    let negative_zero = GpuSpecializationF32::try_new(-0.0).unwrap();
    assert_eq!(positive_zero, negative_zero);
    assert_eq!(positive_zero.canonical_bits(), 0.0f32.to_bits());

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = GpuSpecializationF32::try_new(invalid)
            .expect_err("non-finite specialization values must be rejected");
        assert_eq!(
            error.cause(),
            GpuProgramContractCause::InvalidSpecializationValue
        );
    }
}

#[test]
fn specialization_schema_normalizes_order_and_retains_requirements() {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();
    let compute = GpuSpecializationDeclaration::new(
        key("workgroup_size"),
        GpuSpecializationValueType::U32,
        Some(GpuSpecializationValue::U32(64)),
        requirements,
    )
    .unwrap();
    let enabled = declaration(
        "enabled",
        GpuSpecializationValueType::Bool,
        Some(GpuSpecializationValue::Bool(true)),
    );

    let schema = GpuSpecializationSchema::new([compute.clone(), enabled.clone()]).unwrap();
    let reversed = GpuSpecializationSchema::new([enabled, compute]).unwrap();
    assert_eq!(schema, reversed);
    assert_eq!(hash_of(&schema), hash_of(&reversed));
    assert_eq!(
        schema
            .declarations()
            .map(|declaration| declaration.key().as_str())
            .collect::<Vec<_>>(),
        ["enabled", "workgroup_size"]
    );
    assert!(
        schema
            .declaration(&key("workgroup_size"))
            .unwrap()
            .requirement_implications()
            .get(GpuCapabilityFeature::Compute)
            .is_some()
    );
}

#[test]
fn specialization_requirements_participate_in_semantic_identity() {
    let mut compute_requirement = GpuCapabilityRequirements::new();
    compute_requirement
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();
    let with_requirement = GpuSpecializationDeclaration::new(
        key("enabled"),
        GpuSpecializationValueType::Bool,
        Some(GpuSpecializationValue::Bool(true)),
        compute_requirement,
    )
    .unwrap();
    let without_requirement = declaration(
        "enabled",
        GpuSpecializationValueType::Bool,
        Some(GpuSpecializationValue::Bool(true)),
    );

    let left = GpuSpecializationSchema::new([with_requirement]).unwrap();
    let right = GpuSpecializationSchema::new([without_requirement]).unwrap();

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn specialization_value_sets_fill_defaults_and_ignore_insertion_order() {
    let schema = GpuSpecializationSchema::new([
        declaration(
            "enabled",
            GpuSpecializationValueType::Bool,
            Some(GpuSpecializationValue::Bool(false)),
        ),
        declaration("iterations", GpuSpecializationValueType::U32, None),
    ])
    .unwrap();

    let first = GpuSpecializationValueSet::new(
        schema.clone(),
        [GpuSpecializationEntry::new(
            key("iterations"),
            GpuSpecializationValue::U32(8),
        )],
    )
    .unwrap();
    let second = GpuSpecializationValueSet::new(
        schema,
        [
            GpuSpecializationEntry::new(key("iterations"), GpuSpecializationValue::U32(8)),
            GpuSpecializationEntry::new(key("enabled"), GpuSpecializationValue::Bool(false)),
        ],
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(hash_of(&first), hash_of(&second));
    assert_eq!(
        first.value(&key("enabled")),
        Some(GpuSpecializationValue::Bool(false))
    );
}

#[test]
fn specialization_schema_rejects_duplicate_keys_and_wrong_defaults() {
    let duplicate = declaration("enabled", GpuSpecializationValueType::Bool, None);
    let error = GpuSpecializationSchema::new([duplicate.clone(), duplicate])
        .expect_err("duplicate schema keys must be rejected");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::DuplicateSpecializationKey
    );

    let error = GpuSpecializationDeclaration::new(
        key("enabled"),
        GpuSpecializationValueType::Bool,
        Some(GpuSpecializationValue::U32(1)),
        GpuCapabilityRequirements::new(),
    )
    .expect_err("default type must match the schema");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch
    );
}

#[test]
fn specialization_value_sets_reject_unknown_duplicate_missing_and_wrong_types() {
    let schema = GpuSpecializationSchema::new([declaration(
        "iterations",
        GpuSpecializationValueType::U32,
        None,
    )])
    .unwrap();

    let unknown = GpuSpecializationValueSet::new(
        schema.clone(),
        [GpuSpecializationEntry::new(
            key("other"),
            GpuSpecializationValue::U32(1),
        )],
    )
    .expect_err("unknown keys must be rejected");
    assert_eq!(
        unknown.cause(),
        GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch
    );

    let duplicate = GpuSpecializationValueSet::new(
        schema.clone(),
        [
            GpuSpecializationEntry::new(key("iterations"), GpuSpecializationValue::U32(1)),
            GpuSpecializationEntry::new(key("iterations"), GpuSpecializationValue::U32(2)),
        ],
    )
    .expect_err("duplicate values must be rejected");
    assert_eq!(
        duplicate.cause(),
        GpuProgramContractCause::DuplicateSpecializationKey
    );

    let missing = GpuSpecializationValueSet::new(schema.clone(), [])
        .expect_err("required values must be present");
    assert_eq!(
        missing.cause(),
        GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch
    );

    let wrong_type = GpuSpecializationValueSet::new(
        schema,
        [GpuSpecializationEntry::new(
            key("iterations"),
            GpuSpecializationValue::I32(1),
        )],
    )
    .expect_err("value types must match the schema");
    assert_eq!(
        wrong_type.cause(),
        GpuProgramContractCause::SpecializationUnknownMissingOrTypeMismatch
    );
}

#[test]
fn specialization_schema_rejects_conflicting_capability_implications() {
    let mut required = GpuCapabilityRequirements::new();
    required
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();
    let mut disabled = GpuCapabilityRequirements::new();
    disabled
        .insert(GpuCapabilityRequirement::Disabled(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();

    let first = GpuSpecializationDeclaration::new(
        key("enabled"),
        GpuSpecializationValueType::Bool,
        Some(GpuSpecializationValue::Bool(true)),
        required,
    )
    .unwrap();
    let second = GpuSpecializationDeclaration::new(
        key("iterations"),
        GpuSpecializationValueType::U32,
        Some(GpuSpecializationValue::U32(1)),
        disabled,
    )
    .unwrap();

    let error = GpuSpecializationSchema::new([first, second])
        .expect_err("conflicting capability implications must be rejected");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::SpecializationRequirementConflict
    );
}

#[test]
fn default_values_do_not_require_override_support() {
    let schema = GpuSpecializationSchema::new([declaration(
        "iterations",
        GpuSpecializationValueType::U32,
        Some(GpuSpecializationValue::U32(4)),
    )])
    .unwrap();
    let values = GpuSpecializationValueSet::new(
        schema,
        [GpuSpecializationEntry::new(
            key("iterations"),
            GpuSpecializationValue::U32(4),
        )],
    )
    .unwrap();

    assert!(!values.requires_override_support());
    values
        .validate_override_support(false)
        .expect("effective defaults do not require override support");
}

#[test]
fn unsupported_non_default_overrides_are_rejected_explicitly() {
    let schema = GpuSpecializationSchema::new([declaration(
        "iterations",
        GpuSpecializationValueType::U32,
        Some(GpuSpecializationValue::U32(4)),
    )])
    .unwrap();
    let values = GpuSpecializationValueSet::new(
        schema,
        [GpuSpecializationEntry::new(
            key("iterations"),
            GpuSpecializationValue::U32(8),
        )],
    )
    .unwrap();

    assert!(values.requires_override_support());
    let error = values
        .validate_override_support(false)
        .expect_err("a non-default override requires explicit backend-path support");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::SpecializationOverridesUnsupported
    );
    values
        .validate_override_support(true)
        .expect("a supporting backend path may consume the override");
}
