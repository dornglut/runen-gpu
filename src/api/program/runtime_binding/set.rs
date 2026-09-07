use super::{
    GpuRuntimeBindingDeviceFacts, GpuRuntimeBindingResource, GpuRuntimeBindingValue,
    GpuValidatedBindGroupBindings,
};
use crate::{
    GpuBindingClass, GpuBindingDeclaration, GpuBufferAccess, GpuBufferAccessKind, GpuBufferRange,
    GpuPipelineLayoutDescriptor, GpuProgramContractCause, GpuProgramContractError,
    GpuResourceAccess, GpuSamplerUse, GpuStorageBufferAccess, GpuStorageTextureAccess,
    GpuTextureAccess, GpuTextureAccessKind, GpuTextureAccessResource, GpuTextureAspect,
};
use core::hash::{Hash, Hasher};
use std::collections::BTreeMap;

/// Complete logical runtime binding use for one pipeline invocation.
///
/// This owner is pipeline-layout shaped. Per-group resource compatibility remains owned by
/// [`GpuValidatedBindGroupBindings`]; this type owns complete group coverage, exact G3 resource
/// accesses after per-use dynamic offsets are applied, and binding-set-local writable-alias
/// validity. Admitted positional/dynamic limits, offset alignment, and texture-format capability
/// checks belong to contextual submission preparation and can be exercised explicitly through the
/// internal normalized-facts validation seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuRuntimeBindingSet {
    layout: GpuPipelineLayoutDescriptor,
    groups: Vec<GpuValidatedBindGroupBindings>,
    accesses: Vec<GpuResourceAccess>,
}

impl GpuRuntimeBindingSet {
    pub fn new(
        layout: GpuPipelineLayoutDescriptor,
        values: impl IntoIterator<Item = GpuRuntimeBindingValue>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut values_by_group = BTreeMap::<u32, Vec<GpuRuntimeBindingValue>>::new();
        for value in values {
            values_by_group
                .entry(value.key().group())
                .or_default()
                .push(value);
        }

        let mut groups = Vec::with_capacity(layout.groups().len());
        for group in layout.groups() {
            let values = values_by_group.remove(&group.group()).unwrap_or_default();
            groups.push(GpuValidatedBindGroupBindings::new(group.clone(), values)?);
        }

        if let Some((&group, _)) = values_by_group.first_key_value() {
            return Err(incompatible(
                format!("group {group}"),
                "provide runtime values only for groups declared by the exact pipeline layout",
            ));
        }

        let accesses = derive_effective_accesses(&groups)?;
        validate_writable_aliases(&accesses)?;
        Ok(Self {
            layout,
            groups,
            accesses,
        })
    }

    pub fn layout(&self) -> &GpuPipelineLayoutDescriptor {
        &self.layout
    }

    pub fn groups(&self) -> &[GpuValidatedBindGroupBindings] {
        &self.groups
    }

    pub fn group(&self, group: u32) -> Option<&GpuValidatedBindGroupBindings> {
        self.groups
            .binary_search_by_key(&group, |bindings| bindings.layout().group())
            .ok()
            .map(|index| &self.groups[index])
    }

    pub fn values(&self) -> impl Iterator<Item = &GpuRuntimeBindingValue> {
        self.groups
            .iter()
            .flat_map(GpuValidatedBindGroupBindings::values)
    }

    pub fn accesses(&self) -> &[GpuResourceAccess] {
        &self.accesses
    }

    pub(crate) fn required_bind_group_slots(&self) -> u64 {
        required_bind_group_slots(&self.layout)
    }

    pub(crate) fn validate_device_facts(
        &self,
        device_facts: &GpuRuntimeBindingDeviceFacts,
    ) -> Result<(), GpuProgramContractError> {
        validate_pipeline_binding_limits(&self.layout, device_facts)?;
        for group in &self.groups {
            group.validate_device_facts(device_facts)?;
        }
        Ok(())
    }
}

impl Hash for GpuRuntimeBindingSet {
    fn hash<State: Hasher>(&self, state: &mut State) {
        self.layout.hash(state);
        for group in &self.groups {
            group.layout().hash(state);
            group.values().for_each(|value| value.hash(state));
        }
    }
}

fn derive_effective_accesses(
    groups: &[GpuValidatedBindGroupBindings],
) -> Result<Vec<GpuResourceAccess>, GpuProgramContractError> {
    let mut accesses = Vec::new();
    for group in groups {
        for declaration in group.layout().bindings() {
            let value = group
                .value(declaration.key().binding())
                .ok_or_else(|| effective_access_error(declaration))?;
            for resource in value.resources() {
                accesses.push(derive_effective_access(declaration, resource)?);
            }
        }
    }
    Ok(accesses)
}

fn derive_effective_access(
    declaration: &GpuBindingDeclaration,
    resource: &GpuRuntimeBindingResource,
) -> Result<GpuResourceAccess, GpuProgramContractError> {
    match (declaration.kind().class(), resource) {
        (GpuBindingClass::UniformBuffer, GpuRuntimeBindingResource::Buffer(binding)) => {
            derive_buffer_access(declaration, binding, GpuBufferAccessKind::UniformRead)
        }
        (GpuBindingClass::StorageBuffer, GpuRuntimeBindingResource::Buffer(binding)) => {
            let kind = match declaration.kind().storage_buffer_access() {
                Some(GpuStorageBufferAccess::ReadOnly) => GpuBufferAccessKind::StorageRead,
                Some(GpuStorageBufferAccess::ReadWrite) => GpuBufferAccessKind::StorageReadWrite,
                None => return Err(effective_access_error(declaration)),
            };
            derive_buffer_access(declaration, binding, kind)
        }
        (GpuBindingClass::SampledTexture, GpuRuntimeBindingResource::TextureView(binding)) => {
            derive_texture_access(declaration, binding, GpuTextureAccessKind::SampledRead)
        }
        (GpuBindingClass::StorageTexture, GpuRuntimeBindingResource::TextureView(binding)) => {
            let kind = match declaration.kind().storage_texture_access() {
                Some(GpuStorageTextureAccess::ReadOnly) => GpuTextureAccessKind::StorageRead,
                Some(GpuStorageTextureAccess::WriteOnly) => GpuTextureAccessKind::StorageWrite,
                Some(GpuStorageTextureAccess::ReadWrite) => GpuTextureAccessKind::StorageReadWrite,
                None => return Err(effective_access_error(declaration)),
            };
            derive_texture_access(declaration, binding, kind)
        }
        (GpuBindingClass::Sampler, GpuRuntimeBindingResource::Sampler(handle)) => {
            Ok(GpuResourceAccess::Sampler(GpuSamplerUse::new(handle)))
        }
        _ => Err(effective_access_error(declaration)),
    }
}

fn derive_buffer_access(
    declaration: &GpuBindingDeclaration,
    binding: &super::GpuRuntimeBufferBinding,
    kind: GpuBufferAccessKind,
) -> Result<GpuResourceAccess, GpuProgramContractError> {
    let effective_offset = binding
        .checked_effective_offset()
        .ok_or_else(|| effective_access_error(declaration))?;
    let range = GpuBufferRange::new(binding.handle(), effective_offset, binding.size().get())
        .map_err(|_| effective_access_error(declaration))?;
    let access = GpuBufferAccess::new(binding.handle(), range, kind)
        .map_err(|_| effective_access_error(declaration))?;
    Ok(GpuResourceAccess::Buffer(access))
}

fn derive_texture_access(
    declaration: &GpuBindingDeclaration,
    binding: &super::GpuRuntimeTextureViewBinding,
    kind: GpuTextureAccessKind,
) -> Result<GpuResourceAccess, GpuProgramContractError> {
    let handle = binding.handle();
    let access = GpuTextureAccess::new(
        GpuTextureAccessResource::TextureView(handle.clone()),
        handle.descriptor().subresources(),
        kind,
    )
    .map_err(|_| effective_access_error(declaration))?;
    Ok(GpuResourceAccess::Texture(access))
}

fn validate_writable_aliases(
    accesses: &[GpuResourceAccess],
) -> Result<(), GpuProgramContractError> {
    for left_index in 0..accesses.len() {
        let left = &accesses[left_index];
        for right in &accesses[(left_index + 1)..] {
            if left.resource_identity() != right.resource_identity()
                || (!left.writes() && !right.writes())
                || !binding_accesses_overlap(left, right)
            {
                continue;
            }
            return Err(incompatible(
                format!(
                    "{} overlaps {}",
                    access_evidence(left),
                    access_evidence(right)
                ),
                "bind disjoint effective ranges/subresources whenever either binding is writable",
            ));
        }
    }
    Ok(())
}

fn binding_accesses_overlap(left: &GpuResourceAccess, right: &GpuResourceAccess) -> bool {
    match (left, right) {
        (GpuResourceAccess::Buffer(left), GpuResourceAccess::Buffer(right)) => {
            left.range().overlaps(right.range())
        }
        (GpuResourceAccess::Texture(left), GpuResourceAccess::Texture(right)) => {
            let parent_aspect = if left.normalized_texture().descriptor().format().is_depth() {
                GpuTextureAspect::DepthOnly
            } else {
                GpuTextureAspect::Color
            };
            left.normalized_subresources()
                .overlaps(right.normalized_subresources(), parent_aspect)
        }
        _ => false,
    }
}

fn access_evidence(access: &GpuResourceAccess) -> String {
    match access {
        GpuResourceAccess::Buffer(access) => format!(
            "buffer {:?} {:?} [{}..{})",
            access.resource_identity(),
            access.kind(),
            access.range().offset(),
            access.range().end()
        ),
        GpuResourceAccess::Texture(access) => {
            let range = access.normalized_subresources();
            format!(
                "texture {:?} {:?} mips=[{}..{}) layers=[{}..{}) aspect={:?}",
                access.resource_identity(),
                access.kind(),
                range.base_mip_level(),
                range.mip_end(),
                range.base_array_layer(),
                range.layer_end(),
                range.aspect()
            )
        }
        GpuResourceAccess::Query(access) => format!(
            "query {:?} {:?} [{}..{})",
            access.resource_identity(),
            access.kind(),
            access.range().first(),
            access.range().end()
        ),
        GpuResourceAccess::Sampler(access) => {
            format!("sampler {:?}", access.resource_identity())
        }
    }
}

fn effective_access_error(declaration: &GpuBindingDeclaration) -> GpuProgramContractError {
    incompatible(
        declaration.key().to_string(),
        "keep effective resource access consistent with the validated runtime binding declaration",
    )
}

fn validate_pipeline_binding_limits(
    layout: &GpuPipelineLayoutDescriptor,
    device_facts: &GpuRuntimeBindingDeviceFacts,
) -> Result<(), GpuProgramContractError> {
    let required_bind_group_slots = required_bind_group_slots(layout);
    if required_bind_group_slots > u64::from(device_facts.max_bind_groups()) {
        return Err(device_incompatible(
            "bind groups",
            "keep the highest pipeline bind-group index inside the admitted positional bind-group limit",
        ));
    }

    let mut dynamic_uniform_buffers = 0_u64;
    let mut dynamic_storage_buffers = 0_u64;

    for group in layout.groups() {
        for declaration in group.bindings() {
            if !declaration.kind().uses_dynamic_offset() {
                continue;
            }
            match declaration.kind().class() {
                GpuBindingClass::UniformBuffer => dynamic_uniform_buffers += 1,
                GpuBindingClass::StorageBuffer => dynamic_storage_buffers += 1,
                GpuBindingClass::SampledTexture
                | GpuBindingClass::StorageTexture
                | GpuBindingClass::Sampler => {
                    return Err(incompatible(
                        declaration.key().to_string(),
                        "use dynamic offsets only with uniform or storage buffer declarations",
                    ));
                }
            }
        }
    }

    if dynamic_uniform_buffers
        > u64::from(device_facts.max_dynamic_uniform_buffers_per_pipeline_layout())
    {
        return Err(device_incompatible(
            "dynamic uniform buffers",
            "reduce dynamic uniform-buffer declarations to the admitted pipeline-layout limit",
        ));
    }
    if dynamic_storage_buffers
        > u64::from(device_facts.max_dynamic_storage_buffers_per_pipeline_layout())
    {
        return Err(device_incompatible(
            "dynamic storage buffers",
            "reduce dynamic storage-buffer declarations to the admitted pipeline-layout limit",
        ));
    }
    Ok(())
}

fn required_bind_group_slots(layout: &GpuPipelineLayoutDescriptor) -> u64 {
    layout
        .groups()
        .map(|group| u64::from(group.group()) + 1)
        .max()
        .unwrap_or(0)
}

fn incompatible(label: impl Into<String>, correction: &'static str) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "construct runtime GPU binding set",
        label,
        GpuProgramContractCause::RuntimeBindingIncompatible,
        correction,
    )
}

fn device_incompatible(
    label: impl Into<String>,
    correction: &'static str,
) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "validate runtime GPU binding set against admitted device facts",
        label,
        GpuProgramContractCause::RuntimeBindingIncompatible,
        correction,
    )
}
