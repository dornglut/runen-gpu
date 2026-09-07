use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::interface::{
    GpuBindingClass, GpuBindingDeclaration, GpuSamplerClass, GpuStorageTextureAccess,
    GpuTextureSampleClass, GpuTextureViewDimension,
};
use super::super::layout::GpuBindGroupLayoutDescriptor;
use super::{
    GpuRuntimeBindingDeviceFacts, GpuRuntimeBindingResource, GpuRuntimeBindingValue,
    GpuRuntimeBufferBinding, GpuRuntimeTextureViewBinding,
};
use crate::{
    GpuBufferUsage, GpuFilterMode, GpuTextureDimension, GpuTextureFormat, GpuTextureUsage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuValidatedBindGroupBindings {
    layout: GpuBindGroupLayoutDescriptor,
    values: Vec<GpuRuntimeBindingValue>,
}

impl GpuValidatedBindGroupBindings {
    pub fn new(
        layout: GpuBindGroupLayoutDescriptor,
        values: impl IntoIterator<Item = GpuRuntimeBindingValue>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut values = values.into_iter().collect::<Vec<_>>();
        values.sort_by_key(GpuRuntimeBindingValue::key);

        if let Some(duplicate) = values
            .windows(2)
            .find(|pair| pair[0].key() == pair[1].key())
            .map(|pair| pair[0].key())
        {
            return Err(incompatible(
                duplicate.to_string(),
                "provide each runtime binding key exactly once",
            ));
        }

        if let Some(unexpected) = values.iter().find(|value| {
            value.key().group() != layout.group() || layout.binding(value.key().binding()).is_none()
        }) {
            return Err(incompatible(
                unexpected.key().to_string(),
                "provide values only for declarations in the exact bind-group layout",
            ));
        }

        for declaration in layout.bindings() {
            let value = values
                .binary_search_by_key(&declaration.key(), GpuRuntimeBindingValue::key)
                .ok()
                .map(|index| &values[index])
                .ok_or_else(|| {
                    incompatible(
                        declaration.key().to_string(),
                        "provide one runtime value for every declared binding",
                    )
                })?;

            let expected_count = declaration
                .array_count()
                .map_or(1, |count| count.get() as usize);
            if value.resources().len() != expected_count {
                return Err(incompatible(
                    declaration.key().to_string(),
                    "provide the exact fixed resource-array cardinality declared by the interface",
                ));
            }

            for resource in value.resources() {
                validate_resource_structure(declaration, resource)?;
            }
        }

        Ok(Self { layout, values })
    }

    pub fn layout(&self) -> &GpuBindGroupLayoutDescriptor {
        &self.layout
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = &GpuRuntimeBindingValue> {
        self.values.iter()
    }

    pub fn value(&self, binding: u32) -> Option<&GpuRuntimeBindingValue> {
        self.values
            .binary_search_by_key(&binding, |value| value.key().binding())
            .ok()
            .map(|index| &self.values[index])
    }

    pub(crate) fn validate_device_facts(
        &self,
        device_facts: &GpuRuntimeBindingDeviceFacts,
    ) -> Result<(), GpuProgramContractError> {
        for declaration in self.layout.bindings() {
            let value = self
                .value(declaration.key().binding())
                .expect("checked bind-group bindings retain every declaration");
            for resource in value.resources() {
                validate_resource_device(declaration, resource, device_facts)?;
            }
        }
        Ok(())
    }
}

fn validate_resource_structure(
    declaration: &GpuBindingDeclaration,
    resource: &GpuRuntimeBindingResource,
) -> Result<(), GpuProgramContractError> {
    let kind = *declaration.kind();
    match (kind.class(), resource) {
        (GpuBindingClass::UniformBuffer, GpuRuntimeBindingResource::Buffer(binding)) => {
            validate_buffer_structure(declaration, binding, GpuBufferUsage::Uniform)
        }
        (GpuBindingClass::StorageBuffer, GpuRuntimeBindingResource::Buffer(binding)) => {
            validate_buffer_structure(declaration, binding, GpuBufferUsage::Storage)
        }
        (GpuBindingClass::SampledTexture, GpuRuntimeBindingResource::TextureView(binding)) => {
            validate_sampled_texture_structure(declaration, binding)
        }
        (GpuBindingClass::StorageTexture, GpuRuntimeBindingResource::TextureView(binding)) => {
            validate_storage_texture_structure(declaration, binding)
        }
        (GpuBindingClass::Sampler, GpuRuntimeBindingResource::Sampler(handle)) => {
            validate_sampler(declaration, handle)
        }
        _ => Err(incompatible(
            declaration.key().to_string(),
            "provide a runtime resource whose typed kind matches the declaration",
        )),
    }
}

fn validate_resource_device(
    declaration: &GpuBindingDeclaration,
    resource: &GpuRuntimeBindingResource,
    device_facts: &GpuRuntimeBindingDeviceFacts,
) -> Result<(), GpuProgramContractError> {
    match (declaration.kind().class(), resource) {
        (GpuBindingClass::UniformBuffer, GpuRuntimeBindingResource::Buffer(binding)) => {
            let alignment = device_facts
                .uniform_buffer_offset_alignment()
                .ok_or_else(|| {
                    incompatible(
                        declaration.key().to_string(),
                        "use a context whose admitted device reports the uniform-buffer offset alignment required by this binding",
                    )
                })?;
            validate_buffer_alignment(declaration, binding, alignment.get())
        }
        (GpuBindingClass::StorageBuffer, GpuRuntimeBindingResource::Buffer(binding)) => {
            let alignment = device_facts
                .storage_buffer_offset_alignment()
                .ok_or_else(|| {
                    incompatible(
                        declaration.key().to_string(),
                        "use a context whose admitted device reports the storage-buffer offset alignment required by this binding",
                    )
                })?;
            validate_buffer_alignment(declaration, binding, alignment.get())
        }
        (GpuBindingClass::SampledTexture, GpuRuntimeBindingResource::TextureView(binding)) => {
            validate_sampled_texture_device(declaration, binding, device_facts)
        }
        (GpuBindingClass::StorageTexture, GpuRuntimeBindingResource::TextureView(binding)) => {
            validate_storage_texture_device(declaration, binding, device_facts)
        }
        (GpuBindingClass::Sampler, GpuRuntimeBindingResource::Sampler(_)) => Ok(()),
        _ => Err(incompatible(
            declaration.key().to_string(),
            "retain structurally validated resources before admitted-device validation",
        )),
    }
}

fn validate_buffer_structure(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeBufferBinding,
    required_usage: GpuBufferUsage,
) -> Result<(), GpuProgramContractError> {
    let descriptor = binding.handle().descriptor();
    if !descriptor.usages().contains(required_usage) {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a buffer carrying the usage required by the declaration",
        ));
    }

    let expects_dynamic_offset = declaration.kind().uses_dynamic_offset();
    if expects_dynamic_offset != binding.dynamic_offset().is_some() {
        return Err(incompatible(
            declaration.key().to_string(),
            "match the declaration's dynamic-offset policy exactly",
        ));
    }

    let effective_offset = binding.checked_effective_offset().ok_or_else(|| {
        incompatible(
            declaration.key().to_string(),
            "reduce the buffer offsets so their sum does not overflow",
        )
    })?;
    let end = effective_offset
        .checked_add(binding.size().get())
        .ok_or_else(|| {
            incompatible(
                declaration.key().to_string(),
                "reduce the buffer range so its end does not overflow",
            )
        })?;
    if end > descriptor.size_bytes() {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a range fully contained by the logical buffer descriptor",
        ));
    }

    let compiler_minimum = declaration.compiler_required_minimum_size();
    let host_minimum = declaration.kind().minimum_buffer_size();
    let required_minimum = match (compiler_minimum, host_minimum) {
        (Some(compiler), Some(host)) => Some(compiler.max(host)),
        (Some(compiler), None) => Some(compiler),
        (None, Some(host)) => Some(host),
        (None, None) => None,
    };
    if required_minimum.is_some_and(|minimum| binding.size() < minimum) {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind at least the compiler-required shader size and any stronger host/layout minimum",
        ));
    }

    Ok(())
}

fn validate_buffer_alignment(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeBufferBinding,
    alignment: u64,
) -> Result<(), GpuProgramContractError> {
    if !binding.offset().is_multiple_of(alignment)
        || binding
            .dynamic_offset()
            .is_some_and(|offset| !offset.is_multiple_of(alignment))
    {
        return Err(incompatible(
            declaration.key().to_string(),
            "align static and dynamic buffer offsets to the admitted device requirement",
        ));
    }
    Ok(())
}

fn validate_sampled_texture_structure(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeTextureViewBinding,
) -> Result<(), GpuProgramContractError> {
    let (format, sample_count) = validate_texture_view_shape(declaration, binding)?;
    let texture = binding.handle().descriptor().texture().descriptor();
    if !texture.usages().contains(GpuTextureUsage::Sampled) {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a texture view whose source texture carries Sampled usage",
        ));
    }

    let expected_multisampled = declaration.kind().is_multisampled_texture();
    if expected_multisampled != (sample_count > 1) {
        return Err(incompatible(
            declaration.key().to_string(),
            "match the declaration's multisample state exactly",
        ));
    }

    let sample_class = declaration
        .kind()
        .texture_sample_class()
        .expect("sampled-texture declarations carry a sample class");
    let structurally_compatible = match sample_class {
        GpuTextureSampleClass::FloatFilterable | GpuTextureSampleClass::FloatUnfilterable => {
            is_float_format(format)
        }
        GpuTextureSampleClass::Depth => format.is_depth(),
        GpuTextureSampleClass::Sint => false,
        GpuTextureSampleClass::Uint => format == GpuTextureFormat::R32Uint,
    };
    if !structurally_compatible {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a texture whose normalized sample class matches the declaration",
        ));
    }

    Ok(())
}

fn validate_sampled_texture_device(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeTextureViewBinding,
    device_facts: &GpuRuntimeBindingDeviceFacts,
) -> Result<(), GpuProgramContractError> {
    let view = binding.handle().descriptor();
    let texture = view.texture().descriptor();
    let format = view.format().unwrap_or(texture.format());
    let capabilities = device_facts.format_capabilities(format).ok_or_else(|| {
        incompatible(
            declaration.key().to_string(),
            "supply admitted format capabilities for the bound texture format",
        )
    })?;
    if !capabilities.sampled {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a texture format admitted for sampled access",
        ));
    }
    if declaration.kind().texture_sample_class() == Some(GpuTextureSampleClass::FloatFilterable)
        && !capabilities.filterable
    {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a texture format admitted for filterable sampled access",
        ));
    }
    Ok(())
}

fn validate_storage_texture_structure(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeTextureViewBinding,
) -> Result<(), GpuProgramContractError> {
    let (format, sample_count) = validate_texture_view_shape(declaration, binding)?;
    if sample_count != 1 {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a non-multisampled texture view for storage access",
        ));
    }

    let expected_format = declaration
        .kind()
        .storage_texture_format()
        .expect("storage-texture declarations carry an exact format");
    if format != expected_format {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind the exact storage texture format declared by the interface",
        ));
    }

    let texture = binding.handle().descriptor().texture().descriptor();
    let access = declaration
        .kind()
        .storage_texture_access()
        .expect("storage-texture declarations carry access");
    let usage_compatible = match access {
        GpuStorageTextureAccess::ReadOnly => {
            texture.usages().contains(GpuTextureUsage::StorageRead)
        }
        GpuStorageTextureAccess::WriteOnly => {
            texture.usages().contains(GpuTextureUsage::StorageWrite)
        }
        GpuStorageTextureAccess::ReadWrite => {
            texture.usages().contains(GpuTextureUsage::StorageRead)
                && texture.usages().contains(GpuTextureUsage::StorageWrite)
        }
    };
    if !usage_compatible {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a storage texture whose logical usage satisfies the declared access",
        ));
    }

    Ok(())
}

fn validate_storage_texture_device(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeTextureViewBinding,
    device_facts: &GpuRuntimeBindingDeviceFacts,
) -> Result<(), GpuProgramContractError> {
    let view = binding.handle().descriptor();
    let texture = view.texture().descriptor();
    let format = view.format().unwrap_or(texture.format());
    let capabilities = device_facts.format_capabilities(format).ok_or_else(|| {
        incompatible(
            declaration.key().to_string(),
            "supply admitted format capabilities for the bound storage texture format",
        )
    })?;
    let access = declaration
        .kind()
        .storage_texture_access()
        .expect("storage-texture declarations carry access");
    let compatible = match access {
        GpuStorageTextureAccess::ReadOnly => capabilities.storage_read,
        GpuStorageTextureAccess::WriteOnly => capabilities.storage_write,
        GpuStorageTextureAccess::ReadWrite => {
            capabilities.storage_read && capabilities.storage_write
        }
    };
    if !compatible {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a storage texture whose admitted format capabilities satisfy access",
        ));
    }
    Ok(())
}

fn validate_texture_view_shape(
    declaration: &GpuBindingDeclaration,
    binding: &GpuRuntimeTextureViewBinding,
) -> Result<(GpuTextureFormat, u32), GpuProgramContractError> {
    let expected_dimension = declaration
        .kind()
        .texture_view_dimension()
        .expect("texture declarations carry a view dimension");
    if binding.dimension() != expected_dimension {
        return Err(incompatible(
            declaration.key().to_string(),
            "match the declaration's normalized texture-view dimension exactly",
        ));
    }

    let view = binding.handle().descriptor();
    let layer_count = view.subresources().array_layer_count();
    if !view_dimension_compatible(view.dimension(), layer_count, binding.dimension()) {
        return Err(incompatible(
            declaration.key().to_string(),
            "provide view facts compatible with the logical texture-view descriptor",
        ));
    }

    let texture = view.texture().descriptor();
    Ok((
        view.format().unwrap_or(texture.format()),
        texture.sample_count(),
    ))
}

fn validate_sampler(
    declaration: &GpuBindingDeclaration,
    handle: &crate::GpuSamplerHandle,
) -> Result<(), GpuProgramContractError> {
    let descriptor = handle.descriptor();
    let comparison = descriptor.compare().is_some();
    let (mag_filter, min_filter, mipmap_filter) = descriptor.filters();
    let filtering = [mag_filter, min_filter, mipmap_filter]
        .into_iter()
        .any(|filter| filter == GpuFilterMode::Linear);
    let class = declaration
        .kind()
        .sampler_class()
        .expect("sampler declarations carry a sampler class");
    let compatible = match class {
        GpuSamplerClass::Filtering => filtering && !comparison,
        GpuSamplerClass::NonFiltering => !filtering && !comparison,
        GpuSamplerClass::Comparison => comparison,
    };
    if !compatible {
        return Err(incompatible(
            declaration.key().to_string(),
            "bind a sampler whose filtering/comparison class matches the declaration",
        ));
    }
    Ok(())
}

fn view_dimension_compatible(
    actual: GpuTextureDimension,
    layer_count: u32,
    declared: GpuTextureViewDimension,
) -> bool {
    match declared {
        GpuTextureViewDimension::D1 => actual == GpuTextureDimension::D1 && layer_count == 1,
        GpuTextureViewDimension::D2 => actual == GpuTextureDimension::D2 && layer_count == 1,
        GpuTextureViewDimension::D2Array => actual == GpuTextureDimension::D2,
        GpuTextureViewDimension::Cube => actual == GpuTextureDimension::D2 && layer_count == 6,
        GpuTextureViewDimension::CubeArray => {
            actual == GpuTextureDimension::D2 && layer_count >= 6 && layer_count.is_multiple_of(6)
        }
        GpuTextureViewDimension::D3 => actual == GpuTextureDimension::D3 && layer_count == 1,
    }
}

fn is_float_format(format: GpuTextureFormat) -> bool {
    !format.is_depth() && format != GpuTextureFormat::R32Uint
}

fn incompatible(label: impl Into<String>, correction: &'static str) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "validate runtime GPU binding compatibility",
        label,
        GpuProgramContractCause::RuntimeBindingIncompatible,
        correction,
    )
}
