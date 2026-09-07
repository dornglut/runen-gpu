//! Typed G4B layout/runtime lowering owned by the private G4C2 realization boundary.

use crate::{
    GpuBindGroupLayoutDescriptor, GpuBindingClass, GpuBindingDeclaration, GpuContext,
    GpuProgramBindingRealizationError, GpuProgramBindingRealizationErrorCategory, GpuSamplerClass,
    GpuStorageBufferAccess, GpuStorageTextureAccess, GpuTextureFormat, GpuTextureSampleClass,
    GpuTextureViewDimension,
};
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

pub(super) fn layout_entries(
    context: &GpuContext,
    descriptor: &GpuBindGroupLayoutDescriptor,
) -> Result<Vec<BindGroupLayoutEntry>, GpuProgramBindingRealizationError> {
    let has_binding_array = descriptor
        .bindings()
        .any(|binding| binding.array_count().is_some());
    if has_binding_array
        && descriptor.bindings().any(|binding| {
            binding.kind().uses_dynamic_offset()
                || binding.kind().class() == GpuBindingClass::UniformBuffer
        })
    {
        return Err(layout_error(
            descriptor,
            "a group containing a binding array cannot contain a uniform buffer or dynamic offset",
        ));
    }
    descriptor
        .bindings()
        .map(|binding| layout_entry(context, descriptor, binding))
        .collect()
}

fn layout_entry(
    context: &GpuContext,
    descriptor: &GpuBindGroupLayoutDescriptor,
    binding: &GpuBindingDeclaration,
) -> Result<BindGroupLayoutEntry, GpuProgramBindingRealizationError> {
    let ty =
        match binding.kind().class() {
            GpuBindingClass::UniformBuffer => BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: binding.kind().uses_dynamic_offset(),
                min_binding_size: binding.kind().minimum_buffer_size().map(|size| {
                    wgpu::BufferSize::new(size.get()).expect("minimum sizes are nonzero")
                }),
            },
            GpuBindingClass::StorageBuffer => BindingType::Buffer {
                ty: BufferBindingType::Storage {
                    read_only: matches!(
                        binding.kind().storage_buffer_access(),
                        Some(GpuStorageBufferAccess::ReadOnly)
                    ),
                },
                has_dynamic_offset: binding.kind().uses_dynamic_offset(),
                min_binding_size: binding.kind().minimum_buffer_size().map(|size| {
                    wgpu::BufferSize::new(size.get()).expect("minimum sizes are nonzero")
                }),
            },
            GpuBindingClass::SampledTexture => BindingType::Texture {
                sample_type: match binding.kind().texture_sample_class() {
                    Some(GpuTextureSampleClass::FloatFilterable) => {
                        TextureSampleType::Float { filterable: true }
                    }
                    Some(GpuTextureSampleClass::FloatUnfilterable) => {
                        TextureSampleType::Float { filterable: false }
                    }
                    Some(GpuTextureSampleClass::Depth) => TextureSampleType::Depth,
                    Some(GpuTextureSampleClass::Sint) => TextureSampleType::Sint,
                    Some(GpuTextureSampleClass::Uint) => TextureSampleType::Uint,
                    None => {
                        return Err(layout_error(
                            descriptor,
                            "sampled texture lacks a sample class",
                        ));
                    }
                },
                view_dimension: texture_view_dimension(
                    binding.kind().texture_view_dimension().ok_or_else(|| {
                        layout_error(descriptor, "sampled texture lacks a view dimension")
                    })?,
                ),
                multisampled: binding.kind().is_multisampled_texture(),
            },
            GpuBindingClass::StorageTexture => {
                BindingType::StorageTexture {
                    access: storage_texture_access(
                        binding.kind().storage_texture_access().ok_or_else(|| {
                            layout_error(descriptor, "storage texture lacks access")
                        })?,
                    ),
                    format: texture_format(binding.kind().storage_texture_format().ok_or_else(
                        || layout_error(descriptor, "storage texture lacks a format"),
                    )?),
                    view_dimension: texture_view_dimension(
                        binding.kind().texture_view_dimension().ok_or_else(|| {
                            layout_error(descriptor, "storage texture lacks a view dimension")
                        })?,
                    ),
                }
            }
            GpuBindingClass::Sampler => {
                BindingType::Sampler(match binding.kind().sampler_class() {
                    Some(GpuSamplerClass::Filtering) => SamplerBindingType::Filtering,
                    Some(GpuSamplerClass::NonFiltering) => SamplerBindingType::NonFiltering,
                    Some(GpuSamplerClass::Comparison) => SamplerBindingType::Comparison,
                    None => return Err(layout_error(descriptor, "sampler lacks a sampler class")),
                })
            }
        };
    validate_array_feature(context, descriptor, binding)?;
    Ok(BindGroupLayoutEntry {
        binding: binding.key().binding(),
        visibility: shader_stages(binding.visibility()),
        ty,
        count: binding.array_count(),
    })
}

fn validate_array_feature(
    context: &GpuContext,
    descriptor: &GpuBindGroupLayoutDescriptor,
    binding: &GpuBindingDeclaration,
) -> Result<(), GpuProgramBindingRealizationError> {
    if binding.array_count().is_none() {
        return Ok(());
    }
    let required = match binding.kind().class() {
        GpuBindingClass::UniformBuffer => {
            wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::UNIFORM_BUFFER_BINDING_ARRAYS
        }
        GpuBindingClass::StorageBuffer => {
            wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        }
        GpuBindingClass::SampledTexture | GpuBindingClass::Sampler => {
            wgpu::Features::TEXTURE_BINDING_ARRAY
        }
        GpuBindingClass::StorageTexture => {
            wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        }
    };
    if !context.backend.device.features().contains(required) {
        return Err(layout_error(
            descriptor,
            "the admitted device did not enable the WGPU fixed binding-array features required by this layout",
        ));
    }
    Ok(())
}

pub(super) const fn shader_stages(stages: crate::GpuShaderStages) -> ShaderStages {
    let mut native = ShaderStages::empty();
    if stages.contains(crate::GpuShaderStage::Compute) {
        native = native.union(ShaderStages::COMPUTE);
    }
    if stages.contains(crate::GpuShaderStage::Vertex) {
        native = native.union(ShaderStages::VERTEX);
    }
    if stages.contains(crate::GpuShaderStage::Fragment) {
        native = native.union(ShaderStages::FRAGMENT);
    }
    native
}

pub(super) const fn texture_format(format: GpuTextureFormat) -> TextureFormat {
    match format {
        GpuTextureFormat::R8Unorm => TextureFormat::R8Unorm,
        GpuTextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        GpuTextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        GpuTextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        GpuTextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        GpuTextureFormat::R32Uint => TextureFormat::R32Uint,
        GpuTextureFormat::Depth32Float => TextureFormat::Depth32Float,
    }
}

const fn texture_view_dimension(dimension: GpuTextureViewDimension) -> TextureViewDimension {
    match dimension {
        GpuTextureViewDimension::D1 => TextureViewDimension::D1,
        GpuTextureViewDimension::D2 => TextureViewDimension::D2,
        GpuTextureViewDimension::D2Array => TextureViewDimension::D2Array,
        GpuTextureViewDimension::Cube => TextureViewDimension::Cube,
        GpuTextureViewDimension::CubeArray => TextureViewDimension::CubeArray,
        GpuTextureViewDimension::D3 => TextureViewDimension::D3,
    }
}

const fn storage_texture_access(access: GpuStorageTextureAccess) -> StorageTextureAccess {
    match access {
        GpuStorageTextureAccess::ReadOnly => StorageTextureAccess::ReadOnly,
        GpuStorageTextureAccess::WriteOnly => StorageTextureAccess::WriteOnly,
        GpuStorageTextureAccess::ReadWrite => StorageTextureAccess::ReadWrite,
    }
}

fn layout_error(
    descriptor: &GpuBindGroupLayoutDescriptor,
    detail: impl Into<String>,
) -> GpuProgramBindingRealizationError {
    GpuProgramBindingRealizationError::new(
        GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid,
        format!("bind-group layout group={}", descriptor.group()),
        detail,
    )
}
