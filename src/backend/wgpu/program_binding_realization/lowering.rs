//! Typed G4B layout/runtime lowering owned by the private G4C2 realization boundary.

use super::super::texture_format_mapping::texture_format;
use crate::api::fixed_array_layout_capabilities;
use crate::{
    GpuBindGroupLayoutDescriptor, GpuBindingClass, GpuBindingDeclaration, GpuContext,
    GpuProgramBindingRealizationError, GpuProgramBindingRealizationErrorCategory, GpuSamplerClass,
    GpuStorageBufferAccess, GpuStorageTextureAccess, GpuTextureSampleClass,
    GpuTextureViewDimension,
};
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureSampleType, TextureViewDimension,
};

pub(super) fn layout_entries(
    context: &GpuContext,
    descriptor: &GpuBindGroupLayoutDescriptor,
) -> Result<Vec<BindGroupLayoutEntry>, GpuProgramBindingRealizationError> {
    if has_incompatible_binding_array_mix(descriptor) {
        return Err(layout_error(
            descriptor,
            "a group containing a binding array cannot contain a non-array uniform buffer or dynamic offset",
        ));
    }
    descriptor
        .bindings()
        .map(|binding| layout_entry(context, descriptor, binding))
        .collect()
}

fn has_incompatible_binding_array_mix(descriptor: &GpuBindGroupLayoutDescriptor) -> bool {
    let has_binding_array = descriptor
        .bindings()
        .any(|binding| binding.array_count().is_some());
    has_binding_array
        && descriptor.bindings().any(|binding| {
            binding.kind().uses_dynamic_offset()
                || (binding.kind().class() == GpuBindingClass::UniformBuffer
                    && binding.array_count().is_none())
        })
}

fn layout_entry(
    context: &GpuContext,
    descriptor: &GpuBindGroupLayoutDescriptor,
    binding: &GpuBindingDeclaration,
) -> Result<BindGroupLayoutEntry, GpuProgramBindingRealizationError> {
    validate_array_feature(context, descriptor, binding)?;
    lowered_layout_entry(descriptor, binding)
}

fn lowered_layout_entry(
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
    for feature in fixed_array_layout_capabilities(binding.kind().class()) {
        if !context.device_facts().is_enabled(*feature) {
            return Err(layout_error(
                descriptor,
                format!(
                    "the admitted RunenGPU device did not enable {feature:?}, required by this fixed binding-array layout"
                ),
            ));
        }
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBindingKey, GpuBindingKind, GpuBindingProvenance, GpuShaderStage, GpuShaderStages,
        GpuStorageBufferAccess,
    };
    use core::num::NonZeroU32;

    fn declaration(
        binding: u32,
        class: GpuBindingClass,
        array_count: Option<u32>,
        dynamic_offset: bool,
    ) -> GpuBindingDeclaration {
        let kind = match class {
            GpuBindingClass::UniformBuffer => GpuBindingKind::uniform_buffer(dynamic_offset, None),
            GpuBindingClass::StorageBuffer => GpuBindingKind::storage_buffer(
                GpuStorageBufferAccess::ReadOnly,
                dynamic_offset,
                None,
            ),
            _ => panic!("test helper only needs buffer binding classes"),
        };
        GpuBindingDeclaration::new(
            GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
            GpuShaderStages::one(GpuShaderStage::Compute),
            kind,
            array_count.and_then(NonZeroU32::new),
            format!("binding-{binding}"),
            GpuBindingProvenance::new("fixed-array-mix-test", None).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn fixed_array_lowering_preserves_the_wgpu_array_counter_discriminator() {
        let array = declaration(0, GpuBindingClass::StorageBuffer, Some(9), false);
        let ordinary = declaration(1, GpuBindingClass::StorageBuffer, None, false);
        let descriptor =
            GpuBindGroupLayoutDescriptor::new(0, [array.clone(), ordinary.clone()]).unwrap();

        let array_entry = lowered_layout_entry(&descriptor, &array).unwrap();
        let ordinary_entry = lowered_layout_entry(&descriptor, &ordinary).unwrap();

        assert!(matches!(
            array_entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { .. },
                ..
            }
        ));
        assert!(matches!(
            ordinary_entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { .. },
                ..
            }
        ));
        assert_eq!(array_entry.count.map(|count| count.get()), Some(9));
        assert_eq!(ordinary_entry.count, None);
    }

    #[test]
    fn uniform_binding_array_is_not_mistaken_for_an_ordinary_uniform_buffer() {
        let layout = GpuBindGroupLayoutDescriptor::new(
            0,
            [declaration(
                0,
                GpuBindingClass::UniformBuffer,
                Some(2),
                false,
            )],
        )
        .unwrap();

        assert!(!has_incompatible_binding_array_mix(&layout));
    }

    #[test]
    fn binding_array_mix_rejects_ordinary_uniform_or_dynamic_offset_bindings() {
        let with_uniform = GpuBindGroupLayoutDescriptor::new(
            0,
            [
                declaration(0, GpuBindingClass::StorageBuffer, Some(2), false),
                declaration(1, GpuBindingClass::UniformBuffer, None, false),
            ],
        )
        .unwrap();
        assert!(has_incompatible_binding_array_mix(&with_uniform));

        let with_dynamic = GpuBindGroupLayoutDescriptor::new(
            0,
            [
                declaration(0, GpuBindingClass::StorageBuffer, Some(2), false),
                declaration(1, GpuBindingClass::StorageBuffer, None, true),
            ],
        )
        .unwrap();
        assert!(has_incompatible_binding_array_mix(&with_dynamic));
    }
}
