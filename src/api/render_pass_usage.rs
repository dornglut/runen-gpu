use super::{
    GpuBufferAccessKind, GpuDepthStencilAccess, GpuResourceAccess, GpuTextureAccessKind,
    GpuTextureAspect, GpuWorkOperationCause, GpuWorkOperationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderPassUsageClass {
    ReadOnly,
    Storage,
    Attachment,
}

/// Validate the portable usage scope of one logical render pass.
///
/// WebGPU models one render pass as one usage scope. Buffer subresources are whole buffers, so
/// disjoint byte ranges do not make otherwise-incompatible pass usages compatible. Texture scope is
/// finer grained and only overlapping normalized texture subresources interact. Draw-local writable
/// binding aliasing remains a separate, stricter check owned by `GpuRuntimeBindingSet`.
pub(crate) fn validate_render_pass_usage_scope(
    accesses: &[GpuResourceAccess],
) -> Result<(), GpuWorkOperationError> {
    let classes = accesses
        .iter()
        .map(render_pass_usage_class)
        .collect::<Result<Vec<_>, _>>()?;

    for left_index in 0..accesses.len() {
        for (right_offset, right) in accesses[(left_index + 1)..].iter().enumerate() {
            let left = &accesses[left_index];
            if !same_render_pass_subresource(left, right) {
                continue;
            }

            let Some(left_class) = classes[left_index] else {
                continue;
            };
            let Some(right_class) = classes[left_index + 1 + right_offset] else {
                continue;
            };

            if left_class != right_class {
                return Err(GpuWorkOperationError::invalid(
                    "validate GPU render-pass usage scope",
                    format!("left={left:?}, right={right:?}"),
                    Some(left.resource_identity()),
                    GpuWorkOperationCause::OperationAccessContradiction,
                    "keep each render-pass buffer or overlapping texture subresource within one compatible usage class: read-only, storage, or attachment",
                ));
            }

            if left_class == RenderPassUsageClass::Attachment {
                return Err(GpuWorkOperationError::invalid(
                    "validate GPU render-pass attachment regions",
                    format!("left={left:?}, right={right:?}"),
                    Some(left.resource_identity()),
                    GpuWorkOperationCause::OperationAccessContradiction,
                    "bind each overlapping texture subresource as at most one attachment in the logical render pass",
                ));
            }
        }
    }
    Ok(())
}

fn same_render_pass_subresource(left: &GpuResourceAccess, right: &GpuResourceAccess) -> bool {
    match (left, right) {
        (GpuResourceAccess::Buffer(left), GpuResourceAccess::Buffer(right)) => {
            left.resource_identity() == right.resource_identity()
        }
        (GpuResourceAccess::Texture(left), GpuResourceAccess::Texture(right)) => {
            if left.resource_identity() != right.resource_identity() {
                return false;
            }
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

fn render_pass_usage_class(
    access: &GpuResourceAccess,
) -> Result<Option<RenderPassUsageClass>, GpuWorkOperationError> {
    match access {
        GpuResourceAccess::Buffer(access) => match access.kind() {
            GpuBufferAccessKind::UniformRead
            | GpuBufferAccessKind::VertexRead
            | GpuBufferAccessKind::IndexRead
            | GpuBufferAccessKind::IndirectRead => Ok(Some(RenderPassUsageClass::ReadOnly)),
            GpuBufferAccessKind::StorageRead
            | GpuBufferAccessKind::StorageWrite
            | GpuBufferAccessKind::StorageReadWrite => Ok(Some(RenderPassUsageClass::Storage)),
            GpuBufferAccessKind::CopySource
            | GpuBufferAccessKind::CopyDestination
            | GpuBufferAccessKind::QueryResolveDestination => Err(invalid_render_pass_access(
                access.resource_identity(),
                "buffer copy/query-resolve access",
            )),
        },
        GpuResourceAccess::Texture(access) => match access.kind() {
            GpuTextureAccessKind::SampledRead => Ok(Some(RenderPassUsageClass::ReadOnly)),
            GpuTextureAccessKind::StorageRead
            | GpuTextureAccessKind::StorageWrite
            | GpuTextureAccessKind::StorageReadWrite => Ok(Some(RenderPassUsageClass::Storage)),
            GpuTextureAccessKind::ColorAttachment { .. }
            | GpuTextureAccessKind::MultisampleResolveDestination => {
                Ok(Some(RenderPassUsageClass::Attachment))
            }
            GpuTextureAccessKind::DepthStencilAttachment { access, .. } => match access {
                GpuDepthStencilAccess::ReadOnly => Ok(Some(RenderPassUsageClass::ReadOnly)),
                GpuDepthStencilAccess::ReadWrite => Ok(Some(RenderPassUsageClass::Attachment)),
            },
            GpuTextureAccessKind::CopySource
            | GpuTextureAccessKind::CopyDestination
            | GpuTextureAccessKind::Present => Err(invalid_render_pass_access(
                access.resource_identity(),
                "texture copy/present access",
            )),
        },
        GpuResourceAccess::Query(_) | GpuResourceAccess::Sampler(_) => Ok(None),
    }
}

fn invalid_render_pass_access(
    resource: super::GpuWorkResourceId,
    label: &'static str,
) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        "validate GPU render-pass usage scope",
        label,
        Some(resource),
        GpuWorkOperationCause::OperationAccessContradiction,
        "keep copy, query-resolution, and presentation operations outside render-pass execution",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuAttachmentLoadKind, GpuAttachmentStore, GpuBufferAccess, GpuBufferDescriptor,
        GpuBufferInitialization, GpuBufferRange, GpuBufferUsage, GpuBufferUsages, GpuMemoryIntent,
        GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
        GpuResourceProvenance, GpuTextureAccess, GpuTextureAccessResource, GpuTextureDescriptor,
        GpuTextureDimension, GpuTextureExtent, GpuTextureFormat, GpuTextureInitialization,
        GpuTextureSubresourceRange, GpuTextureUsage, GpuTextureUsages, GpuWorkResourceIdAllocator,
    };
    use std::num::NonZeroU64;

    fn label(value: &str) -> GpuResourceLabel {
        GpuResourceLabel::new(value).unwrap()
    }

    fn common(value: &str) -> GpuResourceCommon {
        let label = label(value);
        GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            GpuResourceProvenance::new(label, None, None),
        )
        .unwrap()
    }

    fn allocator(scope: u64) -> GpuWorkResourceIdAllocator {
        GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(scope).unwrap())
    }

    fn buffer(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        usages: impl IntoIterator<Item = GpuBufferUsage>,
    ) -> super::super::GpuBufferHandle {
        let resource_label = label(name);
        allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common(name),
                    64,
                    GpuBufferUsages::new(&resource_label, usages).unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn texture(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        format: GpuTextureFormat,
        usages: impl IntoIterator<Item = GpuTextureUsage>,
    ) -> super::super::GpuTextureHandle {
        let resource_label = label(name);
        allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common(name),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 16, 16, 1)
                        .unwrap(),
                    1,
                    1,
                    format,
                    GpuTextureUsages::new(&resource_label, usages).unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn writable_storage_uses_share_one_pass_even_for_overlapping_buffer_ranges() {
        let mut allocator = allocator(101);
        let buffer = buffer(&mut allocator, "storage", [GpuBufferUsage::Storage]);
        let first = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 0, 32).unwrap(),
            GpuBufferAccessKind::StorageWrite,
        )
        .unwrap();
        let second = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 16, 32).unwrap(),
            GpuBufferAccessKind::StorageReadWrite,
        )
        .unwrap();

        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Buffer(first),
                GpuResourceAccess::Buffer(second),
            ])
            .is_ok()
        );
    }

    #[test]
    fn read_only_and_writable_storage_uses_share_one_pass() {
        let mut allocator = allocator(107);
        let buffer = buffer(&mut allocator, "mixed storage", [GpuBufferUsage::Storage]);
        let read = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 0, 32).unwrap(),
            GpuBufferAccessKind::StorageRead,
        )
        .unwrap();
        let write = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 16, 32).unwrap(),
            GpuBufferAccessKind::StorageWrite,
        )
        .unwrap();
        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Buffer(read),
                GpuResourceAccess::Buffer(write),
            ])
            .is_ok()
        );

        let texture = texture(
            &mut allocator,
            "mixed storage texture",
            GpuTextureFormat::Rgba8Unorm,
            [GpuTextureUsage::StorageRead, GpuTextureUsage::StorageWrite],
        );
        let subresources = GpuTextureSubresourceRange::whole(&texture).unwrap();
        let read = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture.clone()),
            subresources,
            GpuTextureAccessKind::StorageRead,
        )
        .unwrap();
        let write = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture),
            subresources,
            GpuTextureAccessKind::StorageWrite,
        )
        .unwrap();
        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Texture(read),
                GpuResourceAccess::Texture(write),
            ])
            .is_ok()
        );
    }

    #[test]
    fn buffer_storage_and_vertex_use_conflict_even_for_disjoint_byte_ranges() {
        let mut allocator = allocator(102);
        let buffer = buffer(
            &mut allocator,
            "mixed",
            [GpuBufferUsage::Storage, GpuBufferUsage::Vertex],
        );
        let storage = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 0, 16).unwrap(),
            GpuBufferAccessKind::StorageWrite,
        )
        .unwrap();
        let vertex = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 32, 16).unwrap(),
            GpuBufferAccessKind::VertexRead,
        )
        .unwrap();

        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Buffer(storage),
                GpuResourceAccess::Buffer(vertex),
            ])
            .is_err()
        );
    }

    #[test]
    fn lone_copy_role_rejects_before_pairwise_usage_validation() {
        let mut allocator = allocator(106);
        let buffer = buffer(&mut allocator, "copy", [GpuBufferUsage::CopySource]);
        let copy = GpuBufferAccess::new(
            &buffer,
            GpuBufferRange::new(&buffer, 0, 16).unwrap(),
            GpuBufferAccessKind::CopySource,
        )
        .unwrap();

        assert!(validate_render_pass_usage_scope(&[GpuResourceAccess::Buffer(copy)]).is_err());
    }

    #[test]
    fn attachment_and_sampled_use_conflict_on_the_same_texture_subresource() {
        let mut allocator = allocator(103);
        let texture = texture(
            &mut allocator,
            "color",
            GpuTextureFormat::Rgba8Unorm,
            [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
        );
        let subresources = GpuTextureSubresourceRange::whole(&texture).unwrap();
        let attachment = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture.clone()),
            subresources,
            GpuTextureAccessKind::ColorAttachment {
                load_kind: GpuAttachmentLoadKind::Load,
                store: GpuAttachmentStore::Store,
            },
        )
        .unwrap();
        let sampled = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture),
            subresources,
            GpuTextureAccessKind::SampledRead,
        )
        .unwrap();

        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Texture(attachment),
                GpuResourceAccess::Texture(sampled),
            ])
            .is_err()
        );
    }

    #[test]
    fn read_only_depth_attachment_and_sampled_use_are_compatible() {
        let mut allocator = allocator(104);
        let texture = texture(
            &mut allocator,
            "depth",
            GpuTextureFormat::Depth32Float,
            [
                GpuTextureUsage::DepthStencilAttachment,
                GpuTextureUsage::Sampled,
            ],
        );
        let subresources = GpuTextureSubresourceRange::whole(&texture).unwrap();
        let attachment = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture.clone()),
            subresources,
            GpuTextureAccessKind::DepthStencilAttachment {
                access: GpuDepthStencilAccess::ReadOnly,
                load_kind: GpuAttachmentLoadKind::Load,
                store: GpuAttachmentStore::Store,
            },
        )
        .unwrap();
        let sampled = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture),
            subresources,
            GpuTextureAccessKind::SampledRead,
        )
        .unwrap();

        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Texture(attachment),
                GpuResourceAccess::Texture(sampled),
            ])
            .is_ok()
        );
    }

    #[test]
    fn overlapping_attachment_regions_reject_even_inside_attachment_usage_class() {
        let mut allocator = allocator(105);
        let texture = texture(
            &mut allocator,
            "attachment alias",
            GpuTextureFormat::Rgba8Unorm,
            [GpuTextureUsage::ColorAttachment],
        );
        let subresources = GpuTextureSubresourceRange::whole(&texture).unwrap();
        let first = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture.clone()),
            subresources,
            GpuTextureAccessKind::ColorAttachment {
                load_kind: GpuAttachmentLoadKind::Load,
                store: GpuAttachmentStore::Store,
            },
        )
        .unwrap();
        let second = GpuTextureAccess::new(
            GpuTextureAccessResource::Texture(texture),
            subresources,
            GpuTextureAccessKind::ColorAttachment {
                load_kind: GpuAttachmentLoadKind::Load,
                store: GpuAttachmentStore::Store,
            },
        )
        .unwrap();

        assert!(
            validate_render_pass_usage_scope(&[
                GpuResourceAccess::Texture(first),
                GpuResourceAccess::Texture(second),
            ])
            .is_err()
        );
    }
}
