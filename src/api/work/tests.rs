use super::super::operation::{GpuRenderOperation, GpuWorkOperation};
use super::*;
use crate::{
    GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages, GpuMemoryIntent,
    GpuQuerySetDescriptor, GpuReconstruction, GpuResourceCommon, GpuResourceLabel,
    GpuResourceLifetime, GpuResourceProvenance, GpuTextureDescriptor, GpuTextureExtent,
    GpuTextureInitialization, GpuTextureUsage, GpuTextureUsages, GpuTextureViewDescriptor,
    GpuWorkResourceIdAllocator,
};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    num::NonZeroU64,
};

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

fn allocator() -> GpuWorkResourceIdAllocator {
    GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(73).unwrap())
}

fn semantic_hash(value: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    size: u64,
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let label = label(name);
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(name),
                size,
                GpuBufferUsages::new(&label, usages).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

#[derive(Clone, Copy)]
struct TestTextureShape {
    width: u32,
    height: u32,
    layers: u32,
    mip_levels: u32,
    sample_count: u32,
    format: GpuTextureFormat,
}

fn texture_with_shape(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    shape: TestTextureShape,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> GpuTextureHandle {
    let label = label(name);
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &label,
                    GpuTextureDimension::D2,
                    shape.width,
                    shape.height,
                    shape.layers,
                )
                .unwrap(),
                shape.mip_levels,
                shape.sample_count,
                shape.format,
                GpuTextureUsages::new(&label, usages).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    sample_count: u32,
    format: GpuTextureFormat,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> GpuTextureHandle {
    texture_with_shape(
        allocator,
        name,
        TestTextureShape {
            width: 16,
            height: 16,
            layers: 1,
            mip_levels: 1,
            sample_count,
            format,
        },
        usages,
    )
}

fn texture_view(
    allocator: &mut GpuWorkResourceIdAllocator,
    texture: &GpuTextureHandle,
    name: &str,
    subresources: GpuTextureSubresourceRange,
) -> GpuTextureViewHandle {
    allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common(name),
                texture,
                None,
                GpuTextureDimension::D2,
                subresources,
            )
            .unwrap(),
        )
        .unwrap()
}

fn single_view(
    allocator: &mut GpuWorkResourceIdAllocator,
    texture: &GpuTextureHandle,
    name: &str,
    mip_level: u32,
    array_layer: u32,
    aspect: GpuTextureAspect,
) -> GpuTextureViewHandle {
    let subresources = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        mip_level,
        1,
        array_layer,
        1,
        aspect,
    )
    .unwrap();
    texture_view(allocator, texture, name, subresources)
}

#[test]
fn clear_values_keep_color_generic_depth_normalized_and_signed_zero_canonical() {
    let negative_zero = GpuColorClearValue::new(-0.0, 0.0, 1.0, 1.0).unwrap();
    let positive_zero = GpuColorClearValue::new(0.0, -0.0, 1.0, 1.0).unwrap();
    assert_eq!(negative_zero, positive_zero);
    assert_eq!(semantic_hash(negative_zero), semantic_hash(positive_zero));
    assert_eq!(negative_zero.components()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        GpuColorClearValue::new(-2.0, 3.5, 1.1, 7.0)
            .unwrap()
            .components(),
        [-2.0, 3.5, 1.1, 7.0]
    );
    let negative_zero = GpuDepthClearValue::new(-0.0).unwrap();
    let positive_zero = GpuDepthClearValue::new(0.0).unwrap();
    assert_eq!(negative_zero, positive_zero);
    assert_eq!(semantic_hash(negative_zero), semantic_hash(positive_zero));
    assert_eq!(negative_zero.value().to_bits(), 0.0_f32.to_bits());
    assert!(GpuColorClearValue::new(f64::NAN, 0.0, 0.0, 1.0).is_err());
    assert!(GpuColorClearValue::new(f64::INFINITY, 0.0, 0.0, 1.0).is_err());
    assert!(GpuDepthClearValue::new(f32::INFINITY).is_err());
    assert!(GpuDepthClearValue::new(-0.1).is_err());
}

#[test]
fn dispatch_draw_and_indirect_access_are_checked() {
    assert_eq!(GpuDispatchSize::new(0, 1, 1).as_array(), [0, 1, 1]);
    assert_eq!(GpuDispatchSize::new(2, 3, 4).as_array(), [2, 3, 4]);
    assert!(GpuDrawRange::new(0, 0).is_err());
    let mut allocator = allocator();
    let arguments = buffer(&mut allocator, "arguments", 64, [GpuBufferUsage::Indirect]);
    let range = GpuBufferRange::new(&arguments, 0, 16).unwrap();
    let draw = GpuDrawIntent::indirect(&arguments, range, false).unwrap();
    assert!(draw.derived_access().unwrap().is_some());
    assert!(GpuDrawIntent::indirect(&arguments, range, true).is_err());
    let elements = GpuDrawRange::new(3, 9).unwrap();
    let instances = GpuDrawRange::new(0, 2).unwrap();
    assert!(!GpuDrawIntent::direct(elements, instances).is_indexed());
    assert!(GpuDrawIntent::indexed(elements, -2, instances).is_indexed());
}

#[test]
fn render_attachment_views_select_exactly_one_subresource() {
    let mut allocator = allocator();
    let layered = texture_with_shape(
        &mut allocator,
        "layered target",
        TestTextureShape {
            width: 16,
            height: 16,
            layers: 2,
            mip_levels: 2,
            sample_count: 1,
            format: GpuTextureFormat::Rgba8Unorm,
        },
        [GpuTextureUsage::ColorAttachment],
    );
    let whole = texture_view(
        &mut allocator,
        &layered,
        "whole layered target view",
        GpuTextureSubresourceRange::whole(&layered).unwrap(),
    );
    assert!(
        GpuRenderColorAttachment::new(
            whole,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .is_err()
    );

    let exact = single_view(
        &mut allocator,
        &layered,
        "single layered target view",
        1,
        1,
        GpuTextureAspect::Color,
    );
    assert!(
        GpuRenderColorAttachment::new(
            exact,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .is_ok()
    );
}

#[test]
fn multisample_resolve_is_an_attachment_relation() {
    let mut allocator = allocator();
    let source = texture(
        &mut allocator,
        "msaa",
        4,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let destination = texture(
        &mut allocator,
        "resolved",
        1,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let source_view = single_view(
        &mut allocator,
        &source,
        "msaa view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let destination_view = single_view(
        &mut allocator,
        &destination,
        "resolved view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let resolve = GpuMultisampleResolveTarget::new(destination_view).unwrap();
    let attachment = GpuRenderColorAttachment::new(
        source_view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Discard,
        Some(resolve),
    )
    .unwrap();
    let operation = GpuRenderOperation::new([attachment], None, [], None).unwrap();
    assert_eq!(operation.accesses().len(), 2);
}

#[test]
fn multisample_resolve_rejects_sample_format_and_alias_mismatches() {
    let mut allocator = allocator();
    let single_source = texture(
        &mut allocator,
        "single source",
        1,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let single_destination = texture(
        &mut allocator,
        "single destination",
        1,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let single_source_view = single_view(
        &mut allocator,
        &single_source,
        "single source view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let single_destination_view = single_view(
        &mut allocator,
        &single_destination,
        "single destination view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let resolve = GpuMultisampleResolveTarget::new(single_destination_view).unwrap();
    assert!(
        GpuRenderColorAttachment::new(
            single_source_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            Some(resolve),
        )
        .is_err()
    );

    let multisampled = texture(
        &mut allocator,
        "multisampled",
        4,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let multisampled_view = single_view(
        &mut allocator,
        &multisampled,
        "multisampled view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let wrong_samples = texture(
        &mut allocator,
        "wrong samples",
        4,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let wrong_samples_view = single_view(
        &mut allocator,
        &wrong_samples,
        "wrong samples view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let resolve = GpuMultisampleResolveTarget::new(wrong_samples_view).unwrap();
    assert!(
        GpuRenderColorAttachment::new(
            multisampled_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            Some(resolve),
        )
        .is_err()
    );

    let wrong_format = texture(
        &mut allocator,
        "wrong format",
        1,
        GpuTextureFormat::Bgra8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let wrong_format_view = single_view(
        &mut allocator,
        &wrong_format,
        "wrong format view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let resolve = GpuMultisampleResolveTarget::new(wrong_format_view).unwrap();
    assert!(
        GpuRenderColorAttachment::new(
            multisampled_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            Some(resolve),
        )
        .is_err()
    );

    let alias_view = single_view(
        &mut allocator,
        &multisampled,
        "alias resolve view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let alias = GpuMultisampleResolveTarget::new(alias_view).unwrap();
    assert!(
        GpuRenderColorAttachment::new(
            multisampled_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            Some(alias),
        )
        .is_err()
    );
}

#[test]
fn multisample_resolve_rejects_extent_and_subresource_mismatches() {
    let mut allocator = allocator();
    let source = texture(
        &mut allocator,
        "multisample source",
        4,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let source_view = single_view(
        &mut allocator,
        &source,
        "multisample source view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let wrong_extent = texture_with_shape(
        &mut allocator,
        "wrong extent",
        TestTextureShape {
            width: 8,
            height: 8,
            layers: 1,
            mip_levels: 1,
            sample_count: 1,
            format: GpuTextureFormat::Rgba8Unorm,
        },
        [GpuTextureUsage::ColorAttachment],
    );
    let wrong_extent_view = single_view(
        &mut allocator,
        &wrong_extent,
        "wrong extent view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let wrong_extent_resolve = GpuMultisampleResolveTarget::new(wrong_extent_view).unwrap();
    assert!(
        GpuRenderColorAttachment::new(
            source_view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Discard,
            Some(wrong_extent_resolve),
        )
        .is_err()
    );

    let extra_mip = texture_with_shape(
        &mut allocator,
        "extra destination mip",
        TestTextureShape {
            width: 16,
            height: 16,
            layers: 1,
            mip_levels: 2,
            sample_count: 1,
            format: GpuTextureFormat::Rgba8Unorm,
        },
        [GpuTextureUsage::ColorAttachment],
    );
    let mismatched_view = single_view(
        &mut allocator,
        &extra_mip,
        "mismatched destination mip view",
        1,
        0,
        GpuTextureAspect::Color,
    );
    let mismatched_subresources = GpuMultisampleResolveTarget::new(mismatched_view).unwrap();
    assert!(
        GpuRenderColorAttachment::new(
            source_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Discard,
            Some(mismatched_subresources),
        )
        .is_err()
    );
}

#[test]
fn load_store_only_render_is_rejected_clear_is_work_and_timestamp_requires_attachment() {
    let mut allocator = allocator();
    let target = texture(
        &mut allocator,
        "target",
        1,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::ColorAttachment],
    );
    let target_view = single_view(
        &mut allocator,
        &target,
        "target view",
        0,
        0,
        GpuTextureAspect::Color,
    );
    let load = GpuRenderColorAttachment::new(
        target_view.clone(),
        GpuColorAttachmentLoad::Load,
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    assert!(load.source_access().kind().reads());
    assert!(load.source_access().kind().writes());
    assert!(GpuRenderOperation::new([load.clone()], None, [], None).is_err());
    let clear = GpuRenderColorAttachment::new(
        target_view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    assert!(GpuRenderOperation::new([clear], None, [], None).is_ok());

    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("queries"), GpuQueryKind::Timestamp, 2).unwrap(),
        )
        .unwrap();
    let query = GpuQueryAccess::new(
        &queries,
        GpuQueryRange::new(&queries, 0, 1).unwrap(),
        GpuQueryAccessKind::WriteTimestamp,
    )
    .unwrap();
    assert!(
        GpuResourceAccess::Query(query)
            .derived_requirements()
            .unwrap()
            .get(GpuCapabilityFeature::TimestampQuery)
            .is_some()
    );
    let timestamp_writes = GpuTimestampWrites::new(&queries, Some(0), None).unwrap();
    assert!(GpuRenderOperation::new([], None, [], Some(timestamp_writes.clone())).is_err());
    assert!(GpuRenderOperation::new([load], None, [], Some(timestamp_writes)).is_ok());
}

#[test]
fn depth_attachment_load_clear_store_and_requirements_are_typed() {
    let mut allocator = allocator();
    let depth = texture(
        &mut allocator,
        "depth",
        1,
        GpuTextureFormat::Depth32Float,
        [GpuTextureUsage::DepthStencilAttachment],
    );
    let depth_view = single_view(
        &mut allocator,
        &depth,
        "depth view",
        0,
        0,
        GpuTextureAspect::DepthOnly,
    );
    let read_only = GpuRenderDepthStencilAttachment::new(
        depth_view.clone(),
        GpuDepthStencilAccess::ReadOnly,
        GpuDepthAttachmentLoad::Load,
        GpuAttachmentStore::Store,
    )
    .unwrap();
    assert!(read_only.source_access().kind().reads());
    assert!(!read_only.source_access().kind().writes());
    assert!(
        GpuRenderDepthStencilAttachment::new(
            depth_view.clone(),
            GpuDepthStencilAccess::ReadOnly,
            GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(1.0).unwrap()),
            GpuAttachmentStore::Store,
        )
        .is_err()
    );
    assert!(
        GpuRenderDepthStencilAttachment::new(
            depth_view.clone(),
            GpuDepthStencilAccess::ReadOnly,
            GpuDepthAttachmentLoad::Load,
            GpuAttachmentStore::Discard,
        )
        .is_err()
    );

    let clear = GpuRenderDepthStencilAttachment::new(
        depth_view,
        GpuDepthStencilAccess::ReadWrite,
        GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.5).unwrap()),
        GpuAttachmentStore::Discard,
    )
    .unwrap();
    assert!(!clear.source_access().kind().reads());
    assert!(clear.source_access().kind().writes());
    let operation =
        GpuWorkOperation::Render(GpuRenderOperation::new([], Some(clear), [], None).unwrap());
    let requirements = operation.derived_requirements().unwrap();
    assert!(
        requirements
            .get(GpuCapabilityFeature::RenderPipeline)
            .is_some()
    );
    assert!(
        requirements
            .get(GpuCapabilityFeature::DepthAttachment)
            .is_some()
    );
    assert_eq!(operation.derived_accesses().unwrap().len(), 1);
}

#[test]
fn all_copy_directions_validate_logical_coverage() {
    let mut allocator = allocator();
    let source = buffer(&mut allocator, "source", 2048, [GpuBufferUsage::CopySource]);
    let destination = buffer(
        &mut allocator,
        "destination",
        2048,
        [GpuBufferUsage::CopyDestination],
    );
    let buffer_copy = GpuCopyOperation::buffer_to_buffer(
        GpuBufferRegion::new(&source, GpuBufferRange::new(&source, 0, 64).unwrap()).unwrap(),
        GpuBufferRegion::new(
            &destination,
            GpuBufferRange::new(&destination, 0, 64).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        GpuWorkOperation::Copy(buffer_copy)
            .derived_accesses()
            .unwrap()
            .len(),
        2
    );
    let texture_source = texture(
        &mut allocator,
        "texture source",
        1,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::CopySource],
    );
    let texture_destination = texture(
        &mut allocator,
        "texture destination",
        1,
        GpuTextureFormat::Rgba8Unorm,
        [GpuTextureUsage::CopyDestination],
    );
    let extent = GpuCopyExtent::new(16, 16, 1).unwrap();
    let source_region = GpuTextureCopyRegion::new(
        &texture_source,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        extent,
    )
    .unwrap();
    let destination_region = GpuTextureCopyRegion::new(
        &texture_destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        extent,
    )
    .unwrap();
    let source_layout = GpuBufferTextureLayout::new(&source, 0, 64, 0).unwrap();
    let destination_layout = GpuBufferTextureLayout::new(&destination, 0, 64, 0).unwrap();
    let buffer_texture =
        GpuCopyOperation::buffer_to_texture(source_layout, destination_region.clone()).unwrap();
    let accesses = GpuWorkOperation::Copy(buffer_texture)
        .derived_accesses()
        .unwrap();
    assert_eq!(accesses.len(), 2);
    assert!(matches!(
        &accesses[0],
        GpuResourceAccess::Buffer(access) if access.range().size() == 1_024
    ));
    let texture_buffer =
        GpuCopyOperation::texture_to_buffer(source_region.clone(), destination_layout).unwrap();
    assert_eq!(
        GpuWorkOperation::Copy(texture_buffer)
            .derived_accesses()
            .unwrap()
            .len(),
        2
    );
    let texture_texture =
        GpuCopyOperation::texture_to_texture(source_region, destination_region).unwrap();
    assert_eq!(
        GpuWorkOperation::Copy(texture_texture)
            .derived_accesses()
            .unwrap()
            .len(),
        2
    );
    let invalid_layout = GpuBufferTextureLayout::new(&source, 0, 63, 0).unwrap();
    assert!(
        GpuCopyOperation::buffer_to_texture(
            invalid_layout,
            GpuTextureCopyRegion::new(
                &texture_destination,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                extent,
            )
            .unwrap(),
        )
        .is_err()
    );
}

#[test]
fn buffer_zero_and_query_resolve_derive_exact_accesses() {
    let mut allocator = allocator();
    let zero = buffer(
        &mut allocator,
        "zero",
        64,
        [GpuBufferUsage::CopyDestination],
    );
    let zero_region =
        GpuBufferRegion::new(&zero, GpuBufferRange::new(&zero, 8, 16).unwrap()).unwrap();
    let clear = GpuClearOperation::buffer_zero(zero_region).unwrap();
    assert_eq!(
        GpuWorkOperation::Clear(clear.clone())
            .derived_accesses()
            .unwrap()
            .len(),
        1
    );
    assert!(
        GpuWorkOperation::Clear(clear)
            .derived_requirements()
            .unwrap()
            .get(GpuCapabilityFeature::Copy)
            .is_some()
    );
    let wrong_zero = buffer(&mut allocator, "wrong zero", 64, [GpuBufferUsage::Storage]);
    assert_eq!(
        GpuClearOperation::buffer_zero(
            GpuBufferRegion::new(
                &wrong_zero,
                GpuBufferRange::new(&wrong_zero, 0, 16).unwrap(),
            )
            .unwrap(),
        )
        .unwrap_err()
        .cause(),
        GpuWorkOperationCause::InvalidBufferZero
    );

    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("queries"), GpuQueryKind::Timestamp, 4).unwrap(),
        )
        .unwrap();
    let resolve = buffer(
        &mut allocator,
        "resolve",
        64,
        [GpuBufferUsage::QueryResolve, GpuBufferUsage::CopySource],
    );
    let operation = GpuQueryResolveOperation::new(
        &queries,
        GpuQueryRange::new(&queries, 1, 2).unwrap(),
        &resolve,
        8,
    )
    .unwrap();
    assert_eq!(operation.destination_range().offset(), 8);
    assert_eq!(operation.destination_range().size(), 16);
    assert_eq!(
        operation.source_access().kind(),
        GpuQueryAccessKind::ResolveSource
    );
    assert_eq!(
        operation.destination_access().kind(),
        GpuBufferAccessKind::QueryResolveDestination
    );
    assert_eq!(
        GpuWorkOperation::Resolve(operation.clone())
            .derived_accesses()
            .unwrap()
            .len(),
        2
    );
    assert!(
        GpuWorkOperation::Resolve(operation.clone())
            .derived_requirements()
            .unwrap()
            .get(GpuCapabilityFeature::TimestampQuery)
            .is_some()
    );

    let wrong_usage = buffer(
        &mut allocator,
        "wrong resolve usage",
        64,
        [GpuBufferUsage::CopyDestination],
    );
    assert_eq!(
        GpuQueryResolveOperation::new(
            &queries,
            GpuQueryRange::new(&queries, 0, 1).unwrap(),
            &wrong_usage,
            0,
        )
        .unwrap_err()
        .cause(),
        GpuWorkOperationCause::InvalidQueryResolution
    );
    let too_small = buffer(
        &mut allocator,
        "small resolve",
        8,
        [GpuBufferUsage::QueryResolve],
    );
    assert_eq!(
        GpuQueryResolveOperation::new(
            &queries,
            GpuQueryRange::new(&queries, 0, 2).unwrap(),
            &too_small,
            0,
        )
        .unwrap_err()
        .cause(),
        GpuWorkOperationCause::QueryDestinationOutOfBounds
    );
    let huge = buffer(
        &mut allocator,
        "huge resolve",
        u64::MAX,
        [GpuBufferUsage::QueryResolve],
    );
    assert_eq!(
        GpuQueryResolveOperation::new(
            &queries,
            GpuQueryRange::new(&queries, 0, 2).unwrap(),
            &huge,
            u64::MAX - 7,
        )
        .unwrap_err()
        .cause(),
        GpuWorkOperationCause::QueryDestinationOverflow
    );
}
