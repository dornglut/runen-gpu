use runen_gpu::*;

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

fn texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    format: GpuTextureFormat,
) -> GpuTextureHandle {
    let resource_label = label(name);
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 16, 8, 2).unwrap(),
                2,
                1,
                format,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn depth_copy_region_requires_complete_plane_but_allows_layer_subset() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let depth = texture(&mut allocator, "depth copy", GpuTextureFormat::Depth32Float);

    assert!(
        GpuTextureCopyRegion::new(
            &depth,
            1,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            GpuCopyExtent::new(7, 4, 1).unwrap(),
        )
        .is_err(),
        "partial depth width must not survive into private encoding"
    );
    assert!(
        GpuTextureCopyRegion::new(
            &depth,
            1,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            GpuCopyExtent::new(8, 3, 1).unwrap(),
        )
        .is_err(),
        "partial depth height must not survive into private encoding"
    );

    let layer = GpuTextureCopyRegion::new(
        &depth,
        1,
        GpuTextureOrigin::new(0, 0, 1),
        GpuTextureAspect::DepthOnly,
        GpuCopyExtent::new(8, 4, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(layer.origin(), GpuTextureOrigin::new(0, 0, 1));
    assert_eq!(layer.extent(), GpuCopyExtent::new(8, 4, 1).unwrap());
    assert_eq!(layer.subresources().base_array_layer(), 1);
    assert_eq!(layer.subresources().array_layer_count(), 1);

    let all_layers = GpuTextureCopyRegion::new(
        &depth,
        1,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::All,
        GpuCopyExtent::new(8, 4, 2).unwrap(),
    )
    .unwrap();
    assert_eq!(all_layers.aspect(), GpuTextureAspect::DepthOnly);
}

#[test]
fn color_copy_region_remains_partial_and_layer_scoped() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let color = texture(&mut allocator, "color copy", GpuTextureFormat::Rgba8Unorm);

    let region = GpuTextureCopyRegion::new(
        &color,
        0,
        GpuTextureOrigin::new(3, 2, 1),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(5, 3, 1).unwrap(),
    )
    .unwrap();

    assert_eq!(region.origin(), GpuTextureOrigin::new(3, 2, 1));
    assert_eq!(region.extent(), GpuCopyExtent::new(5, 3, 1).unwrap());
    assert_eq!(region.subresources().base_array_layer(), 1);
    assert_eq!(region.subresources().array_layer_count(), 1);
}

#[test]
fn new_depth_only_formats_preserve_full_plane_and_non_linear_copy_semantics() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    for format in [
        GpuTextureFormat::Depth16Unorm,
        GpuTextureFormat::Depth24Plus,
    ] {
        let source = texture(&mut allocator, &format!("{format:?} source"), format);
        let destination = texture(&mut allocator, &format!("{format:?} destination"), format);

        assert!(
            GpuTextureCopyRegion::new(
                &source,
                0,
                GpuTextureOrigin::new(1, 0, 0),
                GpuTextureAspect::DepthOnly,
                GpuCopyExtent::new(15, 8, 1).unwrap(),
            )
            .is_err()
        );

        let source_region = GpuTextureCopyRegion::new(
            &source,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::All,
            GpuCopyExtent::new(16, 8, 1).unwrap(),
        )
        .unwrap();
        let destination_region = GpuTextureCopyRegion::new(
            &destination,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::DepthOnly,
            GpuCopyExtent::new(16, 8, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(source_region.aspect(), GpuTextureAspect::DepthOnly);
        assert!(
            GpuCopyOperation::texture_to_texture(
                source_region.clone(),
                destination_region.clone(),
            )
            .is_ok()
        );

        let buffer_label = label(&format!("{format:?} staging"));
        let buffer = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common(&format!("{format:?} staging")),
                    4096,
                    GpuBufferUsages::new(
                        &buffer_label,
                        [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
                    )
                    .unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let layout = GpuBufferTextureLayout::new(&buffer, 0, 256, 0).unwrap();
        if format == GpuTextureFormat::Depth16Unorm {
            assert!(
                GpuCopyOperation::buffer_to_texture(layout.clone(), destination_region.clone())
                    .is_ok()
            );
            assert!(GpuCopyOperation::texture_to_buffer(source_region, layout).is_ok());
        } else {
            assert!(
                GpuCopyOperation::buffer_to_texture(layout.clone(), destination_region.clone())
                    .is_err()
            );
            assert!(GpuCopyOperation::texture_to_buffer(source_region.clone(), layout).is_err());

            let payload = PreparedGpuData::<TransferData>::from_pod_transfer(
                "depth24plus upload bytes",
                &[0_u8; 256],
                GpuResourceProvenance::new(label("depth24plus upload bytes"), None, None),
            )
            .unwrap();
            assert!(GpuUploadOperation::new(source_region.clone().into(), payload).is_err());
            assert!(
                GpuReadbackOperation::new(
                    source_region.into(),
                    GpuReadbackId::allocate().unwrap(),
                )
                .is_err()
            );
        }
    }
}

#[test]
fn depth24plus_prepared_bytes_fail_closed() {
    let name = "depth24plus prepared bytes";
    let resource_label = label(name);
    let extent = GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 16, 8, 1).unwrap();
    let data = PreparedGpuData::<TransferData>::from_pod_transfer(
        name,
        &[0_u8; 256],
        GpuResourceProvenance::new(label(name), None, None),
    )
    .unwrap();
    assert!(
        GpuPreparedTextureData::new(
            &resource_label,
            data,
            GpuTextureFormat::Depth24Plus,
            extent,
            256,
            0,
        )
        .is_err()
    );
}
