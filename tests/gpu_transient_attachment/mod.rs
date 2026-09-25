use runen_gpu::*;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const EXPECTED: [u8; 4] = [255, 0, 255, 255];

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    let value = value.as_ref();
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: impl AsRef<str>, lifetime: GpuResourceLifetime) -> GpuResourceCommon {
    let value = value.as_ref();
    GpuResourceCommon::owned(
        label(value),
        lifetime,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn transient_target(
    scope: &mut GpuResourceScope,
    name: &str,
    sample_count: u32,
    lifetime: GpuResourceLifetime,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name, lifetime),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                sample_count,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::TransientAttachment,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = scope
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn ordinary_resolve_target(
    scope: &mut GpuResourceScope,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = "transient resolve target";
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name, GpuResourceLifetime::Transient),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::CopySource,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = scope
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(
                "transient resolve target view",
                &texture,
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn clear() -> GpuColorAttachmentLoad {
    GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(1.0, 0.0, 1.0, 1.0).unwrap())
}

pub(crate) fn descriptor(backend: GpuBackendFamily) -> GpuContextDescriptor {
    GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([backend])
        .with_label("transient attachment retained proof")
}

pub(crate) fn graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let (_single_texture, single_view) = transient_target(
        &mut scope,
        "single-sample transient attachment",
        1,
        GpuResourceLifetime::Retained,
    );
    let (_msaa_texture, msaa_view) = transient_target(
        &mut scope,
        "multisample transient attachment",
        4,
        GpuResourceLifetime::Transient,
    );
    let (resolve_texture, resolve_view) = ordinary_resolve_target(&mut scope);

    let single = GpuRenderOperation::new(
        [
            GpuRenderColorAttachment::new(single_view, clear(), GpuAttachmentStore::Discard, None)
                .unwrap(),
        ],
        None,
        [],
        None,
    )
    .unwrap();

    let resolve = GpuMultisampleResolveTarget::new(resolve_view).unwrap();
    let multisample = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            msaa_view,
            clear(),
            GpuAttachmentStore::Discard,
            Some(resolve),
        )
        .unwrap()],
        None,
        [],
        None,
    )
    .unwrap();

    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuTextureCopyRegion::new(
            &resolve_texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        )
        .unwrap()
        .into(),
        readback_id,
    )
    .unwrap();

    let fragment = GpuWorkFragment::build("transient attachment retained proof", |builder| {
        builder.operation("single-sample transient clear discard", single)?;
        builder.operation("multisample transient resolve", multisample)?;
        builder.operation("read ordinary resolve target", readback)?;
        Ok(())
    })
    .unwrap();

    (
        GpuPreparedWorkGraph::prepare(
            label("transient attachment retained proof graph"),
            [fragment],
        )
        .unwrap(),
        readback_id,
    )
}

pub(crate) fn assert_resolved(bytes: &GpuReadbackBytes) {
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).unwrap()
    );
    for pixel in bytes.as_bytes().chunks_exact(4) {
        assert_eq!(pixel, EXPECTED);
    }
}
