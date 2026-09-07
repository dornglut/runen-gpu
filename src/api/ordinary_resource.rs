use super::{
    GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages, GpuMemoryIntent,
    GpuReconstruction, GpuResourceCommon, GpuResourceDescriptorError, GpuResourceLabel,
    GpuResourceLifetime, GpuResourceProvenance, GpuTextureAspect, GpuTextureDescriptor,
    GpuTextureDimension, GpuTextureExtent, GpuTextureFormat, GpuTextureHandle,
    GpuTextureInitialization, GpuTextureSubresourceRange, GpuTextureUsage, GpuTextureUsages,
    GpuTextureViewDescriptor,
};

fn ordinary_owned_common(
    label: impl AsRef<str>,
    lifetime: GpuResourceLifetime,
    reconstruction: GpuReconstruction,
) -> Result<GpuResourceCommon, GpuResourceDescriptorError> {
    let label = GpuResourceLabel::new(label.as_ref())?;
    let provenance = GpuResourceProvenance::new(label.clone(), None, None);
    GpuResourceCommon::owned(
        label,
        lifetime,
        GpuMemoryIntent::Device,
        reconstruction,
        provenance,
    )
}

impl GpuBufferDescriptor {
    /// Constructs the ordinary owned device-buffer case while keeping the
    /// workload's lifetime, reconstruction, size, usages, and initialization
    /// decisions explicit.
    ///
    /// Ownership, device-memory intent, diagnostic label plumbing, and ordinary
    /// provenance are derived before lowering through [`GpuBufferDescriptor::new`].
    /// Imported resources, upload/readback memory, explicit provenance, and other
    /// non-ordinary cases remain on the explicit common + descriptor constructors.
    pub fn ordinary_owned(
        label: impl AsRef<str>,
        lifetime: GpuResourceLifetime,
        reconstruction: GpuReconstruction,
        size_bytes: u64,
        usages: impl IntoIterator<Item = GpuBufferUsage>,
        initialization: GpuBufferInitialization,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let common = ordinary_owned_common(label, lifetime, reconstruction)?;
        let usages = GpuBufferUsages::new(common.label(), usages)?;
        Self::new(common, size_bytes, usages, initialization)
    }
}

impl GpuTextureDescriptor {
    /// Constructs the ordinary owned 2D texture case: one layer, one mip, and
    /// single-sample rendering.
    ///
    /// Lifetime, reconstruction, extent, format, usages, and initialization stay
    /// explicit. Ownership, device-memory intent, diagnostic provenance, D2
    /// dimension, one layer, one mip, and sample count 1 are the constrained
    /// ordinary defaults. Use [`GpuTextureDescriptor::new`] for arrays, mip chains,
    /// multisampling, imports, explicit provenance, or other advanced semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn ordinary_owned_2d(
        label: impl AsRef<str>,
        lifetime: GpuResourceLifetime,
        reconstruction: GpuReconstruction,
        width: u32,
        height: u32,
        format: GpuTextureFormat,
        usages: impl IntoIterator<Item = GpuTextureUsage>,
        initialization: GpuTextureInitialization,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let common = ordinary_owned_common(label, lifetime, reconstruction)?;
        let extent =
            GpuTextureExtent::new(common.label(), GpuTextureDimension::D2, width, height, 1)?;
        let usages = GpuTextureUsages::new(common.label(), usages)?;
        Self::new(
            common,
            GpuTextureDimension::D2,
            extent,
            1,
            1,
            format,
            usages,
            initialization,
        )
    }
}

impl GpuTextureViewDescriptor {
    /// Constructs the ordinary full-resource view for an owned texture,
    /// preserving the parent's lifetime and reconstruction policy.
    ///
    /// Parent format, dimension, mip range, array-layer range, and aspect are
    /// derived structurally. Canonical common/view validation still rejects
    /// imported/surface ownership mismatches, while retained non-reconstructable
    /// policy continues to require explicit risk acceptance on the advanced path.
    pub fn ordinary_full_owned(
        label: impl AsRef<str>,
        texture: &GpuTextureHandle,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let parent = texture.descriptor();
        let common = ordinary_owned_common(
            label,
            parent.common().lifetime(),
            parent.common().reconstruction(),
        )?;
        let array_layer_count = match parent.dimension() {
            GpuTextureDimension::D2 => parent.extent().depth_or_layers(),
            GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
        };
        let subresources = GpuTextureSubresourceRange::new(
            common.label(),
            0,
            parent.mip_level_count(),
            0,
            array_layer_count,
            GpuTextureAspect::All,
        )?;
        Self::new(common, texture, None, parent.dimension(), subresources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuResourceDescriptorCause, GpuResourceOwnership, GpuWorkResourceIdAllocator};

    #[test]
    fn ordinary_owned_buffer_derives_only_framework_administration() {
        let descriptor = GpuBufferDescriptor::ordinary_owned(
            "particles",
            GpuResourceLifetime::Retained,
            GpuReconstruction::SourceBacked,
            64,
            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap();

        assert_eq!(descriptor.common().label().as_str(), "particles");
        assert_eq!(descriptor.common().ownership(), GpuResourceOwnership::Owned);
        assert_eq!(descriptor.common().memory_intent(), GpuMemoryIntent::Device);
        assert_eq!(
            descriptor.common().reconstruction(),
            GpuReconstruction::SourceBacked
        );
        assert_eq!(
            descriptor.common().provenance().producer().as_str(),
            "particles"
        );
        assert_eq!(descriptor.size_bytes(), 64);
        assert!(descriptor.usages().contains(GpuBufferUsage::Storage));
        assert!(
            descriptor
                .usages()
                .contains(GpuBufferUsage::CopyDestination)
        );
        assert_eq!(
            descriptor.initialization(),
            &GpuBufferInitialization::Uninitialized
        );
    }

    #[test]
    fn ordinary_owned_buffer_retains_non_reconstructable_risk_gate() {
        let error = GpuBufferDescriptor::ordinary_owned(
            "retained device state",
            GpuResourceLifetime::Retained,
            GpuReconstruction::NonReconstructable,
            64,
            [GpuBufferUsage::Storage],
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap_err();

        assert_eq!(
            error.cause(),
            GpuResourceDescriptorCause::InvalidReconstruction
        );
    }

    #[test]
    fn ordinary_owned_2d_texture_materializes_documented_defaults() {
        let descriptor = GpuTextureDescriptor::ordinary_owned_2d(
            "target",
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            64,
            32,
            GpuTextureFormat::Rgba8Unorm,
            [
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopySource,
            ],
            GpuTextureInitialization::Uninitialized,
        )
        .unwrap();

        assert_eq!(descriptor.dimension(), GpuTextureDimension::D2);
        assert_eq!(descriptor.extent().width(), 64);
        assert_eq!(descriptor.extent().height(), 32);
        assert_eq!(descriptor.extent().depth_or_layers(), 1);
        assert_eq!(descriptor.mip_level_count(), 1);
        assert_eq!(descriptor.sample_count(), 1);
        assert_eq!(descriptor.format(), GpuTextureFormat::Rgba8Unorm);
        assert_eq!(descriptor.common().ownership(), GpuResourceOwnership::Owned);
        assert_eq!(descriptor.common().memory_intent(), GpuMemoryIntent::Device);
        assert_eq!(
            descriptor.common().reconstruction(),
            GpuReconstruction::SourceBacked
        );
    }

    #[test]
    fn ordinary_full_owned_view_derives_parent_geometry_and_policy() {
        let texture_descriptor = GpuTextureDescriptor::ordinary_owned_2d(
            "target",
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            64,
            32,
            GpuTextureFormat::Rgba8Unorm,
            [GpuTextureUsage::ColorAttachment],
            GpuTextureInitialization::Uninitialized,
        )
        .unwrap();
        let mut identities = GpuWorkResourceIdAllocator::new();
        let texture = identities
            .allocate_texture_handle(texture_descriptor)
            .unwrap();

        let view = GpuTextureViewDescriptor::ordinary_full_owned("target view", &texture).unwrap();

        assert_eq!(view.texture(), &texture);
        assert_eq!(view.format(), None);
        assert_eq!(view.dimension(), GpuTextureDimension::D2);
        assert_eq!(view.subresources().base_mip_level(), 0);
        assert_eq!(view.subresources().mip_level_count(), 1);
        assert_eq!(view.subresources().base_array_layer(), 0);
        assert_eq!(view.subresources().array_layer_count(), 1);
        assert_eq!(view.subresources().aspect(), GpuTextureAspect::All);
        assert_eq!(
            view.common().lifetime(),
            texture.descriptor().common().lifetime()
        );
        assert_eq!(
            view.common().reconstruction(),
            texture.descriptor().common().reconstruction()
        );
    }

    #[test]
    fn ordinary_full_owned_view_does_not_adopt_imported_parent_ownership() {
        let parent_label = GpuResourceLabel::new("imported target").unwrap();
        let parent_common = GpuResourceCommon::imported(
            parent_label.clone(),
            GpuResourceLifetime::Transient,
            GpuResourceProvenance::new(parent_label.clone(), Some(1), None),
        );
        let extent =
            GpuTextureExtent::new(&parent_label, GpuTextureDimension::D2, 64, 32, 1).unwrap();
        let usages =
            GpuTextureUsages::new(&parent_label, [GpuTextureUsage::ColorAttachment]).unwrap();
        let texture_descriptor = GpuTextureDescriptor::new(
            parent_common,
            GpuTextureDimension::D2,
            extent,
            1,
            1,
            GpuTextureFormat::Rgba8Unorm,
            usages,
            GpuTextureInitialization::Uninitialized,
        )
        .unwrap();
        let mut identities = GpuWorkResourceIdAllocator::new();
        let texture = identities
            .allocate_texture_handle(texture_descriptor)
            .unwrap();

        let error =
            GpuTextureViewDescriptor::ordinary_full_owned("imported view", &texture).unwrap_err();

        assert_eq!(
            error.cause(),
            GpuResourceDescriptorCause::ParentLeaseMismatch
        );
    }
}
