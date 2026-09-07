use super::super::{
    GpuTextureAccess, GpuTextureAccessKind, GpuTextureAccessResource, GpuTextureAspect,
    GpuTextureSubresourceRange, GpuTextureViewHandle, GpuWorkOperationCause, GpuWorkOperationError,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuPresentOperation {
    source: GpuTextureAccessResource,
    subresource: GpuTextureSubresourceRange,
    source_access: GpuTextureAccess,
}

impl GpuPresentOperation {
    pub fn new(
        source: GpuTextureAccessResource,
        subresource: GpuTextureSubresourceRange,
    ) -> Result<Self, GpuWorkOperationError> {
        if subresource.mip_level_count() != 1
            || subresource.array_layer_count() != 1
            || !matches!(
                subresource.aspect(),
                GpuTextureAspect::All | GpuTextureAspect::Color
            )
            || source.parent_texture().descriptor().format().is_depth()
            || source.parent_texture().descriptor().sample_count() != 1
        {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU present operation",
                source
                    .parent_texture()
                    .descriptor()
                    .common()
                    .label()
                    .as_str(),
                Some(source.parent_texture().diagnostic_identity()),
                GpuWorkOperationCause::InvalidAttachment,
                "select exactly one single-sampled color mip and array layer",
            ));
        }
        let source_access =
            GpuTextureAccess::new(source.clone(), subresource, GpuTextureAccessKind::Present)
                .map_err(|source| {
                    GpuWorkOperationError::from_access(
                        "construct GPU present operation",
                        "present",
                        GpuWorkOperationCause::InvalidAttachment,
                        "provide one checked color source subresource",
                        source,
                    )
                })?;
        Ok(Self {
            source,
            subresource,
            source_access,
        })
    }

    /// Constructs Present for the complete checked range already owned by one texture view.
    ///
    /// The view descriptor remains the subresource authority, so ordinary callers do not need to
    /// restate `view.descriptor().subresources()`. Canonical Present validation still requires the
    /// resulting view range to select exactly one single-sampled color mip and array layer. Use
    /// [`Self::new`] for a bare texture source or explicit subset selection inside a broader view.
    pub fn whole_view(source: &GpuTextureViewHandle) -> Result<Self, GpuWorkOperationError> {
        Self::new(
            GpuTextureAccessResource::TextureView(source.clone()),
            source.descriptor().subresources(),
        )
    }

    pub fn source(&self) -> &GpuTextureAccessResource {
        &self.source
    }
    pub const fn subresource(&self) -> GpuTextureSubresourceRange {
        self.subresource
    }
    pub fn source_access(&self) -> &GpuTextureAccess {
        &self.source_access
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuReconstruction, GpuResourceLifetime, GpuResourceScope, GpuTextureDescriptor,
        GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage, GpuTextureViewDescriptor,
    };

    #[test]
    fn whole_view_preserves_exact_view_identity_and_descriptor_range() {
        let mut resources = GpuResourceScope::new();
        let texture = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "present texture",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    32,
                    16,
                    GpuTextureFormat::Rgba8Unorm,
                    [GpuTextureUsage::ColorAttachment],
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let view = resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned("present view", &texture).unwrap(),
            )
            .unwrap();

        let operation = GpuPresentOperation::whole_view(&view).unwrap();

        assert_eq!(
            operation.source(),
            &GpuTextureAccessResource::TextureView(view.clone())
        );
        assert_eq!(operation.subresource(), view.descriptor().subresources());
        assert_eq!(
            operation.source_access().requested_subresources(),
            view.descriptor().subresources()
        );
        let normalized = operation.source_access().normalized_subresources();
        assert_eq!(normalized.base_mip_level(), 0);
        assert_eq!(normalized.mip_level_count(), 1);
        assert_eq!(normalized.base_array_layer(), 0);
        assert_eq!(normalized.array_layer_count(), 1);
        assert_eq!(normalized.aspect(), GpuTextureAspect::Color);
    }
}
