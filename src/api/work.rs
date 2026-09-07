use super::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirementError,
    GpuCapabilityRequirements, GpuQueryAccessKind, GpuResourceAccess, GpuTextureAccessKind,
    GpuTextureDimension, GpuTextureHandle,
};

#[cfg(test)]
use crate::{
    GpuAttachmentStore, GpuBufferAccessKind, GpuBufferHandle, GpuBufferRange,
    GpuDepthStencilAccess, GpuQueryAccess, GpuQueryKind, GpuQueryRange, GpuTextureAspect,
    GpuTextureFormat, GpuTextureSubresourceRange, GpuTextureViewHandle, GpuWorkOperationCause,
};

mod attachment;
mod clear;
mod compute;
mod copy;
mod draw;
mod present;
mod query;

pub use attachment::{
    GpuColorAttachmentLoad, GpuColorClearValue, GpuDepthAttachmentLoad, GpuDepthClearValue,
    GpuMultisampleResolveTarget, GpuRenderColorAttachment, GpuRenderDepthStencilAttachment,
};
pub use clear::GpuClearOperation;
pub use compute::{GpuComputeOperation, GpuDispatchSize};
pub use copy::{
    GpuBufferRegion, GpuBufferTextureLayout, GpuCopyExtent, GpuCopyOperation, GpuTextureCopyRegion,
    GpuTextureOrigin,
};
pub use draw::{GpuDrawIntent, GpuDrawRange};
pub use present::GpuPresentOperation;
pub use query::{GpuQueryResolveOperation, GpuTimestampWrites};

fn mip_extent(texture: &GpuTextureHandle, mip_level: u32) -> (u32, u32, u32) {
    let descriptor = texture.descriptor();
    let extent = descriptor.extent();
    let width = (extent.width() >> mip_level).max(1);
    let height = (extent.height() >> mip_level).max(1);
    let depth = match descriptor.dimension() {
        GpuTextureDimension::D3 => (extent.depth_or_layers() >> mip_level).max(1),
        GpuTextureDimension::D2 => extent.depth_or_layers(),
        GpuTextureDimension::D1 => 1,
    };
    (width, height, depth)
}

pub(crate) fn add_access_requirements(
    requirements: &mut GpuCapabilityRequirements,
    access: &GpuResourceAccess,
) -> Result<(), GpuCapabilityRequirementError> {
    match access {
        GpuResourceAccess::Texture(access)
            if matches!(
                access.kind(),
                GpuTextureAccessKind::StorageRead
                    | GpuTextureAccessKind::StorageWrite
                    | GpuTextureAccessKind::StorageReadWrite
            ) =>
        {
            requirements.insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::StorageTexture,
            ))?;
        }
        GpuResourceAccess::Query(access) if access.kind() == GpuQueryAccessKind::WriteTimestamp => {
            requirements.insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::TimestampQuery,
            ))?;
        }
        _ => {}
    }
    Ok(())
}

impl GpuResourceAccess {
    pub fn derived_requirements(
        &self,
    ) -> Result<GpuCapabilityRequirements, GpuCapabilityRequirementError> {
        let mut requirements = GpuCapabilityRequirements::new();
        add_access_requirements(&mut requirements, self)?;
        Ok(requirements)
    }
}

#[cfg(test)]
mod tests;
