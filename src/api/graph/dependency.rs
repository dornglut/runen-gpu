use super::super::{
    GpuBufferRange, GpuQueryRange, GpuResourceAccess, GpuTextureSubresourceRange, GpuWorkResourceId,
};
use super::coverage::{canonical_texture_aspect, texture_aspect};
use super::identity::GpuPreparedWorkNodeId;
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDependencyRegion {
    Buffer(GpuBufferRange),
    Texture(GpuTextureSubresourceRange),
    Query(GpuQueryRange),
}

impl fmt::Display for GpuDependencyRegion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffer(range) => {
                write!(
                    formatter,
                    "buffer bytes {}..{}",
                    range.offset(),
                    range.end()
                )
            }
            Self::Texture(range) => write!(
                formatter,
                "texture mips {}..{}, layers {}..{}, aspect {:?}",
                range.base_mip_level(),
                range.mip_end(),
                range.base_array_layer(),
                range.layer_end(),
                range.aspect()
            ),
            Self::Query(range) => {
                write!(formatter, "queries {}..{}", range.first(), range.end())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDependencyReason {
    ReadAfterWrite {
        resource: GpuWorkResourceId,
        region: GpuDependencyRegion,
    },
    WriteAfterRead {
        resource: GpuWorkResourceId,
        region: GpuDependencyRegion,
    },
    WriteAfterWrite {
        resource: GpuWorkResourceId,
        region: GpuDependencyRegion,
    },
    ExplicitNonData {
        reason: String,
    },
}

impl GpuDependencyReason {
    pub const fn resource(&self) -> Option<GpuWorkResourceId> {
        match self {
            Self::ReadAfterWrite { resource, .. }
            | Self::WriteAfterRead { resource, .. }
            | Self::WriteAfterWrite { resource, .. } => Some(*resource),
            Self::ExplicitNonData { .. } => None,
        }
    }

    pub const fn region(&self) -> Option<GpuDependencyRegion> {
        match self {
            Self::ReadAfterWrite { region, .. }
            | Self::WriteAfterRead { region, .. }
            | Self::WriteAfterWrite { region, .. } => Some(*region),
            Self::ExplicitNonData { .. } => None,
        }
    }
}

pub(super) fn access_intersection(
    left: &GpuResourceAccess,
    right: &GpuResourceAccess,
) -> Option<(GpuWorkResourceId, GpuDependencyRegion)> {
    if left.resource_identity() != right.resource_identity() {
        return None;
    }
    let resource = left.resource_identity();
    let region = match (left, right) {
        (GpuResourceAccess::Buffer(left), GpuResourceAccess::Buffer(right)) => {
            let start = left.range().offset().max(right.range().offset());
            let end = left.range().end().min(right.range().end());
            if start >= end {
                return None;
            }
            GpuDependencyRegion::Buffer(
                GpuBufferRange::new(left.buffer(), start, end - start).ok()?,
            )
        }
        (GpuResourceAccess::Texture(left), GpuResourceAccess::Texture(right)) => {
            let left_range = left.normalized_subresources();
            let right_range = right.normalized_subresources();
            let texture = left.normalized_texture();
            let parent_aspect = texture_aspect(texture);
            let left_aspect = canonical_texture_aspect(left_range.aspect(), parent_aspect);
            let right_aspect = canonical_texture_aspect(right_range.aspect(), parent_aspect);
            let mip_start = left_range
                .base_mip_level()
                .max(right_range.base_mip_level());
            let mip_end = left_range.mip_end().min(right_range.mip_end());
            let layer_start = left_range
                .base_array_layer()
                .max(right_range.base_array_layer());
            let layer_end = left_range.layer_end().min(right_range.layer_end());
            if mip_start >= mip_end || layer_start >= layer_end || left_aspect != right_aspect {
                return None;
            }
            GpuDependencyRegion::Texture(
                GpuTextureSubresourceRange::new(
                    texture.descriptor().common().label(),
                    mip_start,
                    mip_end - mip_start,
                    layer_start,
                    layer_end - layer_start,
                    left_aspect,
                )
                .ok()?,
            )
        }
        (GpuResourceAccess::Query(left), GpuResourceAccess::Query(right)) => {
            let start = left.range().first().max(right.range().first());
            let end = left.range().end().min(right.range().end());
            if start >= end {
                return None;
            }
            GpuDependencyRegion::Query(
                GpuQueryRange::new(left.query_set(), start, end - start).ok()?,
            )
        }
        _ => return None,
    };
    Some((resource, region))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkDependency {
    pub(super) before: GpuPreparedWorkNodeId,
    pub(super) after: GpuPreparedWorkNodeId,
    pub(super) reasons: Vec<GpuDependencyReason>,
}

impl GpuWorkDependency {
    pub const fn before(&self) -> GpuPreparedWorkNodeId {
        self.before
    }

    pub const fn after(&self) -> GpuPreparedWorkNodeId {
        self.after
    }

    pub fn reasons(&self) -> &[GpuDependencyReason] {
        &self.reasons
    }
}
