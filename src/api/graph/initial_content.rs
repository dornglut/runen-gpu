use super::super::{
    GpuBufferHandle, GpuBufferInitialization, GpuResourceAccess, GpuTextureHandle,
    GpuTextureInitialization, GpuWorkGraphCause, GpuWorkGraphError, GpuWorkGraphErrorContext,
    GpuWorkGraphErrorSource, GpuWorkResourceId,
};
use super::authoring::GpuWorkFragment;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuPreparedInitialContent {
    Buffer(GpuBufferHandle),
    Texture(GpuTextureHandle),
}

impl GpuPreparedInitialContent {
    pub(crate) fn resource_identity(&self) -> GpuWorkResourceId {
        match self {
            Self::Buffer(buffer) => buffer.diagnostic_identity(),
            Self::Texture(texture) => texture.diagnostic_identity(),
        }
    }
}

pub(super) fn derive_prepared_initial_content(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
) -> Result<Vec<GpuPreparedInitialContent>, GpuWorkGraphError> {
    let mut candidates = BTreeMap::<GpuWorkResourceId, GpuPreparedInitialContent>::new();

    for fragment in fragments {
        for node in fragment.nodes() {
            let derived = node.operation().derived_accesses().map_err(|source| {
                let resource = source.resource();
                GpuWorkGraphError::with_source(
                    "derive GPU prepared initial-content candidates",
                    GpuWorkGraphErrorContext::new(
                        graph_label,
                        Some(fragment.label().as_str().to_string()),
                        Some(node.label().as_str().to_string()),
                        None,
                        resource,
                        None,
                        Some(node.provenance().clone()),
                    ),
                    GpuWorkGraphCause::OperationAccessContradiction,
                    "retain the checked operation-derived accesses used by graph preparation",
                    GpuWorkGraphErrorSource::Operation(source),
                )
            })?;

            for access in derived {
                let candidate = match access {
                    GpuResourceAccess::Buffer(access)
                        if matches!(
                            access.buffer().descriptor().initialization(),
                            GpuBufferInitialization::Prepared(_)
                        ) =>
                    {
                        Some(GpuPreparedInitialContent::Buffer(access.buffer().clone()))
                    }
                    GpuResourceAccess::Texture(access)
                        if matches!(
                            access.normalized_texture().descriptor().initialization(),
                            GpuTextureInitialization::Prepared(_)
                        ) =>
                    {
                        Some(GpuPreparedInitialContent::Texture(
                            access.normalized_texture().clone(),
                        ))
                    }
                    _ => None,
                };
                let Some(candidate) = candidate else {
                    continue;
                };
                let identity = candidate.resource_identity();
                if let Some(existing) = candidates.get(&identity) {
                    if existing != &candidate {
                        return Err(GpuWorkGraphError::invalid(
                            "derive GPU prepared initial-content candidates",
                            GpuWorkGraphErrorContext::new(
                                graph_label,
                                Some(fragment.label().as_str().to_string()),
                                Some(node.label().as_str().to_string()),
                                None,
                                Some(identity),
                                None,
                                Some(node.provenance().clone()),
                            ),
                            GpuWorkGraphCause::UnknownIdentity,
                            "retain one kind-preserving prepared resource for each normalized storage identity",
                        ));
                    }
                    continue;
                }
                candidates.insert(identity, candidate);
            }
        }
    }

    Ok(candidates.into_values().collect())
}
