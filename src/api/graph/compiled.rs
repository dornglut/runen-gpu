//! Owner-local, source-neutral G3 *typed hazard* template.
//!
//! This is intentionally not a cached GpuPreparedWorkGraph. It stores no
//! resource handles, raw logical identities, programs, surface leases, upload
//! contents, retained initialization, or previously authored GPU operations.
//! A caller-provided template may only replace single-fragment typed hazard
//! inference after current-frame G3 node/resource/boundary validation. All
//! remaining canonical graph and physical submission validation still runs.
use super::{
    authoring::GpuWorkFragment,
    dependency::{GpuDependencyReason, GpuDependencyRegion},
    hazards::{DependencyEdges, infer_fragment_hazards},
    identity::GpuPreparedWorkNodeId,
};
use crate::api::{
    GpuBufferAccessKind, GpuBufferRange, GpuQueryAccessKind, GpuQueryRange, GpuResourceAccess,
    GpuResourceRef, GpuTextureAccessKind, GpuTextureFormat, GpuTextureSubresourceRange,
    GpuWorkNodeKind, GpuWorkResourceId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceKind {
    Buffer,
    Texture,
    TextureView,
    Sampler,
    QuerySet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResourceSlot {
    kind: ResourceKind,
    /// Position in the frame-local normalized storage owner ordering.
    storage_slot: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AccessSlot {
    Buffer {
        declared: usize,
        storage: usize,
        kind: GpuBufferAccessKind,
        range: GpuBufferRange,
    },
    Texture {
        declared: usize,
        storage: usize,
        kind: GpuTextureAccessKind,
        requested: GpuTextureSubresourceRange,
        normalized: GpuTextureSubresourceRange,
        parent_format: GpuTextureFormat,
    },
    Query {
        declared: usize,
        storage: usize,
        kind: GpuQueryAccessKind,
        range: GpuQueryRange,
    },
    Sampler {
        declared: usize,
        storage: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeSlot {
    kind: GpuWorkNodeKind,
    accesses: Vec<AccessSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HazardShape {
    resources: Vec<ResourceSlot>,
    nodes: Vec<NodeSlot>,
}

struct CurrentFrameSlots {
    shape: HazardShape,
    /// These identities are only for binding the *current* frame. They are
    /// never retained in GpuCompiledHazardTemplate or used as cache keys.
    storage_ids: Vec<GpuWorkResourceId>,
    storage_by_id: BTreeMap<GpuWorkResourceId, usize>,
}

impl CurrentFrameSlots {
    fn capture(fragment: &GpuWorkFragment) -> Option<Self> {
        let mut declared_by_id = BTreeMap::<GpuWorkResourceId, usize>::new();
        let mut storage_by_id = BTreeMap::<GpuWorkResourceId, usize>::new();
        let mut storage_ids = Vec::new();
        let mut resources = Vec::new();
        for (ordinal, resource) in fragment.resources().iter().enumerate() {
            let kind = match resource {
                GpuResourceRef::Buffer(_) => ResourceKind::Buffer,
                GpuResourceRef::Texture(_) => ResourceKind::Texture,
                GpuResourceRef::TextureView(_) => ResourceKind::TextureView,
                GpuResourceRef::Sampler(_) => ResourceKind::Sampler,
                GpuResourceRef::QuerySet(_) => ResourceKind::QuerySet,
            };
            let storage_id = match resource {
                GpuResourceRef::TextureView(view) => {
                    view.descriptor().texture().diagnostic_identity()
                }
                _ => resource.diagnostic_identity(),
            };
            if declared_by_id
                .insert(resource.diagnostic_identity(), ordinal)
                .is_some()
            {
                return None;
            }
            let storage = match storage_by_id.get(&storage_id) {
                Some(slot) => *slot,
                None => {
                    let slot = storage_ids.len();
                    storage_ids.push(storage_id);
                    storage_by_id.insert(storage_id, slot);
                    slot
                }
            };
            resources.push(ResourceSlot {
                kind,
                storage_slot: storage,
            });
        }

        let mut nodes = Vec::with_capacity(fragment.nodes().len());
        for node in fragment.nodes() {
            let mut accesses = Vec::with_capacity(node.accesses().len());
            for access in node.accesses() {
                let declared = *declared_by_id.get(&access.declared_resource_identity())?;
                let storage = *storage_by_id.get(&access.resource_identity())?;
                let pattern = match access {
                    GpuResourceAccess::Buffer(buffer) => AccessSlot::Buffer {
                        declared,
                        storage,
                        kind: buffer.kind(),
                        range: buffer.range(),
                    },
                    GpuResourceAccess::Texture(texture) => AccessSlot::Texture {
                        declared,
                        storage,
                        kind: texture.kind(),
                        requested: texture.requested_subresources(),
                        normalized: texture.normalized_subresources(),
                        parent_format: texture.normalized_texture().descriptor().format(),
                    },
                    GpuResourceAccess::Query(query) => AccessSlot::Query {
                        declared,
                        storage,
                        kind: query.kind(),
                        range: query.range(),
                    },
                    GpuResourceAccess::Sampler(_) => AccessSlot::Sampler { declared, storage },
                };
                accesses.push(pattern);
            }
            nodes.push(NodeSlot {
                kind: node.kind(),
                accesses,
            });
        }

        Some(Self {
            shape: HazardShape { resources, nodes },
            storage_ids,
            storage_by_id,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum TypedReasonKind {
    Raw,
    War,
    Waw,
}

#[derive(Debug, Clone)]
struct TypedReasonSlot {
    storage_slot: usize,
    region: GpuDependencyRegion,
    kind: TypedReasonKind,
}

#[derive(Debug, Clone)]
struct TypedEdgeSlot {
    before: usize,
    after: usize,
    reasons: Vec<TypedReasonSlot>,
}

/// A bounded, single-fragment *hint* for typed access-derived hazards.
///
/// Compilation is sourced from RunenGPU's canonical hazard inference. The
/// template contains only lexical resource/access slots and typed edge
/// topology, never handles or operations from its source frame. It is neither
/// a prepared submission nor a certificate of dynamic initialization.
/// Incompatible frames must take full canonical G3 hazard inference.
#[derive(Debug, Clone)]
pub struct GpuCompiledHazardTemplate {
    shape: HazardShape,
    typed_edges: Vec<TypedEdgeSlot>,
}

impl GpuCompiledHazardTemplate {
    /// Compile a verified author's single-fragment typed access topology.
    /// Returns None if the fragment cannot be represented losslessly by
    /// currently supported resource/access slots; normal preparation remains
    /// available without this optional performance hint.
    pub fn compile(fragment: &GpuWorkFragment) -> Option<Self> {
        let binding = CurrentFrameSlots::capture(fragment)?;
        let mut edges = DependencyEdges::new();
        infer_fragment_hazards(
            fragment.label().as_str(),
            std::slice::from_ref(fragment),
            &mut edges,
        )
        .ok()?;

        let mut typed_edges = Vec::with_capacity(edges.len());
        for ((before, after), reasons) in edges {
            if before.fragment_ordinal() != 0 || after.fragment_ordinal() != 0 {
                return None;
            }
            let before = usize::try_from(before.local_node().checked_sub(1)?).ok()?;
            let after = usize::try_from(after.local_node().checked_sub(1)?).ok()?;
            if before >= binding.shape.nodes.len() || after >= binding.shape.nodes.len() {
                return None;
            }
            let reasons = reasons
                .into_iter()
                .map(|reason| {
                    let storage_slot =
                        *binding.storage_by_id.get(&reason.resource()?)?;
                    let region = reason.region()?;
                    let kind = match reason {
                        GpuDependencyReason::ReadAfterWrite { .. } => TypedReasonKind::Raw,
                        GpuDependencyReason::WriteAfterRead { .. } => TypedReasonKind::War,
                        GpuDependencyReason::WriteAfterWrite { .. } => TypedReasonKind::Waw,
                        GpuDependencyReason::ExplicitNonData { .. } => return None,
                    };
                    Some(TypedReasonSlot {
                        storage_slot,
                        region,
                        kind,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            typed_edges.push(TypedEdgeSlot {
                before,
                after,
                reasons,
            });
        }

        Some(Self {
            shape: binding.shape,
            typed_edges,
        })
    }

    /// Checks only whether the *typed hazard* structure is identical. A true
    /// result does not admit programs, devices, retained content or surfaces.
    pub fn matches_fragment(&self, fragment: &GpuWorkFragment) -> bool {
        CurrentFrameSlots::capture(fragment)
            .is_some_and(|current| current.shape == self.shape)
    }

    pub(super) fn bind_fresh_typed_edges(
        &self,
        fragment: &GpuWorkFragment,
    ) -> Option<DependencyEdges> {
        let current = CurrentFrameSlots::capture(fragment)?;
        if current.shape != self.shape {
            return None;
        }
        let mut edges = DependencyEdges::new();
        for edge in &self.typed_edges {
            let before = GpuPreparedWorkNodeId::new(0, fragment.nodes().get(edge.before)?.id().local);
            let after = GpuPreparedWorkNodeId::new(0, fragment.nodes().get(edge.after)?.id().local);
            let mut reasons = BTreeSet::new();
            for reason in &edge.reasons {
                let resource = *current.storage_ids.get(reason.storage_slot)?;
                let region = reason.region;
                reasons.insert(match reason.kind {
                    TypedReasonKind::Raw => GpuDependencyReason::ReadAfterWrite { resource, region },
                    TypedReasonKind::War => GpuDependencyReason::WriteAfterRead { resource, region },
                    TypedReasonKind::Waw => GpuDependencyReason::WriteAfterWrite { resource, region },
                });
            }
            edges.entry((before, after)).or_default().extend(reasons);
        }
        Some(edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferRegion, GpuBufferUsage,
        GpuClearOperation, GpuReconstruction, GpuResourceLifetime, GpuResourceScope,
    };

    fn fragment(two_writers_share_resource: bool) -> GpuWorkFragment {
        let mut scope = GpuResourceScope::new();
        let first = scope
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "compiled frame slot buffer",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    16,
                    [GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let second = scope
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "alternate compiled frame slot buffer",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    16,
                    [GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        GpuWorkFragment::build("compiled hazards independent source", |builder| {
            builder.operation(
                "first buffer writer",
                GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&first).unwrap()).unwrap(),
            )?;
            builder.operation(
                "second buffer writer",
                GpuClearOperation::buffer_zero(GpuBufferRegion::whole(if two_writers_share_resource {
                    &first
                } else {
                    &second
                }).unwrap()).unwrap(),
            )?;
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn separate_resource_scopes_rebind_only_current_typed_resource_id() {
        let first = fragment(true);
        let second = fragment(true);
        let template = GpuCompiledHazardTemplate::compile(&first).unwrap();
        assert!(template.matches_fragment(&second));
        let rebound = template.bind_fresh_typed_edges(&second).unwrap();
        let mut canonical = DependencyEdges::new();
        infer_fragment_hazards(
            "fresh canonical fixture",
            std::slice::from_ref(&second),
            &mut canonical,
        )
        .unwrap();
        assert_eq!(rebound, canonical);
        assert_eq!(rebound.len(), 1);
        let first_id = first.resources()[0].diagnostic_identity();
        let current_id = second.resources()[0].diagnostic_identity();
        assert_ne!(first_id, current_id);
        assert!(rebound.values().all(|reasons| {
            reasons.iter().all(|reason| reason.resource() == Some(current_id))
        }));
    }

    #[test]
    fn fresh_frame_preparation_matches_canonical_graph_and_reasons() {
        use crate::api::{GpuGraphRequiredOrder, GpuPreparedWorkGraph, GpuResourceLabel};

        let source = fragment(true);
        let fresh = fragment(true);
        let template = GpuCompiledHazardTemplate::compile(&source).unwrap();
        let ordered = GpuGraphRequiredOrder::new(
            fresh.nodes()[0].id(),
            fresh.nodes()[1].id(),
            "current-frame control already guaranteed by typed WAW",
        )
        .unwrap();
        let label = GpuResourceLabel::new("compiled current frame").unwrap();
        let rebound = GpuPreparedWorkGraph::prepare_with_required_orders_using_hazard_template(
            label.clone(),
            fresh.clone(),
            [ordered.clone()],
            &template,
        )
        .unwrap();
        let canonical =
            GpuPreparedWorkGraph::prepare_with_required_orders(label, [fresh], [ordered]).unwrap();
        assert_eq!(rebound.dependencies(), canonical.dependencies());
        assert_eq!(rebound.topological_order(), canonical.topological_order());
        assert_eq!(rebound.initialization(), canonical.initialization());
        assert_eq!(rebound.diagnostics(), canonical.diagnostics());
        assert_eq!(rebound.nodes().len(), canonical.nodes().len());
        assert_eq!(rebound.dependencies().len(), 1);
        assert!(rebound.dependencies()[0]
            .reasons()
            .iter()
            .all(|reason| reason.resource().is_some()));
    }

    #[test]
    fn changed_alias_structure_prepares_canonical_required_controls() {
        use crate::api::{GpuGraphRequiredOrder, GpuPreparedWorkGraph, GpuResourceLabel};

        let source = fragment(true);
        let fresh = fragment(false);
        let template = GpuCompiledHazardTemplate::compile(&source).unwrap();
        let ordered = GpuGraphRequiredOrder::new(
            fresh.nodes()[0].id(),
            fresh.nodes()[1].id(),
            "independent writer control",
        )
        .unwrap();
        let label = GpuResourceLabel::new("changed fresh frame").unwrap();
        let rebound = GpuPreparedWorkGraph::prepare_with_required_orders_using_hazard_template(
            label.clone(),
            fresh.clone(),
            [ordered.clone()],
            &template,
        )
        .unwrap();
        let canonical =
            GpuPreparedWorkGraph::prepare_with_required_orders(label, [fresh], [ordered]).unwrap();
        assert_eq!(rebound.dependencies(), canonical.dependencies());
        assert_eq!(rebound.initialization(), canonical.initialization());
        assert_eq!(rebound.topological_order(), canonical.topological_order());
        assert_eq!(rebound.dependencies().len(), 1);
        assert!(matches!(
            rebound.dependencies()[0].reasons(),
            [GpuDependencyReason::ExplicitNonData { .. }]
        ));
    }

    #[test]
    fn changed_alias_structure_must_use_full_g3_fallback() {
        let base = fragment(true);
        let changed = fragment(false);
        let template = GpuCompiledHazardTemplate::compile(&base).unwrap();
        assert!(!template.matches_fragment(&changed));
        assert!(template.bind_fresh_typed_edges(&changed).is_none());
    }
}
