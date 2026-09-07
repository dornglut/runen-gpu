use super::{
    BufferRealizationRecord, QuerySetRealizationRecord, SamplerRealizationRecord,
    TextureRealizationRecord, TextureViewRealizationRecord,
};
use crate::{
    GpuResourceRealizationError, GpuResourceRealizationErrorCategory, GpuResourceRealizationPolicy,
    GpuResourceRealizationStats, GpuWorkResourceId,
};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) trait RealizationRecord {
    type Descriptor: PartialEq;

    fn descriptor(&self) -> &Self::Descriptor;
}

#[derive(Debug)]
pub(super) struct AuthoritativeRegistry<Record> {
    records: BTreeMap<GpuWorkResourceId, Arc<Record>>,
}

impl<Record> Default for AuthoritativeRegistry<Record> {
    fn default() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }
}

impl<Record: RealizationRecord> AuthoritativeRegistry<Record> {
    pub(super) fn lookup(
        &self,
        identity: GpuWorkResourceId,
        descriptor: &Record::Descriptor,
    ) -> Result<Option<Arc<Record>>, GpuResourceRealizationError> {
        let Some(existing) = self.records.get(&identity) else {
            return Ok(None);
        };
        if existing.descriptor() == descriptor {
            Ok(Some(Arc::clone(existing)))
        } else {
            Err(GpuResourceRealizationError::new(
                GpuResourceRealizationErrorCategory::DescriptorChangedForIdentity,
                Some(identity),
                "the authoritative record retains different complete resource semantics",
            ))
        }
    }

    pub(super) fn insert(&mut self, identity: GpuWorkResourceId, record: Arc<Record>) {
        let replaced = self.records.insert(identity, record);
        debug_assert!(
            replaced.is_none(),
            "registry insertion follows an authoritative miss"
        );
    }

    pub(super) fn contains(&self, identity: GpuWorkResourceId) -> bool {
        self.records.contains_key(&identity)
    }

    pub(super) fn len(&self) -> usize {
        self.records.len()
    }

    pub(super) fn collect_lookup_only(&mut self) -> usize {
        let before = self.records.len();
        self.records
            .retain(|_, record| Arc::strong_count(record) > 1);
        before - self.records.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResourceKind {
    Buffer,
    Texture,
    TextureView,
    Sampler,
    QuerySet,
}

#[derive(Debug)]
pub(super) struct ResourceRegistries<
    Buffer = BufferRealizationRecord,
    Texture = TextureRealizationRecord,
    TextureView = TextureViewRealizationRecord,
    Sampler = SamplerRealizationRecord,
    QuerySet = QuerySetRealizationRecord,
> {
    pub(super) buffers: AuthoritativeRegistry<Buffer>,
    pub(super) textures: AuthoritativeRegistry<Texture>,
    pub(super) texture_views: AuthoritativeRegistry<TextureView>,
    pub(super) samplers: AuthoritativeRegistry<Sampler>,
    pub(super) query_sets: AuthoritativeRegistry<QuerySet>,
}

impl<Buffer, Texture, TextureView, Sampler, QuerySet> Default
    for ResourceRegistries<Buffer, Texture, TextureView, Sampler, QuerySet>
{
    fn default() -> Self {
        Self {
            buffers: AuthoritativeRegistry::default(),
            textures: AuthoritativeRegistry::default(),
            texture_views: AuthoritativeRegistry::default(),
            samplers: AuthoritativeRegistry::default(),
            query_sets: AuthoritativeRegistry::default(),
        }
    }
}

impl<Buffer, Texture, TextureView, Sampler, QuerySet>
    ResourceRegistries<Buffer, Texture, TextureView, Sampler, QuerySet>
where
    Buffer: RealizationRecord,
    Texture: RealizationRecord,
    TextureView: RealizationRecord,
    Sampler: RealizationRecord,
    QuerySet: RealizationRecord,
{
    pub(super) fn reject_other_kind(
        &self,
        requested: ResourceKind,
        identity: GpuWorkResourceId,
    ) -> Result<(), GpuResourceRealizationError> {
        let collision = match requested {
            ResourceKind::Buffer => {
                self.textures.contains(identity)
                    || self.texture_views.contains(identity)
                    || self.samplers.contains(identity)
                    || self.query_sets.contains(identity)
            }
            ResourceKind::Texture => {
                self.buffers.contains(identity)
                    || self.texture_views.contains(identity)
                    || self.samplers.contains(identity)
                    || self.query_sets.contains(identity)
            }
            ResourceKind::TextureView => {
                self.buffers.contains(identity)
                    || self.textures.contains(identity)
                    || self.samplers.contains(identity)
                    || self.query_sets.contains(identity)
            }
            ResourceKind::Sampler => {
                self.buffers.contains(identity)
                    || self.textures.contains(identity)
                    || self.texture_views.contains(identity)
                    || self.query_sets.contains(identity)
            }
            ResourceKind::QuerySet => {
                self.buffers.contains(identity)
                    || self.textures.contains(identity)
                    || self.texture_views.contains(identity)
                    || self.samplers.contains(identity)
            }
        };
        if collision {
            Err(GpuResourceRealizationError::new(
                GpuResourceRealizationErrorCategory::ResourceKindMismatch,
                Some(identity),
                "the logical identity already belongs to another resource family",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn ensure_capacity(
        &mut self,
        identity: GpuWorkResourceId,
        policy: GpuResourceRealizationPolicy,
    ) -> Result<(), GpuResourceRealizationError> {
        if self.total_records() < policy.max_records().get() {
            return Ok(());
        }

        self.collect_lookup_only();
        if self.total_records() < policy.max_records().get() {
            Ok(())
        } else {
            Err(GpuResourceRealizationError::capacity(
                identity,
                self.total_records(),
                policy.max_records(),
            ))
        }
    }

    pub(super) fn stats(
        &self,
        policy: GpuResourceRealizationPolicy,
    ) -> GpuResourceRealizationStats {
        GpuResourceRealizationStats::new(
            policy.max_records(),
            self.buffers.len(),
            self.textures.len(),
            self.texture_views.len(),
            self.samplers.len(),
            self.query_sets.len(),
        )
    }

    fn total_records(&self) -> usize {
        self.buffers.len()
            + self.textures.len()
            + self.texture_views.len()
            + self.samplers.len()
            + self.query_sets.len()
    }

    fn collect_lookup_only(&mut self) -> usize {
        // A view record retains its parent texture. Removing lookup-only views first lets a parent
        // that is otherwise unretained become collectible in the same bounded pressure pass.
        self.texture_views.collect_lookup_only()
            + self.buffers.collect_lookup_only()
            + self.samplers.collect_lookup_only()
            + self.query_sets.collect_lookup_only()
            + self.textures.collect_lookup_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[derive(Debug, Clone)]
    struct SemanticDescriptor {
        semantic_value: u32,
        diagnostic_label: &'static str,
    }

    impl PartialEq for SemanticDescriptor {
        fn eq(&self, other: &Self) -> bool {
            self.semantic_value == other.semantic_value
        }
    }

    #[derive(Debug)]
    struct TestRecord {
        descriptor: SemanticDescriptor,
        retained_parent: Option<Arc<TestRecord>>,
    }

    impl TestRecord {
        fn new(semantic_value: u32, diagnostic_label: &'static str) -> Self {
            Self {
                descriptor: SemanticDescriptor {
                    semantic_value,
                    diagnostic_label,
                },
                retained_parent: None,
            }
        }

        fn view(semantic_value: u32, parent: Arc<TestRecord>) -> Self {
            Self {
                descriptor: SemanticDescriptor {
                    semantic_value,
                    diagnostic_label: "view",
                },
                retained_parent: Some(parent),
            }
        }
    }

    impl RealizationRecord for TestRecord {
        type Descriptor = SemanticDescriptor;

        fn descriptor(&self) -> &Self::Descriptor {
            &self.descriptor
        }
    }

    fn identities() -> impl Iterator<Item = GpuWorkResourceId> {
        let mut allocator = crate::GpuWorkResourceIdAllocator::new();
        (0..8).map(move |_| allocator.allocate().unwrap())
    }

    #[test]
    fn authoritative_lookup_reuses_only_same_identity_and_complete_semantics() {
        let mut identities = identities();
        let first_id = identities.next().unwrap();
        let second_id = identities.next().unwrap();
        let mut registry = AuthoritativeRegistry::<TestRecord>::default();
        let first = Arc::new(TestRecord::new(7, "first label"));
        registry.insert(first_id, Arc::clone(&first));

        let equal_with_new_diagnostics = SemanticDescriptor {
            semantic_value: 7,
            diagnostic_label: "rediscovered label",
        };
        let hit = registry
            .lookup(first_id, &equal_with_new_diagnostics)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&hit, &first));
        assert_eq!(hit.descriptor.diagnostic_label, "first label");

        let changed = registry
            .lookup(
                first_id,
                &SemanticDescriptor {
                    semantic_value: 8,
                    diagnostic_label: "changed",
                },
            )
            .unwrap_err();
        assert_eq!(
            changed.category(),
            GpuResourceRealizationErrorCategory::DescriptorChangedForIdentity
        );

        assert!(
            registry
                .lookup(second_id, &equal_with_new_diagnostics)
                .unwrap()
                .is_none(),
            "equal descriptors must not alias distinct logical identities"
        );
    }

    #[test]
    fn total_capacity_collects_views_before_their_lookup_only_parent() {
        type TestRegistries =
            ResourceRegistries<TestRecord, TestRecord, TestRecord, TestRecord, TestRecord>;
        let mut identities = identities();
        let texture_id = identities.next().unwrap();
        let view_id = identities.next().unwrap();
        let replacement_id = identities.next().unwrap();
        let parent = Arc::new(TestRecord::new(1, "parent"));
        let view = Arc::new(TestRecord::view(2, Arc::clone(&parent)));
        assert!(view.retained_parent.is_some());

        let mut registries = TestRegistries::default();
        registries.textures.insert(texture_id, parent);
        registries.texture_views.insert(view_id, view);

        registries
            .ensure_capacity(
                replacement_id,
                GpuResourceRealizationPolicy::new(NonZeroUsize::new(2).unwrap()),
            )
            .unwrap();
        assert_eq!(
            registries
                .stats(GpuResourceRealizationPolicy::new(
                    NonZeroUsize::new(2).unwrap()
                ))
                .retained_records(),
            0
        );
    }

    #[test]
    fn live_records_are_never_evicted_and_capacity_fails_before_creation() {
        type TestRegistries =
            ResourceRegistries<TestRecord, TestRecord, TestRecord, TestRecord, TestRecord>;
        let mut identities = identities();
        let live_id = identities.next().unwrap();
        let attempted_id = identities.next().unwrap();
        let live = Arc::new(TestRecord::new(1, "live"));
        let retained_handle = Arc::clone(&live);
        let mut registries = TestRegistries::default();
        registries.buffers.insert(live_id, live);

        let error = registries
            .ensure_capacity(
                attempted_id,
                GpuResourceRealizationPolicy::new(NonZeroUsize::MIN),
            )
            .unwrap_err();
        assert_eq!(
            error.category(),
            GpuResourceRealizationErrorCategory::RegistryCapacityExceeded
        );
        assert!(registries.buffers.contains(live_id));
        assert_eq!(retained_handle.descriptor.semantic_value, 1);
    }

    #[test]
    fn one_identity_cannot_cross_typed_registry_families() {
        type TestRegistries =
            ResourceRegistries<TestRecord, TestRecord, TestRecord, TestRecord, TestRecord>;
        let identity = identities().next().unwrap();
        let mut registries = TestRegistries::default();
        registries
            .buffers
            .insert(identity, Arc::new(TestRecord::new(1, "buffer")));

        let error = registries
            .reject_other_kind(ResourceKind::Texture, identity)
            .unwrap_err();
        assert_eq!(
            error.category(),
            GpuResourceRealizationErrorCategory::ResourceKindMismatch
        );
    }
}
