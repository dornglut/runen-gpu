//! Private ready-plus-in-flight single-flight registries for G4C2 authority.

use super::records::{
    BindGroupLayoutRealizationRecord, BindGroupRealizationRecord, PipelineLayoutRealizationRecord,
    ProgramRealizationRecord, StaticBindGroupValue, static_bind_group_values,
};
use crate::{
    GpuBindGroupLayoutDescriptor, GpuContextAffinity, GpuPipelineLayoutDescriptor,
    GpuProgramBindingRealizationError, GpuProgramBindingRealizationPolicy,
    GpuProgramBindingRealizationStats, GpuProgramDescriptor, GpuProgramSourceDigest,
    GpuRuntimeBindingValue,
};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

const G4C2_WGPU_REALIZATION_COMPATIBILITY_REVISION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ProgramRequestKey {
    affinity: GpuContextAffinity,
    descriptor: GpuProgramDescriptor,
    source_digest: GpuProgramSourceDigest,
    wgpu_realization_compatibility_revision: u32,
}

impl ProgramRequestKey {
    pub(super) fn new(affinity: GpuContextAffinity, descriptor: GpuProgramDescriptor) -> Self {
        Self {
            affinity,
            source_digest: descriptor.source().digest(),
            descriptor,
            wgpu_realization_compatibility_revision: G4C2_WGPU_REALIZATION_COMPATIBILITY_REVISION,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BindGroupLayoutRequestKey {
    affinity: GpuContextAffinity,
    descriptor: GpuBindGroupLayoutDescriptor,
}

impl BindGroupLayoutRequestKey {
    pub(super) fn new(
        affinity: GpuContextAffinity,
        descriptor: GpuBindGroupLayoutDescriptor,
    ) -> Self {
        Self {
            affinity,
            descriptor,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineLayoutRequestKey {
    affinity: GpuContextAffinity,
    descriptor: GpuPipelineLayoutDescriptor,
}

impl PipelineLayoutRequestKey {
    pub(super) fn new(
        affinity: GpuContextAffinity,
        descriptor: GpuPipelineLayoutDescriptor,
    ) -> Self {
        Self {
            affinity,
            descriptor,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BindGroupRequestKey {
    affinity: GpuContextAffinity,
    layout: GpuBindGroupLayoutDescriptor,
    static_values: Vec<StaticBindGroupValue>,
}

impl BindGroupRequestKey {
    pub(super) fn new(
        affinity: GpuContextAffinity,
        layout: GpuBindGroupLayoutDescriptor,
        values: Vec<GpuRuntimeBindingValue>,
    ) -> Self {
        Self {
            affinity,
            layout,
            static_values: static_bind_group_values(values),
        }
    }

    pub(super) fn from_static(
        affinity: GpuContextAffinity,
        layout: GpuBindGroupLayoutDescriptor,
        static_values: Vec<StaticBindGroupValue>,
    ) -> Self {
        Self {
            affinity,
            layout,
            static_values,
        }
    }
}

#[derive(Default)]
pub(super) struct ProgramBindingRegistries {
    programs: SingleFlightRegistry<ProgramRequestKey, ProgramRealizationRecord>,
    bind_group_layouts:
        SingleFlightRegistry<BindGroupLayoutRequestKey, BindGroupLayoutRealizationRecord>,
    pipeline_layouts:
        SingleFlightRegistry<PipelineLayoutRequestKey, PipelineLayoutRealizationRecord>,
    bind_groups: SingleFlightRegistry<BindGroupRequestKey, BindGroupRealizationRecord>,
}

impl ProgramBindingRegistries {
    pub(super) fn stats(
        &self,
        policy: GpuProgramBindingRealizationPolicy,
    ) -> GpuProgramBindingRealizationStats {
        GpuProgramBindingRealizationStats::new(
            policy.max_records(),
            self.programs.in_flight_len()
                + self.bind_group_layouts.in_flight_len()
                + self.pipeline_layouts.in_flight_len()
                + self.bind_groups.in_flight_len(),
            self.programs.ready_len(),
            self.bind_group_layouts.ready_len(),
            self.pipeline_layouts.ready_len(),
            self.bind_groups.ready_len(),
        )
    }

    fn total_records(&self) -> usize {
        self.programs.total_len()
            + self.bind_group_layouts.total_len()
            + self.pipeline_layouts.total_len()
            + self.bind_groups.total_len()
    }

    fn collect_lookup_only(&mut self) {
        self.bind_groups.collect_lookup_only();
        self.pipeline_layouts.collect_lookup_only();
        self.bind_group_layouts.collect_lookup_only();
        self.programs.collect_lookup_only();
    }

    fn ensure_capacity(
        &mut self,
        policy: GpuProgramBindingRealizationPolicy,
        request: impl Into<String>,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        if self.total_records() >= policy.max_records().get() {
            self.collect_lookup_only();
        }
        if self.total_records() < policy.max_records().get() {
            Ok(())
        } else {
            Err(GpuProgramBindingRealizationError::capacity(
                request,
                self.total_records(),
                policy.max_records(),
            ))
        }
    }

    pub(super) fn contains_program(&self, record: &Arc<ProgramRealizationRecord>) -> bool {
        let key = ProgramRequestKey::new(record.affinity(), record.descriptor().clone());
        self.programs
            .ready
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, record))
    }

    pub(super) fn contains_pipeline_layout(
        &self,
        record: &Arc<PipelineLayoutRealizationRecord>,
    ) -> bool {
        let key = PipelineLayoutRequestKey::new(record.affinity(), record.descriptor().clone());
        self.pipeline_layouts
            .ready
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, record))
    }

    pub(super) fn contains_bind_group(&self, record: &Arc<BindGroupRealizationRecord>) -> bool {
        let key = BindGroupRequestKey::from_static(
            record.affinity(),
            record.layout_descriptor().clone(),
            record.static_values().to_vec(),
        );
        self.bind_groups
            .ready
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, record))
    }
}

struct SingleFlightRegistry<K, R> {
    ready: HashMap<K, Arc<R>>,
    in_flight: HashMap<K, Arc<InFlight<R>>>,
}

impl<K, R> Default for SingleFlightRegistry<K, R> {
    fn default() -> Self {
        Self {
            ready: HashMap::new(),
            in_flight: HashMap::new(),
        }
    }
}

impl<K, R> SingleFlightRegistry<K, R> {
    fn ready_len(&self) -> usize {
        self.ready.len()
    }

    fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    fn total_len(&self) -> usize {
        self.ready_len() + self.in_flight_len()
    }

    fn collect_lookup_only(&mut self) {
        self.ready.retain(|_, record| Arc::strong_count(record) > 1);
    }
}

pub(super) enum Reservation<K: Clone + Eq + Hash, R> {
    Ready(Arc<R>),
    Waiter(Arc<InFlight<R>>),
    Owner(OwnerReservation<K, R>),
}

pub(super) struct InFlight<R> {
    outcome: Mutex<InFlightOutcome<R>>,
    notify: Notify,
}

impl<R> Default for InFlight<R> {
    fn default() -> Self {
        Self {
            outcome: Mutex::new(InFlightOutcome::Pending),
            notify: Notify::new(),
        }
    }
}

pub(super) enum InFlightOutcome<R> {
    Pending,
    Complete(Result<Arc<R>, GpuProgramBindingRealizationError>),
    Abandoned,
}

impl<R> Clone for InFlightOutcome<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Pending => Self::Pending,
            Self::Complete(outcome) => Self::Complete(outcome.clone()),
            Self::Abandoned => Self::Abandoned,
        }
    }
}

impl<R> InFlight<R> {
    pub(super) async fn wait(&self) -> InFlightOutcome<R> {
        loop {
            let notified = self.notify.notified();
            let outcome = self
                .outcome
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if !matches!(outcome, InFlightOutcome::Pending) {
                return outcome;
            }
            notified.await;
        }
    }

    fn complete(&self, outcome: InFlightOutcome<R>) {
        *self
            .outcome
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = outcome;
        self.notify.notify_waiters();
    }
}

pub(super) struct OwnerReservation<K: Clone + Eq + Hash, R> {
    registries: Arc<Mutex<ProgramBindingRegistries>>,
    key: K,
    attempt: Arc<InFlight<R>>,
    select: fn(&mut ProgramBindingRegistries) -> &mut SingleFlightRegistry<K, R>,
    active: bool,
}

impl<K: Clone + Eq + Hash, R> core::fmt::Debug for OwnerReservation<K, R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OwnerReservation")
            .finish_non_exhaustive()
    }
}

impl<K: Clone + Eq + Hash, R> OwnerReservation<K, R> {
    pub(super) fn finish(
        mut self,
        outcome: Result<Arc<R>, GpuProgramBindingRealizationError>,
    ) -> Result<Arc<R>, GpuProgramBindingRealizationError> {
        let mut all = self
            .registries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let registry = (self.select)(&mut all);
        let owner_is_current = registry
            .in_flight
            .get(&self.key)
            .is_some_and(|current| Arc::ptr_eq(current, &self.attempt));
        if owner_is_current {
            registry.in_flight.remove(&self.key);
            if let Ok(record) = &outcome {
                registry.ready.insert(self.key.clone(), Arc::clone(record));
            }
        }
        self.active = false;
        drop(all);
        self.attempt
            .complete(InFlightOutcome::Complete(outcome.clone()));
        outcome
    }
}

impl<K: Clone + Eq + Hash, R> Drop for OwnerReservation<K, R> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut all = self
            .registries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let registry = (self.select)(&mut all);
        let owner_is_current = registry
            .in_flight
            .get(&self.key)
            .is_some_and(|current| Arc::ptr_eq(current, &self.attempt));
        if owner_is_current {
            registry.in_flight.remove(&self.key);
        }
        drop(all);
        self.attempt.complete(InFlightOutcome::Abandoned);
    }
}

impl ProgramBindingRegistries {
    pub(super) fn reserve_program(
        registries: &Arc<Mutex<Self>>,
        policy: GpuProgramBindingRealizationPolicy,
        key: ProgramRequestKey,
        request: impl Into<String>,
    ) -> Result<
        Reservation<ProgramRequestKey, ProgramRealizationRecord>,
        GpuProgramBindingRealizationError,
    > {
        reserve(registries, policy, key, request.into(), |all| {
            &mut all.programs
        })
    }

    pub(super) fn reserve_bind_group_layout(
        registries: &Arc<Mutex<Self>>,
        policy: GpuProgramBindingRealizationPolicy,
        key: BindGroupLayoutRequestKey,
        request: impl Into<String>,
    ) -> Result<
        Reservation<BindGroupLayoutRequestKey, BindGroupLayoutRealizationRecord>,
        GpuProgramBindingRealizationError,
    > {
        reserve(registries, policy, key, request.into(), |all| {
            &mut all.bind_group_layouts
        })
    }

    pub(super) fn reserve_pipeline_layout(
        registries: &Arc<Mutex<Self>>,
        policy: GpuProgramBindingRealizationPolicy,
        key: PipelineLayoutRequestKey,
        request: impl Into<String>,
    ) -> Result<
        Reservation<PipelineLayoutRequestKey, PipelineLayoutRealizationRecord>,
        GpuProgramBindingRealizationError,
    > {
        reserve(registries, policy, key, request.into(), |all| {
            &mut all.pipeline_layouts
        })
    }

    pub(super) fn reserve_bind_group(
        registries: &Arc<Mutex<Self>>,
        policy: GpuProgramBindingRealizationPolicy,
        key: BindGroupRequestKey,
        request: impl Into<String>,
    ) -> Result<
        Reservation<BindGroupRequestKey, BindGroupRealizationRecord>,
        GpuProgramBindingRealizationError,
    > {
        reserve(registries, policy, key, request.into(), |all| {
            &mut all.bind_groups
        })
    }
}

fn reserve<K, R>(
    registries: &Arc<Mutex<ProgramBindingRegistries>>,
    policy: GpuProgramBindingRealizationPolicy,
    key: K,
    request: String,
    select: fn(&mut ProgramBindingRegistries) -> &mut SingleFlightRegistry<K, R>,
) -> Result<Reservation<K, R>, GpuProgramBindingRealizationError>
where
    K: Clone + Eq + Hash + 'static,
    R: 'static,
{
    let mut all = registries
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(record) = select(&mut all).ready.get(&key).cloned() {
        return Ok(Reservation::Ready(record));
    }
    if let Some(attempt) = select(&mut all).in_flight.get(&key).cloned() {
        return Ok(Reservation::Waiter(attempt));
    }
    all.ensure_capacity(policy, request)?;
    let attempt = Arc::new(InFlight::default());
    select(&mut all)
        .in_flight
        .insert(key.clone(), Arc::clone(&attempt));
    Ok(Reservation::Owner(OwnerReservation {
        registries: Arc::clone(registries),
        key,
        attempt,
        select,
        active: true,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBindGroupLayoutDescriptor, GpuBindingLayoutRefinement, GpuContextAffinity, GpuContextId,
        GpuDeviceGeneration, GpuEntryPointName, GpuProgramBindingRealizationErrorCategory,
        GpuProgramDescriptor, GpuProgramSourceIdentity, GpuProgramSourceKey,
        GpuProgramSourceOwnerId, GpuProgramSourceProvenance, GpuProgramSourceRegistry,
        GpuProgramSourceRevision,
    };
    use core::future::Future;
    use core::task::{Context, Poll};
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    fn affinity() -> GpuContextAffinity {
        GpuContextAffinity::test_value(
            GpuContextId::test_value(NonZeroU64::new(41).expect("test context ID is nonzero")),
            GpuDeviceGeneration::test_value(
                NonZeroU64::new(3).expect("test generation is nonzero"),
            ),
        )
    }

    fn policy(max_records: usize) -> GpuProgramBindingRealizationPolicy {
        GpuProgramBindingRealizationPolicy::new(
            NonZeroUsize::new(max_records).expect("test record policy is nonzero"),
        )
    }

    fn empty_layout(group: u32) -> GpuBindGroupLayoutDescriptor {
        GpuBindGroupLayoutDescriptor::new(group, std::iter::empty::<crate::GpuBindingDeclaration>())
            .expect("empty test layout is valid")
    }

    fn program_key(name: &str) -> ProgramRequestKey {
        let source_owner = GpuProgramSourceOwnerId::allocate().expect("test source owner");
        let identity = GpuProgramSourceIdentity::new(
            source_owner,
            GpuProgramSourceKey::new(name).expect("test source key"),
            GpuProgramSourceRevision::try_from_raw(1).expect("test source revision"),
        );
        let source = GpuProgramSourceRegistry::new(1, 1024)
            .expect("test source registry")
            .admit_wgsl(
                identity,
                "@compute @workgroup_size(1) fn cs_main() {}",
                GpuProgramSourceProvenance::new("g4c2-registry-test", None)
                    .expect("test source provenance"),
            )
            .expect("test source admission");
        let entry = GpuEntryPointName::new("cs_main").expect("test entry point");
        let descriptor = GpuProgramDescriptor::new(
            source,
            [entry],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )
        .expect("test program descriptor");
        ProgramRequestKey::new(affinity(), descriptor)
    }

    macro_rules! assert_family_reserves_one_owner_and_one_waiter {
        ($reserve:ident, $key:expr, $request:literal) => {{
            let registries = Arc::new(Mutex::new(ProgramBindingRegistries::default()));
            let key = $key;
            let owner = match ProgramBindingRegistries::$reserve(
                &registries,
                policy(4),
                key.clone(),
                $request,
            )
            .expect("first reservation should succeed")
            {
                Reservation::Owner(owner) => owner,
                Reservation::Ready(_) | Reservation::Waiter(_) => {
                    panic!("first reservation must own the in-flight attempt")
                }
            };
            assert!(matches!(
                ProgramBindingRegistries::$reserve(&registries, policy(4), key, $request)
                    .expect("equal reservation should succeed"),
                Reservation::Waiter(_)
            ));
            assert_eq!(
                registries
                    .lock()
                    .expect("test registry lock")
                    .stats(policy(4))
                    .in_flight_records(),
                1,
                "equal requests must share one counted in-flight slot"
            );
            drop(owner);
            assert_eq!(
                registries
                    .lock()
                    .expect("test registry lock")
                    .stats(policy(4))
                    .retained_records(),
                0,
                "abandoning the owner must release its counted slot"
            );
        }};
    }

    #[test]
    fn every_g4c2_family_reserves_one_owner_and_equal_waiter() {
        assert_family_reserves_one_owner_and_one_waiter!(
            reserve_program,
            program_key("registry.program"),
            "program"
        );
        assert_family_reserves_one_owner_and_one_waiter!(
            reserve_bind_group_layout,
            BindGroupLayoutRequestKey::new(affinity(), empty_layout(0)),
            "bind-group layout"
        );
        assert_family_reserves_one_owner_and_one_waiter!(
            reserve_pipeline_layout,
            PipelineLayoutRequestKey::new(
                affinity(),
                GpuPipelineLayoutDescriptor::new([empty_layout(0)]).expect("test pipeline layout"),
            ),
            "pipeline layout"
        );
        assert_family_reserves_one_owner_and_one_waiter!(
            reserve_bind_group,
            BindGroupRequestKey::new(affinity(), empty_layout(0), Vec::new()),
            "bind group"
        );
    }

    #[test]
    fn distinct_keys_reserve_independently_before_backend_scope_dispatch() {
        let registries = Arc::new(Mutex::new(ProgramBindingRegistries::default()));
        let first = BindGroupLayoutRequestKey::new(affinity(), empty_layout(0));
        let second = BindGroupLayoutRequestKey::new(affinity(), empty_layout(1));
        let first = match ProgramBindingRegistries::reserve_bind_group_layout(
            &registries,
            policy(2),
            first,
            "first layout",
        )
        .expect("first distinct layout reservation")
        {
            Reservation::Owner(owner) => owner,
            Reservation::Ready(_) | Reservation::Waiter(_) => panic!("first distinct key owns"),
        };
        let second = match ProgramBindingRegistries::reserve_bind_group_layout(
            &registries,
            policy(2),
            second,
            "second layout",
        )
        .expect("second distinct layout reservation")
        {
            Reservation::Owner(owner) => owner,
            Reservation::Ready(_) | Reservation::Waiter(_) => panic!("second distinct key owns"),
        };
        assert_eq!(
            registries
                .lock()
                .expect("test registry lock")
                .stats(policy(2))
                .in_flight_records(),
            2,
            "distinct keys must not serialize on registry coordination"
        );
        drop(first);
        drop(second);
    }

    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn owner_abandonment_wakes_waiters_releases_capacity_and_allows_retry() {
        let registries = Arc::new(Mutex::new(ProgramBindingRegistries::default()));
        let key = BindGroupLayoutRequestKey::new(affinity(), empty_layout(0));
        let owner = match ProgramBindingRegistries::reserve_bind_group_layout(
            &registries,
            policy(1),
            key.clone(),
            "layout",
        )
        .expect("first reservation")
        {
            Reservation::Owner(owner) => owner,
            Reservation::Ready(_) | Reservation::Waiter(_) => panic!("first caller owns"),
        };
        let attempt = match ProgramBindingRegistries::reserve_bind_group_layout(
            &registries,
            policy(1),
            key.clone(),
            "layout",
        )
        .expect("equal reservation")
        {
            Reservation::Waiter(attempt) => attempt,
            Reservation::Ready(_) | Reservation::Owner(_) => panic!("equal caller waits"),
        };
        let counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let mut context = Context::from_waker(&waker);
        let mut waiter = std::pin::pin!(attempt.wait());
        assert!(matches!(waiter.as_mut().poll(&mut context), Poll::Pending));

        drop(owner);

        assert!(
            counter.0.load(Ordering::SeqCst) > 0,
            "owner cleanup must notify an already waiting equal caller"
        );
        assert!(matches!(
            waiter.as_mut().poll(&mut context),
            Poll::Ready(InFlightOutcome::Abandoned)
        ));
        assert_eq!(
            registries
                .lock()
                .expect("test registry lock")
                .stats(policy(1))
                .retained_records(),
            0,
            "abandoned owners must return their capacity slot"
        );
        assert!(matches!(
            ProgramBindingRegistries::reserve_bind_group_layout(
                &registries,
                policy(1),
                key,
                "layout retry",
            )
            .expect("retry reservation"),
            Reservation::Owner(_)
        ));
    }

    #[test]
    fn failed_attempts_have_no_negative_cache_and_dropped_waiters_do_not_cancel_owners() {
        let registries = Arc::new(Mutex::new(ProgramBindingRegistries::default()));
        let key = BindGroupRequestKey::new(affinity(), empty_layout(0), Vec::new());
        let owner = match ProgramBindingRegistries::reserve_bind_group(
            &registries,
            policy(1),
            key.clone(),
            "bind group",
        )
        .expect("first reservation")
        {
            Reservation::Owner(owner) => owner,
            Reservation::Ready(_) | Reservation::Waiter(_) => panic!("first caller owns"),
        };
        let waiter = match ProgramBindingRegistries::reserve_bind_group(
            &registries,
            policy(1),
            key.clone(),
            "bind group",
        )
        .expect("equal reservation")
        {
            Reservation::Waiter(attempt) => attempt,
            Reservation::Ready(_) | Reservation::Owner(_) => panic!("equal caller waits"),
        };
        drop(waiter);
        assert_eq!(
            registries
                .lock()
                .expect("test registry lock")
                .stats(policy(1))
                .in_flight_records(),
            1,
            "dropping a waiter must not cancel the owner reservation"
        );

        let failure = GpuProgramBindingRealizationError::new(
            GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            "bind group",
            "test failure",
        );
        match owner.finish(Err(failure.clone())) {
            Err(observed) => assert_eq!(observed, failure),
            Ok(_) => panic!("a failed owner attempt must not publish a ready record"),
        }
        assert_eq!(
            registries
                .lock()
                .expect("test registry lock")
                .stats(policy(1))
                .retained_records(),
            0,
            "a failed attempt must remove its in-flight slot rather than cache a failure"
        );
        assert!(matches!(
            ProgramBindingRegistries::reserve_bind_group(
                &registries,
                policy(1),
                key,
                "bind group retry",
            )
            .expect("retry reservation"),
            Reservation::Owner(_)
        ));
    }

    #[test]
    fn completed_waiters_receive_the_exact_shared_arc_outcome() {
        let attempt = InFlight::<usize>::default();
        let first = Arc::new(17usize);
        let second = Arc::clone(&first);
        attempt.complete(InFlightOutcome::Complete(Ok(first)));

        let outcome = {
            let mut future = std::pin::pin!(attempt.wait());
            let waker = Waker::from(Arc::new(WakeCounter(AtomicUsize::new(0))));
            let mut context = Context::from_waker(&waker);
            match future.as_mut().poll(&mut context) {
                Poll::Ready(outcome) => outcome,
                Poll::Pending => panic!("completed in-flight attempt must resolve immediately"),
            }
        };
        match outcome {
            InFlightOutcome::Complete(Ok(record)) => {
                assert!(Arc::ptr_eq(&record, &second));
            }
            InFlightOutcome::Complete(Err(_))
            | InFlightOutcome::Abandoned
            | InFlightOutcome::Pending => {
                panic!("completed success must retain the shared authoritative Arc")
            }
        }
    }
}
