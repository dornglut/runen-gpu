use super::diagnostics::PipelineCacheObservation;
use crate::GpuPipelineRealizationError;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use tokio::sync::watch;

pub(super) struct SingleFlightRegistry<K, R> {
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
    pub(super) fn total_len(&self) -> usize {
        self.ready.len() + self.in_flight.len()
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
    outcome: watch::Sender<InFlightOutcome<R>>,
}

impl<R> Default for InFlight<R> {
    fn default() -> Self {
        let (outcome, _receiver) = watch::channel(InFlightOutcome::Pending);
        Self { outcome }
    }
}

pub(super) enum InFlightOutcome<R> {
    Pending,
    Complete(Result<Arc<R>, GpuPipelineRealizationError>),
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
        let mut receiver = self.outcome.subscribe();
        loop {
            let outcome = receiver.borrow_and_update().clone();
            if !matches!(outcome, InFlightOutcome::Pending) {
                return outcome;
            }
            if receiver.changed().await.is_err() {
                return InFlightOutcome::Abandoned;
            }
        }
    }

    fn complete(&self, outcome: InFlightOutcome<R>) {
        self.outcome.send_replace(outcome);
    }
}

pub(super) struct OwnerReservation<K: Clone + Eq + Hash, R> {
    registry: Arc<Mutex<SingleFlightRegistry<K, R>>>,
    key: K,
    attempt: Arc<InFlight<R>>,
    active: bool,
}

impl<K: Clone + Eq + Hash, R> OwnerReservation<K, R> {
    pub(super) fn finish(
        mut self,
        outcome: Result<Arc<R>, GpuPipelineRealizationError>,
    ) -> Result<Arc<R>, GpuPipelineRealizationError> {
        let mut registry = self
            .registry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        drop(registry);
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
        let mut registry = self
            .registry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let owner_is_current = registry
            .in_flight
            .get(&self.key)
            .is_some_and(|current| Arc::ptr_eq(current, &self.attempt));
        if owner_is_current {
            registry.in_flight.remove(&self.key);
        }
        drop(registry);
        self.attempt.complete(InFlightOutcome::Abandoned);
    }
}

pub(super) fn reserve<K, R>(
    registry: &Arc<Mutex<SingleFlightRegistry<K, R>>>,
    max_records: NonZeroUsize,
    key: K,
    request: impl Into<String>,
    ready_matches: impl FnOnce(&K, &R) -> bool,
) -> Result<(Reservation<K, R>, PipelineCacheObservation), GpuPipelineRealizationError>
where
    K: Clone + Eq + Hash + 'static,
    R: 'static,
{
    let request = request.into();
    let mut locked = registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rejected = false;
    if let Some(record) = locked.ready.get(&key).cloned() {
        if ready_matches(&key, &record) {
            return Ok((Reservation::Ready(record), PipelineCacheObservation::Hit));
        }
        // A mismatched ready record is removed only from future lookup. Any live opaque handle
        // keeps its Arc and remains valid; ordinary realization below reconstructs this key.
        locked.ready.remove(&key);
        rejected = true;
    }
    if let Some(attempt) = locked.in_flight.get(&key).cloned() {
        return Ok((Reservation::Waiter(attempt), PipelineCacheObservation::Hit));
    }
    if locked.total_len() >= max_records.get() {
        locked.collect_lookup_only();
    }
    if locked.total_len() >= max_records.get() {
        return Err(GpuPipelineRealizationError::capacity(
            request,
            locked.total_len(),
            max_records,
        ));
    }
    let attempt = Arc::new(InFlight::default());
    locked.in_flight.insert(key.clone(), Arc::clone(&attempt));
    drop(locked);
    Ok((
        Reservation::Owner(OwnerReservation {
            registry: Arc::clone(registry),
            key,
            attempt,
            active: true,
        }),
        if rejected {
            PipelineCacheObservation::Rejected
        } else {
            PipelineCacheObservation::Miss
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuPipelineRealizationErrorCategory;
    use core::future::Future;
    use core::task::{Context, Poll};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

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
    fn equal_requests_single_flight_and_owner_abandonment_releases_capacity() {
        let registry = Arc::new(Mutex::new(SingleFlightRegistry::<u32, ()>::default()));
        let max = NonZeroUsize::new(1).unwrap();
        let (owner, observation) = reserve(&registry, max, 7, "compute 7", |_, _| true).unwrap();
        assert_eq!(observation, PipelineCacheObservation::Miss);
        let owner = match owner {
            Reservation::Owner(owner) => owner,
            Reservation::Ready(_) | Reservation::Waiter(_) => panic!("first caller owns"),
        };
        let (attempt, observation) = reserve(&registry, max, 7, "compute 7", |_, _| true).unwrap();
        assert_eq!(observation, PipelineCacheObservation::Hit);
        let attempt = match attempt {
            Reservation::Waiter(attempt) => attempt,
            Reservation::Ready(_) | Reservation::Owner(_) => panic!("equal caller waits"),
        };
        assert_eq!(registry.lock().unwrap().total_len(), 1);
        let capacity_error = match reserve(&registry, max, 8, "compute 8", |_, _| true) {
            Err(error) => error,
            Ok(_) => panic!("distinct request must observe occupied capacity"),
        };
        assert_eq!(
            capacity_error.category(),
            GpuPipelineRealizationErrorCategory::RegistryCapacityExceeded
        );

        let counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let mut context = Context::from_waker(&waker);
        let mut waiter = std::pin::pin!(attempt.wait());
        assert!(matches!(waiter.as_mut().poll(&mut context), Poll::Pending));
        drop(owner);
        assert!(counter.0.load(Ordering::SeqCst) > 0);
        assert!(matches!(
            waiter.as_mut().poll(&mut context),
            Poll::Ready(InFlightOutcome::Abandoned)
        ));
        assert_eq!(registry.lock().unwrap().total_len(), 0);
        let (reservation, observation) =
            reserve(&registry, max, 8, "compute 8", |_, _| true).unwrap();
        assert_eq!(observation, PipelineCacheObservation::Miss);
        assert!(matches!(reservation, Reservation::Owner(_)));
    }

    #[test]
    fn completion_is_retained_when_it_precedes_waiter_polling() {
        let attempt = InFlight::<u32>::default();
        attempt.complete(InFlightOutcome::Complete(Ok(Arc::new(11))));

        let counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let waker = Waker::from(counter);
        let mut context = Context::from_waker(&waker);
        let mut waiter = std::pin::pin!(attempt.wait());
        let outcome = waiter.as_mut().poll(&mut context);

        match outcome {
            Poll::Ready(InFlightOutcome::Complete(Ok(record))) => assert_eq!(*record, 11),
            Poll::Ready(InFlightOutcome::Complete(Err(error))) => {
                panic!("completion unexpectedly failed: {error}")
            }
            Poll::Ready(InFlightOutcome::Pending) => panic!("pending is not terminal"),
            Poll::Ready(InFlightOutcome::Abandoned) => panic!("completion was not abandoned"),
            Poll::Pending => panic!("stored completion must be observed without another wake"),
        }
    }

    #[test]
    fn incompatible_ready_candidate_is_rejected_then_realized_ordinally() {
        let registry = Arc::new(Mutex::new(SingleFlightRegistry::<u32, u32>::default()));
        let max = NonZeroUsize::new(2).unwrap();
        let stale = Arc::new(8_u32);
        registry.lock().unwrap().ready.insert(7, Arc::clone(&stale));

        let (reservation, observation) =
            reserve(&registry, max, 7, "compute 7", |key, record| key == record).unwrap();
        assert_eq!(observation, PipelineCacheObservation::Rejected);
        assert!(matches!(reservation, Reservation::Owner(_)));
        assert_eq!(Arc::strong_count(&stale), 1);
        assert_eq!(registry.lock().unwrap().total_len(), 1);
    }
}
