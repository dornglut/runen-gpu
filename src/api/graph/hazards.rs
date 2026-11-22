use super::super::{
    GpuResourceAccess, GpuWorkGraphCause, GpuWorkGraphError, GpuWorkGraphErrorContext,
    GpuWorkResourceId,
};
use super::{
    authoring::{GpuGraphExplicitOrder, GpuGraphRequiredOrder, GpuWorkFragment, GpuWorkNode},
    composition::FragmentRelations,
    dependency::{GpuDependencyReason, GpuDependencyRegion, access_intersection},
    diagnostics::{GraphErrorOrigin, graph_error, graph_error_with_region},
    identity::{GpuPreparedWorkNodeId, GpuWorkNodeId},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) type DependencyEdges =
    BTreeMap<(GpuPreparedWorkNodeId, GpuPreparedWorkNodeId), BTreeSet<GpuDependencyReason>>;

pub(super) fn infer_fragment_hazards(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    edges: &mut DependencyEdges,
) -> Result<(), GpuWorkGraphError> {
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        let mut accesses_by_resource =
            BTreeMap::<GpuWorkResourceId, Vec<(usize, &GpuResourceAccess)>>::new();

        for (later_index, later) in fragment.nodes().iter().enumerate() {
            for later_access in later.accesses() {
                if matches!(later_access, GpuResourceAccess::Sampler(_)) {
                    continue;
                }
                let resource = later_access.resource_identity();
                if let Some(earlier_accesses) = accesses_by_resource.get(&resource) {
                    for &(earlier_index, earlier_access) in earlier_accesses {
                        if earlier_index == later_index
                            || (!earlier_access.writes() && !later_access.writes())
                        {
                            continue;
                        }
                        let Some((resource, region)) =
                            access_intersection(earlier_access, later_access)
                        else {
                            continue;
                        };
                        let reasons = access_pair_hazard_reasons(
                            earlier_access,
                            later_access,
                            resource,
                            region,
                        );
                        if reasons.is_empty() {
                            continue;
                        }
                        let before = prepared_node_id(
                            graph_label,
                            fragment_index,
                            fragment,
                            &fragment.nodes()[earlier_index],
                        )?;
                        let after = prepared_node_id(graph_label, fragment_index, fragment, later)?;
                        edges.entry((before, after)).or_default().extend(reasons);
                    }
                }
                accesses_by_resource
                    .entry(resource)
                    .or_default()
                    .push((later_index, later_access));
            }
        }
    }
    Ok(())
}

pub(super) fn infer_cross_fragment_hazards(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    relations: &FragmentRelations,
    edges: &mut DependencyEdges,
) -> Result<(), GpuWorkGraphError> {
    for left_fragment_index in 0..fragments.len() {
        for right_fragment_index in (left_fragment_index + 1)..fragments.len() {
            let left_fragment = &fragments[left_fragment_index];
            let right_fragment = &fragments[right_fragment_index];
            for left_node in left_fragment.nodes() {
                for right_node in right_fragment.nodes() {
                    for left_access in left_node.accesses() {
                        for right_access in right_node.accesses() {
                            let Some((resource, region)) =
                                access_intersection(left_access, right_access)
                            else {
                                continue;
                            };
                            if !left_access.writes() && !right_access.writes() {
                                continue;
                            }
                            let left_to_right = relations.contains(&(
                                left_fragment_index,
                                right_fragment_index,
                                resource,
                            ));
                            let right_to_left = relations.contains(&(
                                right_fragment_index,
                                left_fragment_index,
                                resource,
                            ));
                            if left_to_right == right_to_left {
                                let cause = if left_access.writes() && right_access.writes() {
                                    GpuWorkGraphCause::AmbiguousWriter
                                } else {
                                    GpuWorkGraphCause::MissingCrossFragmentCausality
                                };
                                return Err(graph_error_with_region(
                                    "infer cross-fragment GPU hazard",
                                    graph_label,
                                    GraphErrorOrigin::new(Some(left_fragment), Some(left_node)),
                                    None,
                                    (resource, region.to_string()),
                                    cause,
                                    "bind one unique typed producer output to the consumer import for this storage resource",
                                ));
                            }
                            let (
                                before_fragment,
                                before_node,
                                before_access,
                                after_fragment,
                                after_node,
                                after_access,
                            ) = if left_to_right {
                                (
                                    left_fragment_index,
                                    left_node,
                                    left_access,
                                    right_fragment_index,
                                    right_node,
                                    right_access,
                                )
                            } else {
                                (
                                    right_fragment_index,
                                    right_node,
                                    right_access,
                                    left_fragment_index,
                                    left_node,
                                    left_access,
                                )
                            };
                            let before = prepared_node_id(
                                graph_label,
                                before_fragment,
                                &fragments[before_fragment],
                                before_node,
                            )?;
                            let after = prepared_node_id(
                                graph_label,
                                after_fragment,
                                &fragments[after_fragment],
                                after_node,
                            )?;
                            edges.entry((before, after)).or_default().extend(
                                access_pair_hazard_reasons(
                                    before_access,
                                    after_access,
                                    resource,
                                    region,
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn access_pair_hazard_reasons(
    earlier: &GpuResourceAccess,
    later: &GpuResourceAccess,
    resource: GpuWorkResourceId,
    region: GpuDependencyRegion,
) -> BTreeSet<GpuDependencyReason> {
    let mut reasons = BTreeSet::new();
    if earlier.writes() && later.reads() {
        reasons.insert(GpuDependencyReason::ReadAfterWrite { resource, region });
    }
    if earlier.reads() && later.writes() {
        reasons.insert(GpuDependencyReason::WriteAfterRead { resource, region });
    }
    if earlier.writes() && later.writes() {
        reasons.insert(GpuDependencyReason::WriteAfterWrite { resource, region });
    }
    reasons
}

fn prepared_node_id(
    graph_label: &str,
    fragment_index: usize,
    fragment: &GpuWorkFragment,
    node: &GpuWorkNode,
) -> Result<GpuPreparedWorkNodeId, GpuWorkGraphError> {
    let ordinal = u32::try_from(fragment_index).map_err(|_| {
        graph_error(
            "assign prepared GPU work-node identity",
            graph_label,
            GraphErrorOrigin::new(Some(fragment), Some(node)),
            None,
            None,
            GpuWorkGraphCause::UnknownIdentity,
            "prepare fewer fragments in one bounded graph",
        )
    })?;
    Ok(GpuPreparedWorkNodeId::new(ordinal, node.id().local))
}

pub(super) fn add_explicit_orders(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    edges: &mut DependencyEdges,
    paths: &mut DataDependencyPaths,
) -> Result<(), GpuWorkGraphError> {
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        for order in fragment.explicit_orders() {
            if !order.before().belongs_to(&fragment.identity)
                || !order.after().belongs_to(&fragment.identity)
            {
                return Err(graph_error(
                    "validate explicit GPU work order",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    None,
                    GpuWorkGraphCause::ForeignIdentity,
                    "use endpoints allocated by the containing fragment",
                ));
            }
            let before_node = fragment
                .nodes()
                .iter()
                .find(|node| node.id() == order.before())
                .ok_or_else(|| {
                    graph_error(
                        "validate explicit GPU work-order endpoint",
                        graph_label,
                        GraphErrorOrigin::new(Some(fragment), None),
                        None,
                        None,
                        GpuWorkGraphCause::UnknownIdentity,
                        "retain both typed endpoints in the immutable fragment",
                    )
                })?;
            let after_node = fragment
                .nodes()
                .iter()
                .find(|node| node.id() == order.after())
                .ok_or_else(|| {
                    graph_error(
                        "validate explicit GPU work-order endpoint",
                        graph_label,
                        GraphErrorOrigin::new(Some(fragment), None),
                        None,
                        None,
                        GpuWorkGraphCause::UnknownIdentity,
                        "retain both typed endpoints in the immutable fragment",
                    )
                })?;
            let before = prepared_node_id(graph_label, fragment_index, fragment, before_node)?;
            let after = prepared_node_id(graph_label, fragment_index, fragment, after_node)?;
            if paths.contains_data_path(before, after) {
                return Err(graph_error(
                    "add explicit GPU work order",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), Some(after_node)),
                    Some(after),
                    None,
                    GpuWorkGraphCause::RedundantExplicitDataOrder,
                    "remove the explicit edge and rely on typed access-derived dependency",
                ));
            }
            if paths.contains_data_path(after, before) {
                return Err(graph_error(
                    "add explicit GPU work order",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), Some(after_node)),
                    Some(after),
                    None,
                    GpuWorkGraphCause::ExplicitOrderConflict,
                    "orient the non-data constraint consistently with inferred data order",
                ));
            }
            edges.entry((before, after)).or_default().insert(
                GpuDependencyReason::ExplicitNonData {
                    reason: order.reason().to_string(),
                },
            );
            paths.add_non_data_edge(before, after);
        }
    }
    Ok(())
}

pub(super) fn add_graph_explicit_orders(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    orders: &[GpuGraphExplicitOrder],
    edges: &mut DependencyEdges,
    paths: &mut DataDependencyPaths,
) -> Result<(), GpuWorkGraphError> {
    for order in orders {
        let (_, _, _, before) =
            resolve_graph_order_endpoint(graph_label, fragments, order.before())?;
        let (_, after_fragment, after_node, after) =
            resolve_graph_order_endpoint(graph_label, fragments, order.after())?;

        if paths.contains_data_path(before, after) {
            return Err(graph_error(
                "add graph-scope explicit GPU work order",
                graph_label,
                GraphErrorOrigin::new(Some(after_fragment), Some(after_node)),
                Some(after),
                None,
                GpuWorkGraphCause::RedundantExplicitDataOrder,
                "remove the graph-scope explicit edge and rely on typed access-derived dependency",
            ));
        }
        if paths.contains_data_path(after, before) {
            return Err(graph_error(
                "add graph-scope explicit GPU work order",
                graph_label,
                GraphErrorOrigin::new(Some(after_fragment), Some(after_node)),
                Some(after),
                None,
                GpuWorkGraphCause::ExplicitOrderConflict,
                "orient the graph-scope non-data constraint consistently with inferred data order",
            ));
        }

        edges
            .entry((before, after))
            .or_default()
            .insert(GpuDependencyReason::ExplicitNonData {
                reason: order.reason().to_string(),
            });
        paths.add_non_data_edge(before, after);
    }
    Ok(())
}

/// Required order admission differs from strict explicit orders: a preexisting
/// path of typed dependencies or admitted controls fulfills the requirement.
/// A reverse path is rejected rather than creating a cycle. All accepted
/// orders are still authored as canonical RunenGPU non-data dependency reasons.
pub(super) fn add_graph_required_orders(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    orders: &[GpuGraphRequiredOrder],
    node_locations: &BTreeMap<GpuPreparedWorkNodeId, (usize, usize)>,
    edges: &mut DependencyEdges,
    paths: &mut DataDependencyPaths,
) -> Result<(), GpuWorkGraphError> {
    // Bounded, canonical reachability for repeated required orders. A one-off
    // request and large graphs retain the original adjacency DFS behavior.
    // Both paths compare typed-vs-nondata causality identically.
    let mut indexed = (orders.len() >= 4)
        .then(|| RequiredReachability::from_paths(paths, node_locations))
        .flatten();
    for order in orders {
        let (_, _, _, before) =
            resolve_graph_order_endpoint(graph_label, fragments, order.before())?;
        let (_, after_fragment, after_node, after) =
            resolve_graph_order_endpoint(graph_label, fragments, order.after())?;

        if required_path_exists(paths, indexed.as_ref(), before, after, false) {
            continue;
        }
        if required_path_exists(paths, indexed.as_ref(), after, before, true) {
            return Err(graph_error(
                "admit required GPU work order",
                graph_label,
                GraphErrorOrigin::new(Some(after_fragment), Some(after_node)),
                Some(after),
                None,
                GpuWorkGraphCause::ExplicitOrderConflict,
                "orient the required control consistently with typed resource causality",
            ));
        }
        if required_path_exists(paths, indexed.as_ref(), after, before, false) {
            return Err(graph_error(
                "admit required GPU work order",
                graph_label,
                GraphErrorOrigin::new(Some(after_fragment), Some(after_node)),
                Some(after),
                None,
                GpuWorkGraphCause::Cycle,
                "remove cyclic required control ordering",
            ));
        }

        edges
            .entry((before, after))
            .or_default()
            .insert(GpuDependencyReason::ExplicitNonData {
                reason: order.reason().to_string(),
            });
        paths.add_non_data_edge(before, after);
        if indexed
            .as_mut()
            .is_some_and(|index| !index.add_non_data_edge(before, after))
        {
            // A missing, unexpectedly introduced endpoint cannot justify a
            // cache hit. Fall back to the original authoritative path checks.
            indexed = None;
        }
    }
    Ok(())
}

fn resolve_graph_order_endpoint<'a>(
    graph_label: &str,
    fragments: &'a [GpuWorkFragment],
    endpoint: &GpuWorkNodeId,
) -> Result<
    (
        usize,
        &'a GpuWorkFragment,
        &'a GpuWorkNode,
        GpuPreparedWorkNodeId,
    ),
    GpuWorkGraphError,
> {
    let mut matching_fragments = fragments
        .iter()
        .enumerate()
        .filter(|(_, fragment)| endpoint.belongs_to(&fragment.identity));
    let Some((fragment_index, fragment)) = matching_fragments.next() else {
        return Err(graph_error(
            "resolve graph-scope explicit GPU work-order fragment",
            graph_label,
            GraphErrorOrigin::new(None, None),
            None,
            None,
            GpuWorkGraphCause::ForeignIdentity,
            "include the endpoint's originating immutable fragment in the prepared graph",
        ));
    };
    if matching_fragments.next().is_some() {
        return Err(graph_error(
            "resolve graph-scope explicit GPU work-order fragment",
            graph_label,
            GraphErrorOrigin::new(Some(fragment), None),
            None,
            None,
            GpuWorkGraphCause::UnknownIdentity,
            "include each originating immutable fragment identity at most once when graph-scope orders reference it",
        ));
    }
    let Some(node) = fragment.nodes().iter().find(|node| node.id() == endpoint) else {
        return Err(graph_error(
            "resolve graph-scope explicit GPU work-order endpoint",
            graph_label,
            GraphErrorOrigin::new(Some(fragment), None),
            None,
            None,
            GpuWorkGraphCause::UnknownIdentity,
            "retain the authored endpoint node in its immutable fragment",
        ));
    };
    let prepared = prepared_node_id(graph_label, fragment_index, fragment, node)?;
    Ok((fragment_index, fragment, node, prepared))
}

/// Derived adjacency for explicit-order validation only. The original typed
/// dependency map remains the graph authority and is never replaced by this
/// index. Edges retain whether at least one reason is data-derived.
pub(super) struct DataDependencyPaths {
    outgoing: BTreeMap<GpuPreparedWorkNodeId, Vec<(GpuPreparedWorkNodeId, bool)>>,
}

impl DataDependencyPaths {
    pub(super) fn from_edges(edges: &DependencyEdges) -> Self {
        let mut outgoing =
            BTreeMap::<GpuPreparedWorkNodeId, Vec<(GpuPreparedWorkNodeId, bool)>>::new();
        for (&(before, after), reasons) in edges {
            let has_data = reasons
                .iter()
                .any(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. }));
            outgoing.entry(before).or_default().push((after, has_data));
        }
        Self { outgoing }
    }

    pub(super) fn add_non_data_edge(
        &mut self,
        before: GpuPreparedWorkNodeId,
        after: GpuPreparedWorkNodeId,
    ) {
        // Explicit orders may only add a non-data reason. Preserve the
        // distinction across subsequent queries, including when an edge is
        // already present from an earlier control request.
        self.outgoing
            .entry(before)
            .or_default()
            .push((after, false));
    }

    /// Tests for a positive-length directed path, regardless of whether it
    /// contains typed data edges or only non-data controls.
    pub(super) fn contains_path(
        &self,
        start: GpuPreparedWorkNodeId,
        target: GpuPreparedWorkNodeId,
    ) -> bool {
        let mut ready = vec![start];
        let mut visited = BTreeSet::new();
        while let Some(node) = ready.pop() {
            if !visited.insert(node) {
                continue;
            }
            if let Some(successors) = self.outgoing.get(&node) {
                for &(after, _) in successors {
                    if after == target {
                        return true;
                    }
                    ready.push(after);
                }
            }
        }
        false
    }

    pub(super) fn contains_data_path(
        &self,
        start: GpuPreparedWorkNodeId,
        target: GpuPreparedWorkNodeId,
    ) -> bool {
        let mut ready = vec![(start, false)];
        let mut visited = BTreeSet::new();
        while let Some((node, path_has_data)) = ready.pop() {
            if !visited.insert((node, path_has_data)) {
                continue;
            }
            if let Some(successors) = self.outgoing.get(&node) {
                for &(after, edge_has_data) in successors {
                    let next_has_data = path_has_data || edge_has_data;
                    if after == target && next_has_data {
                        return true;
                    }
                    ready.push((after, next_has_data));
                }
            }
        }
        false
    }
}

/// Dense reachability is only a derivative of G3's canonical typed dependency
/// map. The entire index is local to a single preparation and never stores
/// frame resources, retained coverage or operations.
const MAX_INDEXED_REQUIRED_NODES: usize = 256;

struct RequiredReachability {
    index: BTreeMap<GpuPreparedWorkNodeId, usize>,
    any: Vec<Vec<u64>>,
    typed: Vec<Vec<u64>>,
}

impl RequiredReachability {
    fn from_paths(
        paths: &DataDependencyPaths,
        node_locations: &BTreeMap<GpuPreparedWorkNodeId, (usize, usize)>,
    ) -> Option<Self> {
        let node_count = node_locations.len();
        if node_count > MAX_INDEXED_REQUIRED_NODES {
            return None;
        }
        let index = node_locations
            .keys()
            .copied()
            .enumerate()
            .map(|(slot, id)| (id, slot))
            .collect::<BTreeMap<_, _>>();
        let word_count = node_count.div_ceil(64);
        let mut adjacency = vec![Vec::<(usize, bool)>::new(); node_count];
        for (before, successors) in &paths.outgoing {
            let &source = index.get(before)?;
            for &(after, has_data) in successors {
                let &target = index.get(&after)?;
                adjacency[source].push((target, has_data));
            }
        }

        // Fixed-point propagation over two independent reachability states:
        // any positive-length path, and a path containing at least one typed
        // data dependency. Unlike a simple Floyd variant, this also handles
        // typed cycles and non-data cycles without guessing path semantics.
        let mut any = vec![vec![0_u64; word_count]; node_count];
        let mut typed = vec![vec![0_u64; word_count]; node_count];
        for _ in 0..=node_count.saturating_mul(2) {
            let mut changed = false;
            for source in (0..node_count).rev() {
                for &(target, has_data) in &adjacency[source] {
                    for word in 0..word_count {
                        let immediate = if word == target / 64 {
                            1_u64 << (target % 64)
                        } else {
                            0
                        };
                        let reachable = immediate | any[target][word];
                        let typed_successor = typed[target][word];
                        let old_any = any[source][word];
                        let old_typed = typed[source][word];
                        any[source][word] |= reachable;
                        typed[source][word] |=
                            typed_successor | (if has_data { reachable } else { 0 });
                        changed |= any[source][word] != old_any || typed[source][word] != old_typed;
                    }
                }
            }
            if !changed {
                return Some(Self { index, any, typed });
            }
        }
        // The fallback keeps error behavior identical even if this bounded
        // acceleration cannot establish a complete closure.
        None
    }

    fn contains(
        &self,
        before: GpuPreparedWorkNodeId,
        after: GpuPreparedWorkNodeId,
        requires_data: bool,
    ) -> Option<bool> {
        let from = *self.index.get(&before)?;
        let to = *self.index.get(&after)?;
        let row = if requires_data {
            &self.typed[from]
        } else {
            &self.any[from]
        };
        Some(row[to / 64] & (1_u64 << (to % 64)) != 0)
    }

    /// Caller has already rejected reverse reachability; the new non-data edge
    /// cannot participate in a cycle. Every newly admitted positive path can
    /// therefore be expressed with exactly one occurrence of this new edge.
    /// Preserve typed prefixes and suffixes but never mark this edge as typed.
    fn add_non_data_edge(
        &mut self,
        before: GpuPreparedWorkNodeId,
        after: GpuPreparedWorkNodeId,
    ) -> bool {
        let Some(&source) = self.index.get(&before) else {
            return false;
        };
        let Some(&target) = self.index.get(&after) else {
            return false;
        };
        let mut suffix_any = self.any[target].clone();
        suffix_any[target / 64] |= 1_u64 << (target % 64);
        let suffix_typed = self.typed[target].clone();
        for from in 0..self.any.len() {
            if from != source && self.any[from][source / 64] & (1_u64 << (source % 64)) == 0 {
                continue;
            }
            let typed_prefix = self.typed[from][source / 64] & (1_u64 << (source % 64)) != 0;
            for word in 0..suffix_any.len() {
                self.any[from][word] |= suffix_any[word];
                self.typed[from][word] |=
                    suffix_typed[word] | (if typed_prefix { suffix_any[word] } else { 0 });
            }
        }
        true
    }
}

fn required_path_exists(
    paths: &DataDependencyPaths,
    index: Option<&RequiredReachability>,
    before: GpuPreparedWorkNodeId,
    after: GpuPreparedWorkNodeId,
    requires_data: bool,
) -> bool {
    if let Some(result) = index.and_then(|index| index.contains(before, after, requires_data)) {
        return result;
    }
    if requires_data {
        paths.contains_data_path(before, after)
    } else {
        paths.contains_path(before, after)
    }
}

pub(super) fn topological_node_order(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    node_locations: &BTreeMap<GpuPreparedWorkNodeId, (usize, usize)>,
    edges: &DependencyEdges,
) -> Result<Vec<GpuPreparedWorkNodeId>, GpuWorkGraphError> {
    let mut outgoing = BTreeMap::<GpuPreparedWorkNodeId, BTreeSet<GpuPreparedWorkNodeId>>::new();
    let mut indegree = node_locations
        .keys()
        .copied()
        .map(|node| (node, 0_usize))
        .collect::<BTreeMap<_, _>>();
    for &(before, after) in edges.keys() {
        if !indegree.contains_key(&before) || !indegree.contains_key(&after) {
            return Err(GpuWorkGraphError::invalid(
                "validate GPU work dependency identity",
                GpuWorkGraphErrorContext::new(
                    graph_label,
                    None,
                    None,
                    Some(after),
                    None,
                    None,
                    None,
                ),
                GpuWorkGraphCause::UnknownIdentity,
                "retain dependency endpoints in the prepared node set",
            ));
        }
        if outgoing.entry(before).or_default().insert(after) {
            if let Some(degree) = indegree.get_mut(&after) {
                *degree += 1;
            }
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(&node, &degree)| (degree == 0).then_some(node))
        .collect::<BTreeSet<_>>();
    let mut ordered = Vec::with_capacity(node_locations.len());
    while let Some(node) = ready.pop_first() {
        ordered.push(node);
        if let Some(dependents) = outgoing.get(&node) {
            for &dependent in dependents {
                if let Some(degree) = indegree.get_mut(&dependent) {
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(dependent);
                    }
                }
            }
        }
    }
    if ordered.len() != node_locations.len() {
        let cycle_node = indegree
            .iter()
            .find_map(|(&node, &degree)| (degree > 0).then_some(node));
        let fragment = cycle_node
            .and_then(|node| node_locations.get(&node))
            .map(|&(fragment, _)| &fragments[fragment]);
        return Err(GpuWorkGraphError::invalid(
            "topologically order GPU work",
            GpuWorkGraphErrorContext::new(
                graph_label,
                fragment.map(|fragment| fragment.label().as_str().to_string()),
                None,
                cycle_node,
                None,
                None,
                fragment.map(|fragment| fragment.provenance().clone()),
            ),
            GpuWorkGraphCause::Cycle,
            "remove cyclic explicit order or cyclic typed resource causality",
        ));
    }
    Ok(ordered)
}

#[cfg(test)]
mod dependency_path_tests {
    use super::*;
    use core::num::NonZeroU64;

    fn node(local: u64) -> GpuPreparedWorkNodeId {
        GpuPreparedWorkNodeId::new(0, NonZeroU64::new(local).expect("positive local ID"))
    }

    // The old algorithm is kept solely as a differential correctness oracle.
    fn original_full_edge_scan(
        edges: &[(GpuPreparedWorkNodeId, GpuPreparedWorkNodeId, bool)],
        start: GpuPreparedWorkNodeId,
        target: GpuPreparedWorkNodeId,
    ) -> bool {
        let mut ready = vec![(start, false)];
        let mut visited = BTreeSet::new();
        while let Some((current, path_has_data)) = ready.pop() {
            if !visited.insert((current, path_has_data)) {
                continue;
            }
            for &(before, after, edge_has_data) in edges {
                if before != current {
                    continue;
                }
                let next_has_data = path_has_data || edge_has_data;
                if after == target && next_has_data {
                    return true;
                }
                ready.push((after, next_has_data));
            }
        }
        false
    }

    fn assert_matches_reference(
        indexed: &DataDependencyPaths,
        edges: &[(GpuPreparedWorkNodeId, GpuPreparedWorkNodeId, bool)],
    ) {
        for from in 1..=16 {
            for to in 1..=16 {
                let source = node(from);
                let dest = node(to);
                assert_eq!(
                    indexed.contains_data_path(source, dest),
                    original_full_edge_scan(edges, source, dest),
                    "typed path {source} -> {dest}"
                );
            }
        }
    }

    #[test]
    fn typed_path_index_matches_full_edge_scan_with_dense_sparse_cycles_and_incremental_orders() {
        let mut edges = Vec::new();
        let mut index = DataDependencyPaths {
            outgoing: BTreeMap::new(),
        };
        // A dense typed region with enough same-source successors to exercise
        // revisiting a source via non-data and data-bearing paths.
        for before in 1..8 {
            for after in (before + 1)..8 {
                let has_data = (before + after) % 3 != 0;
                edges.push((node(before), node(after), has_data));
                index
                    .outgoing
                    .entry(node(before))
                    .or_default()
                    .push((node(after), has_data));
            }
        }
        // A separate all-non-data cycle must not count as typed causality.
        for (before, after) in [(8, 9), (9, 10), (10, 8)] {
            edges.push((node(before), node(after), false));
            index.add_non_data_edge(node(before), node(after));
        }
        edges.push((node(11), node(12), true));
        index
            .outgoing
            .entry(node(11))
            .or_default()
            .push((node(12), true));
        assert_matches_reference(&index, &edges);
        assert!(!index.contains_data_path(node(8), node(8)));
        assert!(!index.contains_data_path(node(8), node(10)));

        // The path index and canonical edge map both incorporate each newly
        // admitted explicit edge *before* the next query.
        for (before, after) in [(7, 8), (10, 11), (12, 1), (12, 13), (13, 13)] {
            edges.push((node(before), node(after), false));
            index.add_non_data_edge(node(before), node(after));
            assert_matches_reference(&index, &edges);
        }
        assert!(index.contains_data_path(node(8), node(8)));
    }
}

#[cfg(test)]
mod required_reachability_tests {
    use super::*;
    use core::num::NonZeroU64;

    fn node(index: usize) -> GpuPreparedWorkNodeId {
        GpuPreparedWorkNodeId::new(
            0,
            NonZeroU64::new(u64::try_from(index + 1).unwrap()).unwrap(),
        )
    }

    fn locations(count: usize) -> BTreeMap<GpuPreparedWorkNodeId, (usize, usize)> {
        (0..count).map(|index| (node(index), (0, index))).collect()
    }

    fn compare_against_original(
        paths: &DataDependencyPaths,
        indexed: &RequiredReachability,
        count: usize,
    ) {
        for from in 0..count {
            for to in 0..count {
                let before = node(from);
                let after = node(to);
                assert_eq!(
                    indexed.contains(before, after, false),
                    Some(paths.contains_path(before, after)),
                    "positive-length path {from}->{to}"
                );
                assert_eq!(
                    indexed.contains(before, after, true),
                    Some(paths.contains_data_path(before, after)),
                    "typed-bearing path {from}->{to}"
                );
            }
        }
    }

    fn next_random(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed
    }

    #[test]
    fn required_reachability_matches_dfs_in_cycles_and_all_nondata_regions() {
        let mut seed = 0x1f70_9bad_84ac_c331_u64;
        for node_count in 1..=11 {
            let locations = locations(node_count);
            for _ in 0..96 {
                let mut paths = DataDependencyPaths {
                    outgoing: BTreeMap::new(),
                };
                for from in 0..node_count {
                    for to in 0..node_count {
                        if next_random(&mut seed) % 7 >= 3 {
                            continue;
                        }
                        let data = next_random(&mut seed).is_multiple_of(3);
                        paths
                            .outgoing
                            .entry(node(from))
                            .or_default()
                            .push((node(to), data));
                        if next_random(&mut seed).is_multiple_of(5) {
                            // Distinct typed/non-data reasons may share endpoints.
                            paths
                                .outgoing
                                .entry(node(from))
                                .or_default()
                                .push((node(to), false));
                        }
                    }
                }
                let index = RequiredReachability::from_paths(&paths, &locations)
                    .expect("small cycles must be bounded and convergent");
                compare_against_original(&paths, &index, node_count);
            }
        }
    }

    #[test]
    fn required_reachability_updates_after_each_accepted_nondata_edge() {
        let count = 16;
        let locations = locations(count);
        let mut paths = DataDependencyPaths {
            outgoing: BTreeMap::new(),
        };
        // Existing strict controls can have a cycle; the index must preserve
        // typed paths through the cycle even if a required edge is added later.
        for (from, to, typed) in [
            (0, 1, false),
            (1, 0, false),
            (1, 1, true),
            (3, 4, true),
            (4, 5, false),
            (7, 8, true),
            (8, 9, false),
            (11, 12, true),
        ] {
            paths
                .outgoing
                .entry(node(from))
                .or_default()
                .push((node(to), typed));
        }
        let mut index = RequiredReachability::from_paths(&paths, &locations).unwrap();
        compare_against_original(&paths, &index, count);
        let mut inserted = 0_usize;
        for before in 0..count {
            for after in 0..count {
                if before == after
                    || paths.contains_path(node(before), node(after))
                    || paths.contains_path(node(after), node(before))
                {
                    continue;
                }
                let source = node(before);
                let target = node(after);
                assert!(index.add_non_data_edge(source, target));
                paths.add_non_data_edge(source, target);
                compare_against_original(&paths, &index, count);
                inserted += 1;
            }
        }
        assert!(
            inserted >= 3,
            "fixture must actually admit new control edges"
        );
    }

    #[test]
    fn bounded_index_retains_adjacency_fallback_for_large_graphs() {
        let paths = DataDependencyPaths {
            outgoing: BTreeMap::new(),
        };
        let over_limit = locations(MAX_INDEXED_REQUIRED_NODES + 1);
        assert!(
            RequiredReachability::from_paths(&paths, &over_limit).is_none(),
            "avoid unbounded dense reachability memory and construction work"
        );
        assert_eq!(
            required_path_exists(&paths, None, node(0), node(1), false),
            paths.contains_path(node(0), node(1))
        );
    }

    #[test]
    fn dense_typed_dependency_fixture_preserves_all_path_states() {
        let count = 104;
        let mut paths = DataDependencyPaths {
            outgoing: BTreeMap::new(),
        };
        // 68 densely dependent writers yield 2,278 typed dependencies,
        // comparable to the measured 2,327-edge native Boids graph.
        for from in 0..68 {
            for to in (from + 1)..68 {
                paths
                    .outgoing
                    .entry(node(from))
                    .or_default()
                    .push((node(to), true));
            }
        }
        let indexed = RequiredReachability::from_paths(&paths, &locations(count)).unwrap();
        compare_against_original(&paths, &indexed, count);
    }
}
