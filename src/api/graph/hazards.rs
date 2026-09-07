use super::super::{
    GpuResourceAccess, GpuWorkGraphCause, GpuWorkGraphError, GpuWorkGraphErrorContext,
    GpuWorkResourceId,
};
use super::{
    authoring::{GpuWorkFragment, GpuWorkNode},
    composition::FragmentRelations,
    dependency::{GpuDependencyReason, GpuDependencyRegion, access_intersection},
    diagnostics::{GraphErrorOrigin, graph_error, graph_error_with_region},
    identity::GpuPreparedWorkNodeId,
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
            if dependency_data_path_exists(edges, before, after) {
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
            if dependency_data_path_exists(edges, after, before) {
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
        }
    }
    Ok(())
}

fn dependency_data_path_exists(
    edges: &DependencyEdges,
    start: GpuPreparedWorkNodeId,
    target: GpuPreparedWorkNodeId,
) -> bool {
    let mut ready = vec![(start, false)];
    let mut visited = BTreeSet::new();
    while let Some((node, path_has_data)) = ready.pop() {
        if !visited.insert((node, path_has_data)) {
            continue;
        }
        for (&(before, after), reasons) in edges {
            if before != node {
                continue;
            }
            let edge_has_data = reasons
                .iter()
                .any(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. }));
            let next_has_data = path_has_data || edge_has_data;
            if after == target && next_has_data {
                return true;
            }
            ready.push((after, next_has_data));
        }
    }
    false
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
