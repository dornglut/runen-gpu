use super::super::coverage::intersect_buffer_coverage;
use super::support::*;

fn test_buffer() -> GpuBufferHandle {
    let mut allocator = allocator();
    buffer(
        &mut allocator,
        "coverage intersection",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    )
}

#[test]
fn dense_buffer_intersection_keeps_only_the_shared_range() {
    let buffer = test_buffer();
    let left = [GpuBufferCoverage::dense(
        GpuBufferRange::new(&buffer, 0, 32).unwrap(),
    )];
    let right = [GpuBufferCoverage::dense(
        GpuBufferRange::new(&buffer, 8, 16).unwrap(),
    )];

    let intersection = intersect_buffer_coverage(&buffer, &left, &right);

    assert_eq!(intersection, right);
}

#[test]
fn disjoint_buffer_intersection_is_empty() {
    let buffer = test_buffer();
    let left = [GpuBufferCoverage::dense(
        GpuBufferRange::new(&buffer, 0, 8).unwrap(),
    )];
    let right = [GpuBufferCoverage::dense(
        GpuBufferRange::new(&buffer, 16, 8).unwrap(),
    )];

    let intersection = intersect_buffer_coverage(&buffer, &left, &right);

    assert!(intersection.is_empty());
}

#[test]
fn partial_strided_buffer_intersection_remains_exact() {
    let buffer = test_buffer();
    let strided = GpuBufferStridedCoverage::new(&buffer, 0, 4, 8, 4, 0, 1).unwrap();
    let left = [GpuBufferCoverage::strided(strided)];
    let right = [GpuBufferCoverage::dense(
        GpuBufferRange::new(&buffer, 2, 16).unwrap(),
    )];

    let intersection = intersect_buffer_coverage(&buffer, &left, &right);

    assert_eq!(
        intersection,
        [
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 2, 2).unwrap()),
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 8, 4).unwrap()),
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 16, 2).unwrap()),
        ]
    );
}
