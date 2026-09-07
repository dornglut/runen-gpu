use super::*;
use crate::*;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

#[test]
fn retained_sampler_does_not_enter_storage_continuity() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let sampler_label = label("retained sampler");
    let common = GpuResourceCommon::owned(
        sampler_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance("retained sampler"),
    )
    .unwrap();
    let sampler = allocator
        .allocate_sampler_handle(
            GpuSamplerDescriptor::new(
                common,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                0.0,
                1.0,
                None,
            )
            .unwrap(),
        )
        .unwrap();

    let mut fragment = GpuWorkFragmentBuilder::new(
        label("retained sampler fragment"),
        provenance("retained sampler fragment"),
    );
    fragment
        .declare_resource(GpuResourceRef::Sampler(sampler))
        .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        label("retained sampler graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();

    assert!(PreparedRetainedContinuity::from_graph(&graph).is_empty());
}
