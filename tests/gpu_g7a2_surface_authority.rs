//! Structural G7A2 surface-acquisition and presentation cutover guards.
//!
//! Behavioural tests cover lease state and execution categories. These guards keep the accepted
//! backend-neutral/public boundary and the private G4/G5/G7 ownership topology from regressing.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_rust_sources(root: &Path, paths: &mut Vec<PathBuf>) {
    for entry in
        fs::read_dir(root).unwrap_or_else(|error| panic!("cannot read {}: {error}", root.display()))
    {
        let path = entry.expect("source entry should be readable").path();
        if path.is_dir() {
            collect_rust_sources(&path, paths);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path);
        }
    }
}

fn compact(source: &str) -> String {
    source
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            !line.starts_with("//") && !line.starts_with("* ") && !line.starts_with("*/")
        })
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn read(manifest: &Path, relative: &str) -> String {
    fs::read_to_string(manifest.join(relative))
        .unwrap_or_else(|error| panic!("cannot read {relative}: {error}"))
}

fn token_paths(root: &Path, manifest: &Path, token: &str) -> BTreeSet<String> {
    let mut paths = Vec::new();
    collect_rust_sources(root, &mut paths);
    paths
        .into_iter()
        .filter_map(|path| {
            let source =
                compact(&fs::read_to_string(&path).expect("Rust source should be readable"));
            source.contains(token).then(|| {
                path.strip_prefix(manifest)
                    .expect("source stays in engine")
                    .to_string_lossy()
                    .into_owned()
            })
        })
        .collect()
}

#[test]
fn g7a2_public_surface_contract_stays_backend_neutral_and_nonowning() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let surface = compact(&read(&manifest, "src/api/surface.rs"));
    let acquisition = compact(&read(&manifest, "src/api/surface_acquisition.rs"));

    assert!(
        surface.contains("raw_window_handle::{HasDisplayHandle,HasWindowHandle}"),
        "the public host target must remain the standardized raw-window-handle pair"
    );
    for forbidden in [
        "wgpu::",
        "winit::",
        "SurfaceTexture",
        "CurrentRenderDeviceQueue",
        "CurrentRenderExecutionBridge",
        "RenderSurfaceId",
        "WindowId",
    ] {
        assert!(
            !surface.contains(forbidden) && !acquisition.contains(forbidden),
            "backend/window-system authority leaked into the public G7 surface API via {forbidden}"
        );
    }

    assert!(acquisition.contains("pubstructGpuAcquiredSurfaceImage"));
    assert!(acquisition.contains("pub(crate)structGpuSurfaceResourceLease"));
    assert!(acquisition.contains("pub(crate)structGpuSurfaceLeaseOwner"));
    assert_eq!(
        acquisition
            .matches("pub(crate)traitGpuSurfaceLeaseReleaser")
            .count(),
        2,
        "native and Wasm releaser contracts must both remain crate-private"
    );
    assert!(!acquisition.contains("pubstructGpuSurfaceResourceLease"));
    assert!(!acquisition.contains("pubstructGpuSurfaceLeaseOwner"));
}

#[test]
fn g7a2_physical_acquire_and_present_have_one_private_owner_without_renderer_bridges() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wgpu_root = manifest.join("src/backend/wgpu");
    let surface_path = "src/backend/wgpu/surface.rs";
    let surface_execution_path = "src/backend/wgpu/surface/execution.rs";

    assert_eq!(
        token_paths(&wgpu_root, &manifest, ".get_current_texture("),
        BTreeSet::from([surface_path.to_owned()]),
        "physical surface acquisition must remain owned by the one private G7 surface state"
    );
    assert_eq!(
        token_paths(&wgpu_root, &manifest, "queue.present("),
        BTreeSet::from([surface_execution_path.to_owned()]),
        "physical Present must remain owned by the one private G7 execution lease owner"
    );

    let surface = compact(&read(&manifest, surface_path));
    let present = compact(&read(&manifest, surface_execution_path));
    let prepared = compact(&read(
        &manifest,
        "src/backend/wgpu/execution/surface_resources.rs",
    ));
    for forbidden in [
        "CurrentRenderDeviceQueue",
        "current_render_device_queue",
        "CurrentRenderExecutionBridge",
        "current_render_execution_bridge",
    ] {
        assert!(
            !surface.contains(forbidden)
                && !present.contains(forbidden)
                && !prepared.contains(forbidden),
            "G7A2 regained a renderer-owned device/queue/execution seam via {forbidden}"
        );
    }
    assert!(
        !present.contains(".configure(") && !present.contains(".get_current_texture("),
        "Present must consume the validated lease without implicit reconfiguration or reacquisition"
    );
    assert_eq!(
        present.matches(".create_view(").count(),
        1,
        "G7A2 permits exactly the acquired default-view creation path"
    );
    assert!(present.contains(".create_view(&TextureViewDescriptor::default())"));
    assert!(present.contains("lease.mark_presented()"));
    assert!(present.contains("queue.present(active.texture)"));
}

#[test]
fn g7a2_surface_resources_route_around_g4_and_present_rejects_ordinary_resources() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let prepared = compact(&read(
        &manifest,
        "src/backend/wgpu/execution/surface_resources.rs",
    ));
    let g4 = compact(&read(
        &manifest,
        "src/backend/wgpu/resource_realization/lowering.rs",
    ));

    assert!(prepared.contains("ifletSome(lease)=handle.surface_lease()"));
    assert!(prepared.contains("PreparedTexture::Surface("));
    assert!(prepared.contains("PreparedTextureView::Surface("));
    assert!(
        prepared.matches(".validate_execution_lease(").count() >= 3,
        "surface texture, surface view, and Present preparation must all validate through G7"
    );
    assert!(
        prepared.contains("context.realize_texture(handle)")
            && prepared.contains(".realize_texture_view(handle,&parent)"),
        "ordinary resources must continue to use the existing G4 realization path"
    );
    assert!(prepared.contains(
        "GpuPresentOperationrequiresanactiveSurfaceAcquiredtextureoritsexplicitacquireddefaultview"
    ));

    assert!(g4.contains("GpuResourceOwnership::SurfaceAcquired=>Err("));
    assert!(g4.contains("GpuResourceRealizationErrorCategory::RequirementNotAdmitted"));
    assert!(g4.contains("surface-acquiredresourcerealizationremainsownedbyG7"));
}
