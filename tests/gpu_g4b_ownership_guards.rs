use std::fs;
use std::path::{Path, PathBuf};

const PROGRAM_SOURCE_FORBIDDEN: &[(&str, &str)] = &[
    (
        "wgpu::",
        "raw WGPU authority belongs to private backend realization",
    ),
    (
        "crate::plugins::render",
        "RunenGPU program contracts must remain renderer-neutral",
    ),
    (
        "std::any::TypeId",
        "Rust runtime type identity is not a shader ABI contract",
    ),
    (
        "include_str!",
        "shader filesystem and embedding policy belong outside RunenGPU",
    ),
    (
        "assets/shaders",
        "shader asset discovery belongs outside RunenGPU",
    ),
    ("admit_wesl", "G4B is WGSL-first and admits no WESL surface"),
    (
        "admit_slang",
        "G4B is WGSL-first and admits no Slang surface",
    ),
    ("admit_glsl", "G4B is WGSL-first and admits no GLSL surface"),
    ("admit_hlsl", "G4B is WGSL-first and admits no HLSL surface"),
    (
        "admit_spirv",
        "G4B is WGSL-first and admits no SPIR-V surface",
    ),
];

const PRIVATE_COMPILER_ANALYSIS: &str = "src/api/program/analysis.rs";

#[test]
fn g4b_program_source_remains_backend_and_renderer_neutral() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let program_root = manifest_dir.join("src/api/program");
    let mut files = Vec::new();
    collect_rust_files(&program_root, &mut files);
    files.sort();
    assert!(!files.is_empty(), "G4B program source census is empty");

    let mut violations = Vec::new();
    for file in files {
        let content = fs::read_to_string(&file).expect("G4B source should remain readable");
        let executable = without_line_comments(&content);
        let relative = file
            .strip_prefix(&manifest_dir)
            .expect("G4B source should remain inside the engine crate");

        for (token, reason) in PROGRAM_SOURCE_FORBIDDEN {
            if executable.contains(token) {
                violations.push(format!(
                    "{} contains {token:?}: {reason}",
                    relative.display()
                ));
            }
        }

        if executable.contains("naga::") && relative != Path::new(PRIVATE_COMPILER_ANALYSIS) {
            violations.push(format!(
                "{} contains \"naga::\": compiler analysis must remain private to {PRIVATE_COMPILER_ANALYSIS}",
                relative.display()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "G4B source-boundary violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn g4b_adds_no_alternate_shader_language_frontend_dependency() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(manifest_dir.join("Cargo.toml"))
        .expect("engine manifest should remain readable");
    let forbidden = ["wesl =", "slang =", "glsl =", "hlsl ="];
    let violations = forbidden
        .into_iter()
        .filter(|token| {
            manifest
                .lines()
                .any(|line| line.trim_start().starts_with(token))
        })
        .collect::<Vec<_>>();

    assert!(
        violations.is_empty(),
        "G4B introduced alternate shader-language frontend dependencies: {}",
        violations.join(", ")
    );
}

fn collect_rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()));
    for entry in entries {
        let path = entry
            .expect("G4B directory entry should be readable")
            .path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn without_line_comments(content: &str) -> String {
    content
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}
