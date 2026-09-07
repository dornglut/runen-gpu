use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const REQUIRED_FILES: &[&str] = &[
    ".cargo/config.toml",
    ".github/workflows/validation.yml",
    ".github/workflows/runengpu-conformance.yml",
    ".gitignore",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "BOOTSTRAP.md",
    "Cargo.lock",
    "Cargo.toml",
    "conformance/downstream/Cargo.lock",
    "conformance/downstream/Cargo.toml",
    "conformance/downstream/src/main.rs",
    "LICENSE",
    "LICENSING.md",
    "README.md",
    "TESTING.md",
    "rust-toolchain.toml",
    "src/lib.rs",
    "xtask/Cargo.toml",
    "xtask/src/main.rs",
];

const ACTIVE_IDENTITY_FILES: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "TESTING.md",
    "src/lib.rs",
];

const STALE_ACTIVE_IDENTITY: &[&str] = &[
    "rust-framework-template",
    "MIT",
    "MIT OR Apache-2.0",
    "Apache-2.0",
    "Apache License 2.0",
];

fn main() {
    let result = match env::args().nth(1).as_deref() {
        Some("validate") => validate(),
        _ => Err("usage: cargo xtask validate".to_owned()),
    };

    if let Err(error) = result {
        eprintln!("validation failed: {error}");
        std::process::exit(1);
    }
}

fn validate() -> Result<(), String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest must have a workspace root")
        .to_path_buf();

    validate_required_files(&root)?;
    validate_product_identity(&root)?;
    validate_extraction_boundary(&root)?;

    let initial_state = git_status(&root)?;
    if !initial_state.is_empty() {
        return Err(format!(
            "repository must be clean before validation:\n{initial_state}"
        ));
    }

    run(&root, "cargo", &["fmt", "--all", "--", "--check"])?;
    run(
        &root,
        "cargo",
        &["metadata", "--locked", "--format-version", "1"],
    )?;
    run(
        &root,
        "cargo",
        &["tree", "-p", "runen-gpu", "-e", "normal", "--locked"],
    )?;
    run(
        &root,
        "cargo",
        &[
            "tree",
            "-p",
            "runen-gpu",
            "--target",
            "wasm32-unknown-unknown",
            "-e",
            "normal",
            "--locked",
        ],
    )?;
    run(&root, "cargo", &["test", "--workspace", "--locked"])?;
    run(
        &root,
        "cargo",
        &[
            "test",
            "--manifest-path",
            "conformance/downstream/Cargo.toml",
            "--locked",
        ],
    )?;
    run(
        &root,
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run_with_env(
        &root,
        "cargo",
        &["doc", "--workspace", "--no-deps", "--locked"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )?;
    run(
        &root,
        "cargo",
        &["+1.87", "check", "--workspace", "--all-targets", "--locked"],
    )?;
    run(&root, "git", &["diff", "--check"])?;
    run(&root, "git", &["diff", "--cached", "--check"])?;

    let final_state = git_status(&root)?;
    if final_state != initial_state {
        return Err(format!(
            "validation changed repository state:\nbefore:\n{initial_state}after:\n{final_state}"
        ));
    }

    Ok(())
}

fn validate_required_files(root: &Path) -> Result<(), String> {
    for relative_path in REQUIRED_FILES {
        let path = root.join(relative_path);
        if !path.is_file() {
            return Err(format!("required file is missing: {relative_path}"));
        }
    }

    Ok(())
}

fn validate_product_identity(root: &Path) -> Result<(), String> {
    let manifest = read_file(root, "Cargo.toml")?;
    for required in [
        "name = \"runen-gpu\"",
        "version = \"0.1.0\"",
        "edition = \"2024\"",
        "rust-version = \"1.87\"",
        "license.workspace = true",
        "repository = \"https://github.com/dornglut/runen-gpu\"",
        "description = \"Backend-neutral GPU execution framework\"",
        "publish = false",
        "[workspace.package]",
        "license = \"GPL-3.0-only\"",
        "[features]\ndefault = []",
    ] {
        require_contains("Cargo.toml", &manifest, required)?;
    }

    let license = read_file(root, "LICENSE")?;
    for required in [
        "GNU GENERAL PUBLIC LICENSE",
        "Version 3, 29 June 2007",
        "END OF TERMS AND CONDITIONS",
        "How to Apply These Terms to Your New Programs",
    ] {
        require_contains("LICENSE", &license, required)?;
    }
    if license.len() < 10_000 {
        return Err("LICENSE is shorter than a complete GPLv3 text".to_owned());
    }

    for relative_path in ACTIVE_IDENTITY_FILES {
        let contents = read_file(root, relative_path)?;
        for stale_identity in STALE_ACTIVE_IDENTITY {
            if contents.contains(stale_identity) {
                return Err(format!(
                    "active identity file {relative_path} contains stale product identity or license: {stale_identity}"
                ));
            }
        }
    }

    Ok(())
}

fn validate_extraction_boundary(root: &Path) -> Result<(), String> {
    let manifest = read_file(root, "Cargo.toml")?;
    let dependencies = section(&manifest, "[dependencies]")?;
    let allowed_production = [
        "bytemuck",
        "naga",
        "raw-window-handle",
        "tokio",
        "tracing",
        "wgpu",
    ];
    for line in dependencies.lines().filter(|line| !line.trim().is_empty()) {
        let name = line
            .split_once('=')
            .map(|(name, _)| name.trim())
            .unwrap_or_default();
        if !allowed_production.contains(&name) {
            return Err(format!(
                "Cargo.toml contains an unaccepted production dependency: {name}"
            ));
        }
    }
    if dependencies.contains("git =") || dependencies.contains("path =") {
        return Err(
            "standalone production manifest contains a moving/source-coupled dependency".to_owned(),
        );
    }

    let source_root = root.join("src");
    let mut source_paths = Vec::new();
    collect_files(&source_root, &mut source_paths)?;
    for path in source_paths {
        let source = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let executable = without_line_comments(&source);
        let relative = path
            .strip_prefix(root)
            .map_err(|error| format!("failed to relativize {}: {error}", path.display()))?;
        let is_private_backend = relative.starts_with(Path::new("src/backend/"));
        if !is_private_backend && executable.contains("wgpu::") {
            return Err(format!(
                "public RunenGPU source exposes raw WGPU usage: {}",
                relative.display()
            ));
        }
        for forbidden in [
            "engine::",
            "crate::plugins::",
            "runenwerk",
            "include!(",
            "include_str!(",
        ] {
            if executable.contains(forbidden) {
                return Err(format!(
                    "standalone source contains forbidden coupling {forbidden:?}: {}",
                    relative.display()
                ));
            }
        }
    }
    let lib = read_file(root, "src/lib.rs")?;
    if lib.contains("pub mod backend") || lib.contains("pub use backend") {
        return Err("private backend module is publicly wired".to_owned());
    }
    if root.join(".gitmodules").exists() {
        return Err("standalone repository must not contain a submodule".to_owned());
    }

    let downstream_manifest = read_file(root, "conformance/downstream/Cargo.toml")?;
    let downstream_dependencies = section(&downstream_manifest, "[dependencies]")?;
    if downstream_dependencies.contains("workspace = true")
        || downstream_dependencies.contains("wgpu")
        || downstream_dependencies.contains("engine")
        || !downstream_dependencies.contains("path = \"../..\"")
    {
        return Err("independent downstream dependency boundary is not isolated".to_owned());
    }
    Ok(())
}

fn section<'a>(manifest: &'a str, header: &str) -> Result<&'a str, String> {
    let start = manifest
        .find(header)
        .ok_or_else(|| format!("manifest is missing {header}"))?;
    let contents = &manifest[start + header.len()..];
    let end = contents.find("\n[").unwrap_or(contents.len());
    Ok(&contents[..end])
}

fn collect_files(root: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(root).map_err(|error| format!("failed to read {}: {error}", root.display()))?
    {
        let path = entry
            .map_err(|error| format!("failed to read source entry: {error}"))?
            .path();
        if path.is_dir() {
            collect_files(&path, paths)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path);
        }
    }
    Ok(())
}

fn without_line_comments(contents: &str) -> String {
    contents
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//") && !trimmed.starts_with('*')
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn read_file(root: &Path, relative_path: &str) -> Result<String, String> {
    fs::read_to_string(root.join(relative_path))
        .map_err(|error| format!("failed to read {relative_path}: {error}"))
}

fn require_contains(file: &str, contents: &str, required: &str) -> Result<(), String> {
    if contents.contains(required) {
        Ok(())
    } else {
        Err(format!(
            "{file} is missing required product contract: {required}"
        ))
    }
}

fn git_status(root: &Path) -> Result<String, String> {
    output(
        root,
        "git",
        &["status", "--porcelain", "--untracked-files=all"],
    )
}

fn run(root: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    run_with_env(root, program, args, &[])
}

fn run_with_env(
    root: &Path,
    program: &str,
    args: &[&str],
    environment: &[(&str, &str)],
) -> Result<(), String> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(root)
        .envs(environment.iter().copied())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command
        .status()
        .map_err(|error| format!("failed to execute {program}: {error}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} {} exited with {status}", args.join(" ")))
    }
}

fn output(root: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let result = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("failed to execute {program}: {error}"))?;

    if !result.status.success() {
        return Err(format!(
            "{program} {} exited with {}:\n{}",
            args.join(" "),
            result.status,
            String::from_utf8_lossy(&result.stderr)
        ));
    }

    String::from_utf8(result.stdout)
        .map_err(|error| format!("{program} produced invalid UTF-8: {error}"))
}
