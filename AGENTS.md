# Rust Framework Template Agent Guide

Start with `README.md`, `ARCHITECTURE.md`, `TESTING.md`, and `BOOTSTRAP.md`.

## Scope

This repository owns a generic, one-time bootstrap baseline for Dornglut Rust
framework repositories. It does not own product behavior, product architecture,
product dependencies, releases, or ongoing synchronization.

## Rules

- Keep the template generic and one-shot.
- Do not add product-specific implementation or dependencies.
- Do not copy sibling repository roadmaps, status, issues, or live Project state.
- Do not create compatibility forwarders, aliases, mirrors, or synchronization
  machinery.
- Preserve one semantic authority for each concern.
- Keep the canonical validation command as `cargo validate`.
- Keep CI a thin read-only caller of repository-owned validation.
- Resolve generated repository identity, license, MSRV, settings, conformance,
  provenance, and deviations during bootstrap rather than freezing them here.

## Required workflow

1. Read the current repository authority and accepted Engineering standards.
2. Keep changes bounded to the template's generic bootstrap purpose.
3. Run `cargo validate` from a suitable checked-out Rust executor when one is
   available; do not simulate local validation when the selected procedure lacks
   local execution.
4. Validate the exact feature head through repository-owned CI before acceptance.
5. Merge only the exact reviewed head after reconciling current `main`.

The template is not an ongoing authority for repositories generated from it.
