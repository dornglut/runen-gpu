# RunenGPU executor contract

Begin with `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `TESTING.md`, and
the current owning issue. Consult `BOOTSTRAP.md` for provenance or source-transfer
work. Confirm the accepted base, repository state, and applicable Engineering
authority before editing.

## Durable constraints

- Keep one semantic authority per concern and preserve one-way dependencies.
- Keep public RunenGPU contracts backend-neutral. WGPU is a private implementation
  authority; no public raw-device, raw-queue, or generic backend escape hatch is
  allowed.
- Keep one RunenGPU product package unless later independent authority proves a
  separate release boundary. `xtask` is repository tooling only.
- Do not place Runenwerk, RunenRender, ECS, UI, SDF, world, or application types
  in public RunenGPU contracts.
- Do not add compatibility aliases, forwarding packages/modules, mirrors, source
  includes, submodules, moving-branch dependencies, or duplicate execution
  authority.
- Keep tracked-content contributions `owner-only` until an accepted inbound
  contribution mechanism preserves commercial relicensing rights.

## Source authority

`dornglut/runen-gpu` owns the RunenGPU implementation. Runenwerk consumes an
exact accepted revision and owns only downstream integration. The source transfer
is complete; `BOOTSTRAP.md` records its provenance and Engineering ADR-0008
governs any future authority reversal. The historical framework template is not
an active dependency or synchronization source.

## Validation and evidence

Run the canonical command from a clean checkout:

```text
cargo validate
```

CI must validate the exact reviewed feature head through the thin immutable
caller. Reconcile `main`, branch state, repository settings, and the observed
check context before guarded squash merge. Do not claim CI, runtime, browser, or
platform evidence that was not observed.
