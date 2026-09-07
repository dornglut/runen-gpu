# RunenGPU executor contract

Begin with `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `BOOTSTRAP.md`, and the
current owning issue. Confirm the accepted base, repository state, and applicable
Engineering and Runenwerk authority before editing.

## Durable constraints

- Keep one semantic authority per concern and preserve one-way dependencies.
- Keep public RunenGPU contracts backend-neutral. After extraction, WGPU is a
  private implementation authority; no public raw-device, raw-queue, or generic
  backend escape hatch is allowed.
- Keep one RunenGPU product package unless later independent authority proves a
  separate release boundary. `xtask` is repository tooling only.
- Do not place Runenwerk, RunenRender, ECS, UI, SDF, world, or application types
  in public RunenGPU contracts.
- Do not add compatibility aliases, forwarding packages/modules, mirrors, source
  includes, submodules, moving-branch dependencies, or duplicate execution
  authority.
- Do not transfer implementation under the bootstrap boundary. Source extraction
  requires a separately accepted repository-local issue and successor branch.
- Keep tracked-content contributions `owner-only` until an accepted inbound
  contribution mechanism preserves commercial relicensing rights.

## ADR-0008 sequence

```text
accepted Runenwerk implementation
    -> sole semantic source authority
unmerged runen-gpu extraction candidate
    -> candidate only
accepted runen-gpu successor on default branch
    -> semantic authority switches
Runenwerk predecessor
    -> frozen and deletion-bound
```

The historical framework template is one-time provenance, not synchronization or
architecture authority. Do not begin the ADR-0008 authority switch or modify
Runenwerk during bootstrap.

## Validation and evidence

Run the canonical command from a clean checkout:

```text
cargo validate
```

CI must validate the exact reviewed feature head through the thin immutable
caller. Reconcile `main`, branch state, repository settings, and the observed
check context before guarded squash merge. Do not claim CI, runtime, browser, or
platform evidence that was not observed.
