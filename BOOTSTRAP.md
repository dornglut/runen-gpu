# Bootstrap procedure

This template is a one-time starting point. A generated repository must establish
its own current authority before accepting substantive implementation.

## 1. Identity

Replace the placeholder repository, package, crate, version, description, and
repository metadata with the generated repository's accepted identity.

Do not leave `rust-framework-template` as an active product identity.

## 2. Classification

Resolve and record:

- repository profile;
- lifecycle;
- contribution mode;
- owning domain;
- relationship to existing repositories.

Use the accepted organization vocabulary rather than inventing local property
values.

## 3. Licensing

Select the product license class before substantive implementation.

Establish the required current `LICENSE`, package SPDX metadata, README license
statement, and any required additional licensing documentation.

The template's Apache-2.0 license is the template's own current license; it is
not a generic product-license choice.

License changes are prospective. Historical grants remain historical evidence,
and third-party material keeps its own license.

## 4. Toolchain

Resolve the generated repository's MSRV and toolchain from product evidence.

Do not assume the template's current toolchain is the generated product's final
MSRV. Update the package metadata and toolchain declaration together.

## 5. Repository settings

Establish the repository's accepted GitHub posture:

- public or explicitly accepted visibility;
- default branch `main`;
- squash merge enabled;
- merge commits disabled;
- rebase merge disabled unless specifically justified;
- merged head branches deleted;
- normal changes through pull requests;
- canonical validation required;
- conversations resolved before merge;
- force pushes and default-branch deletion blocked;
- linear history preferred;
- no meaningless solo-maintainer approval count;
- applicable security controls enabled or recorded as platform deviations.

## 6. Validation

Keep `cargo validate` as the canonical command.

Extend the local `xtask` only when the product has a demonstrated validation
requirement. Product-specific checks remain repository-local and are not moved
into shared CI.

## 7. Downstream conformance

When the framework exposes a public contract consumed by another repository,
create an independent downstream conformance package or workload. Do not use
the template itself as conformance evidence.

## 8. Extraction and provenance

If implementation is transferred from another repository, record:

- source repository and accepted source revision;
- source path or boundary;
- ownership and licensing disposition;
- transfer rationale and scope;
- consumer migration boundary;
- deviations introduced during extraction.

The template does not grant authority to transfer source. Extraction remains
owned by the accepted source and destination work.

## 9. Deviations

Record every intentional deviation from this template baseline in the generated
repository's appropriate authority.

Do not preserve obsolete template material merely for tree similarity.

## Completion

After these decisions are accepted, the template ceases to be relevant. The
generated repository becomes the sole authority for its code, architecture,
validation semantics, roadmap, releases, and compatibility.
