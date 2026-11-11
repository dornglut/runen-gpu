# Streaming adaptive background — RunenGPU #172

**Purpose:** a useful foreground-change mask generator and an exact public-RunenGPU retained-state execution proof. The implementation does **not** provide ordered dependent submissions. It compares actual supported terminal deferral and within-graph batching; no renderer, device handles, or custom GPU scheduler is involved.

## Run

```sh
cargo run --example gpu_streaming_background -- --mode cpu --width 256 --height 144 --frames 64
cargo run --example gpu_streaming_background -- --mode serial --frames 64 --output-dir target/streaming-serial
cargo run --example gpu_streaming_background -- --mode batch --batch 4 --frames 64 --output-dir target/streaming-batch
```

Use `--help` for options. The `--mode serial` path processes one retained background step per accepted GPU submission and correctly waits for observed terminal success before preparing a new one. `--mode batch --batch 4` combines **already received** frames into one G3 graph and one GPU submission. Every frame retains an independent foreground-mask readback. The program polls nonblocking, may progress the next batch while prior readback mapping is pending, verifies each entire mask and the final retained background against CPU, then emits one JSON line per verified frame and a final summary. The CPU mode measures the same scalar recurrence without GPU execution.

Masks are binary PGM files (`P5`, values 0/255); any ordinary image viewer able to read PGM can inspect them, or `ffmpeg`/ImageMagick can convert them to PNG. A mismatch, resource invalidity, failed submission, timeout or error in producer framing ends the program nonzero; nothing is silently counted as verified. `--timeout-ms N` controls the GPU-example session deadline (default 120000 ms). It must be positive. If a producer holds stdin open without sending the remaining frames, the GPU session fails on that deadline **without blocking on a reader-thread join**; the executable exits with failure and does not claim a complete output. This is a finite-session example, not a promise of generic interruptible stdin reading or transparent recovery of a failed GPU submission.

The generated fixture includes a moving foreground, discrete lighting changes, alternating per-frame adaptation factors and fixed-seed per-pixel grayscale perturbations; repeating the same frame IDs yields identical bytes. The adaptive model deliberately uses **integer** grayscale pixel values, not a production-quality mixture-of-Gaussians classifier:

```text
old = H[t-1][p]
M[t][p] = 255 if abs(F[t][p] - old) >= 32 else 0
H[t][p] = floor(((q[t]-1)*old + F[t][p])/q[t]), q[t] in {2,4}, H[0] = 0
```

All pixels and background results must match the independent CPU oracle exactly. The program uses one per-pixel `u32` retained background buffer and one distinct input/mask pair per frame. It does not synthesize GPU-completion, device-loss recovery or application publication.

## Source modes and evidence

**`--source generated` is a reproducible OFFLINE CONTROL.** A bounded independent producer thread generates a predictable moving bright square against a static grayscale room, but its future frames are mathematically knowable. Its timing is **not proof that an actual camera input cannot be batched**. The bounded channel limits outstanding received inputs; producer backpressure may affect measured arrival timing. Batching from external `stdin` additionally requires an explicit `--batch-wait-ms N` application policy to prevent indefinite waiting for a full batch on an open, low-rate stream. A maximum wait of zero deliberately submits available partial batches immediately; the example does not assume a universal FPS/deadline target.

For genuinely external input, use `--source stdin` and send the following little-endian binary packet repeatedly; EOF **between** packets ends the stream:

| Bytes | Meaning |
|---|---|
| 8 | `u64` external capture Unix microseconds (metadata, no clock-sync assertion) |
| 4 | `u32` adaptation q, exactly 2 or 4 |
| `width * height` | raw 8-bit grayscale, row-major |

With `--source stdin --mode serial --producer-signals`, the application emits a `RUNEN_GPU_172_READY` diagnostic to stderr **after** GPU context initialization and `RUNEN_GPU_172_ACCEPTED <frame_id>` after each accepted submission. An external producer can wait for these acknowledgements before creating a subsequent packet; this proves actual post-acceptance arrival without inventing GPU overlap. The acknowledgment is an **application/proof transport signal**, not a public RunenGPU API, frame-completion signal, physical capture timestamp, or performance-neutral interaction. It must never be interpreted as proving that the predecessor is still in flight when a packet is received. A webcam/decoder process can write these packets only as frames are captured. There is no bundled camera SDK, hidden prerecorded lookahead or forced frame rate. An independent producer may still be a synthetic timed source, **not** independently proved physical camera capture. Report the distinction.

The JSONL report records input receipt, accepted GPU submission, terminal observation and CPU verification times from the consumer's local monotonic clock, plus the externally supplied capture timestamp **without converting it into a trusted same-clock latency**. The entire completed GPU batch is independently compared against the CPU oracle and each per-frame verification instant is captured **before writing any** PGM/JSONL output. Thus earlier mask file writes cannot artificially delay later frames' verification records within the same ready batch. Each completed run includes nearest-rank p50/p95/p99 receipt-to-verified latency, total CPU reference time, progress time, combined application-authoring+submission-call time, adapter/backend/driver facts, peak application backlog/pending results, capacity retries, and counters distinguishing frames arriving **after** earlier acceptance from those also observed while an earlier submission remains `Accepted`. Those counters do not prove hardware overlap at the instant the producer created the packet. They are diagnostics rather than inferred GPU-only timings; logging, input delivery and output processing still perturb application scheduling. Compare `receipt_to_verified_us` under equivalent source/size/throughput conditions, quantify batch wait and validation/readback overhead, and retain raw traces. A single hosted Lavapipe run proves the documented behavior **on that software adapter only**, not a particular discrete GPU or real-time performance target. No arbitrary FPS/percentage requirement is adopted.

## Scope and current design decision

- [#171's candidate investigation](https://github.com/dornglut/runen-gpu/issues/171#issuecomment-6311374607) selected this example.
- [#167's decision](https://github.com/dornglut/runen-gpu/issues/167#issuecomment-6223645650) keeps conservative G5 until measured workload pressure warrants a successor.
- RunenGPU G3 may reject a dependent pending read/write **during preparation** because the predecessor's coverage is not yet established; otherwise G5 may reject with typed `InFlightParticipation`. Record the **first observed** boundary, never assume a specific one.
- Current RunenGPU G5 has no public in-flight same-resource ordered continuation. Accepting another GPU submission before the predecessor terminalizes is **NOT** achieved by this demo. Separate design/prototype authorization is required.
- Ring-renamed resources are not inherently equivalent: a new frame still needs the previous background state. A genuine alternative must preserve exact masks and resource/latency costs.

**Conformance:** On the declared Mesa Vulkan fallback runner, the retained exact per-frame CPU-oracle controls additionally cover 640×360 (8 frames in serial and K=4) and 1280×720 (4 frames in K=2). These are scale/functional qualification cases, not throughput benchmarks or evidence that an arbitrary physical GPU admits the same budgets. `cargo test --example gpu_streaming_background` runs CPU tests; `RUNEN_GPU_STREAMING_REQUIRE_GPU=1 cargo test --example gpu_streaming_background -- --ignored` runs the GPU tests and requires a real declared Vulkan software-fallback adapter. The source-owned Conformance workflow runs these plus serial/batch CLI fixtures and retains JSONL/PGM artifacts. Normal canonical validation remains `cargo validate`.

**Not claimed by this example:** direct camera driver, on-glass Present, synchronized capture clocks, new G5 semantics, partial-GPU-submit fault injection, or a measured need for ordered continuation. The tests exercise the available public boundary and explicitly distinguish valid terminal status from readback and domain verification.
