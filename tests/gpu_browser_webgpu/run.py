#!/usr/bin/env python3
"""Execute the retained RunenGPU browser WebGPU proof through ChromeDriver."""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import io
import json
import pathlib
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    method: str,
    url: str,
    payload: object | None = None,
    *,
    timeout: float = 10,
) -> object:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"WebDriver HTTP {error.code}: {detail}") from error
    decoded = json.loads(raw.decode("utf-8")) if raw else {}
    if isinstance(decoded, dict):
        value = decoded.get("value")
        if isinstance(value, dict):
            if value.get("error"):
                raise RuntimeError(f"WebDriver error: {value}")
    return decoded


def wait_for_driver(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ChromeDriver exited early with code {process.returncode}")
        try:
            response = request_json("GET", f"{base_url}/status")
            if isinstance(response, dict):
                value = response.get("value")
                if isinstance(value, dict) and value.get("ready"):
                    return
        except (OSError, RuntimeError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise RuntimeError("ChromeDriver did not become ready")


def browser_script() -> str:
    return r"""
const done = arguments[arguments.length - 1];
(async () => {
  try {
    if (!navigator.gpu) {
      throw new Error("navigator.gpu is unavailable in the declared browser-conformance environment");
    }
    const module = await import("./gpu_browser_webgpu.js");
    const wasm = await module.default();
    if (typeof wasm.runengpu_browser_start !== "function" ||
        typeof wasm.runengpu_browser_poll !== "function" ||
        typeof wasm.runengpu_browser_rgba16_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_rgba8_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_r8_new_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_rg8_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_r16_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_rg16_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_packed32_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_packed32_sampled_mask !== "function" ||
        typeof wasm.runengpu_browser_packed32_color_attachment_mask !== "function" ||
        typeof wasm.runengpu_browser_bc_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_blend_state_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_sampler_anisotropy_exercised !== "function" ||
        typeof wasm.runengpu_browser_vertex8_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_vertex16_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_vertex_packed_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_depth_sampled_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_depth_attachment_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_depth_copy_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_depth_linear_exercised_mask !== "function" ||
        typeof wasm.runengpu_browser_stencil8_exercised !== "function" ||
        typeof wasm.runengpu_browser_depth24plus_stencil8_exercised !== "function" ||
        typeof wasm.runengpu_browser_depth24plus_stencil8_sampled_exercised !== "function" ||
        typeof wasm.runengpu_browser_depth32float_stencil8_exercised !== "function" ||
        typeof wasm.runengpu_browser_depth32float_stencil8_sampled_exercised !== "function") {
      throw new Error("RunenGPU browser proof control exports are absent");
    }
    wasm.runengpu_browser_start();
    for (let tick = 0; tick < 5000; tick += 1) {
      const status = wasm.runengpu_browser_poll();
      if (status === 1) {
        done({
          ok: true,
          rgba16Mask: wasm.runengpu_browser_rgba16_exercised_mask(),
          rgba8Mask: wasm.runengpu_browser_rgba8_exercised_mask(),
          r8NewMask: wasm.runengpu_browser_r8_new_exercised_mask(),
          rg8Mask: wasm.runengpu_browser_rg8_exercised_mask(),
          r16Mask: wasm.runengpu_browser_r16_exercised_mask(),
          rg16Mask: wasm.runengpu_browser_rg16_exercised_mask(),
          packed32Mask: wasm.runengpu_browser_packed32_exercised_mask(),
          packed32SampledMask: wasm.runengpu_browser_packed32_sampled_mask(),
          packed32ColorAttachmentMask: wasm.runengpu_browser_packed32_color_attachment_mask(),
          bcMask: wasm.runengpu_browser_bc_exercised_mask(),
          blendStateMask: wasm.runengpu_browser_blend_state_exercised_mask(),
          samplerAnisotropyExercised: wasm.runengpu_browser_sampler_anisotropy_exercised(),
          vertex8Mask: wasm.runengpu_browser_vertex8_exercised_mask(),
          vertex16Mask: wasm.runengpu_browser_vertex16_exercised_mask(),
          vertexPackedMask: wasm.runengpu_browser_vertex_packed_exercised_mask(),
          depthSampledMask: wasm.runengpu_browser_depth_sampled_exercised_mask(),
          depthAttachmentMask: wasm.runengpu_browser_depth_attachment_exercised_mask(),
          depthCopyMask: wasm.runengpu_browser_depth_copy_exercised_mask(),
          depthLinearMask: wasm.runengpu_browser_depth_linear_exercised_mask(),
          stencil8Exercised: wasm.runengpu_browser_stencil8_exercised(),
          depth24PlusStencil8Exercised: wasm.runengpu_browser_depth24plus_stencil8_exercised(),
          depth24PlusStencil8SampledExercised: wasm.runengpu_browser_depth24plus_stencil8_sampled_exercised(),
          depth32FloatStencil8Exercised: wasm.runengpu_browser_depth32float_stencil8_exercised(),
          depth32FloatStencil8SampledExercised: wasm.runengpu_browser_depth32float_stencil8_sampled_exercised(),
        });
        return;
      }
      if (status !== 0) {
        throw new Error(`RunenGPU browser proof returned unexpected status ${status}`);
      }
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    throw new Error("RunenGPU browser proof exceeded the bounded event-loop progress budget");
  } catch (error) {
    done({ok: false, error: String(error && error.stack ? error.stack : error)});
  }
})();
"""


def read_exercised_mask(
    value: dict[str, object],
    mask_key: str,
    family: str,
    format_count: int,
) -> int:
    if format_count < 1 or format_count > 32:
        raise ValueError("format reporter requires 1..32 formats")
    mask = value.get(mask_key)
    full_mask = (1 << format_count) - 1
    if type(mask) is not int or mask < 0 or mask > full_mask:
        raise RuntimeError(
            f"actual-browser {family} proof did not report valid execution evidence: {mask!r}"
        )
    return mask


def report_format_family(
    value: dict[str, object],
    mask_key: str,
    family: str,
    format_names: tuple[str, ...],
    evidence: str,
    skip_reason: str = "copy roles not both advertised",
) -> None:
    mask = read_exercised_mask(value, mask_key, family, len(format_names))
    for index, format_name in enumerate(format_names):
        if mask & (1 << index):
            print(f"RunenGPU actual-browser {format_name}: EXERCISED ({evidence})")
        else:
            print(f"RunenGPU actual-browser {format_name}: SKIPPED ({skip_reason})")
    if mask == 0:
        message = (
            f"RunenGPU actual-browser {family}: NOT QUALIFIED "
            f"(all {len(format_names)} formats skipped)"
        )
        print(message)
        raise RuntimeError(message)


def report_optional_format_roles(
    value: dict[str, object],
    mask_key: str,
    family: str,
    format_names: tuple[str, ...],
    role: str,
    evidence: str,
) -> None:
    mask = read_exercised_mask(value, mask_key, f"{family} {role}", len(format_names))
    for index, format_name in enumerate(format_names):
        if mask & (1 << index):
            print(
                f"RunenGPU actual-browser {format_name} {role}: "
                f"EXERCISED ({evidence})"
            )
        else:
            print(
                f"RunenGPU actual-browser {format_name} {role}: "
                f"SKIPPED ({role} role not advertised)"
            )


def report_depth_proofs(value: dict[str, object]) -> None:
    formats = ("Depth16Unorm", "Depth24Plus")
    sampled = read_exercised_mask(value, "depthSampledMask", "Depth sampled", len(formats))
    attachment = read_exercised_mask(
        value, "depthAttachmentMask", "Depth attachment", len(formats)
    )
    copy = read_exercised_mask(value, "depthCopyMask", "Depth copy", len(formats))
    linear = read_exercised_mask(value, "depthLinearMask", "Depth16 linear", len(formats))

    if linear & ~copy or linear & ~1:
        raise RuntimeError("actual-browser Depth16 linear evidence is inconsistent")

    for index, format_name in enumerate(formats):
        bit = 1 << index
        if sampled & bit:
            print(
                f"RunenGPU actual-browser {format_name} Sampled: "
                "EXERCISED (D2 sampled-usage realization)"
            )
        else:
            print(
                f"RunenGPU actual-browser {format_name} Sampled: "
                "SKIPPED (sampled role not advertised)"
            )

        if attachment & bit:
            print(
                f"RunenGPU actual-browser {format_name} DepthStencil: "
                "EXERCISED (clear-only depth attachment pass)"
            )
        else:
            print(
                f"RunenGPU actual-browser {format_name} DepthStencil: "
                "SKIPPED (depth-attachment role not advertised)"
            )

        if copy & bit:
            print(
                f"RunenGPU actual-browser {format_name} Copy: "
                "EXERCISED (full-plane texture-to-texture copy)"
            )
        else:
            print(
                f"RunenGPU actual-browser {format_name} Copy: "
                "SKIPPED (CopySource + CopyDestination not both advertised)"
            )

    if linear & 1:
        print(
            "RunenGPU actual-browser Depth16Unorm linear transfer: "
            "EXERCISED (127px and 128px buffer -> texture -> texture -> buffer round trips)"
        )
    elif copy & 1:
        raise RuntimeError("actual-browser Depth16 copy executed without required linear proof")
    else:
        print(
            "RunenGPU actual-browser Depth16Unorm linear transfer: "
            "SKIPPED (CopySource + CopyDestination not both advertised)"
        )

    if attachment == 0:
        message = "RunenGPU actual-browser Depth: NOT QUALIFIED (all formats skipped)"
        print(message)
        raise RuntimeError(message)


def report_stencil8_proof(value: dict[str, object]) -> None:
    exercised = value.get("stencil8Exercised")
    if type(exercised) is not int or exercised not in (0, 1):
        raise RuntimeError(
            f"actual-browser Stencil8 proof did not report valid execution evidence: {exercised!r}"
        )
    if exercised:
        print(
            "RunenGPU actual-browser Stencil8: EXERCISED "
            "(clear + dynamic Replace + read-only test + exact 255px/256px readback)"
        )
    else:
        print(
            "RunenGPU actual-browser Stencil8: SKIPPED "
            "(DepthStencil + CopySource + CopyDestination roles not all advertised)"
        )


def report_depth24plus_stencil8_proof(value: dict[str, object]) -> None:
    exercised = value.get("depth24PlusStencil8Exercised")
    sampled = value.get("depth24PlusStencil8SampledExercised")
    if type(exercised) is not int or exercised not in (0, 1):
        raise RuntimeError(
            "actual-browser Depth24PlusStencil8 proof did not report valid "
            f"execution evidence: {exercised!r}"
        )
    if type(sampled) is not int or sampled not in (0, 1):
        raise RuntimeError(
            "actual-browser Depth24PlusStencil8 sampled proof did not report valid "
            f"execution evidence: {sampled!r}"
        )
    if sampled and not exercised:
        raise RuntimeError(
            "actual-browser Depth24PlusStencil8 sampled evidence requires core execution"
        )
    if exercised:
        print(
            "RunenGPU actual-browser Depth24PlusStencil8: EXERCISED "
            "(mixed depth-read-only/stencil-write + all-aspect copy + "
            "copied stencil snapshot + copied-depth gate + exact 255px/256px readback)"
        )
    else:
        print(
            "RunenGPU actual-browser Depth24PlusStencil8: SKIPPED "
            "(DepthStencil + CopySource + CopyDestination roles not all advertised)"
        )
    if sampled:
        print(
            "RunenGPU actual-browser Depth24PlusStencil8 Sampled: EXERCISED "
            "(DepthOnly texture_depth_2d + StencilOnly texture_2d<u32> compute bindings)"
        )
    else:
        print(
            "RunenGPU actual-browser Depth24PlusStencil8 Sampled: SKIPPED "
            "(sampled role not advertised)"
        )


def report_depth32float_stencil8_proof(value: dict[str, object]) -> None:
    exercised = value.get("depth32FloatStencil8Exercised")
    sampled = value.get("depth32FloatStencil8SampledExercised")
    if type(exercised) is not int or exercised not in (0, 1):
        raise RuntimeError(
            "actual-browser Depth32FloatStencil8 proof did not report valid "
            f"execution evidence: {exercised!r}"
        )
    if type(sampled) is not int or sampled not in (0, 1):
        raise RuntimeError(
            "actual-browser Depth32FloatStencil8 sampled proof did not report valid "
            f"execution evidence: {sampled!r}"
        )
    if sampled and not exercised:
        raise RuntimeError(
            "actual-browser Depth32FloatStencil8 sampled evidence requires core execution"
        )
    if exercised:
        print(
            "RunenGPU actual-browser Depth32FloatStencil8: EXERCISED "
            "(combined attachment + all-aspect copy + DepthOnly 4-byte readback + "
            "StencilOnly 1-byte readback + copied-depth gate at 255px/256px)"
        )
    else:
        print(
            "RunenGPU actual-browser Depth32FloatStencil8: SKIPPED "
            "(optional backend prerequisite or DepthStencil/Copy roles not advertised)"
        )
    if sampled:
        print(
            "RunenGPU actual-browser Depth32FloatStencil8 Sampled: EXERCISED "
            "(DepthOnly texture_depth_2d + StencilOnly texture_2d<u32> compute bindings)"
        )
    else:
        print(
            "RunenGPU actual-browser Depth32FloatStencil8 Sampled: SKIPPED "
            "(sampled role not advertised)"
        )


def verify_format_reporter() -> None:
    """CI-invoked zero/partial/full/out-of-range regression for 3 and 4 formats."""
    families = (
        ("RGBA8", ("Rgba8Snorm", "Rgba8Uint", "Rgba8Sint"), (0, 1, 5, 7)),
        ("RGBA16", ("Rgba16Uint", "Rgba16Sint", "Rgba16Float"), (0, 1, 5, 7)),
        ("R8-new", ("R8Snorm", "R8Uint", "R8Sint"), (0, 1, 5, 7)),
        ("RG8", ("Rg8Unorm", "Rg8Snorm", "Rg8Uint", "Rg8Sint"), (0, 1, 9, 15)),
        ("R16", ("R16Uint", "R16Sint", "R16Float"), (0, 1, 5, 7)),
        ("RG16", ("Rg16Uint", "Rg16Sint", "Rg16Float"), (0, 1, 5, 7)),
        (
            "Packed32",
            ("Rgb9e5Ufloat", "Rgb10a2Uint", "Rgb10a2Unorm", "Rg11b10Ufloat"),
            (0, 1, 9, 15),
        ),
    )
    for family, names, masks in families:
        full_mask = (1 << len(names)) - 1
        for mask in masks:
            captured = io.StringIO()
            failed = False
            try:
                with contextlib.redirect_stdout(captured):
                    report_format_family({"mask": mask}, "mask", family, names, "test evidence")
            except RuntimeError as error:
                if mask != 0 or "NOT QUALIFIED" not in str(error):
                    raise AssertionError(
                        f"unexpected reporter failure for {family} mask {mask}"
                    ) from error
                failed = True
            if failed != (mask == 0):
                raise AssertionError(f"incorrect reporter qualification for {family} mask {mask}")
            transcript = captured.getvalue()
            count = mask.bit_count()
            if transcript.count(": EXERCISED (") != count:
                raise AssertionError(f"incorrect exercised count for {family} mask {mask}")
            if transcript.count(": SKIPPED (") != len(names) - count:
                raise AssertionError(f"incorrect skipped count for {family} mask {mask}")
            if ("NOT QUALIFIED" in transcript) != (mask == 0):
                raise AssertionError(
                    f"incorrect not-qualified evidence for {family} mask {mask}"
                )

        for invalid in (
            {},
            {"mask": -1},
            {"mask": full_mask + 1},
            {"mask": "1"},
            {"mask": True},
        ):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    report_format_family(invalid, "mask", family, names, "test evidence")
            except RuntimeError as error:
                if "valid execution evidence" not in str(error):
                    raise AssertionError(
                        f"incorrect invalid-mask failure for {family}: {invalid!r}"
                    ) from error
            else:
                raise AssertionError(f"invalid {family} mask was accepted: {invalid!r}")
    print("RunenGPU actual-browser format reporter regression: PASS")


def verify_depth_reporter() -> None:
    valid = {
        "depthSampledMask": 3,
        "depthAttachmentMask": 1,
        "depthCopyMask": 1,
        "depthLinearMask": 1,
    }
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        report_depth_proofs(valid)
    transcript = captured.getvalue()
    if transcript.count(": EXERCISED (") != 5:
        raise AssertionError("incorrect exercised count for depth proof reporter")
    if transcript.count(": SKIPPED (") != 2:
        raise AssertionError("incorrect skipped count for depth proof reporter")

    for invalid in (
        {**valid, "depthAttachmentMask": 0, "depthCopyMask": 0, "depthLinearMask": 0},
        {**valid, "depthCopyMask": 0},
        {**valid, "depthLinearMask": 2},
        {**valid, "depthSampledMask": 4},
    ):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                report_depth_proofs(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid depth proof evidence was accepted: {invalid!r}")
    print("RunenGPU actual-browser depth reporter regression: PASS")


def verify_stencil8_reporter() -> None:
    for exercised in (0, 1):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            report_stencil8_proof({"stencil8Exercised": exercised})
        transcript = captured.getvalue()
        expected = "EXERCISED" if exercised else "SKIPPED"
        if expected not in transcript:
            raise AssertionError(
                f"incorrect Stencil8 reporter output for evidence {exercised}"
            )
    for invalid in ({}, {"stencil8Exercised": -1}, {"stencil8Exercised": 2},
                    {"stencil8Exercised": True}, {"stencil8Exercised": "1"}):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                report_stencil8_proof(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid Stencil8 evidence was accepted: {invalid!r}")
    print("RunenGPU actual-browser Stencil8 reporter regression: PASS")


def verify_depth24plus_stencil8_reporter() -> None:
    cases = (
        (0, 0, "SKIPPED", "SKIPPED"),
        (1, 0, "EXERCISED", "SKIPPED"),
        (1, 1, "EXERCISED", "EXERCISED"),
    )
    for exercised, sampled, core_expected, sampled_expected in cases:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            report_depth24plus_stencil8_proof(
                {
                    "depth24PlusStencil8Exercised": exercised,
                    "depth24PlusStencil8SampledExercised": sampled,
                }
            )
        transcript = captured.getvalue()
        lines = [line for line in transcript.splitlines() if line]
        if core_expected not in lines[0] or sampled_expected not in lines[1]:
            raise AssertionError(
                "incorrect Depth24PlusStencil8 reporter output "
                f"for evidence {(exercised, sampled)}"
            )
    invalid = (
        {},
        {"depth24PlusStencil8Exercised": 1},
        {
            "depth24PlusStencil8Exercised": 0,
            "depth24PlusStencil8SampledExercised": 1,
        },
        {
            "depth24PlusStencil8Exercised": 2,
            "depth24PlusStencil8SampledExercised": 0,
        },
        {
            "depth24PlusStencil8Exercised": 1,
            "depth24PlusStencil8SampledExercised": True,
        },
    )
    for evidence in invalid:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                report_depth24plus_stencil8_proof(evidence)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"invalid Depth24PlusStencil8 evidence was accepted: {evidence!r}"
            )
    print("RunenGPU actual-browser Depth24PlusStencil8 reporter regression: PASS")


def verify_depth32float_stencil8_reporter() -> None:
    cases = (
        (0, 0, "SKIPPED", "SKIPPED"),
        (1, 0, "EXERCISED", "SKIPPED"),
        (1, 1, "EXERCISED", "EXERCISED"),
    )
    for exercised, sampled, core_expected, sampled_expected in cases:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            report_depth32float_stencil8_proof(
                {
                    "depth32FloatStencil8Exercised": exercised,
                    "depth32FloatStencil8SampledExercised": sampled,
                }
            )
        lines = [line for line in captured.getvalue().splitlines() if line]
        if core_expected not in lines[0] or sampled_expected not in lines[1]:
            raise AssertionError(
                "incorrect Depth32FloatStencil8 reporter output "
                f"for evidence {(exercised, sampled)}"
            )
    invalid = (
        {},
        {"depth32FloatStencil8Exercised": 1},
        {
            "depth32FloatStencil8Exercised": 0,
            "depth32FloatStencil8SampledExercised": 1,
        },
        {
            "depth32FloatStencil8Exercised": 2,
            "depth32FloatStencil8SampledExercised": 0,
        },
        {
            "depth32FloatStencil8Exercised": 1,
            "depth32FloatStencil8SampledExercised": True,
        },
    )
    for evidence in invalid:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                report_depth32float_stencil8_proof(evidence)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"invalid Depth32FloatStencil8 evidence was accepted: {evidence!r}"
            )
    print("RunenGPU actual-browser Depth32FloatStencil8 reporter regression: PASS")


def main() -> int:
    args = parse_args()
    verify_format_reporter()
    verify_depth_reporter()
    verify_stencil8_reporter()
    verify_depth24plus_stencil8_reporter()
    verify_depth32float_stencil8_reporter()
    out_dir = args.out_dir.resolve()
    js_path = out_dir / "gpu_browser_webgpu.js"
    wasm_path = out_dir / "gpu_browser_webgpu_bg.wasm"
    if not js_path.is_file() or not wasm_path.is_file():
        raise RuntimeError(f"wasm-bindgen browser output is incomplete in {out_dir}")

    chromedriver = shutil.which("chromedriver")
    chrome = (
        shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chromedriver is None or chrome is None:
        raise RuntimeError("Chrome and ChromeDriver must be installed by the conformance environment")

    index_path = out_dir / "index.html"
    index_path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>RunenGPU Browser WebGPU</title>\n"
    )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    driver_port = free_port()
    driver_log = tempfile.NamedTemporaryFile(
        prefix="runengpu-chromedriver-", suffix=".log", delete=False
    )
    driver_log_path = pathlib.Path(driver_log.name)
    driver_log.close()
    process: subprocess.Popen[str] | None = None
    driver_base = f"http://127.0.0.1:{driver_port}"
    session_id: str | None = None

    try:
        process = subprocess.Popen(
            [chromedriver, f"--port={driver_port}", f"--log-path={driver_log_path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        wait_for_driver(driver_base, process)
        response = request_json(
            "POST",
            f"{driver_base}/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "chrome",
                        "goog:chromeOptions": {
                            "binary": chrome,
                            "args": [
                                "--headless=new",
                                "--no-sandbox",
                                "--disable-dev-shm-usage",
                                "--enable-unsafe-webgpu",
                                "--use-webgpu-adapter=swiftshader",
                                "--use-gpu-in-tests",
                            ],
                        },
                    }
                }
            },
            timeout=30,
        )
        if not isinstance(response, dict) or not isinstance(response.get("value"), dict):
            raise RuntimeError(f"unexpected ChromeDriver session response: {response!r}")
        session_id = response["value"].get("sessionId")
        if not isinstance(session_id, str):
            raise RuntimeError(f"ChromeDriver did not return a session id: {response!r}")

        session_base = f"{driver_base}/session/{session_id}"
        request_json(
            "POST",
            f"{session_base}/timeouts",
            {"script": 120000, "pageLoad": 30000},
        )
        browser_url = f"http://127.0.0.1:{server.server_port}/index.html"
        request_json("POST", f"{session_base}/url", {"url": browser_url})
        result = request_json(
            "POST",
            f"{session_base}/execute/async",
            {"script": browser_script(), "args": []},
            timeout=130,
        )
        if not isinstance(result, dict) or not isinstance(result.get("value"), dict):
            raise RuntimeError(f"unexpected browser proof response: {result!r}")
        value = result["value"]
        if value.get("ok") is not True:
            raise RuntimeError(
                f"actual-browser RunenGPU proof failed: {value.get('error', value)!s}"
            )
        report_format_family(
            value,
            "rgba16Mask",
            "RGBA16",
            ("Rgba16Uint", "Rgba16Sint", "Rgba16Float"),
            "31px and 32px copy round trips",
        )
        report_format_family(
            value,
            "rgba8Mask",
            "RGBA8",
            ("Rgba8Snorm", "Rgba8Uint", "Rgba8Sint"),
            "63px and 64px copy round trips",
        )
        report_format_family(
            value,
            "r8NewMask",
            "R8-new",
            ("R8Snorm", "R8Uint", "R8Sint"),
            "255px and 256px copy round trips",
        )
        report_format_family(
            value,
            "rg8Mask",
            "RG8",
            ("Rg8Unorm", "Rg8Snorm", "Rg8Uint", "Rg8Sint"),
            "127px and 128px copy round trips",
        )
        report_format_family(
            value,
            "r16Mask",
            "R16",
            ("R16Uint", "R16Sint", "R16Float"),
            "127px and 128px copy round trips",
        )
        report_format_family(
            value,
            "rg16Mask",
            "RG16",
            ("Rg16Uint", "Rg16Sint", "Rg16Float"),
            "63px and 64px copy round trips",
        )
        packed_names = (
            "Rgb9e5Ufloat",
            "Rgb10a2Uint",
            "Rgb10a2Unorm",
            "Rg11b10Ufloat",
        )
        report_format_family(
            value,
            "packed32Mask",
            "Packed32",
            packed_names,
            "zero-valued 63px and 64px copy round trips",
        )
        report_optional_format_roles(
            value,
            "packed32SampledMask",
            "Packed32",
            packed_names,
            "Sampled",
            "sampled texture-view and bind-group realization",
        )
        report_optional_format_roles(
            value,
            "packed32ColorAttachmentMask",
            "Packed32",
            packed_names,
            "ColorAttachment",
            "submitted color-attachment clear render pass with admitted role",
        )
        bc_names = (
            "Bc1RgbaUnorm",
            "Bc1RgbaUnormSrgb",
            "Bc2RgbaUnorm",
            "Bc2RgbaUnormSrgb",
            "Bc3RgbaUnorm",
            "Bc3RgbaUnormSrgb",
            "Bc4RUnorm",
            "Bc4RSnorm",
            "Bc5RgUnorm",
            "Bc5RgSnorm",
            "Bc6hRgbUfloat",
            "Bc6hRgbFloat",
            "Bc7RgbaUnorm",
            "Bc7RgbaUnormSrgb",
        )
        bc_mask = read_exercised_mask(value, "bcMask", "BC", len(bc_names))
        bc_full_mask = (1 << len(bc_names)) - 1
        if bc_mask == 0:
            print(
                "RunenGPU actual-browser BC texture family: "
                "UNSUPPORTED (adapter does not expose portable BC roles)"
            )
        elif bc_mask == bc_full_mask:
            for format_name in bc_names:
                print(
                    f"RunenGPU actual-browser {format_name}: "
                    "EXERCISED (compressed upload + copy + exact readback)"
                )
        else:
            raise RuntimeError(
                "RunenGPU actual-browser BC: PARTIAL SUPPORT IS NOT QUALIFIED "
                f"(mask={bc_mask:#x}, expected 0 or {bc_full_mask:#x})"
            )

        blend_state_names = ("independent_subtract", "min_max")
        blend_state_mask = read_exercised_mask(
            value, "blendStateMask", "BlendState", len(blend_state_names)
        )
        blend_state_full_mask = (1 << len(blend_state_names)) - 1
        for index, case_name in enumerate(blend_state_names):
            if blend_state_mask & (1 << index):
                print(
                    f"RunenGPU actual-browser {case_name} blend: "
                    "EXERCISED (independent normalized blend state + exact readback)"
                )
            else:
                print(f"RunenGPU actual-browser {case_name} blend: NOT EXERCISED")
        if blend_state_mask != blend_state_full_mask:
            raise RuntimeError(
                "RunenGPU actual-browser BlendState: NOT QUALIFIED "
                f"(mask={blend_state_mask:#x}, expected={blend_state_full_mask:#x})"
            )

        sampler_anisotropy = value.get("samplerAnisotropyExercised")
        if type(sampler_anisotropy) is not int or sampler_anisotropy != 1:
            raise RuntimeError(
                "RunenGPU actual-browser sampler anisotropy: NOT QUALIFIED "
                f"(value={sampler_anisotropy!r}, expected=1)"
            )
        print(
            "RunenGPU actual-browser sampler anisotropy: "
            "EXERCISED (requested max=8 through public sampler realization)"
        )

        vertex8_names = (
            "Uint8",
            "Uint8x2",
            "Uint8x4",
            "Sint8",
            "Sint8x2",
            "Sint8x4",
            "Unorm8",
            "Unorm8x2",
            "Unorm8x4",
            "Snorm8",
            "Snorm8x2",
            "Snorm8x4",
        )
        vertex8_mask = read_exercised_mask(
            value, "vertex8Mask", "Vertex8", len(vertex8_names)
        )
        vertex8_full_mask = (1 << len(vertex8_names)) - 1
        for index, format_name in enumerate(vertex8_names):
            if vertex8_mask & (1 << index):
                print(
                    f"RunenGPU actual-browser {format_name} vertex: "
                    "EXERCISED (compact-offset vertex draw + exact readback)"
                )
            else:
                print(
                    f"RunenGPU actual-browser {format_name} vertex: NOT EXERCISED"
                )
        if vertex8_mask != vertex8_full_mask:
            raise RuntimeError(
                "RunenGPU actual-browser Vertex8: NOT QUALIFIED "
                f"(mask={vertex8_mask:#x}, expected={vertex8_full_mask:#x})"
            )

        vertex16_names = (
            "Uint16",
            "Uint16x2",
            "Uint16x4",
            "Sint16",
            "Sint16x2",
            "Sint16x4",
            "Unorm16",
            "Unorm16x2",
            "Unorm16x4",
            "Snorm16",
            "Snorm16x2",
            "Snorm16x4",
            "Float16",
            "Float16x2",
            "Float16x4",
        )
        vertex16_mask = read_exercised_mask(
            value, "vertex16Mask", "Vertex16", len(vertex16_names)
        )
        vertex16_full_mask = (1 << len(vertex16_names)) - 1
        for index, format_name in enumerate(vertex16_names):
            if vertex16_mask & (1 << index):
                print(
                    f"RunenGPU actual-browser {format_name} vertex: "
                    "EXERCISED (16-bit compact-offset vertex draw + exact readback)"
                )
            else:
                print(
                    f"RunenGPU actual-browser {format_name} vertex: NOT EXERCISED"
                )
        if vertex16_mask != vertex16_full_mask:
            raise RuntimeError(
                "RunenGPU actual-browser Vertex16: NOT QUALIFIED "
                f"(mask={vertex16_mask:#x}, expected={vertex16_full_mask:#x})"
            )

        vertex_packed_names = ("Unorm10_10_10_2", "Unorm8x4Bgra")
        vertex_packed_mask = read_exercised_mask(
            value, "vertexPackedMask", "VertexPacked", len(vertex_packed_names)
        )
        vertex_packed_full_mask = (1 << len(vertex_packed_names)) - 1
        for index, format_name in enumerate(vertex_packed_names):
            if vertex_packed_mask & (1 << index):
                print(
                    f"RunenGPU actual-browser {format_name} vertex: "
                    "EXERCISED (non-symmetric packed decode + exact readback)"
                )
            else:
                print(
                    f"RunenGPU actual-browser {format_name} vertex: NOT EXERCISED"
                )
        if vertex_packed_mask != vertex_packed_full_mask:
            raise RuntimeError(
                "RunenGPU actual-browser VertexPacked: NOT QUALIFIED "
                f"(mask={vertex_packed_mask:#x}, expected={vertex_packed_full_mask:#x})"
            )

        report_depth_proofs(value)
        report_stencil8_proof(value)
        report_depth24plus_stencil8_proof(value)
        report_depth32float_stencil8_proof(value)
        print("RunenGPU actual-browser WebGPU conformance: PASS")
        return 0
    except Exception:
        if driver_log_path.exists():
            log = driver_log_path.read_text(errors="replace")
            if log:
                print("--- ChromeDriver log ---")
                print(log)
        raise
    finally:
        if session_id is not None and process is not None and process.poll() is None:
            try:
                request_json("DELETE", f"{driver_base}/session/{session_id}")
            except Exception:
                pass
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        index_path.unlink(missing_ok=True)
        driver_log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
