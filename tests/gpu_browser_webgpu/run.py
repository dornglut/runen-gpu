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
        typeof wasm.runengpu_browser_rg16_exercised_mask !== "function") {
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


def report_format_family(
    value: dict[str, object],
    mask_key: str,
    family: str,
    format_names: tuple[str, ...],
    widths: str,
) -> None:
    if not format_names or len(format_names) > 32:
        raise ValueError("format reporter requires 1..32 names")
    mask = value.get(mask_key)
    full_mask = (1 << len(format_names)) - 1
    if type(mask) is not int or mask < 0 or mask > full_mask:
        raise RuntimeError(
            f"actual-browser {family} proof did not report valid execution evidence: {mask!r}"
        )
    for index, format_name in enumerate(format_names):
        if mask & (1 << index):
            print(f"RunenGPU actual-browser {format_name}: EXERCISED ({widths} copy round trips)")
        else:
            print(f"RunenGPU actual-browser {format_name}: SKIPPED (copy roles not both advertised)")
    if mask == 0:
        message = (
            f"RunenGPU actual-browser {family}: NOT QUALIFIED "
            f"(all {len(format_names)} formats skipped)"
        )
        print(message)
        raise RuntimeError(message)


def verify_format_reporter() -> None:
    """CI-invoked zero/partial/full/out-of-range regression for 3 and 4 formats."""
    families = (
        ("RGBA8", ("Rgba8Snorm", "Rgba8Uint", "Rgba8Sint"), (0, 1, 5, 7)),
        ("RGBA16", ("Rgba16Uint", "Rgba16Sint", "Rgba16Float"), (0, 1, 5, 7)),
        ("R8-new", ("R8Snorm", "R8Uint", "R8Sint"), (0, 1, 5, 7)),
        ("RG8", ("Rg8Unorm", "Rg8Snorm", "Rg8Uint", "Rg8Sint"), (0, 1, 9, 15)),
        ("R16", ("R16Uint", "R16Sint", "R16Float"), (0, 1, 5, 7)),
        ("RG16", ("Rg16Uint", "Rg16Sint", "Rg16Float"), (0, 1, 5, 7)),
    )
    for family, names, masks in families:
        full_mask = (1 << len(names)) - 1
        for mask in masks:
            captured = io.StringIO()
            failed = False
            try:
                with contextlib.redirect_stdout(captured):
                    report_format_family({"mask": mask}, "mask", family, names, "test widths")
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
                    report_format_family(invalid, "mask", family, names, "test widths")
            except RuntimeError as error:
                if "valid execution evidence" not in str(error):
                    raise AssertionError(
                        f"incorrect invalid-mask failure for {family}: {invalid!r}"
                    ) from error
            else:
                raise AssertionError(f"invalid {family} mask was accepted: {invalid!r}")
    print("RunenGPU actual-browser format reporter regression: PASS")


def main() -> int:
    args = parse_args()
    verify_format_reporter()
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
            "31px and 32px",
        )
        report_format_family(
            value,
            "rgba8Mask",
            "RGBA8",
            ("Rgba8Snorm", "Rgba8Uint", "Rgba8Sint"),
            "63px and 64px",
        )
        report_format_family(
            value,
            "r8NewMask",
            "R8-new",
            ("R8Snorm", "R8Uint", "R8Sint"),
            "255px and 256px",
        )
        report_format_family(
            value,
            "rg8Mask",
            "RG8",
            ("Rg8Unorm", "Rg8Snorm", "Rg8Uint", "Rg8Sint"),
            "127px and 128px",
        )
        report_format_family(
            value,
            "r16Mask",
            "R16",
            ("R16Uint", "R16Sint", "R16Float"),
            "127px and 128px",
        )
        report_format_family(
            value,
            "rg16Mask",
            "RG16",
            ("Rg16Uint", "Rg16Sint", "Rg16Float"),
            "63px and 64px",
        )
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
