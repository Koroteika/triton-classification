from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


CONFIGS = {
    "none": {
        "title": "No batching",
        "preferred_batch_size": None,
        "max_queue_delay_microseconds": None,
    },
    "small": {
        "title": "Small batch",
        "preferred_batch_size": [2, 4],
        "max_queue_delay_microseconds": 50_000,
    },
    "medium": {
        "title": "Medium batch",
        "preferred_batch_size": [4, 8, 16],
        "max_queue_delay_microseconds": 100_000,
    },
    "large": {
        "title": "Large batch",
        "preferred_batch_size": [8, 16, 32],
        "max_queue_delay_microseconds": 200_000,
    },
}


def config_pbtxt(mode: str) -> str:
    settings = CONFIGS[mode]
    dynamic_batching = ""
    if settings["preferred_batch_size"] is not None:
        preferred = ", ".join(str(item) for item in settings["preferred_batch_size"])
        delay = settings["max_queue_delay_microseconds"]
        dynamic_batching = f"""
# Dynamic Batching collects individual requests into larger batches.
dynamic_batching {{
  preferred_batch_size: [{preferred}]
  max_queue_delay_microseconds: {delay}
}}
"""

    return f"""name: "image_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "input_image"
    data_type: TYPE_FP32
    dims: [128, 128, 3]
  }}
]

output [
  {{
    name: "Identity:0"
    data_type: TYPE_FP32
    dims: [3]
  }}
]
{dynamic_batching}
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""


def write_model_config(project_dir: Path, mode: str) -> None:
    config_path = project_dir / "model_repository" / "image_classifier" / "config.pbtxt"
    config_path.write_text(config_pbtxt(mode), encoding="utf-8")


def run_command(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def http_get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


def wait_for_url(url: str, timeout_seconds: float, expect_healthy: bool = False) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            if expect_healthy:
                payload = http_get_json(url)
                if payload.get("status") == "healthy":
                    return
            else:
                with urllib.request.urlopen(url, timeout=5.0) as response:
                    if 200 <= response.status < 300:
                        return
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)

    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def restart_stack_for_config(project_dir: Path, mode: str) -> None:
    print(f"\n=== Applying config: {mode} ({CONFIGS[mode]['title']}) ===")
    write_model_config(project_dir, mode)

    run_command(["docker", "compose", "restart", "triton"], cwd=project_dir)
    wait_for_url("http://localhost:8000/v2/health/ready", timeout_seconds=120)

    run_command(["docker", "compose", "restart", "api"], cwd=project_dir)
    wait_for_url("http://localhost:8080/health", timeout_seconds=120, expect_healthy=True)


def build_payload(image_path: Path) -> bytes:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return json.dumps({"image": encoded}).encode("utf-8")


def post_prediction(api_url: str, payload: bytes, timeout: float) -> dict:
    request = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
        latency_ms = (time.perf_counter() - start) * 1000
        return {"ok": 200 <= status < 300, "status": status, "latency_ms": latency_ms}
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "status": exc.code, "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "status": None, "latency_ms": latency_ms, "error": str(exc)}


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percent / 100)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_load_test(
    api_url: str,
    image_path: Path,
    total_requests: int,
    concurrency: int,
    warmup_requests: int,
    timeout: float,
) -> dict:
    payload = build_payload(image_path)

    for _ in range(warmup_requests):
        post_prediction(api_url, payload, timeout)

    started = time.perf_counter()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(post_prediction, api_url, payload, timeout)
            for _ in range(total_requests)
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if index % max(1, total_requests // 10) == 0:
                print(f"  completed {index}/{total_requests}")

    duration_seconds = time.perf_counter() - started
    successful = [item["latency_ms"] for item in results if item["ok"]]
    failed = len(results) - len(successful)

    avg_ms = sum(successful) / len(successful) if successful else 0.0
    throughput = len(successful) / duration_seconds if duration_seconds > 0 else 0.0

    return {
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "failed_requests": failed,
        "duration_seconds": round(duration_seconds, 4),
        "throughput_rps": round(throughput, 4),
        "avg_latency_ms": round(avg_ms, 4),
        "p50_latency_ms": round(percentile(successful, 50), 4),
        "p95_latency_ms": round(percentile(successful, 95), 4),
        "p99_latency_ms": round(percentile(successful, 99), 4),
        "min_latency_ms": round(min(successful), 4) if successful else 0.0,
        "max_latency_ms": round(max(successful), 4) if successful else 0.0,
    }


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| Configuration | preferred_batch_size | max_queue_delay_microseconds | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) | Success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        settings = CONFIGS[item["mode"]]
        preferred = settings["preferred_batch_size"]
        preferred_text = "-" if preferred is None else str(preferred)
        delay = settings["max_queue_delay_microseconds"]
        delay_text = "-" if delay is None else str(delay)
        summary = item["summary"]
        success = f"{summary['successful_requests']}/{summary['total_requests']}"
        lines.append(
            "| {title} | {preferred} | {delay} | {avg:.2f} | {p95:.2f} | {p99:.2f} | {rps:.2f} | {success} |".format(
                title=settings["title"],
                preferred=preferred_text,
                delay=delay_text,
                avg=summary["avg_latency_ms"],
                p95=summary["p95_latency_ms"],
                p99=summary["p99_latency_ms"],
                rps=summary["throughput_rps"],
                success=success,
            )
        )
    return "\n".join(lines) + "\n"


def svg_bar_chart(results: list[dict], output_path: Path) -> None:
    width = 1100
    height = 560
    margin_left = 80
    margin_bottom = 110
    plot_width = width - margin_left - 50
    plot_height = 360

    labels = [CONFIGS[item["mode"]]["title"] for item in results]
    avg_values = [item["summary"]["avg_latency_ms"] for item in results]
    rps_values = [item["summary"]["throughput_rps"] for item in results]
    max_avg = max(avg_values) if avg_values else 1
    max_rps = max(rps_values) if rps_values else 1
    group_width = plot_width / max(1, len(results))
    bar_width = min(70, group_width * 0.25)

    def avg_y(value: float) -> float:
        return 60 + plot_height - (value / max_avg) * plot_height

    def rps_y(value: float) -> float:
        return 60 + plot_height - (value / max_rps) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="50" y="35" font-family="Arial" font-size="24" font-weight="700">Dynamic Batching Benchmark</text>',
        '<text x="50" y="535" font-family="Arial" font-size="14" fill="#555">Blue: avg latency (ms), Green: throughput (RPS)</text>',
        f'<line x1="{margin_left}" y1="60" x2="{margin_left}" y2="{60 + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{60 + plot_height}" x2="{width - 50}" y2="{60 + plot_height}" stroke="#333"/>',
    ]

    for index, label in enumerate(labels):
        x_center = margin_left + group_width * index + group_width / 2
        avg = avg_values[index]
        rps = rps_values[index]
        avg_height = 60 + plot_height - avg_y(avg)
        rps_height = 60 + plot_height - rps_y(rps)
        avg_x = x_center - bar_width - 6
        rps_x = x_center + 6

        parts.extend(
            [
                f'<rect x="{avg_x:.1f}" y="{avg_y(avg):.1f}" width="{bar_width:.1f}" height="{avg_height:.1f}" fill="#4e79a7"/>',
                f'<rect x="{rps_x:.1f}" y="{rps_y(rps):.1f}" width="{bar_width:.1f}" height="{rps_height:.1f}" fill="#59a14f"/>',
                f'<text x="{avg_x + bar_width / 2:.1f}" y="{avg_y(avg) - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{avg:.1f}</text>',
                f'<text x="{rps_x + bar_width / 2:.1f}" y="{rps_y(rps) - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{rps:.1f}</text>',
                f'<text x="{x_center:.1f}" y="{60 + plot_height + 35}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{margin_left - 50}" y="70" font-family="Arial" font-size="12">{max_avg:.0f} ms</text>',
            f'<text x="{width - 130}" y="70" font-family="Arial" font-size="12">{max_rps:.0f} RPS</text>',
            '<rect x="780" y="18" width="16" height="16" fill="#4e79a7"/>',
            '<text x="802" y="31" font-family="Arial" font-size="13">Avg latency</text>',
            '<rect x="900" y="18" width="16" height="16" fill="#59a14f"/>',
            '<text x="922" y="31" font-family="Arial" font-size="13">Throughput</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8080/predict")
    parser.add_argument("--image", required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--mode", choices=["all", *CONFIGS.keys()], default="all")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--restore", choices=["keep", *CONFIGS.keys()], default="medium")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (project_dir / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    modes = list(CONFIGS.keys()) if args.mode == "all" else [args.mode]
    results_dir = (project_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for mode in modes:
        if not args.skip_docker:
            restart_stack_for_config(project_dir, mode)

        print(f"Running load test for {mode}: {args.requests} requests, concurrency={args.concurrency}")
        summary = run_load_test(
            api_url=args.api,
            image_path=image_path,
            total_requests=args.requests,
            concurrency=args.concurrency,
            warmup_requests=args.warmup,
            timeout=args.timeout,
        )
        result = {
            "mode": mode,
            "title": CONFIGS[mode]["title"],
            "settings": CONFIGS[mode],
            "summary": summary,
        }
        all_results.append(result)
        (results_dir / f"{mode}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))

    (results_dir / "dynamic_batching_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_dir / "dynamic_batching_results.md").write_text(
        markdown_table(all_results),
        encoding="utf-8",
    )
    svg_bar_chart(all_results, results_dir / "dynamic_batching_comparison.svg")

    if args.restore != "keep" and not args.skip_docker:
        restart_stack_for_config(project_dir, args.restore)

    print("\nSaved:")
    print(results_dir / "dynamic_batching_results.json")
    print(results_dir / "dynamic_batching_results.md")
    print(results_dir / "dynamic_batching_comparison.svg")
    print("\n" + markdown_table(all_results))


if __name__ == "__main__":
    main()
