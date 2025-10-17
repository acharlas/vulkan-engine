#!/usr/bin/env python3
"""
Simple harness to execute VulkanEngine benchmark runs with consistent settings.
"""

import argparse
import pathlib
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VulkanEngine in benchmark mode and collect CSV stats."
    )
    parser.add_argument(
        "--exe",
        type=pathlib.Path,
        default=pathlib.Path("../VulkanEngine"),
        help="Path to the VulkanEngine executable (default: ../VulkanEngine)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs to execute (default: 1)",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=30,
        help="Benchmark duration in seconds (measured time, default: 30)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Optional frame limit (0 disables, default: 0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Number of warmup frames to ignore (default: 60)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width for the benchmark run (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height for the benchmark run (default: 720)",
    )
    parser.add_argument(
        "--csv-dir",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks"),
        help="Directory where CSV logs will be written (default: ./benchmarks)",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.0,
        help="Optional fixed timestep in milliseconds (0 keeps real frame delta)",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run benchmark with the window hidden (default: enabled)",
    )
    parser.add_argument(
        "--show-window",
        dest="headless",
        action="store_false",
        help="Display the window during the benchmark run",
    )
    parser.add_argument(
        "--no-vsync",
        action="store_true",
        help="Disable v-sync during the benchmark run",
    )
    return parser.parse_args()


def run_single(
    executable: pathlib.Path,
    csv_path: pathlib.Path,
    seconds: int,
    frames: int,
    warmup: int,
    width: int,
    height: int,
    fixed_delta: float,
    headless: bool,
    no_vsync: bool,
) -> None:
    cmd = [
        str(executable),
        "--benchmark",
        f"--benchmark-csv={csv_path}",
        f"--benchmark-seconds={seconds}",
        f"--benchmark-warmup={warmup}",
        f"--benchmark-width={width}",
        f"--benchmark-height={height}",
    ]

    if frames > 0:
        cmd.append(f"--benchmark-frames={frames}")

    if headless:
        cmd.append("--benchmark-headless")

    if no_vsync:
        cmd.append("--benchmark-no-vsync")

    if fixed_delta > 0.0:
        cmd.append(f"--benchmark-fixed-delta={fixed_delta}")

    print(f"[Harness] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=executable.parent)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark run failed with exit code {result.returncode}. Command: {' '.join(cmd)}"
        )


def main() -> int:
    args = parse_args()
    exe_path = args.exe.resolve()
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}", file=sys.stderr)
        return 1

    csv_dir = args.csv_dir.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    for run_index in range(1, args.runs + 1):
        csv_path = csv_dir / f"benchmark_run_{run_index}.csv"
        run_single(
            exe_path,
            csv_path,
            seconds=args.seconds,
            frames=args.frames,
            warmup=args.warmup,
            width=args.width,
            height=args.height,
            fixed_delta=args.fixed_delta,
            headless=args.headless,
            no_vsync=args.no_vsync,
        )

    print("[Harness] Benchmark runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
