#!/usr/bin/env python3
"""Benchmark CUDA pRRTC on MBM problems from pyronot resources."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyronot as pk
import yourdfpy
from tqdm import tqdm

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
except Exception:
    pass

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ModuleNotFoundError:
    HAS_TABULATE = False

    def _fmt_cell(v: Any) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if abs(fv) >= 1000.0:
                return f"{fv:.1f}"
            return f"{fv:.6f}".rstrip("0").rstrip(".")
        return str(v)

    def _github_table_from_df(df: pd.DataFrame, headers: list[str]) -> str:
        cols = list(df.columns)
        display_headers = [""] + list(headers or cols)
        lines = ["| " + " | ".join(display_headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(display_headers)) + " |")

        for idx, row in df.iterrows():
            values = [f"{idx}"] + [_fmt_cell(row[c]) for c in cols]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def tabulate(data, headers=(), tablefmt=None):
        if isinstance(data, pd.DataFrame) and tablefmt == "github":
            return _github_table_from_df(data, list(headers))
        if isinstance(data, pd.DataFrame):
            return data.to_string()
        return str(data)

from pyronot.collision._obstacles import create_collision_environment
from pyronot.collision._robot_collision import RobotCollisionSpherized


ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "pyronot" / "resources"
PRRTC_ROOT = ROOT / "cuda-rrtc" / "jax"
PRRTC_SCRIPTS = ROOT / "pRRTC" / "scripts"
TQDM_DISABLE = not sys.stdout.isatty()


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRRTC = _load_module("cuda_rrtc_prrtc", PRRTC_ROOT / "prrtc.py")
PRRTC_UTILS = _load_module("cuda_rrtc_utils", PRRTC_ROOT / "utils.py")


def load_robot_dataset(robot: str) -> dict[str, Any]:
    """Load MBM dataset from pyronot/resources, with robust fallbacks for baxter."""
    robot_dir = RESOURCES / robot
    pkl_path = robot_dir / "problems.pkl"
    json_path = robot_dir / "problems.json"
    tar_path = robot_dir / "problems.tar.bz2"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "problems" in data:
            return data

    if tar_path.exists():
        with tempfile.TemporaryDirectory(prefix=f"{robot}_mbm_") as tmpdir:
            with tarfile.open(tar_path, mode="r:bz2") as tf:
                tf.extractall(tmpdir)
            extracted = Path(tmpdir)
            pkl_candidates = list(extracted.rglob("problems.pkl"))
            json_candidates = list(extracted.rglob("problems.json"))

            if pkl_candidates:
                with open(pkl_candidates[0], "rb") as f:
                    return pickle.load(f)

            if json_candidates:
                with open(json_candidates[0], "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "problems" in data:
                    return data

    fallback = PRRTC_SCRIPTS / f"{robot}_problems.json"
    if fallback.exists():
        print(
            f"Warning: no parsed problems file found in {robot_dir}; "
            f"falling back to {fallback}"
        )
        with open(fallback, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError(
        f"Unable to load MBM dataset for robot '{robot}' from {robot_dir}"
    )


def load_robot_models(robot: str):
    urdf_path = RESOURCES / robot / f"{robot}_spherized.urdf"
    srdf_path = RESOURCES / robot / f"{robot}.srdf"

    if not urdf_path.exists():
        raise RuntimeError(f"URDF not found: {urdf_path}")
    if not srdf_path.exists():
        raise RuntimeError(f"SRDF not found: {srdf_path}")

    urdf = yourdfpy.URDF.load(str(urdf_path))
    robot_model = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(srdf_path))

    lo = jnp.asarray(robot_model.joints.lower_limits, dtype=jnp.float32)
    hi = jnp.asarray(robot_model.joints.upper_limits, dtype=jnp.float32)

    return robot_model, robot_coll, lo, hi


def evaluate_robot(
    robot: str,
    *,
    max_iterations: int,
    step_size: float,
    num_new_samples: int,
    granularity: int,
    max_nodes: int,
    balance_mode: int,
    tree_ratio: float,
    dynamic_domain: bool,
    dd_alpha: float,
    dd_radius: float,
    dd_min_radius: float,
    warmup: bool,
    timing_source: str,
    max_problems_per_set: int,
    print_failures: bool,
):
    data = load_robot_dataset(robot)
    robot_model, robot_coll, lo, hi = load_robot_models(robot)

    problems = data.get("problems", {})
    if not isinstance(problems, dict):
        raise RuntimeError(f"Unexpected problems structure for robot '{robot}'")

    total_problems = 0
    valid_problems = 0
    failed_problems = 0
    did_warmup = not warmup

    results: list[dict[str, Any]] = []

    for problem_name, pset in problems.items():
        if not isinstance(pset, list):
            continue

        failures: list[int] = []
        invalids: list[int] = []

        print(f"Evaluating {robot} on {problem_name}:")
        iterator = pset if max_problems_per_set <= 0 else pset[:max_problems_per_set]

        for i, problem_data in tqdm(list(enumerate(iterator)), disable=TQDM_DISABLE):
            total_problems += 1

            if not problem_data.get("valid", False):
                invalids.append(i)
                continue

            valid_problems += 1

            start = jnp.asarray(problem_data["start"], dtype=jnp.float32)
            goals = jnp.asarray(problem_data["goals"], dtype=jnp.float32)
            obstacles = create_collision_environment(problem_data)
            collision_context = PRRTC_UTILS.build_prrtc_collision_context(
                robot_model,
                robot_coll,
                obstacles,
            )

            plan_kwargs = dict(
                start_config=start,
                goal_configs=goals,
                max_iterations=max_iterations,
                step_size=step_size,
                num_new_samples=num_new_samples,
                granularity=granularity,
                max_nodes=max_nodes,
                balance_mode=balance_mode,
                tree_ratio=tree_ratio,
                dynamic_domain=dynamic_domain,
                dd_alpha=dd_alpha,
                dd_radius=dd_radius,
                dd_min_radius=dd_min_radius,
                min_vals=lo,
                max_vals=hi,
                collision_context=collision_context,
            )

            if not did_warmup:
                _ = PRRTC.prrtc_plan(**plan_kwargs)
                did_warmup = True

            t0 = time.perf_counter_ns()
            result = PRRTC.prrtc_plan(**plan_kwargs)
            PRRTC_UTILS.synchronize_prrtc_result(result)
            host_planning_ns = time.perf_counter_ns() - t0
            kernel_planning_ns = None
            if getattr(result, "kernel_time_ms", None) is not None:
                kernel_planning_ns = int(float(result.kernel_time_ms) * 1e6)

            if timing_source == "kernel" and kernel_planning_ns is not None:
                planning_ns = kernel_planning_ns
            else:
                planning_ns = host_planning_ns

            if not result.solved:
                failures.append(i)
                continue

            planning_td = pd.Timedelta(nanoseconds=int(planning_ns))
            simplification_td = pd.Timedelta(0)
            total_td = planning_td + simplification_td

            results.append(
                {
                    "robot": robot,
                    "problem": problem_name,
                    "planning_time": planning_td,
                    "simplification_time": simplification_td,
                    "total_time": total_td,
                    "planning_iterations": int(result.iterations),
                    "initial_path_cost": float(result.cost),
                    "simplified_path_cost": float(result.cost),
                }
            )

        failed_problems += len(failures)

        if print_failures:
            if invalids:
                print(f"  Invalid problems: {invalids}")
            if failures:
                print(f"  Failed on {failures}")

    return results, total_problems, valid_problems, failed_problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pRRTC on MBM datasets")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["panda", "fetch", "baxter"],
        choices=["panda", "fetch", "baxter"],
        help="Robots to evaluate",
    )
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--num-new-samples", type=int, default=64)
    parser.add_argument("--granularity", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--balance-mode", type=int, default=2)
    parser.add_argument("--tree-ratio", type=float, default=1.0)
    parser.add_argument("--dynamic-domain", action="store_true", default=True)
    parser.add_argument("--no-dynamic-domain", action="store_false", dest="dynamic_domain")
    parser.add_argument("--dd-alpha", type=float, default=1e-4)
    parser.add_argument("--dd-radius", type=float, default=4.0)
    parser.add_argument("--dd-min-radius", type=float, default=1.0)
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_false", dest="warmup")
    parser.add_argument(
        "--timing-source",
        choices=["host", "kernel"],
        default="host",
        help="Use host wall time (default) or kernel-reported time for planning_time.",
    )
    parser.add_argument("--max-problems-per-set", type=int, default=0)
    parser.add_argument("--print-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tick = time.perf_counter()

    all_results: list[dict[str, Any]] = []
    total = 0
    valid = 0
    failed = 0

    for robot in args.robots:
        (
            robot_results,
            robot_total,
            robot_valid,
            robot_failed,
        ) = evaluate_robot(
            robot,
            max_iterations=args.max_iterations,
            step_size=args.step_size,
            num_new_samples=args.num_new_samples,
            granularity=args.granularity,
            max_nodes=args.max_nodes,
            balance_mode=args.balance_mode,
            tree_ratio=args.tree_ratio,
            dynamic_domain=args.dynamic_domain,
            dd_alpha=args.dd_alpha,
            dd_radius=args.dd_radius,
            dd_min_radius=args.dd_min_radius,
            warmup=args.warmup,
            timing_source=args.timing_source,
            max_problems_per_set=args.max_problems_per_set,
            print_failures=args.print_failures,
        )
        all_results.extend(robot_results)
        total += robot_total
        valid += robot_valid
        failed += robot_failed

    if not all_results:
        raise RuntimeError("No solved plans were collected; cannot summarize results")

    df = pd.DataFrame.from_dict(all_results)

    # Match vamp_evaluate_mbm.py output math and units.
    df["planning_time"] = df["planning_time"].dt.microseconds
    df["simplification_time"] = df["simplification_time"].dt.microseconds
    df["avg_time_per_iteration"] = df["planning_iterations"] / df["planning_time"]
    df["total_time"] = df["total_time"].dt.microseconds

    time_stats = df[
        [
            "planning_time",
            "simplification_time",
            "total_time",
            "planning_iterations",
            "avg_time_per_iteration",
        ]
    ].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
    time_stats.drop(index=["count"], inplace=True)

    cost_stats = df[["initial_path_cost", "simplified_path_cost"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95]
    )
    cost_stats.drop(index=["count"], inplace=True)

    print()
    print(
        tabulate(
            time_stats,
            headers=[
                "Planning Time (us)",
                "Simplification Time (us)",
                "Total Time (us)",
                "Planning Iters.",
                "Time per Iter. (us)",
            ],
            tablefmt="github",
            **({"floatfmt": ".6f"} if HAS_TABULATE else {}),
        )
    )

    print(
        tabulate(
            cost_stats,
            headers=[
                " Initial Cost (L2)",
                "    Simplified Cost (L2)",
            ],
            tablefmt="github",
            **({"floatfmt": ".6f"} if HAS_TABULATE else {}),
        )
    )

    tock = time.perf_counter()

    print(f"Timing source for planning_time: {args.timing_source}")
    print(f"Solved / Valid / Total # Problems: {valid - failed} / {valid} / {total}")
    print(f"Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    print(f"Total time including Python overhead: {(tock - tick) * 1000:.3f} milliseconds")


if __name__ == "__main__":
    main()
