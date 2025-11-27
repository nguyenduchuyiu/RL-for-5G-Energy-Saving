#!/usr/bin/env python3
"""
Benchmark different control strategies across all scenarios.

Policies compared:
1. Random actions (serves as an untrained baseline).
2. Final trained RL model (loaded from a checkpoint).
3. Max-power heuristic (always run every cell at maximum power).

The script reports per-scenario metrics and relative energy savings of
the final model versus the baselines.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulation import FiveGEnvironment
from energy_agent import RLAgent
from energy_agent import rl_agent as rl_module


def load_scenarios_from_directory(scenarios_dir: str, base_seed: int = 42) -> List[Dict[str, object]]:
    """Collect every scenario JSON file inside scenarios_dir."""
    scenarios_path = Path(scenarios_dir)
    scenario_files = sorted(scenarios_path.glob("*.json"))

    suite = []
    for scenario_file in scenario_files:
        suite.append({"name": scenario_file.stem, "seed": base_seed})
    return suite


def default_empty_result() -> Dict[str, float]:
    """Return a zeroed-out metrics dict matching FiveGEnvironment.get_results()."""
    return {
        "final_energy_consumption": 0.0,
        "final_drop_rate": 0.0,
        "final_latency": 0.0,
        "total_handovers": 0,
        "successful_handovers": 0,
        "handover_success_rate": 0.0,
        "kpi_violations": 0,
        "violated": False,
        "e_thisinh": 0.0,
        "metrics_history": [],
    }


def rollout_environment(
    env: FiveGEnvironment,
    seed: int,
    action_fn,
) -> Dict[str, float]:
    """Run a full episode inside env, delegating action selection to action_fn."""
    state, _ = env.reset(seed=seed)
    done = False

    while not done:
        action = action_fn(state, env)
        next_state, _, done, _, _ = env.step(action)
        state = next_state

    return env.get_results()


def evaluate_random_policy(
    scenario: str,
    seed: int,
    scenarios_dir: Optional[str],
    rng_seed: Optional[int],
) -> Dict[str, float]:
    """Randomly sample power ratios for every cell at each step."""
    env = FiveGEnvironment(scenario=scenario, seed=seed, scenarios_dir=scenarios_dir)

    if env.sim_params.sim_time <= 0:
        print(f"Skipping scenario {scenario} (simTime={env.sim_params.sim_time})")
        return default_empty_result()

    rng = np.random.default_rng(rng_seed if rng_seed is not None else seed)

    def random_action_fn(_: np.ndarray, local_env: FiveGEnvironment) -> np.ndarray:
        return rng.uniform(low=0.0, high=1.0, size=local_env.n_cells).astype(np.float32)

    return rollout_environment(env, seed, random_action_fn)


def evaluate_max_power_policy(
    scenario: str,
    seed: int,
    scenarios_dir: Optional[str],
) -> Dict[str, float]:
    """Always command every cell to operate at maximum transmit power."""
    env = FiveGEnvironment(scenario=scenario, seed=seed, scenarios_dir=scenarios_dir)

    if env.sim_params.sim_time <= 0:
        print(f"Skipping scenario {scenario} (simTime={env.sim_params.sim_time})")
        return default_empty_result()

    def max_power_fn(_: np.ndarray, local_env: FiveGEnvironment) -> np.ndarray:
        return np.ones(local_env.n_cells, dtype=np.float32)

    return rollout_environment(env, seed, max_power_fn)


def evaluate_trained_policy(
    scenario: str,
    seed: int,
    scenarios_dir: Optional[str],
    checkpoint_path: Path,
) -> Dict[str, float]:
    """Run RLAgent with weights loaded from checkpoint_path."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")

    env = FiveGEnvironment(scenario=scenario, seed=seed, scenarios_dir=scenarios_dir)

    if env.sim_params.sim_time <= 0:
        print(f"Skipping scenario {scenario} (simTime={env.sim_params.sim_time})")
        return default_empty_result()

    # Temporarily override RLAgent global config so that we load the desired checkpoint.
    original_load_path = rl_module.config.get("checkpoint_load_path")
    original_training_mode = rl_module.config.get("training_mode", False)

    try:
        rl_module.config["checkpoint_load_path"] = str(checkpoint_path)
        rl_module.config["training_mode"] = False

        agent = RLAgent(
            n_cells=env.n_cells,
            n_ues=env.sim_params.num_ues,
            max_time=env.sim_params.total_steps,
        )
        agent.set_training_mode(False)

        def agent_action_fn(state: np.ndarray, _: FiveGEnvironment) -> np.ndarray:
            return agent.get_action(state)

        results = rollout_environment(env, seed, agent_action_fn)
        return results
    finally:
        rl_module.config["checkpoint_load_path"] = original_load_path
        rl_module.config["training_mode"] = original_training_mode


def format_energy(value: float) -> str:
    return f"{value:.6f}"


def print_scenario_summary(scenario_name: str, scenario_results: Dict[str, Dict[str, float]]):
    """Nicely format per-scenario metrics and energy gains."""
    header = f"=== {scenario_name} ==="
    print(header)
    print("-" * len(header))
    print(f"{'Policy':<12}{'Energy (kWh)':>15}{'Drop %':>12}{'Latency ms':>15}{'KPI Violations':>16}")

    for policy_name, metrics in scenario_results.items():
        print(
            f"{policy_name:<12}"
            f"{format_energy(metrics['final_energy_consumption']):>15}"
            f"{metrics['final_drop_rate']:>12.2f}"
            f"{metrics['final_latency']:>15.1f}"
            f"{metrics['kpi_violations']:>16}"
        )

    final_metrics = scenario_results.get("final")
    if final_metrics:
        base_energy = final_metrics["final_energy_consumption"]
        for baseline in ("random", "max_power"):
            if baseline in scenario_results and scenario_results[baseline]["final_energy_consumption"] > 0:
                baseline_energy = scenario_results[baseline]["final_energy_consumption"]
                absolute_gain = baseline_energy - base_energy
                percent_gain = (absolute_gain / baseline_energy) * 100.0
                print(
                    f"  -> Energy gain vs {baseline}: {absolute_gain:.6f} kWh "
                    f"({percent_gain:.2f}%)"
                )

    print()


def write_csv_report(csv_path: Path, rows: List[Dict[str, object]]):
    """Persist benchmark metrics to a CSV file."""
    if not rows:
        print("No rows to write, skipping CSV export.")
        return

    fieldnames = [
        "scenario",
        "policy",
        "final_energy_consumption",
        "final_drop_rate",
        "final_latency",
        "total_handovers",
        "handover_success_rate",
        "kpi_violations",
        "violated",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV report saved to {csv_path}")


def benchmark_policies(
    scenarios_dir: str,
    base_seed: int,
    final_checkpoint: Path,
    csv_output: Optional[Path],
    rng_seed: Optional[int],
):
    suite = load_scenarios_from_directory(scenarios_dir, base_seed)
    if not suite:
        raise RuntimeError(f"No scenarios found in '{scenarios_dir}'.")

    aggregate_rows: List[Dict[str, object]] = []

    for scenario_config in suite:
        scenario_name = scenario_config["name"]
        seed = int(scenario_config["seed"])
        scenario_results: Dict[str, Dict[str, float]] = {}

        print(f"\nRunning scenario '{scenario_name}' (seed={seed})")

        scenario_results["random"] = evaluate_random_policy(
            scenario=scenario_name,
            seed=seed,
            scenarios_dir=scenarios_dir,
            rng_seed=rng_seed,
        )

        scenario_results["final"] = evaluate_trained_policy(
            scenario=scenario_name,
            seed=seed,
            scenarios_dir=scenarios_dir,
            checkpoint_path=final_checkpoint,
        )

        scenario_results["max_power"] = evaluate_max_power_policy(
            scenario=scenario_name,
            seed=seed,
            scenarios_dir=scenarios_dir,
        )

        print_scenario_summary(scenario_name, scenario_results)

        for policy_name, metrics in scenario_results.items():
            aggregate_rows.append(
                {
                    "scenario": scenario_name,
                    "policy": policy_name,
                    "final_energy_consumption": metrics["final_energy_consumption"],
                    "final_drop_rate": metrics["final_drop_rate"],
                    "final_latency": metrics["final_latency"],
                    "total_handovers": metrics["total_handovers"],
                    "handover_success_rate": metrics["handover_success_rate"],
                    "kpi_violations": metrics["kpi_violations"],
                    "violated": metrics["violated"],
                }
            )

    if csv_output:
        write_csv_report(csv_output, aggregate_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple control strategies.")
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default="app/scenarios",
        help="Directory containing scenario JSON files (default: app/scenarios).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed reused for every scenario (default: 42).",
    )
    parser.add_argument(
        "--final-checkpoint",
        type=str,
        required=True,
        help="Path to the trained RL model checkpoint.",
    )
    parser.add_argument(
        "--csv-report",
        type=str,
        default=None,
        help="Optional path to write a CSV summary.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional RNG seed for the random baseline (defaults to base seed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    final_checkpoint = Path(args.final_checkpoint).expanduser().resolve()
    csv_output = Path(args.csv_report).expanduser().resolve() if args.csv_report else None

    benchmark_policies(
        scenarios_dir=args.scenarios_dir,
        base_seed=args.base_seed,
        final_checkpoint=final_checkpoint,
        csv_output=csv_output,
        rng_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

