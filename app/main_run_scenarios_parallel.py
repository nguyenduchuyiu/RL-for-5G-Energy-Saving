#!/usr/bin/env python3
"""
Run all scenarios with RL agent using parallel environments
Parallel version of main_run_scenarios_python.py
"""

import sys
from pathlib import Path
import numpy as np
import os
import glob

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_parallel_env import SimpleParallelEnv
from energy_agent import RLAgent


def load_scenarios_from_directory(scenarios_dir: str = 'scenarios', base_seed: int = 42):
    """
    Load all scenario files from directory
    
    Args:
        scenarios_dir: Directory containing scenario JSON files
        base_seed: Base seed for scenarios
    
    Returns:
        List of scenario configs with name and seed
    """
    # Get all JSON files in scenarios directory
    scenario_files = sorted(glob.glob(os.path.join(scenarios_dir, '*.json')))
    
    suite = []
    for scenario_file in scenario_files:
        # Extract scenario name from filename (remove .json extension)
        scenario_name = Path(scenario_file).stem
        suite.append({'name': scenario_name, 'seed': base_seed})
    
    return suite


def main(n_parallel_envs: int = 4, scenarios_dir: str = 'scenarios'):
    """Run all scenarios and generate energies.txt"""
    
    # Load scenarios from directory
    suite = load_scenarios_from_directory(scenarios_dir)
    
    if not suite:
        print(f'Error: No scenario files found in {scenarios_dir}/')
        return
    
    energies = []
    
    print(f'\n=== Running Benchmark Suite ({len(suite)} scenarios) ===')
    print(f'Scenarios directory: {scenarios_dir}/')
    print(f'Using {n_parallel_envs} parallel environments per scenario')
    print(f'Loaded scenarios: {[s["name"] for s in suite]}\n')
    
    for i, scenario_config in enumerate(suite, 1):
        name = scenario_config['name']
        seed = scenario_config['seed']
        
        print(f'\n--- Scenario {i}/{len(suite)}: {name} ---')
        
        try:
            results = run_scenario_with_rl_agent_parallel(
                scenario=name, 
                seed=seed,
                n_envs=n_parallel_envs,
                scenarios_dir=scenarios_dir
            )
            
            if results['violated']:
                print(f'⚠️  Scenario {name} has {results["kpi_violations"]} KPI violations.')
                energies.append(0.0)
            else:
                energies.append(results['final_energy_consumption'])
            
            print(f'\nResults for {name}:')
            print(f'  Energy: {results["final_energy_consumption"]:.6f} kWh')
            print(f'  Drop Rate: {results["final_drop_rate"]:.2f}%')
            print(f'  Latency: {results["final_latency"]:.1f} ms')
            print(f'  Handovers: {results["total_handovers"]} '
                  f'(Success: {results["handover_success_rate"]*100:.1f}%)')
            
        except Exception as e:
            print(f'Error in scenario {name}: {e}')
            import traceback
            traceback.print_exc()
            energies.append(0.0)
    
    # Write energies.txt
    write_energies_file(energies)
    
    print(f'\nenergies.txt generated with {len(energies)} values')
    print('\nEnergy values written to energies.txt:')
    for i, (scenario_config, energy) in enumerate(zip(suite, energies), 1):
        print(f'  Scenario {i} ({scenario_config["name"]}): {energy:.6f} kWh')


def run_scenario_with_rl_agent_parallel(scenario: str, seed: int, n_envs: int = 4, scenarios_dir: str = None):
    """
    Run scenario with RL agent using parallel environments
    
    Args:
        scenario: Scenario name
        seed: Base random seed
        n_envs: Number of parallel environments
        scenarios_dir: Custom scenarios directory path
    
    Returns:
        Results dictionary from the first environment (primary)
    """
    
    # Create parallel environments
    envs = SimpleParallelEnv(
        n_envs=n_envs,
        scenario=scenario,
        base_seed=seed,
        max_steps=None,  # Use scenario default
        scenarios_dir=scenarios_dir  # Pass custom scenarios directory
    )
    
    # Check if scenario has valid simTime
    # We'll use the first environment's params
    from simulation import FiveGEnvironment
    test_env = FiveGEnvironment(scenario=scenario, seed=seed, scenarios_dir=scenarios_dir)
    
    if test_env.sim_params.sim_time <= 0:
        print(f'Skipping scenario {scenario}: simTime={test_env.sim_params.sim_time}')
        envs.close()
        return {
            'final_energy_consumption': 0.0,
            'final_drop_rate': 0.0,
            'final_latency': 0.0,
            'total_handovers': 0,
            'handover_success_rate': 0.0,
            'kpi_violations': 0,
            'violated': False,
            'e_thisinh': 0.0,
            'metrics_history': []
        }
    
    max_steps = test_env.sim_params.total_steps
    
    # Create RL agent (single agent controlling all environments)
    agent = RLAgent(
        n_cells=test_env.n_cells,
        n_ues=test_env.sim_params.num_ues,
        max_time=max_steps
    )
    
    # Reset all environments
    obs = envs.reset()  # Shape: (n_envs, obs_dim)
    
    # Start scenario
    agent.start_scenario()
    
    print(f'Training with {n_envs} parallel environments...')
    print(f'Total steps: {max_steps}')
    
    # Track which environments are done
    dones = np.zeros(n_envs, dtype=bool)
    step_count = 0
    
    # Run simulation with RL agent
    while not dones.all():
        # Get actions from RL agent for all environments
        actions = []
        for i in range(n_envs):
            if not dones[i]:
                action = agent.get_action(obs[i])
                actions.append(action)
            else:
                # Environment is done, send dummy action
                actions.append(np.ones(envs.action_dim) * 0.7)
        
        actions = np.array(actions)
        
        # Step all environments in parallel
        next_obs, rewards, new_dones, infos = envs.step(actions)
        
        # Update agent with experiences from all environments
        for i in range(n_envs):
            if not dones[i]:
                agent.update(obs[i], actions[i], next_obs[i], new_dones[i])
        
        # Update done flags
        dones = np.logical_or(dones, new_dones)
        obs = next_obs
        step_count += 1
        
        # Progress update
        if step_count % 100 == 0:
            active_envs = n_envs - dones.sum()
            print(f'  Step {step_count}/{max_steps}: {active_envs}/{n_envs} envs active')
    
    print(f'Training completed after {step_count} steps')
    
    # End scenario
    agent.end_scenario()
    
    # Get results from all environments
    all_results = envs.get_results()
    
    # Close environments
    envs.close()
    
    # Return results from first environment as primary result
    # (Could also aggregate or pick best one)
    primary_result = all_results[0]
    
    # Log results from all environments for comparison
    print(f'\nResults from all {n_envs} parallel environments:')
    for i, result in enumerate(all_results):
        print(f'  Env {i}: Energy={result["final_energy_consumption"]:.6f} kWh, '
              f'Drop={result["final_drop_rate"]:.2f}%, '
              f'Violations={result["kpi_violations"]}')
    
    # Optionally: Use average or best result
    # Here we use the first environment's result
    return primary_result


def write_energies_file(energies):
    """Write energies to energies.txt file"""
    
    filename = 'energies.txt'
    
    try:
        with open(filename, 'w') as f:
            for energy in energies:
                f.write(f'{energy:.6f}\n')
        
        print(f'\nWritten {len(energies)} energy values to {filename}')
    
    except Exception as e:
        print(f'Error writing {filename}: {e}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run scenarios with parallel environments')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments per scenario (default: 4)')
    parser.add_argument('--scenarios-dir', type=str, default='scenarios',
                       help='Directory containing scenario files (default: scenarios)')
    
    args = parser.parse_args()
    
    main(n_parallel_envs=args.n_envs, scenarios_dir=args.scenarios_dir)

