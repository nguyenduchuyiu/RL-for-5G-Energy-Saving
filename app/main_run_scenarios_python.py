#!/usr/bin/env python3
"""
Run all scenarios with RL agent and output energies.txt
Python port of main_run_scenarios.m
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simulation import FiveGEnvironment
from energy_agent import RLAgent


def main():
    """Run all scenarios and generate energies.txt"""
    
    # Define test suite (matching opts.txt format)
    suite = [
        {'name': 'indoor_hotspot', 'seed': 42},
        {'name': 'dense_urban', 'seed': 42},
        {'name': 'rural', 'seed': 42},
        {'name': 'urban_macro', 'seed': 42},
    ]
    
    energies = []
    
    print(f'\n=== Running Benchmark Suite ({len(suite)} scenarios) ===\n')
    
    for i, scenario_config in enumerate(suite, 1):
        name = scenario_config['name']
        seed = scenario_config['seed']
        
        print(f'\n--- Scenario {i}/{len(suite)}: {name} ---')
        
        try:
            results = run_scenario_with_rl_agent(scenario=name, seed=seed)
            
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


def run_scenario_with_rl_agent(scenario: str, seed: int):
    """Run scenario with RL agent"""
    
    # Create environment
    env = FiveGEnvironment(scenario=scenario, seed=seed)
    
    # Skip scenario if simTime is 0 or negative
    if env.sim_params.sim_time <= 0:
        print(f'Skipping scenario {scenario}: simTime={env.sim_params.sim_time}')
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
    
    # Create RL agent
    agent = RLAgent(
        n_cells=env.n_cells,
        n_ues=env.sim_params.num_ues,
        max_time=env.sim_params.total_steps
    )
    
    # Reset environment
    state, _ = env.reset(seed=seed)
    
    # Start scenario
    agent.start_scenario()
    
    # Run simulation with RL agent
    done = False
    while not done:
        # Get action from RL agent
        action = agent.get_action(state)
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Update agent
        agent.update(state, action, next_state, done)
        
        state = next_state
    
    # End scenario
    agent.end_scenario()
    
    # Get results
    return env.get_results()


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
    main()

