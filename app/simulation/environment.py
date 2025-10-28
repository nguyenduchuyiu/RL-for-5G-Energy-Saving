"""5G Network Environment - Gym interface"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .network import Site, Cell, UE
from .scenario_loader import SimParams, load_scenario_config
from .network_init import create_layout, configure_cells, initialize_ues
from .mobility import update_ue_mobility
from .traffic import update_traffic_generation, update_cell_resource_usage
from .signal_measurement import update_signal_measurements
from .handover import (check_handover_events, handle_disconnected_ues, 
                      update_ue_drop_events)
from .metrics import compute_energy_saving_metrics, create_rl_state


class FiveGEnvironment:
    """5G Network simulation environment (compatible with RL agents)"""
    
    def __init__(self, scenario: str = 'indoor_hotspot', seed: int = 42, scenarios_dir: str = None):
        self.scenario_name = scenario
        self.seed_value = seed
        
        # Load scenario configuration
        self.sim_params = load_scenario_config(scenario, scenarios_dir=scenarios_dir)
        
        # Initialize network components
        self.sites: List[Site] = []
        self.cells: List[Cell] = []
        self.ues: List[UE] = []
        
        # Simulation state
        self.current_step = 0
        # Align first decision timeProgress to MATLAB (step 1 / simTime)
        self.current_time = self.sim_params.time_step
        self.cumulative_energy = 0.0
        
        # Results tracking
        self.handover_events = []
        self.energy_metrics_history = []
        self.total_handovers = 0
        self.successful_handovers = 0
        self.kpi_violations = 0
        
        # Define action and observation dimensions
        self.n_cells = self.sim_params.num_sites * self.sim_params.num_sectors
        self.action_dim = self.n_cells
        self.state_dim = 17 + 11 + 12 * self.n_cells
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        
        if seed is not None:
            self.seed_value = seed
        
        np.random.seed(self.seed_value)
        
        # Initialize network
        self.sites = create_layout(self.sim_params, self.seed_value)
        self.cells = configure_cells(self.sites, self.sim_params)
        self.ues = initialize_ues(self.sim_params, self.sites, self.seed_value)
        
        # Reset simulation state (align first state and step index with MATLAB)
        self.current_step = 1
        self.current_time = self.sim_params.time_step
        self.cumulative_energy = 0.0
        self.handover_events = []
        self.energy_metrics_history = []
        self.total_handovers = 0
        self.successful_handovers = 0
        self.kpi_violations = 0
        
        # Initial measurements
        update_signal_measurements(
            self.ues, self.cells, 
            self.sim_params.rsrp_measurement_threshold,
            self.current_time, self.seed_value
        )
        
        # Initial connection setup
        self.ues = handle_disconnected_ues(
            self.ues, self.cells, self.sim_params,
            self.sim_params.time_step, self.current_time
        )
        
        # Populate initial traffic and resource usage to align first RL decision with MATLAB
        update_traffic_generation(
            self.ues, self.cells, self.current_time, self.sim_params, self.seed_value
        )
        update_cell_resource_usage(self.cells, self.ues)
        
        # Get initial state
        state = create_rl_state(self.cells, self.ues, self.current_time, self.sim_params)
        
        info = {
            'step': self.current_step,
            'time': self.current_time,
            'cumulative_energy': self.cumulative_energy
        }
        
        return state.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one simulation step"""
        
        # Apply actions (power control)
        self._apply_action(action)
        
        # Advance time
        self.current_step += 1
        self.current_time = self.current_step * self.sim_params.time_step
        
        # Update UE mobility
        self.ues = update_ue_mobility(
            self.ues, self.sim_params.time_step, 
            self.current_time, self.seed_value, self.sim_params
        )
        
        # Update traffic generation
        update_traffic_generation(
            self.ues, self.cells, self.current_time, self.sim_params, self.seed_value
        )
        
        # Update signal measurements
        update_signal_measurements(
            self.ues, self.cells,
            self.sim_params.rsrp_measurement_threshold,
            self.current_time, self.seed_value
        )
        
        # Handle disconnected UEs
        self.ues = handle_disconnected_ues(
            self.ues, self.cells, self.sim_params,
            self.sim_params.time_step, self.current_time
        )
        
        # Update cell resource usage
        update_cell_resource_usage(self.cells, self.ues)
        
        # Update UE drop events (deterministic)
        update_ue_drop_events(self.ues, self.cells, self.current_time, self.seed_value)
        
        # Calculate energy consumption
        total_power = sum(cell.energy_consumption for cell in self.cells)
        time_hours = self.sim_params.time_step / 3600
        energy_this_step = (total_power / 1000) * time_hours  # kWh
        self.cumulative_energy += energy_this_step
        
        # Check handover events
        handover_events, self.ues = check_handover_events(
            self.ues, self.cells, self.current_time,
            self.sim_params, self.seed_value
        )
        
        for event in handover_events:
            self.total_handovers += 1
            if event.ho_success:
                self.successful_handovers += 1
            self.handover_events.append(event)

        
        # Compute metrics every 10 steps
        if self.current_step % 10 == 0:
            metrics = compute_energy_saving_metrics(
                self.ues, self.cells, self.sim_params
            )
            metrics['time'] = self.current_time
            metrics['total_energy'] = self.cumulative_energy
            metrics['instantaneous_power'] = total_power
            self.energy_metrics_history.append(metrics)
            
            # Log only every 50 steps to reduce I/O overhead
            if self.current_step % 50 == 0:
                print(f'Step {self.current_step}/{self.sim_params.total_steps}: '
                      f'Energy: {self.cumulative_energy:.3f} kWh, '
                      f'Power: {total_power/1000:.1f} kW, '
                      f'Drop Rate: {metrics["avg_drop_rate"]:.2f}%')
            
            # Check KPI violations (but don't print every time)
            if metrics['avg_drop_rate'] > self.sim_params.drop_call_threshold:
                if self.current_step % 50 == 0:
                    print(f'Drop rate violation: {metrics["avg_drop_rate"]:.2f}% > '
                          f'{self.sim_params.drop_call_threshold}%')
                self.kpi_violations += 1
            
            if metrics['avg_latency'] > self.sim_params.latency_threshold:
                if self.current_step % 50 == 0:
                    print(f'Latency violation: {metrics["avg_latency"]:.1f} ms > '
                          f'{self.sim_params.latency_threshold} ms')
                self.kpi_violations += 1
        
        # Get next state
        next_state = create_rl_state(
            self.cells, self.ues, self.current_time, self.sim_params
        )
        
        # Check if episode is done
        done = self.current_step >= self.sim_params.total_steps
        truncated = False
        
        # Calculate reward (negative energy consumption + penalties)
        reward = -energy_this_step
        
        # Add penalties for KPI violations
        final_metrics = compute_energy_saving_metrics(
            self.ues, self.cells, self.sim_params
        )
        if final_metrics['avg_drop_rate'] > self.sim_params.drop_call_threshold:
            reward -= 10.0
        if final_metrics['avg_latency'] > self.sim_params.latency_threshold:
            reward -= 5.0
        
        info = {
            'step': self.current_step,
            'time': self.current_time,
            'cumulative_energy': self.cumulative_energy,
            'instantaneous_power': total_power,
            'metrics': final_metrics,
            'handovers': self.total_handovers,
            'kpi_violations': self.kpi_violations
        }
        
        return next_state.astype(np.float32), float(reward), done, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to cells (power control)"""
        
        action = np.clip(action, 0.0, 1.0)
        
        # Log RL decision every step to mirror MATLAB verbosity
        log_decision = False
        
        for i, cell in enumerate(self.cells):
            if i < len(action):
                power_ratio = action[i]
                # Map ratio to actual power
                prev_power = cell.tx_power
                new_power = (cell.min_tx_power + 
                               power_ratio * (cell.max_tx_power - cell.min_tx_power))
                cell.tx_power = new_power
                if log_decision:
                    print(f'Step {self.current_step}/{self.sim_params.total_steps}: '
                          f'RL Agent: Adjusting cell {cell.id} power from {prev_power:.1f} to {new_power:.1f} dBm')
    
    def get_results(self) -> Dict[str, Any]:
        """Get final simulation results"""
        
        final_metrics = compute_energy_saving_metrics(
            self.ues, self.cells, self.sim_params
        )
        
        return {
            'final_energy_consumption': self.cumulative_energy,
            'final_drop_rate': final_metrics['avg_drop_rate'],
            'final_latency': final_metrics['avg_latency'],
            'total_handovers': self.total_handovers,
            'successful_handovers': self.successful_handovers,
            'handover_success_rate': (self.successful_handovers / max(1, self.total_handovers)),
            'kpi_violations': self.kpi_violations,
            'violated': self.kpi_violations > 0,
            'e_thisinh': self.cumulative_energy,
            'metrics_history': self.energy_metrics_history
        }


def run_simulation(scenario: str = 'indoor_hotspot', seed: int = 42) -> Dict[str, Any]:
    """Run a complete simulation without RL agent (for testing)"""
    
    env = FiveGEnvironment(scenario=scenario, seed=seed)
    obs, info = env.reset(seed=seed)
    
    done = False
    while not done:
        # Default action: maintain current power
        action = np.ones(env.action_dim)
        obs, reward, done, truncated, info = env.step(action)
    
    return env.get_results()

