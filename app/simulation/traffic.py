"""Traffic generation and cell resource usage - ported from updateTrafficGeneration.m and updateCellResourceUsage.m"""

import numpy as np
from typing import List, Optional
from .network import Cell, UE
from .scenario_loader import SimParams


def update_traffic_generation(ues: List[UE], cells: List[Cell], current_time: float, sim_params: SimParams, seed: Optional[int] = None):
    """Generate traffic for each UE and update cell loads"""
    
    peak_hour_factor = sim_params.peak_hour_multiplier
    lambda_val = sim_params.traffic_lambda * peak_hour_factor
    
    # Reset cell loads
    for cell in cells:
        cell.current_load = 0
        cell.connected_ues = []
    
    # Prepare deterministic RNG per-step (mimic MATLAB's global stream) when seed provided
    step_rng = None
    if seed is not None:
        step_seed = seed + 7000 + int(current_time * 100)
        step_rng = np.random.RandomState(step_seed)

    # Generate traffic for each UE
    for ue in ues:
        if ue.serving_cell is not None and ue.serving_cell > 0:
            # Generate Poisson traffic demand
            if step_rng is not None:
                traffic_demand = step_rng.poisson(lambda_val / max(1, len(ues)))
            else:
                traffic_demand = np.random.poisson(lambda_val / max(1, len(ues)))
            ue.traffic_demand = float(traffic_demand)
            ue.session_active = traffic_demand > 0
            
            # Add to serving cell load
            cell_idx = next((i for i, c in enumerate(cells) if c.id == ue.serving_cell), None)
            if cell_idx is not None:
                cells[cell_idx].current_load += ue.traffic_demand
                cells[cell_idx].connected_ues.append(ue.id)
        else:
            ue.traffic_demand = 0
            ue.session_active = False


def update_cell_resource_usage(cells: List[Cell], ues: List[UE]):
    """Update cell resource usage based on connected UEs"""
    
    for cell in cells:
        connected_ues = []
        total_traffic_demand = 0
        total_sinr = 0
        valid_sinr_count = 0
        
        for ue in ues:
            if ue.serving_cell == cell.id:
                connected_ues.append(ue.id)
                total_traffic_demand += ue.traffic_demand
                
                if ue.sinr is not None and not np.isnan(ue.sinr):
                    total_sinr += ue.sinr
                    valid_sinr_count += 1
        
        cell.connected_ues = connected_ues
        cell.current_load = total_traffic_demand
        
        load_ratio = min(1.0, total_traffic_demand / cell.max_capacity)
        num_ues = len(connected_ues)
        
        # Power-dependent calculations
        power_ratio = (cell.tx_power - cell.min_tx_power) / (cell.max_tx_power - cell.min_tx_power)
        
        # CPU usage: base processing + per UE overhead + load processing + power scaling
        base_cpu = 10 + power_ratio * 5
        per_ue_cpu = num_ues * 2.5
        load_cpu = load_ratio * 50
        cell.cpu_usage = min(95, base_cpu + per_ue_cpu + load_cpu)
        
        # PRB usage: directly related to traffic demand and number of UEs
        base_prb = num_ues * 3
        load_prb = load_ratio * 60
        cell.prb_usage = min(95, base_prb + load_prb)
        
        # Energy consumption: base + transmit power + per UE + traffic load
        base_energy = cell.base_energy_consumption
        tx_power_consumption = 10**((cell.tx_power - 30)/10)
        per_ue_energy = num_ues * 15
        load_energy = load_ratio * 200
        cell.energy_consumption = base_energy + tx_power_consumption + per_ue_energy + load_energy
        
        # Average SINR
        if num_ues > 0 and valid_sinr_count > 0:
            avg_sinr = total_sinr / valid_sinr_count
            cell.avg_sinr = avg_sinr
        else:
            cell.avg_sinr = 0
        
        # Calculate drop rate
        base_drop_rate = 0.1  # Base 0.1% drop rate
        
        # Power factor - Penalty for low power
        if cell.tx_power <= cell.min_tx_power + 1:
            power_drop_penalty = (cell.min_tx_power + 1 - cell.tx_power + 1) * 4.0
        elif cell.tx_power <= cell.min_tx_power + 3:
            power_drop_penalty = (cell.min_tx_power + 3 - cell.tx_power) * 1.5
        else:
            power_drop_penalty = 0
        
        # Congestion factors
        congestion_factor = 0
        if cell.cpu_usage > 90:
            congestion_factor += (cell.cpu_usage - 90) * 0.4
        elif cell.cpu_usage > 85:
            congestion_factor += (cell.cpu_usage - 85) * 0.2
        
        if cell.prb_usage > 90:
            congestion_factor += (cell.prb_usage - 90) * 0.3
        elif cell.prb_usage > 85:
            congestion_factor += (cell.prb_usage - 85) * 0.15
        
        # Signal quality factor
        signal_factor = 0
        if valid_sinr_count > 0:
            if avg_sinr < 0:
                signal_factor += abs(avg_sinr) * 0.2
            elif avg_sinr < 5:
                signal_factor += (5 - avg_sinr) * 0.1
        
        cell.drop_rate = min(25.0, base_drop_rate + power_drop_penalty + congestion_factor + signal_factor)
        
        # Latency calculation - power-sensitive
        base_latency = 10
        load_latency = load_ratio * 25
        ue_latency = min(15, num_ues * 0.8)
        
        # Power latency penalty
        if cell.tx_power <= cell.min_tx_power + 1:
            power_latency_penalty = (cell.min_tx_power + 1 - cell.tx_power + 1) * 15
        elif cell.tx_power <= cell.min_tx_power + 3:
            power_latency_penalty = (cell.min_tx_power + 3 - cell.tx_power) * 8
        else:
            power_latency_penalty = (1 - power_ratio) * 3
        
        cell.avg_latency = base_latency + load_latency + ue_latency + power_latency_penalty
        
        # Update additional fields for RL state
        cell.load_ratio = load_ratio
        cell.total_traffic_demand = total_traffic_demand

