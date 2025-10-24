"""Metrics computation - ported from computeEnergySavingMetrics.m"""

from typing import List, Dict, Any
from .network import Cell, UE
from .scenario_loader import SimParams
import numpy as np


def compute_energy_saving_metrics(ues: List[UE], cells: List[Cell], sim_params: SimParams) -> Dict[str, Any]:
    """Compute comprehensive energy saving metrics"""
    
    total_energy = 0
    active_cells = 0
    total_drop_rate = 0
    total_latency = 0
    total_traffic = 0
    connected_ues = 0
    cpu_violations = 0
    prb_violations = 0
    max_cpu_usage = 0
    max_prb_usage = 0
    
    # Calculate cell-level metrics
    for cell in cells:
        if cell.cpu_usage > sim_params.cpu_threshold:
            cpu_violations += 1
        if cell.prb_usage > sim_params.prb_threshold:
            prb_violations += 1
        
        max_cpu_usage = max(max_cpu_usage, cell.cpu_usage)
        max_prb_usage = max(max_prb_usage, cell.prb_usage)
        
        total_energy += cell.energy_consumption
        active_cells += 1
        total_drop_rate += cell.drop_rate
        total_latency += cell.avg_latency
        total_traffic += cell.current_load
    
    # Calculate UE-level metrics
    for ue in ues:
        if ue.serving_cell is not None and ue.serving_cell > 0:
            connected_ues += 1
    
    # Compute averages
    metrics = {
        'total_energy': total_energy,
        'active_cells': active_cells,
        'avg_drop_rate': total_drop_rate / max(1, active_cells),
        'avg_latency': total_latency / max(1, active_cells),
        'total_traffic': total_traffic,
        'connected_ues': connected_ues,
        'connection_rate': (connected_ues / len(ues)) * 100 if len(ues) > 0 else 0,
        'cpu_violations': cpu_violations,
        'prb_violations': prb_violations,
        'max_cpu_usage': max_cpu_usage,
        'max_prb_usage': max_prb_usage,
    }
    
    # Check KPI violations
    kpi_violations = 0
    if metrics['avg_drop_rate'] > sim_params.drop_call_threshold:
        kpi_violations += 1
    if metrics['avg_latency'] > sim_params.latency_threshold:
        kpi_violations += 1
    
    metrics['kpi_violations'] = kpi_violations
    
    return metrics


def create_rl_state(cells: List[Cell], ues: List[UE], current_time: float, sim_params: SimParams) -> Dict[str, Any]:
    """Create RL state representation matching MATLAB ESInterface format"""
    
    # Compute network metrics
    energy_metrics = compute_energy_saving_metrics(ues, cells, sim_params)
    
    # Simulation features
    simulation_features = [
        len(cells),  # totalCells
        len(ues),  # totalUEs
        sim_params.sim_time,  # simTime
        sim_params.time_step,  # timeStep
        current_time / max(sim_params.sim_time, 1.0),  # timeProgress (avoid division by zero)
        sim_params.carrier_frequency,  # carrierFrequency
        sim_params.isd,  # isd
        sim_params.min_tx_power,  # minTxPower
        sim_params.max_tx_power,  # maxTxPower
        sim_params.base_power,  # basePower
        sim_params.idle_power,  # idlePower
        sim_params.drop_call_threshold,  # dropCallThreshold
        sim_params.latency_threshold,  # latencyThreshold
        sim_params.cpu_threshold,  # cpuThreshold
        sim_params.prb_threshold,  # prbThreshold
        sim_params.traffic_lambda,  # trafficLambda
        sim_params.peak_hour_multiplier  # peakHourMultiplier
    ]
    
    # Network features (scalar metrics from energy_metrics)
    # Calculate total TX power and avg power ratio
    total_tx_power = sum(cell.tx_power for cell in cells)
    avg_power_ratio = np.mean([cell.tx_power / cell.max_tx_power for cell in cells]) if cells else 0
    kpi_violations = energy_metrics['cpu_violations'] + energy_metrics['prb_violations']
    
    network_features = [
        energy_metrics['total_energy'],
        energy_metrics['active_cells'],
        energy_metrics['avg_drop_rate'],
        energy_metrics['avg_latency'],
        energy_metrics['total_traffic'],
        energy_metrics['connected_ues'],
        energy_metrics['connection_rate'],
        energy_metrics['cpu_violations'],
        energy_metrics['prb_violations'],
        energy_metrics['max_cpu_usage'],
        energy_metrics['max_prb_usage'],
        kpi_violations,
        total_tx_power,
        avg_power_ratio
    ]
    
    # Per-cell features (12 features per cell)
    n_cells = len(cells)
    cell_features = np.zeros((n_cells, 12))
    
    for i, cell in enumerate(cells):
        # Calculate averages from connected UEs
        connected_ue_objects = [ue for ue in ues if ue.serving_cell == cell.id]
        
        avg_rsrp = np.nanmean([ue.rsrp for ue in connected_ue_objects if ue.rsrp is not None]) if connected_ue_objects else 0
        avg_rsrq = np.nanmean([ue.rsrq for ue in connected_ue_objects if ue.rsrq is not None]) if connected_ue_objects else 0
        avg_sinr = np.nanmean([ue.sinr for ue in connected_ue_objects if ue.sinr is not None]) if connected_ue_objects else 0
        
        cell_features[i] = [
            cell.cpu_usage,
            cell.prb_usage,
            cell.current_load,
            cell.max_capacity,
            len(cell.connected_ues),
            cell.tx_power,
            cell.energy_consumption,
            avg_rsrp if not np.isnan(avg_rsrp) else 0,
            avg_rsrq if not np.isnan(avg_rsrq) else 0,
            avg_sinr if not np.isnan(avg_sinr) else 0,
            cell.current_load,  # totalTrafficDemand
            cell.current_load / max(1, cell.max_capacity)  # loadRatio
        ]
    
    # Handle NaN/Inf values
    simulation_features = [0 if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in simulation_features]
    network_features = [0 if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in network_features]
    cell_features[np.isnan(cell_features) | np.isinf(cell_features)] = 0
    
    # Combine all features into single vector (matching MATLAB format)
    # CRITICAL: Use 'F' order (Fortran/column-major) to match MATLAB's cellFeatures(:)
    # This gives: cell1_f1, cell2_f1, ..., cellN_f1, cell1_f2, cell2_f2, ..., cellN_f2, ...
    state_vector = np.concatenate([
        simulation_features,
        network_features,
        cell_features.flatten('F')  # ← Fortran order for MATLAB compatibility
    ])
    
    return state_vector

