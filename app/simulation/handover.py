"""Handover logic - ported from checkHandoverEvents.m, evaluateHandoverSuccess.m, etc."""

import numpy as np
from typing import List, Tuple, Optional
from .network import Cell, UE, Measurement, HandoverEvent
from .scenario_loader import SimParams


def check_handover_events(ues: List[UE], cells: List[Cell], current_time: float, 
                          sim_params: SimParams, seed: int) -> Tuple[List[HandoverEvent], List[UE]]:
    """Check for handover events"""
    
    handover_events = []
    
    for ue in ues:
        if ue.serving_cell is None:
            continue
        
        # Find serving cell
        serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
        if serving_cell is None:
            continue
        
        # Check if serving cell signal is too weak
        if ue.rsrp is None or np.isnan(ue.rsrp) or ue.rsrp < sim_params.rsrp_serving_threshold:
            best_cell = find_best_cell_for_connection(ue, sim_params.rsrp_target_threshold, cells)
            if best_cell is not None:
                ue.serving_cell = best_cell['cell_id']
                ue.rsrp = best_cell['rsrp']
                ue.rsrq = best_cell['rsrq']
                ue.sinr = best_cell['sinr']
            else:
                ue.serving_cell = None
            continue
        
        # Check A3 handover conditions
        for neighbor in ue.neighbor_measurements:
            if neighbor.cell_id == ue.serving_cell:
                continue
            
            if neighbor.rsrp < sim_params.rsrp_target_threshold:
                continue
            
            # Find target cell
            target_cell = next((c for c in cells if c.id == neighbor.cell_id), None)
            if target_cell is None:
                continue
            
            # A3 condition: neighbor RSRP > serving RSRP + offset
            a3_condition = (neighbor.rsrp > (ue.rsrp + serving_cell.a3_offset) and
                           neighbor.rsrp >= sim_params.rsrp_target_threshold)
            
            if a3_condition:
                if ue.ho_timer == 0:
                    ue.ho_timer = current_time
                
                # Check if TTT has elapsed
                if (current_time - ue.ho_timer) >= (serving_cell.ttt / 1000):
                    # Evaluate handover success
                    ho_success = evaluate_handover_success(ue, neighbor, serving_cell, target_cell, current_time, seed)
                    
                    # Create handover event
                    ho_event = create_handover_event(ue, neighbor, serving_cell, current_time, ho_success)
                    handover_events.append(ho_event)
                    
                    if ho_success:
                        ue.serving_cell = neighbor.cell_id
                        ue.rsrp = neighbor.rsrp
                        ue.rsrq = neighbor.rsrq
                        ue.sinr = neighbor.sinr
                    
                    ue.handover_history.append(ho_event)
                    ue.ho_timer = 0
                    break
            else:
                ue.ho_timer = 0
    
    return handover_events, ues


def evaluate_handover_success(ue: UE, neighbor: Measurement, serving_cell: Cell, 
                              target_cell: Cell, current_time: float, seed: int) -> bool:
    """Evaluate if handover succeeds"""
    
    # Create deterministic RNG
    ho_seed = seed + ue.id + neighbor.cell_id + int(current_time)
    rng = np.random.RandomState(ho_seed)
    
    # Base success probability
    base_success_prob = 0.98
    
    # Source cell power penalty
    if serving_cell.tx_power <= serving_cell.min_tx_power + 2:
        source_power_penalty = (serving_cell.min_tx_power + 2 - serving_cell.tx_power) * 0.15
        base_success_prob -= source_power_penalty
    
    # Target cell power penalty
    if target_cell.tx_power <= target_cell.min_tx_power + 3:
        target_power_penalty = (target_cell.min_tx_power + 3 - target_cell.tx_power) * 0.10
        base_success_prob -= target_power_penalty
    
    # Signal quality factors
    if neighbor.rsrp >= -75:
        signal_bonus = 0.02
    elif neighbor.rsrp >= -85:
        signal_bonus = 0.01
    elif neighbor.rsrp >= -95:
        signal_bonus = 0.0
    elif neighbor.rsrp >= -105:
        signal_bonus = -0.05
    else:
        signal_bonus = -0.15
    
    # SINR penalty
    if neighbor.sinr >= 15:
        sinr_bonus = 0.02
    elif neighbor.sinr >= 5:
        sinr_bonus = 0.01
    elif neighbor.sinr >= 0:
        sinr_bonus = 0.0
    elif neighbor.sinr >= -5:
        sinr_bonus = -0.03
    else:
        sinr_bonus = -0.10
    
    # Additional penalty if both cells are at minimum power
    if (serving_cell.tx_power <= serving_cell.min_tx_power + 1 and
        target_cell.tx_power <= target_cell.min_tx_power + 1):
        base_success_prob -= 0.20
    
    # Resource congestion
    if target_cell.cpu_usage > 85 or target_cell.prb_usage > 85:
        congestion_penalty = 0.05
        base_success_prob -= congestion_penalty
    
    final_success_prob = base_success_prob + signal_bonus + sinr_bonus
    final_success_prob = max(0.25, min(0.98, final_success_prob))
    
    return rng.rand() < final_success_prob


def create_handover_event(ue: UE, neighbor: Measurement, serving_cell: Cell, 
                         current_time: float, ho_success: bool) -> HandoverEvent:
    """Create handover event structure"""
    
    return HandoverEvent(
        ue_id=ue.id,
        cell_source=serving_cell.id,
        cell_target=neighbor.cell_id,
        rsrp_source=ue.rsrp if ue.rsrp is not None else 0,
        rsrp_target=neighbor.rsrp,
        rsrq_source=ue.rsrq if ue.rsrq is not None else 0,
        rsrq_target=neighbor.rsrq,
        sinr_source=ue.sinr if ue.sinr is not None else 0,
        sinr_target=neighbor.sinr,
        a3_offset=serving_cell.a3_offset,
        ttt=serving_cell.ttt,
        ho_success=ho_success,
        timestamp=current_time
    )


def find_best_cell_for_connection(ue: UE, rsrp_threshold: float, cells: List[Cell]) -> Optional[dict]:
    """Find best cell for connection based on RSRP"""
    
    best_cell = None
    best_rsrp = -np.inf
    
    for cell in cells:
        # Check if we have measurement for this cell
        cell_measurement = next((m for m in ue.neighbor_measurements if m.cell_id == cell.id), None)
        
        if cell_measurement is not None and cell_measurement.rsrp >= rsrp_threshold:
            if cell_measurement.rsrp > best_rsrp:
                best_rsrp = cell_measurement.rsrp
                best_cell = {
                    'cell_id': cell.id,
                    'rsrp': cell_measurement.rsrp,
                    'rsrq': cell_measurement.rsrq,
                    'sinr': cell_measurement.sinr
                }
    
    return best_cell


def handle_disconnected_ues(ues: List[UE], cells: List[Cell], sim_params: SimParams, 
                            time_step: float, current_time: float) -> List[UE]:
    """Handle disconnected UEs and connection attempts"""
    
    disconnection_timeout = 5.0
    connection_timeout = 2.0
    hysteresis_margin = 3.0
    
    for ue in ues:
        # Check for disconnection
        if ue.serving_cell is not None:
            if ue.rsrp is not None and not np.isnan(ue.rsrp):
                if ue.rsrp < (sim_params.rsrp_serving_threshold - hysteresis_margin):
                    if ue.disconnection_timer == 0:
                        ue.disconnection_timer = disconnection_timeout
                    else:
                        ue.disconnection_timer -= time_step
                        
                        if ue.disconnection_timer <= 0:
                            # Reduce logging to avoid I/O overhead
                            # print(f'UE {ue.id} disconnected from cell {ue.serving_cell} (RSRP: {ue.rsrp:.1f} dBm)')
                            ue.serving_cell = None
                            ue.rsrp = None
                            ue.rsrq = None
                            ue.sinr = None
                            ue.disconnection_timer = 0
                            ue.connection_timer = 0
                            ue.session_active = False
                            ue.traffic_demand = 0
                            ue.drop_count += 1
                else:
                    ue.disconnection_timer = 0
        
        # Try to connect if disconnected
        if ue.serving_cell is None:
            best_cell = find_best_cell_for_connection(
                ue, sim_params.rsrp_serving_threshold + hysteresis_margin, cells)
            
            if best_cell is not None:
                if ue.connection_timer == 0:
                    ue.connection_timer = connection_timeout
                else:
                    ue.connection_timer -= time_step
                    
                    if ue.connection_timer <= 0:
                        current_best_cell = find_best_cell_for_connection(
                            ue, sim_params.rsrp_serving_threshold + hysteresis_margin, cells)
                        if current_best_cell is not None and current_best_cell['cell_id'] == best_cell['cell_id']:
                            ue.serving_cell = best_cell['cell_id']
                            ue.rsrp = best_cell['rsrp']
                            ue.rsrq = best_cell['rsrq']
                            ue.sinr = best_cell['sinr']
                            ue.connection_timer = 0
                        else:
                            ue.connection_timer = 0
            else:
                ue.connection_timer = 0
    
    return ues


def update_ue_drop_events(ues: List[UE], cells: List[Cell], current_time: float):
    """Update UE drop events based on cell conditions"""
    
    for ue in ues:
        if ue.serving_cell is not None and ue.session_active:
            # Find serving cell
            serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
            
            if serving_cell is not None:
                drop_prob = 0.001  # Base 0.1%
                
                # Power penalty
                if serving_cell.tx_power <= serving_cell.min_tx_power + 1:
                    power_penalty = (serving_cell.min_tx_power + 1 - serving_cell.tx_power + 1) * 0.08
                    drop_prob += power_penalty
                elif serving_cell.tx_power <= serving_cell.min_tx_power + 3:
                    power_penalty = (serving_cell.min_tx_power + 3 - serving_cell.tx_power) * 0.03
                    drop_prob += power_penalty
                
                # Signal quality factor
                if ue.sinr is not None and not np.isnan(ue.sinr):
                    if ue.sinr < -10:
                        drop_prob += 0.15 + abs(ue.sinr + 10) * 0.02
                    elif ue.sinr < -5:
                        drop_prob += 0.08 + abs(ue.sinr + 5) * 0.015
                    elif ue.sinr < 0:
                        drop_prob += 0.04
                    elif ue.sinr < 5:
                        drop_prob += 0.01
                
                if ue.rsrp is not None and not np.isnan(ue.rsrp):
                    if ue.rsrp < -120:
                        drop_prob += 0.12 + abs(ue.rsrp + 120) * 0.01
                    elif ue.rsrp < -115:
                        drop_prob += 0.06 + abs(ue.rsrp + 115) * 0.008
                    elif ue.rsrp < -110:
                        drop_prob += 0.03
                
                # Cell congestion
                if serving_cell.cpu_usage > 95:
                    drop_prob += (serving_cell.cpu_usage - 95) * 0.015
                elif serving_cell.cpu_usage > 90:
                    drop_prob += (serving_cell.cpu_usage - 90) * 0.01
                
                if serving_cell.prb_usage > 95:
                    drop_prob += (serving_cell.prb_usage - 95) * 0.012
                elif serving_cell.prb_usage > 90:
                    drop_prob += (serving_cell.prb_usage - 90) * 0.008
                
                # Traffic load factor
                load_ratio = serving_cell.current_load / serving_cell.max_capacity
                if load_ratio > 0.98:
                    drop_prob += (load_ratio - 0.98) * 1.0
                elif load_ratio > 0.95:
                    drop_prob += (load_ratio - 0.95) * 0.6
                elif load_ratio > 0.90:
                    drop_prob += (load_ratio - 0.90) * 0.2
                
                # Critical case: minimum power with poor signal
                if (serving_cell.tx_power <= serving_cell.min_tx_power and
                    ((ue.rsrp is not None and ue.rsrp < -110) or (ue.sinr is not None and ue.sinr < -5))):
                    drop_prob += 0.25
                
                # Apply drop event
                if np.random.rand() < min(0.45, drop_prob):
                    ue.serving_cell = None
                    ue.rsrp = None
                    ue.rsrq = None
                    ue.sinr = None
                    ue.session_active = False
                    ue.traffic_demand = 0
                    ue.drop_count += 1
            else:
                # Cell is inactive - force drop
                ue.serving_cell = None
                ue.rsrp = None
                ue.rsrq = None
                ue.sinr = None
                ue.session_active = False
                ue.traffic_demand = 0
                ue.drop_count += 1

