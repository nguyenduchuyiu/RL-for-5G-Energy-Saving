"""Signal measurements update - ported from updateSignalMeasurements.m"""

import numpy as np
from typing import List
from .network import Cell, UE, Measurement
from .path_loss import calculate_path_loss, calculate_path_loss_vectorized


def update_signal_measurements(ues: List[UE], cells: List[Cell], rsrp_measurement_threshold: float, 
                               current_time: float, seed: int):
    """Update signal measurements for all UEs - Vectorized version"""
    
    if not ues or not cells:
        return
    
    n_ues = len(ues)
    n_cells = len(cells)
    
    # Extract UE and Cell positions into arrays
    ue_positions = np.array([[ue.x, ue.y] for ue in ues])  # (n_ues, 2)
    cell_positions = np.array([[cell.x, cell.y] for cell in cells])  # (n_cells, 2)
    
    # Calculate distance matrix (n_ues, n_cells)
    # Use broadcasting: (n_ues, 1, 2) - (1, n_cells, 2) -> (n_ues, n_cells, 2)
    diff = ue_positions[:, np.newaxis, :] - cell_positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))  # (n_ues, n_cells)
    
    # Extract cell properties
    cell_frequencies = np.array([cell.frequency for cell in cells])
    cell_tx_powers = np.array([cell.tx_power for cell in cells])
    cell_min_tx_powers = np.array([cell.min_tx_power for cell in cells])
    cell_ids = np.array([cell.id for cell in cells])
    ue_ids = np.array([ue.id for ue in ues])
    
    # Calculate path loss matrix (n_ues, n_cells)
    path_loss_matrix = calculate_path_loss_vectorized(
        distances, cell_frequencies, ue_ids, cell_ids, current_time, seed
    )
    
    # Calculate RSRP matrix
    # Generate random noise for RSRP
    rsrp_noise = np.zeros((n_ues, n_cells))
    power_penalty_matrix = np.zeros((n_ues, n_cells))
    extra_noise = np.zeros((n_ues, n_cells))
    
    for i in range(n_ues):
        for j in range(n_cells):
            rsrp_seed = seed + 6000 + ue_ids[i] + cell_ids[j] + int(current_time * 100)
            rng = np.random.RandomState(rsrp_seed)
            rsrp_noise[i, j] = rng.randn() * 1.5
            
            # Power penalty for cells at minimum power
            if cell_tx_powers[j] <= cell_min_tx_powers[j] + 2:
                power_penalty_matrix[i, j] = (cell_min_tx_powers[j] + 2 - cell_tx_powers[j]) * 8
                extra_noise[i, j] = rng.randn() * 3
    
    # Calculate RSRP: tx_power - path_loss + noise - power_penalty + extra_noise
    rsrp_matrix = (cell_tx_powers[np.newaxis, :] - path_loss_matrix + 
                   rsrp_noise - power_penalty_matrix + extra_noise)
    
    # Process measurements for each UE
    for i, ue in enumerate(ues):
        measurements = []
        
        for j, cell in enumerate(cells):
            rsrp = rsrp_matrix[i, j]
            
            if rsrp >= (rsrp_measurement_threshold - 5):
                # Generate RSSI and RSRQ noise
                rsrp_seed = seed + 6000 + ue_ids[i] + cell_ids[j] + int(current_time * 100)
                rng = np.random.RandomState(rsrp_seed)
                rng.randn()  # Skip first (already used for rsrp_noise)
                if cell_tx_powers[j] <= cell_min_tx_powers[j] + 2:
                    rng.randn()  # Skip second (already used for extra_noise)
                
                # Calculate RSSI
                rssi = rsrp + 10*np.log10(12) + rng.randn() * 0.5
                
                # Calculate RSRQ
                rsrq = max(-20, min(-3, 10*np.log10(12) + rsrp - rssi))
                
                # SINR also affected by tx power
                base_sinr = rsrp - (-110)
                if cell_tx_powers[j] <= cell_min_tx_powers[j] + 2:
                    sinr_penalty = (cell_min_tx_powers[j] + 2 - cell_tx_powers[j]) * 6
                    base_sinr = base_sinr - sinr_penalty
                sinr = base_sinr + rng.randn() * 2
                
                measurements.append(Measurement(
                    cell_id=cell.id,
                    rsrp=rsrp,
                    rsrq=rsrq,
                    sinr=sinr
                ))
        
        # Update UE measurements
        if not measurements:
            ue.serving_cell = None
            ue.rsrp = None
            ue.rsrq = None
            ue.sinr = None
            ue.neighbor_measurements = []
            continue
        
        # Find serving cell measurement
        serving_cell_measurement = None
        if ue.serving_cell is not None:
            serving_cell_measurement = next(
                (m for m in measurements if m.cell_id == ue.serving_cell), None
            )
        
        if serving_cell_measurement is not None:
            ue.rsrp = serving_cell_measurement.rsrp
            ue.rsrq = serving_cell_measurement.rsrq
            ue.sinr = serving_cell_measurement.sinr
        else:
            # Serving cell not found in measurements, UE loses connection
            ue.serving_cell = None
            ue.rsrp = None
            ue.rsrq = None
            ue.sinr = None
        
        ue.neighbor_measurements = measurements

