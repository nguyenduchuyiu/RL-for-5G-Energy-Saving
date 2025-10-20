"""Signal measurements update - ported from updateSignalMeasurements.m"""

import numpy as np
from typing import List
from .network import Cell, UE, Measurement
from .path_loss import calculate_path_loss


def update_signal_measurements(ues: List[UE], cells: List[Cell], rsrp_measurement_threshold: float, 
                               current_time: float, seed: int):
    """Update signal measurements for all UEs"""
    
    for ue in ues:
        measurements = []
        
        for cell in cells:
            # Calculate distance
            distance = np.sqrt((ue.x - cell.x)**2 + (ue.y - cell.y)**2)
            
            # Calculate path loss
            path_loss = calculate_path_loss(distance, cell.frequency, ue.id, current_time, seed)
            
            # Set up RNG for RSRP calculation
            rsrp_seed = seed + 6000 + ue.id + cell.id + int(current_time * 100)
            rng = np.random.RandomState(rsrp_seed)
            
            # RSRP calculation - sensitive to tx power changes
            rsrp = cell.tx_power - path_loss + rng.randn() * 1.5
            
            # If cell is at minimum power, severely degrade signal quality
            if cell.tx_power <= cell.min_tx_power + 2:
                power_penalty = (cell.min_tx_power + 2 - cell.tx_power) * 8
                rsrp = rsrp - power_penalty
                # Add extra noise for unstable connection
                rsrp = rsrp + rng.randn() * 3
            
            if rsrp >= (rsrp_measurement_threshold - 5):
                # Calculate RSSI
                rssi = rsrp + 10*np.log10(12) + rng.randn() * 0.5
                
                # Calculate RSRQ
                rsrq = max(-20, min(-3, 10*np.log10(12) + rsrp - rssi))
                
                # SINR also affected by tx power
                base_sinr = rsrp - (-110)
                if cell.tx_power <= cell.min_tx_power + 2:
                    sinr_penalty = (cell.min_tx_power + 2 - cell.tx_power) * 6
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

