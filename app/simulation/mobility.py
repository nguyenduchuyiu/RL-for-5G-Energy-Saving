"""UE mobility update - ported from updateUEMobility.m"""

import numpy as np
from typing import List
from .network import UE
from .scenario_loader import SimParams


def update_ue_mobility(ues: List[UE], time_step: float, current_time: float, seed: int, sim_params: SimParams) -> List[UE]:
    """Update UE mobility with scenario-specific patterns"""
    
    for ue in ues:
        # Set up mobility-specific RNG
        mobility_seed = ue.rng_seed + int(current_time * 1000)
        rng = np.random.RandomState(mobility_seed)
        
        ue.step_counter += 1
        
        # Update position based on mobility pattern
        _update_ue_position(ue, time_step, current_time, rng)
        
        # Enforce scenario-specific boundaries
        _enforce_scenario_bounds(ue, sim_params, rng)
        
        # Normalize direction
        ue.direction = ue.direction % (2 * np.pi)
    
    return ues


def _update_ue_position(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Update UE position based on mobility pattern"""
    
    pattern = ue.mobility_pattern
    
    if pattern == 'stationary':
        _handle_stationary_mobility(ue, time_step, rng)
    elif pattern == 'pedestrian':
        _handle_pedestrian_mobility(ue, time_step, current_time, rng)
    elif pattern == 'slow_walk':
        _handle_slow_walk_mobility(ue, time_step, rng)
    elif pattern == 'normal_walk':
        _handle_normal_walk_mobility(ue, time_step, rng)
    elif pattern == 'fast_walk':
        _handle_fast_walk_mobility(ue, time_step, rng)
    elif pattern == 'slow_vehicle':
        _handle_slow_vehicle_mobility(ue, time_step, current_time, rng)
    elif pattern == 'fast_vehicle':
        _handle_fast_vehicle_mobility(ue, time_step, current_time, rng)
    elif pattern == 'indoor_pedestrian':
        _handle_indoor_pedestrian_mobility(ue, time_step, current_time, rng)
    elif pattern == 'indoor_mobile':
        _handle_indoor_mobile_mobility(ue, time_step, rng)
    elif pattern == 'outdoor_vehicle':
        _handle_outdoor_vehicle_mobility(ue, time_step, current_time, rng)
    elif pattern == 'vehicle':
        _handle_vehicle_mobility(ue, time_step, current_time, rng)
    else:
        _handle_pedestrian_mobility(ue, time_step, current_time, rng)


def _move_ue(ue: UE, distance: float):
    """Move UE in its current direction"""
    ue.x += distance * np.cos(ue.direction)
    ue.y += distance * np.sin(ue.direction)


def _handle_stationary_mobility(ue: UE, time_step: float, rng: np.random.RandomState):
    """Stationary UE with minimal random movement"""
    if rng.rand() < 0.05:
        ue.x += (rng.rand() - 0.5) * 2
        ue.y += (rng.rand() - 0.5) * 2


def _handle_pedestrian_mobility(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Standard pedestrian mobility with pause periods"""
    distance = ue.velocity * time_step
    
    if ue.pause_timer > 0:
        ue.pause_timer -= time_step
    elif rng.rand() < 0.1:
        ue.pause_timer = 5 + rng.rand() * 10
    elif rng.rand() < 0.3:
        ue.direction += (rng.rand() - 0.5) * np.pi
        _move_ue(ue, distance)
    else:
        _move_ue(ue, distance)


def _handle_slow_walk_mobility(ue: UE, time_step: float, rng: np.random.RandomState):
    """Slow walking pattern"""
    distance = ue.velocity * time_step
    if rng.rand() < 0.3:
        ue.direction += (rng.rand() - 0.5) * np.pi / 2
    _move_ue(ue, distance)


def _handle_normal_walk_mobility(ue: UE, time_step: float, rng: np.random.RandomState):
    """Normal walking pattern"""
    distance = ue.velocity * time_step
    if rng.rand() < 0.4:
        ue.direction += (rng.rand() - 0.5) * np.pi / 2
    _move_ue(ue, distance)


def _handle_fast_walk_mobility(ue: UE, time_step: float, rng: np.random.RandomState):
    """Fast walking/jogging pattern"""
    distance = ue.velocity * time_step
    if rng.rand() < 0.2:
        ue.direction += (rng.rand() - 0.5) * np.pi / 4
    _move_ue(ue, distance)


def _handle_slow_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Slow vehicle mobility pattern"""
    distance = ue.velocity * time_step
    
    if current_time - ue.last_direction_change > 20 + rng.rand() * 30:
        ue.direction += (rng.rand() - 0.5) * np.pi / 2
        ue.last_direction_change = current_time
    
    _move_ue(ue, distance)


def _handle_fast_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Fast vehicle mobility pattern"""
    distance = ue.velocity * time_step
    
    if current_time - ue.last_direction_change > 40 + rng.rand() * 20:
        ue.direction += (rng.rand() - 0.5) * np.pi / 4
        ue.last_direction_change = current_time
    
    _move_ue(ue, distance)


def _handle_indoor_pedestrian_mobility(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Indoor pedestrian with pause and direction changes"""
    distance = ue.velocity * time_step
    
    if ue.pause_timer > 0:
        ue.pause_timer -= time_step
    elif rng.rand() < 0.15:
        ue.pause_timer = 2 + rng.rand() * 8
    elif rng.rand() < 0.4:
        ue.direction += (rng.rand() - 0.5) * np.pi
        _move_ue(ue, distance)
    else:
        _move_ue(ue, distance)


def _handle_indoor_mobile_mobility(ue: UE, time_step: float, rng: np.random.RandomState):
    """General indoor mobile pattern"""
    distance = ue.velocity * time_step
    
    if rng.rand() < 0.2:
        ue.direction += (rng.rand() - 0.5) * np.pi
    
    _move_ue(ue, distance)


def _handle_outdoor_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Outdoor vehicle mobility for urban scenarios"""
    distance = ue.velocity * time_step
    
    if current_time - ue.last_direction_change > 30 + rng.rand() * 20:
        ue.direction += (rng.rand() - 0.5) * np.pi / 6
        ue.last_direction_change = current_time
    
    _move_ue(ue, distance)


def _handle_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng: np.random.RandomState):
    """Generic vehicle mobility pattern"""
    distance = ue.velocity * time_step
    
    if current_time - ue.last_direction_change > 25 + rng.rand() * 15:
        ue.direction += (rng.rand() - 0.5) * np.pi / 3
        ue.last_direction_change = current_time
    
    _move_ue(ue, distance)


def _enforce_scenario_bounds(ue: UE, sim_params: SimParams, rng: np.random.RandomState):
    """Enforce boundaries based on deployment scenario"""
    
    scenario = sim_params.deployment_scenario
    
    if scenario == 'indoor_hotspot':
        _enforce_indoor_bounds(ue, rng)
    elif scenario == 'dense_urban':
        _enforce_urban_bounds(ue, sim_params, rng)
    elif scenario == 'rural':
        _enforce_rural_bounds(ue, sim_params, rng)
    elif scenario == 'urban_macro':
        _enforce_urban_macro_bounds(ue, sim_params, rng)
    else:
        _enforce_default_bounds(ue, rng)


def _enforce_indoor_bounds(ue: UE, rng: np.random.RandomState):
    """Indoor building bounds (120m x 50m office)"""
    min_x, max_x = 5, 115
    min_y, max_y = 5, 45
    
    # Bounce off walls
    if ue.x <= min_x:
        ue.x = min_x + 1
        ue.direction = np.pi - ue.direction
    elif ue.x >= max_x:
        ue.x = max_x - 1
        ue.direction = np.pi - ue.direction
    
    if ue.y <= min_y:
        ue.y = min_y + 1
        ue.direction = -ue.direction
    elif ue.y >= max_y:
        ue.y = max_y - 1
        ue.direction = -ue.direction


def _enforce_urban_bounds(ue: UE, sim_params: SimParams, rng: np.random.RandomState):
    """Urban area bounds"""
    max_radius = sim_params.max_radius
    distance = np.sqrt(ue.x**2 + ue.y**2)
    
    if distance > max_radius:
        angle = np.arctan2(ue.y, ue.x)
        ue.x = (max_radius - 10) * np.cos(angle)
        ue.y = (max_radius - 10) * np.sin(angle)
        ue.direction = angle + np.pi + (rng.rand() - 0.5) * np.pi / 2


def _enforce_rural_bounds(ue: UE, sim_params: SimParams, rng: np.random.RandomState):
    """Rural area bounds"""
    max_radius = sim_params.max_radius
    distance = np.sqrt(ue.x**2 + ue.y**2)
    
    if distance > max_radius:
        angle = np.arctan2(ue.y, ue.x)
        ue.x = (max_radius - 50) * np.cos(angle)
        ue.y = (max_radius - 50) * np.sin(angle)
        ue.direction = angle + np.pi + (rng.rand() - 0.5) * np.pi / 4


def _enforce_urban_macro_bounds(ue: UE, sim_params: SimParams, rng: np.random.RandomState):
    """Urban macro bounds"""
    max_radius = sim_params.max_radius
    distance = np.sqrt(ue.x**2 + ue.y**2)
    
    if distance > max_radius:
        angle = np.arctan2(ue.y, ue.x)
        ue.x = (max_radius - 20) * np.cos(angle)
        ue.y = (max_radius - 20) * np.sin(angle)
        ue.direction = angle + np.pi + (rng.rand() - 0.5) * np.pi / 3


def _enforce_default_bounds(ue: UE, rng: np.random.RandomState):
    """Default bounds for unknown scenarios"""
    max_bound = 1000
    if abs(ue.x) > max_bound or abs(ue.y) > max_bound:
        ue.x = np.clip(ue.x, -max_bound, max_bound)
        ue.y = np.clip(ue.y, -max_bound, max_bound)
        ue.direction = ue.direction + np.pi + (rng.rand() - 0.5) * np.pi / 4

