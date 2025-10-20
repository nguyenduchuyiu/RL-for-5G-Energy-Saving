"""Scenario configuration loader - ported from loadScenarioConfig.m"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SimParams:
    # Basic information
    name: str
    description: str
    deployment_scenario: str
    
    # Network topology
    num_sites: int
    num_sectors: int
    isd: float
    antenna_height: float
    cell_radius: float
    
    # RF parameters
    carrier_frequency: float
    system_bandwidth: float
    
    # User parameters
    num_ues: int
    ue_speed: float
    indoor_ratio: float = 0.8
    outdoor_speed: float = 30.0
    
    # Power parameters
    min_tx_power: float = 30.0
    max_tx_power: float = 46.0
    base_power: float = 800.0
    idle_power: float = 200.0
    
    # Simulation parameters
    sim_time: float = 600.0
    time_step: float = 1.0
    
    # Thresholds
    rsrp_serving_threshold: float = -110.0
    rsrp_target_threshold: float = -100.0
    rsrp_measurement_threshold: float = -115.0
    drop_call_threshold: float = 1.0
    latency_threshold: float = 50.0
    cpu_threshold: float = 80.0
    prb_threshold: float = 80.0
    
    # Traffic parameters
    traffic_lambda: float = 30.0
    peak_hour_multiplier: float = 1.5
    
    # Derived parameters
    total_steps: int = 0
    max_radius: float = 0.0
    expected_cells: int = 0
    
    # Optional fields
    layout: Optional[Dict[str, Any]] = None
    user_distribution: Optional[Dict[str, Any]] = None
    mobility_model: Optional[Dict[str, Any]] = None
    
    # Logging configuration
    log_file: str = ''
    ue_log_file: str = ''
    cell_log_file: str = ''
    agent_log_file: str = ''
    handover_log_file: str = ''
    enable_logging: bool = True
    log_level: str = 'INFO'


SCENARIO_MAPPINGS = {
    'indoor_hotspot': 'indoor_hotspot.json',
    'dense_urban': 'dense_urban.json',
    'rural': 'rural.json',
    'urban_macro': 'urban_macro.json'
}


def load_scenario_config(scenario_input: str, scenarios_dir: Optional[Path] = None) -> SimParams:
    """Load scenario configuration from JSON file"""
    
    if scenarios_dir is None:
        # Default to app/scenarios/
        scenarios_dir = Path(__file__).parent.parent / 'scenarios'
    
    # Resolve JSON file path
    json_path = _resolve_scenario_path(scenario_input, scenarios_dir)
    
    # Load and parse configuration
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    
    # Convert to SimParams
    sim_params = _convert_json_to_params(cfg)
    
    # Validate and enhance
    sim_params = _validate_and_enhance_config(sim_params)
    
    print(f'Loaded scenario: {sim_params.name}')
    return sim_params


def _resolve_scenario_path(scenario_input: str, scenarios_dir: Path) -> Path:
    """Resolve scenario input to actual JSON file path"""
    
    path = Path(scenario_input)
    
    # Direct file path provided
    if path.is_file():
        return path
    
    # Known scenario name
    if scenario_input in SCENARIO_MAPPINGS:
        return scenarios_dir / SCENARIO_MAPPINGS[scenario_input]
    
    # Try appending .json extension
    candidate = scenarios_dir / f'{scenario_input}.json'
    if candidate.is_file():
        return candidate
    
    available = ', '.join(SCENARIO_MAPPINGS.keys())
    raise ValueError(f'Unknown scenario: {scenario_input}\nAvailable scenarios: {available}')


def _get_field_or_default(cfg: Dict, field: str, default: Any) -> Any:
    """Get field value or default"""
    return cfg.get(field, default)


def _convert_json_to_params(cfg: Dict) -> SimParams:
    """Convert JSON configuration to SimParams"""
    
    return SimParams(
        # Basic information
        name=_get_field_or_default(cfg, 'name', 'Unnamed Scenario'),
        description=_get_field_or_default(cfg, 'description', 'No description'),
        deployment_scenario=_get_field_or_default(cfg, 'deploymentScenario', 'custom'),
        
        # Network topology
        num_sites=_get_field_or_default(cfg, 'numSites', 7),
        num_sectors=_get_field_or_default(cfg, 'numSectors', 3),
        isd=_get_field_or_default(cfg, 'isd', 200),
        antenna_height=_get_field_or_default(cfg, 'antennaHeight', 25),
        cell_radius=_get_field_or_default(cfg, 'cellRadius', 200),
        
        # RF parameters
        carrier_frequency=_get_field_or_default(cfg, 'carrierFrequency', 3.5e9),
        system_bandwidth=_get_field_or_default(cfg, 'systemBandwidth', 100e6),
        
        # User parameters
        num_ues=_get_field_or_default(cfg, 'numUEs', 210),
        ue_speed=_get_field_or_default(cfg, 'ueSpeed', 3),
        indoor_ratio=_get_field_or_default(cfg, 'indoorRatio', 0.8),
        outdoor_speed=_get_field_or_default(cfg, 'outdoorSpeed', 30),
        
        # Power parameters
        min_tx_power=_get_field_or_default(cfg, 'minTxPower', 30),
        max_tx_power=_get_field_or_default(cfg, 'maxTxPower', 46),
        base_power=_get_field_or_default(cfg, 'basePower', 800),
        idle_power=_get_field_or_default(cfg, 'idlePower', 200),
        
        # Simulation parameters
        sim_time=_get_field_or_default(cfg, 'simTime', 600),
        time_step=_get_field_or_default(cfg, 'timeStep', 1),
        
        # Thresholds
        rsrp_serving_threshold=_get_field_or_default(cfg, 'rsrpServingThreshold', -110),
        rsrp_target_threshold=_get_field_or_default(cfg, 'rsrpTargetThreshold', -100),
        rsrp_measurement_threshold=_get_field_or_default(cfg, 'rsrpMeasurementThreshold', -115),
        drop_call_threshold=_get_field_or_default(cfg, 'dropCallThreshold', 1),
        latency_threshold=_get_field_or_default(cfg, 'latencyThreshold', 50),
        cpu_threshold=_get_field_or_default(cfg, 'cpuThreshold', 80),
        prb_threshold=_get_field_or_default(cfg, 'prbThreshold', 80),
        
        # Traffic parameters
        traffic_lambda=_get_field_or_default(cfg, 'trafficLambda', 30),
        peak_hour_multiplier=_get_field_or_default(cfg, 'peakHourMultiplier', 1.5),
        
        # Optional fields
        layout=cfg.get('layout'),
        user_distribution=cfg.get('userDistribution'),
        mobility_model=cfg.get('mobilityModel'),
    )


def _validate_and_enhance_config(sim_params: SimParams) -> SimParams:
    """Validate configuration and add derived parameters"""
    
    # Validate basic parameters
    assert sim_params.num_sites > 0, 'numSites must be positive'
    assert sim_params.num_ues > 0, 'numUEs must be positive'
    if sim_params.sim_time > 0:
        assert 0 < sim_params.time_step <= sim_params.sim_time, 'invalid timeStep'
    
    # Add derived parameters
    if sim_params.sim_time <= 0 or sim_params.time_step <= 0:
        sim_params.total_steps = 0
    else:
        sim_params.total_steps = int(np.ceil(sim_params.sim_time / sim_params.time_step))
    sim_params.expected_cells = sim_params.num_sites * sim_params.num_sectors
    
    # Set scenario-specific defaults
    if sim_params.deployment_scenario == 'indoor_hotspot':
        sim_params.max_radius = 100
    elif sim_params.deployment_scenario == 'dense_urban':
        sim_params.max_radius = 500
    elif sim_params.deployment_scenario == 'rural':
        sim_params.max_radius = 2000
    elif sim_params.deployment_scenario == 'urban_macro':
        sim_params.max_radius = 800
    else:
        sim_params.max_radius = 1000
    
    # Configure logging (simplified for Python)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_params.log_file = f'logs/{timestamp}_energy_saving.log'
    sim_params.ue_log_file = f'logs/{timestamp}_ue.log'
    sim_params.cell_log_file = f'logs/{timestamp}_cell.log'
    sim_params.agent_log_file = f'logs/{timestamp}_agent.log'
    sim_params.handover_log_file = f'logs/{timestamp}_handover.log'
    
    return sim_params


import numpy as np

