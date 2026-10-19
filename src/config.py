"""Simulation configuration module for RESCO datasets and SUMO/TraCI execution."""

from dataclasses import dataclass, field
import math
import os
import shutil
from typing import List, Optional

try:
    import sumolib
except ImportError:
    sumolib = None

from src.data_inspector import ScenarioInfo, inspect_scenario

DEFAULT_SAFETY_CONFIG = {
    "yellow_duration": 3.0,
    "all_red_duration": 1.0,
    "min_green": 5.0,
    "max_green": 60.0,
}


@dataclass
class SimulationConfig:
    """Configuration settings for a SUMO simulation run."""
    dataset_name: str
    data_dir: str = "data"
    gui: bool = False
    gui_delay: float = 0.0
    step_length: float = 1.0
    yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"]
    all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"]
    min_green: float = DEFAULT_SAFETY_CONFIG["min_green"]
    max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"]
    max_steps: Optional[int] = None
    seed: Optional[int] = 42
    output_dir: str = "results"
    begin_time: Optional[float] = None
    end_time: Optional[float] = None
    sumocfg_path: str = ""
    scenario_dir: str = ""
    sumo_binary: str = "sumo"
    control_mode: str = "original"
    use_tripinfo: bool = False
    decision_interval: float = 1.0
    target_tls_id: Optional[str] = None
    controlled_tls_ids: Optional[List[str]] = None
    use_libsumo: bool = True
    save_run_details: bool = False
    scenario_info: Optional[ScenarioInfo] = field(default=None, repr=False)

    def __post_init__(self):
        if self.step_length <= 0:
            raise ValueError(f"step_length must be positive, got {self.step_length}")
        if self.decision_interval <= 0:
            raise ValueError(f"decision_interval must be positive, got {self.decision_interval}")
        if self.yellow_duration <= 0:
            raise ValueError(f"yellow_duration must be positive, got {self.yellow_duration}")
        if self.control_mode not in ("original", "agent", "cyclic"):
            raise ValueError(f"Unknown control_mode: {self.control_mode}")

        self.scenario_dir = os.path.abspath(os.path.join(self.data_dir, self.dataset_name))
        if not os.path.isdir(self.scenario_dir):
            raise ValueError(f"Dataset '{self.dataset_name}' not found under '{self.data_dir}'")
        self.scenario_info = self.scenario_info or inspect_scenario(self.dataset_name, self.data_dir)
        self.sumocfg_path = os.path.abspath(self.sumocfg_path or self.scenario_info.cfg_path)
        self.begin_time = self.scenario_info.begin_time if self.begin_time is None else self.begin_time
        self.end_time = self.scenario_info.end_time if self.end_time is None else self.end_time

        # Resolve SUMO binary
        binary = "sumo-gui" if self.gui else "sumo"
        self.sumo_binary = (sumolib.checkBinary(binary) if sumolib else None) or shutil.which(binary) or binary


def get_scenario_config(dataset_name: str, data_dir: str = "data", **kwargs) -> SimulationConfig:
    """Creates a SimulationConfig populated with metadata from dataset inspection."""
    return SimulationConfig(dataset_name=dataset_name, data_dir=data_dir, **kwargs)


def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration format in {config_path}: expected dictionary, got {type(config)}")
    return config
