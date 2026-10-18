"""Simulation configuration module for RESCO datasets and SUMO/TraCI execution."""

from dataclasses import dataclass
import os
import shutil
from typing import List, Optional

try:
    import sumolib
except ImportError:
    sumolib = None

from src.data_inspector import inspect_scenario

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
    begin_time: float = 0.0
    end_time: float = 3600.0
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

    def __post_init__(self):
        # Basic sanity checks
        if self.step_length <= 0:
            raise ValueError(f"step_length must be positive, got {self.step_length}")
        if self.decision_interval <= 0:
            raise ValueError(f"decision_interval must be positive, got {self.decision_interval}")
        if self.yellow_duration <= 0:
            raise ValueError(f"yellow_duration must be positive, got {self.yellow_duration}")

        # Resolve scenario and configuration paths
        self.scenario_dir = os.path.abspath(os.path.join(self.data_dir, self.dataset_name))
        if not os.path.isdir(self.scenario_dir):
            raise ValueError(f"Dataset '{self.dataset_name}' not found under '{self.data_dir}'")

        if not self.sumocfg_path:
            candidate = os.path.join(self.scenario_dir, f"{self.dataset_name}.sumocfg")
            if os.path.isfile(candidate):
                self.sumocfg_path = os.path.abspath(candidate)
            else:
                for f in os.listdir(self.scenario_dir):
                    if f.endswith(".sumocfg"):
                        self.sumocfg_path = os.path.abspath(os.path.join(self.scenario_dir, f))
                        break

        if not self.sumocfg_path or not os.path.isfile(self.sumocfg_path):
            raise ValueError(f"No valid .sumocfg found for dataset '{self.dataset_name}'")

        # Resolve SUMO binary
        binary = "sumo-gui" if self.gui else "sumo"
        self.sumo_binary = (sumolib.checkBinary(binary) if sumolib else None) or shutil.which(binary) or binary


def get_scenario_config(dataset_name: str, data_dir: str = "data", **kwargs) -> SimulationConfig:
    """Creates a SimulationConfig populated with metadata from dataset inspection."""
    try:
        info = inspect_scenario(dataset_name, data_dir)
        kwargs.setdefault("begin_time", info.begin_time)
        kwargs.setdefault("end_time", info.end_time)
        kwargs.setdefault("sumocfg_path", info.cfg_path)
    except Exception:
        pass
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
