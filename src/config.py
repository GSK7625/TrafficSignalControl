"""Simulation configuration module for RESCO datasets and SUMO/TraCI execution."""

from dataclasses import dataclass
import os
import shutil
from typing import Optional

try:
    import sumolib
except ImportError:
    sumolib = None

from src.data_inspector import inspect_scenario


@dataclass
class SimulationConfig:
    """Configuration settings for a SUMO simulation run."""
    dataset_name: str
    data_dir: str = "data"
    gui: bool = False
    gui_delay: float = 0.0
    step_length: float = 1.0
    yellow_duration: float = 3.0
    all_red_duration: float = 0.0
    min_green: float = 5.0
    max_green: Optional[float] = 60.0
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

    def __post_init__(self):
        if self.step_length <= 0:
            raise ValueError(f"step_length must be positive, got {self.step_length}")
        if self.decision_interval <= 0 or self.decision_interval != self.decision_interval or self.decision_interval == float("inf"):
            raise ValueError(f"decision_interval must be positive and finite, got {self.decision_interval}")
        if self.gui_delay < 0:
            raise ValueError(f"gui_delay must be non-negative, got {self.gui_delay}")
        if self.yellow_duration <= 0:
            raise ValueError(f"yellow_duration must be positive, got {self.yellow_duration}")
        if self.all_red_duration < 0:
            raise ValueError(f"all_red_duration must be non-negative, got {self.all_red_duration}")
        if self.min_green < 0:
            raise ValueError(f"min_green must be non-negative, got {self.min_green}")
        if self.max_green is not None and self.max_green < self.min_green:
            raise ValueError(f"max_green ({self.max_green}) cannot be less than min_green ({self.min_green})")

        # Resolve paths
        self.scenario_dir = os.path.abspath(os.path.join(self.data_dir, self.dataset_name))
        if not os.path.isdir(self.scenario_dir):
            raise ValueError(f"Dataset '{self.dataset_name}' not found under '{self.data_dir}'")

        if not self.sumocfg_path:
            candidates = [
                os.path.join(self.scenario_dir, f"{self.dataset_name}.sumocfg"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    self.sumocfg_path = os.path.abspath(c)
                    break
            if not self.sumocfg_path:
                for f in os.listdir(self.scenario_dir):
                    if f.endswith(".sumocfg"):
                        self.sumocfg_path = os.path.abspath(os.path.join(self.scenario_dir, f))
                        break

        if not self.sumocfg_path or not os.path.isfile(self.sumocfg_path):
            raise ValueError(f"No valid .sumocfg found for dataset '{self.dataset_name}'")

        # Automatically extract begin and end time from .sumocfg if kept at defaults
        if self.begin_time == 0.0 and self.end_time == 3600.0:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(self.sumocfg_path)
                root = tree.getroot()
                b_node = root.find(".//begin")
                e_node = root.find(".//end")
                if b_node is not None and b_node.get("value"):
                    self.begin_time = float(b_node.get("value"))
                if e_node is not None and e_node.get("value"):
                    self.end_time = float(e_node.get("value"))
            except Exception:
                pass

        # Resolve binary
        binary_name = "sumo-gui" if self.gui else "sumo"
        resolved_bin = None
        if sumolib is not None:
            try:
                resolved_bin = sumolib.checkBinary(binary_name)
            except Exception:
                resolved_bin = None

        if not resolved_bin:
            resolved_bin = shutil.which(binary_name) or binary_name
        self.sumo_binary = resolved_bin


def get_scenario_config(
    dataset_name: str,
    data_dir: str = "data",
    gui: bool = False,
    gui_delay: float = 0.0,
    step_length: float = 1.0,
    yellow_duration: float = 3.0,
    all_red_duration: float = 0.0,
    min_green: float = 5.0,
    max_green: Optional[float] = 60.0,
    max_steps: Optional[int] = None,
    seed: Optional[int] = 42,
    output_dir: str = "results",
    control_mode: str = "original",
    use_tripinfo: bool = False,
    decision_interval: float = 1.0,
    **kwargs,
) -> SimulationConfig:
    """Creates a SimulationConfig populated with metadata from the dataset inspection."""
    try:
        scenario_info = inspect_scenario(dataset_name, data_dir)
    except FileNotFoundError as e:
        raise ValueError(f"Dataset '{dataset_name}' not found under '{data_dir}'") from e

    begin_time = kwargs.get("begin_time", scenario_info.begin_time)
    end_time = kwargs.get("end_time", scenario_info.end_time)

    return SimulationConfig(
        dataset_name=dataset_name,
        data_dir=data_dir,
        gui=gui,
        gui_delay=kwargs.get("gui_delay", gui_delay),
        step_length=step_length,
        yellow_duration=yellow_duration,
        all_red_duration=all_red_duration,
        min_green=min_green,
        max_green=kwargs.get("max_green", max_green),
        max_steps=max_steps,
        seed=seed,
        output_dir=output_dir,
        begin_time=begin_time,
        end_time=end_time,
        sumocfg_path=scenario_info.cfg_path,
        control_mode=control_mode,
        use_tripinfo=use_tripinfo,
        decision_interval=decision_interval,
    )


def load_yaml_config(config_path: str) -> dict:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML does not parse into a dictionary.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration format in {config_path}: expected dictionary, got {type(config)}")
    return config
