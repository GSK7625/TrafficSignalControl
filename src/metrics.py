"""Metrics Collection and Calculation module for SUMO Traffic Signal Evaluation.

Tracks 5 standard traffic metrics:
  1. Average Waiting Time (s) [via SUMO tripinfo output]
  2. Average Travel Time (s) [via SUMO tripinfo output]
  3. Mean Queue Length & Max Queue Length (veh) [via TraCI step sampling]
  4. Average Time Loss (s) [via SUMO tripinfo output]
  5. Throughput (completed vehicles) [via SUMO tripinfo output]
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET


@dataclass
class StepRecord:
    """Snapshot of traffic conditions at an individual simulation step."""
    step_time: float
    queue_length: float
    halted_vehicles: Optional[int] = None
    running_vehicles: Optional[int] = None


@dataclass
class TripRecord:
    """Detailed trip data for an individual completed vehicle."""
    vehicle_id: str
    depart: float
    arrival: float
    duration: float
    waiting_time: float
    time_loss: float
    route_length: float


class MetricsCollector:
    """Collects real-time TraCI step metrics and parses post-simulation TripInfo XML.

    Provides combined summary of the 5 key traffic performance metrics.
    """

    def __init__(self, monitored_lanes: Optional[Sequence[str]] = None) -> None:
        """Initializes MetricsCollector.

        Args:
            monitored_lanes: Optional list of lane IDs to monitor during simulation.
        """
        self.monitored_lanes: List[str] = list(monitored_lanes) if monitored_lanes else []
        self.step_records: List[StepRecord] = []
        self.trip_records: List[TripRecord] = []
        self.tripinfo_metrics: Dict[str, Any] = {}

    @property
    def total_steps(self) -> int:
        """Returns the total number of simulation steps recorded."""
        return len(self.step_records)

    @property
    def queue_history(self) -> List[float]:
        """Returns the instantaneous queue length recorded at each step."""
        return [record.queue_length for record in self.step_records]

    @property
    def mean_queue_length(self) -> float:
        """Calculates average queue length across all recorded simulation steps."""
        if not self.step_records:
            return 0.0
        return float(sum(self.queue_history) / len(self.step_records))

    @property
    def max_queue_length(self) -> int:
        """Returns the peak (maximum) queue length observed across all steps."""
        if not self.step_records:
            return 0
        return int(max(self.queue_history))

    def record_step(
        self,
        step_time: float,
        queue_length: float,
        halted_vehicles: Optional[int] = None,
        running_vehicles: Optional[int] = None,
    ) -> None:
        """Records traffic metrics for an individual simulation step.

        Args:
            step_time: Simulation timestamp in seconds.
            queue_length: Queue length / number of halted vehicles at this step.
            halted_vehicles: Optional explicit halted vehicles count.
            running_vehicles: Optional count of actively running vehicles.
        """
        record = StepRecord(
            step_time=float(step_time),
            queue_length=float(queue_length),
            halted_vehicles=halted_vehicles,
            running_vehicles=running_vehicles,
        )
        self.step_records.append(record)

    def collect_step_from_traci(
        self,
        traci_module: Any,
        incoming_lanes: Optional[Sequence[str]] = None,
    ) -> int:
        """Queries TraCI directly to record queue length and running vehicles.

        Args:
            traci_module: Active TraCI module or mock.
            incoming_lanes: Lanes to inspect for halted vehicles. Defaults to self.monitored_lanes.

        Returns:
            Total halted vehicles (queue length) at this step.
        """
        lanes = list(incoming_lanes) if incoming_lanes is not None else self.monitored_lanes

        step_time = 0.0
        try:
            step_time = float(traci_module.simulation.getTime())
        except Exception:
            pass

        total_halted = 0
        for lane_id in lanes:
            try:
                total_halted += int(traci_module.lane.getLastStepHaltingNumber(lane_id))
            except Exception:
                # Safely ignore lane query errors (e.g. unknown lane)
                pass

        running_count: Optional[int] = None
        try:
            running_count = len(traci_module.vehicle.getIDList())
        except Exception:
            try:
                running_count = int(traci_module.simulation.getMinExpectedNumber())
            except Exception:
                running_count = None

        self.record_step(
            step_time=step_time,
            queue_length=total_halted,
            halted_vehicles=total_halted,
            running_vehicles=running_count,
        )
        return total_halted

    def parse_tripinfo(self, xml_path: str) -> Dict[str, Any]:
        """Parses SUMO --tripinfo-output XML file and calculates trip metrics.

        Computes:
          - throughput: number of arrived/completed vehicles
          - average_travel_time: mean trip duration (s)
          - average_waiting_time: mean waiting time with speed < 0.1 m/s (s)
          - average_time_loss: mean time loss due to traffic/signals (s)
          - total_travel_time, total_waiting_time, total_time_loss (s)

        Args:
            xml_path: Absolute or relative path to SUMO tripinfo XML file.

        Returns:
            Dictionary containing calculated trip metrics.

        Raises:
            FileNotFoundError: If the specified XML file does not exist.
            ValueError: If the file is not valid XML.
        """
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Tripinfo file not found: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as err:
            raise ValueError(f"Failed to parse XML from {xml_path}: {err}") from err
        except Exception as err:
            raise ValueError(f"Error reading tripinfo file {xml_path}: {err}") from err

        total_duration = 0.0
        total_waiting = 0.0
        total_loss = 0.0
        throughput = 0
        self.trip_records.clear()

        # Iterate over all tripinfo elements
        for elem in root.findall(".//tripinfo"):
            try:
                veh_id = elem.get("id", f"veh_{throughput}")
                duration = float(elem.get("duration", "0.0"))
                waiting_time = float(elem.get("waitingTime", "0.0"))
                time_loss = float(elem.get("timeLoss", "0.0"))
                depart = float(elem.get("depart", "0.0"))
                arrival = float(elem.get("arrival", "0.0"))
                route_length = float(elem.get("routeLength", "0.0"))

                record = TripRecord(
                    vehicle_id=veh_id,
                    depart=depart,
                    arrival=arrival,
                    duration=duration,
                    waiting_time=waiting_time,
                    time_loss=time_loss,
                    route_length=route_length,
                )
                self.trip_records.append(record)

                total_duration += duration
                total_waiting += waiting_time
                total_loss += time_loss
                throughput += 1
            except (ValueError, TypeError):
                continue

        avg_travel = (total_duration / throughput) if throughput > 0 else 0.0
        avg_waiting = (total_waiting / throughput) if throughput > 0 else 0.0
        avg_loss = (total_loss / throughput) if throughput > 0 else 0.0

        metrics = {
            "throughput": throughput,
            "average_travel_time": avg_travel,
            "average_waiting_time": avg_waiting,
            "average_time_loss": avg_loss,
            "total_travel_time": total_duration,
            "total_waiting_time": total_waiting,
            "total_time_loss": total_loss,
        }

        self.tripinfo_metrics = metrics
        return metrics

    def get_summary(self, tripinfo_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns consolidated summary of the 5 standard traffic metrics.

        1. average_waiting_time (s)
        2. average_travel_time (s)
        3. mean_queue_length (veh) & max_queue_length (veh)
        4. average_time_loss (s)
        5. throughput (completed vehicles)

        Args:
            tripinfo_path: Optional path to tripinfo XML to parse before summarizing.

        Returns:
            Dictionary containing 5 core metrics plus simulation tracking metadata.
        """
        if tripinfo_path is not None:
            self.parse_tripinfo(tripinfo_path)

        avg_waiting = float(self.tripinfo_metrics.get("average_waiting_time", 0.0))
        avg_travel = float(self.tripinfo_metrics.get("average_travel_time", 0.0))
        avg_loss = float(self.tripinfo_metrics.get("average_time_loss", 0.0))
        throughput = int(self.tripinfo_metrics.get("throughput", 0))

        return {
            "average_waiting_time": avg_waiting,
            "average_travel_time": avg_travel,
            "mean_queue_length": self.mean_queue_length,
            "max_queue_length": self.max_queue_length,
            "average_time_loss": avg_loss,
            "throughput": throughput,
            "total_steps": self.total_steps,
            "queue_history": self.queue_history,
            "tripinfo_metrics": dict(self.tripinfo_metrics),
        }

    def reset(self) -> None:
        """Resets all collected step and tripinfo metrics to empty state."""
        self.step_records.clear()
        self.trip_records.clear()
        self.tripinfo_metrics.clear()

    def to_markdown_table(self, title: str = "Traffic Evaluation Summary") -> str:
        """Formats the 5 primary metrics as a clean Markdown table.

        Args:
            title: Table header title or scenario name.

        Returns:
            Markdown formatted string representation.
        """
        summary = self.get_summary()
        lines = [
            f"### {title}",
            "",
            "| Metric | Value | Unit | Description |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Average Waiting Time** | `{summary['average_waiting_time']:.2f}` | s | Mean time halting (speed < 0.1 m/s) |",
            f"| **Average Travel Time** | `{summary['average_travel_time']:.2f}` | s | Mean duration from depart to arrival |",
            f"| **Mean Queue Length** | `{summary['mean_queue_length']:.2f}` | veh | Time-averaged queue across incoming lanes |",
            f"| **Peak Queue Length** | `{summary['max_queue_length']}` | veh | Maximum instantaneous queue observed |",
            f"| **Average Time Loss** | `{summary['average_time_loss']:.2f}` | s | Mean delay compared to ideal free-flow |",
            f"| **Throughput** | `{summary['throughput']}` | veh | Total completed trips |",
        ]
        return "\n".join(lines)
