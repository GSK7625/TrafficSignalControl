"""Simulation metrics collection and result export module."""

import csv
from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


def parse_tripinfo_output(xml_path: str) -> Dict[str, Any]:
    """Parses SUMO native --tripinfo-output XML file for ground-truth statistics."""
    if not os.path.isfile(xml_path):
        return {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    trips = root.findall("tripinfo")
    if not trips:
        return {
            "tripinfo_completed_vehicles": 0,
            "tripinfo_mean_travel_time": 0.0,
            "tripinfo_mean_waiting_time": 0.0,
            "tripinfo_total_waiting_time": 0.0,
            "tripinfo_mean_time_loss": 0.0,
            "tripinfo_total_time_loss": 0.0,
        }

    durations = [float(t.get("duration", 0.0)) for t in trips]
    waiting_times = [float(t.get("waitingTime", 0.0)) for t in trips]
    time_losses = [float(t.get("timeLoss", 0.0)) for t in trips]
    count = len(trips)

    return {
        "tripinfo_completed_vehicles": count,
        "tripinfo_mean_travel_time": round(sum(durations) / count, 2),
        "tripinfo_mean_waiting_time": round(sum(waiting_times) / count, 2),
        "tripinfo_total_waiting_time": round(sum(waiting_times), 2),
        "tripinfo_mean_time_loss": round(sum(time_losses) / count, 2),
        "tripinfo_total_time_loss": round(sum(time_losses), 2),
    }


@dataclass
class StepMetrics:
    """Metrics recorded at an individual simulation step."""
    step: int
    sim_time: float
    running_vehicles: int
    queue_length: int
    waiting_time: float
    time_loss: float
    step_completed_vehicles: int
    cumulative_completed_vehicles: int
    pending_vehicles: int = 0


class MetricsCollector:
    """Collects traffic performance metrics from TraCI and exports formatted results."""

    def __init__(self, scenario_name: str, tripinfo_file: Optional[str] = None):
        self.scenario_name = scenario_name
        self.tripinfo_file = tripinfo_file
        self.step_records: List[StepMetrics] = []
        self.vehicle_departure_times: Dict[str, float] = {}
        self.completed_travel_times: List[float] = []
        self.completed_time_losses: List[float] = []
        self.vehicle_last_time_loss: Dict[str, float] = {}
        self.completed_waiting_times: List[float] = []
        self.vehicle_last_waiting_time: Dict[str, float] = {}
        self.total_delay_seconds: float = 0.0
        self.pending_vehicles_history: List[int] = []
        self.total_pending_delay: float = 0.0
        self._cumulative_completed: int = 0
        self._cached_tripinfo_summary: Optional[Dict[str, Any]] = None
        self.run_metadata: Dict[str, Any] = {}
        self.measurement_errors: int = 0
        self.is_valid: bool = True
        self.error_messages: List[str] = []

    def set_run_metadata(self, metadata: Dict[str, Any]) -> None:
        """Sets external simulation run metadata (seed, control mode, unserved vehicles, etc.)."""
        self.run_metadata.update(metadata)

    def record_error(self, message: str) -> None:
        """Records an observation or measurement error."""
        self.measurement_errors += 1
        self.is_valid = False
        self.error_messages.append(message)

    def register_departed_vehicles(self, departures: Dict[str, float]) -> None:
        """Registers departure times for vehicles."""
        self.vehicle_departure_times.update(departures)

    def record_step(
        self,
        step: int,
        sim_time: float,
        controlled_lanes: Sequence[str],
        traci_module: Any,
        step_length: float = 1.0,
    ) -> StepMetrics:
        """Queries TraCI for current step performance metrics and records them.

        Args:
            step: Simulation step counter (1, 2, ...).
            sim_time: Current simulation timestamp in seconds.
            controlled_lanes: List of incoming lane IDs controlled by intersections.
            traci_module: Active TraCI interface object.
            step_length: Duration of the simulation step in seconds.

        Returns:
            StepMetrics instance for the current step.
        """
        # 1. Track newly departed vehicles
        try:
            departed_ids = traci_module.simulation.getDepartedIDList()
            for vid in departed_ids:
                if vid not in self.vehicle_departure_times:
                    # Query exact departure time if available, fallback to sim_time
                    try:
                        dep = traci_module.vehicle.getDeparture(vid)
                        dep_val = float(dep) if isinstance(dep, (int, float)) else sim_time
                        self.vehicle_departure_times[vid] = dep_val if dep_val >= 0 else sim_time
                    except Exception:
                        self.vehicle_departure_times[vid] = sim_time
        except Exception as e:
            self.measurement_errors += 1
            self.is_valid = False
            self.error_messages.append(f"Step {step} getDepartedIDList error: {e}")

        # 2. Track arrived (completed) vehicles and calculate travel times & time loss
        try:
            arrived_ids = traci_module.simulation.getArrivedIDList()
        except Exception as e:
            self.measurement_errors += 1
            self.is_valid = False
            self.error_messages.append(f"Step {step} getArrivedIDList error: {e}")
            arrived_ids = []

        step_completed = len(arrived_ids)
        self._cumulative_completed += step_completed

        for vid in arrived_ids:
            dept_time = self.vehicle_departure_times.pop(vid, None)
            if dept_time is not None:
                try:
                    dept_time_val = float(dept_time) if isinstance(dept_time, (int, float)) else sim_time
                    # In SUMO, when getArrivedIDList() reports a vehicle at sim_time,
                    # the vehicle arrived at the destination in the step that just ended,
                    # so the actual arrival timestamp is (sim_time - step_length).
                    arrival_time = sim_time - step_length
                    travel_time = max(0.0, arrival_time - dept_time_val)
                    self.completed_travel_times.append(travel_time)
                except Exception as e:
                    self.measurement_errors += 1
                    self.is_valid = False
                    self.error_messages.append(f"Step {step} vehicle {vid} travel time calc error: {e}")

            # Record final time loss & waiting time of arrived vehicle
            loss = self.vehicle_last_time_loss.pop(vid, 0.0)
            self.completed_time_losses.append(loss)
            wait = self.vehicle_last_waiting_time.pop(vid, 0.0)
            self.completed_waiting_times.append(wait)

        # 3. Calculate queue length (halting vehicles with speed < 0.1 m/s on controlled lanes)
        total_queue = 0
        if controlled_lanes:
            for lane_id in controlled_lanes:
                try:
                    total_queue += traci_module.lane.getLastStepHaltingNumber(lane_id)
                except Exception as e:
                    self.measurement_errors += 1
                    self.is_valid = False
                    self.error_messages.append(f"Step {step} lane {lane_id} queue read error: {e}")
        else:
            # Fallback: halting vehicles across all running vehicles
            try:
                all_vehs = traci_module.vehicle.getIDList()
                for vid in all_vehs:
                    if traci_module.vehicle.getSpeed(vid) < 0.1:
                        total_queue += 1
            except Exception as e:
                self.measurement_errors += 1
                self.is_valid = False
                self.error_messages.append(f"Step {step} fallback queue read error: {e}")

        # Accumulate vehicle-seconds of queue delay
        self.total_delay_seconds += total_queue * step_length

        # 4. Calculate total waiting time and time loss of currently active vehicles
        total_waiting_time = 0.0
        total_time_loss = 0.0
        running_vehs_count = 0

        try:
            running_vehs = traci_module.vehicle.getIDList()
            running_vehs_count = len(running_vehs)
            for vid in running_vehs:
                w = 0.0
                try:
                    w = traci_module.vehicle.getAccumulatedWaitingTime(vid)
                    if not isinstance(w, (int, float)):
                        w = traci_module.vehicle.getWaitingTime(vid)
                except Exception as e:
                    self.measurement_errors += 1
                    self.is_valid = False
                    self.error_messages.append(
                        f"Step {step} vehicle {vid} getAccumulatedWaitingTime failed: {e}; fell back to snapshot getWaitingTime"
                    )
                    try:
                        w = traci_module.vehicle.getWaitingTime(vid)
                    except Exception as err:
                        self.measurement_errors += 1
                        self.error_messages.append(f"Step {step} vehicle {vid} getWaitingTime read error: {err}")
                        w = 0.0

                if not isinstance(w, (int, float)):
                    try:
                        w = float(w)
                    except Exception:
                        w = 0.0

                total_waiting_time += w
                self.vehicle_last_waiting_time[vid] = w

                try:
                    loss = traci_module.vehicle.getTimeLoss(vid)
                    if not isinstance(loss, (int, float)):
                        loss = float(loss)
                except Exception as e:
                    self.measurement_errors += 1
                    self.is_valid = False
                    self.error_messages.append(f"Step {step} vehicle {vid} time loss read error: {e}")
                    loss = 0.0

                total_time_loss += loss
                self.vehicle_last_time_loss[vid] = loss

                if vid not in self.vehicle_departure_times:
                    self.vehicle_departure_times[vid] = sim_time
        except Exception as e:
            self.measurement_errors += 1
            self.is_valid = False
            self.error_messages.append(f"Step {step} vehicle getIDList read error: {e}")

        # Query pending insertion vehicles
        pending_count = 0
        try:
            pending_ids = traci_module.simulation.getPendingVehicles()
            pending_count = len(pending_ids) if pending_ids is not None else 0
        except Exception as e:
            self.measurement_errors += 1
            self.is_valid = False
            self.error_messages.append(f"Step {step} simulation getPendingVehicles read error: {e}")
            pending_count = 0
        self.pending_vehicles_history.append(pending_count)
        self.total_pending_delay += pending_count * step_length

        record = StepMetrics(
            step=step,
            sim_time=sim_time,
            running_vehicles=running_vehs_count,
            queue_length=total_queue,
            waiting_time=total_waiting_time,
            time_loss=total_time_loss,
            step_completed_vehicles=step_completed,
            cumulative_completed_vehicles=self._cumulative_completed,
            pending_vehicles=pending_count,
        )
        self.step_records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Calculates aggregate statistical metrics across the recorded simulation run."""
        total_steps = len(self.step_records)
        if total_steps == 0:
            return {
                "scenario_name": self.scenario_name,
                "total_steps": 0,
                "total_completed_vehicles": 0,
                "mean_queue_length": 0.0,
                "max_queue_length": 0,
                "mean_waiting_time": 0.0,
                "completed_mean_waiting_time": 0.0,
                "system_mean_delay": 0.0,
                "system_total_delay": 0.0,
                "pending_insertion_vehicles": 0,
                "total_pending_delay": 0.0,
                "completed_vehicles_waiting_time": 0.0,
                "running_vehicles_waiting_time": 0.0,
                "total_network_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "total_queue_delay": 0.0,
                "mean_travel_time": 0.0,
                "total_time_loss": 0.0,
                "is_valid": self.is_valid,
                "measurement_errors": self.measurement_errors,
                "status": "OK" if self.is_valid else "INVALID",
            }

        queue_lengths = [r.queue_length for r in self.step_records]

        total_completed = (
            self._cumulative_completed
            if self._cumulative_completed > 0
            else (self.step_records[-1].cumulative_completed_vehicles if self.step_records else 0)
        )

        mean_travel = (
            sum(self.completed_travel_times) / len(self.completed_travel_times)
            if self.completed_travel_times
            else 0.0
        )

        # Aggregate total time loss: completed vehicles + remaining running vehicles
        total_loss = sum(self.completed_time_losses) + sum(self.vehicle_last_time_loss.values())
        if total_loss == 0.0 and self.step_records:
            total_loss = self.step_records[-1].time_loss

        # Metric Definitions & Accounting:
        # - completed_mean_wait (mean_waiting_time): average stopped time (speed < 0.1 m/s)
        #   for completed vehicles that exited the network.
        # - system_total_delay: total stopped delay inside network (completed + running) plus
        #   total pending delay (delay of vehicles queued to enter the network).
        # - system_mean_delay: system_total_delay averaged across all demand vehicles
        #   (completed + currently running + pending insertion at simulation end).
        completed_wait = sum(self.completed_waiting_times)
        running_wait = sum(self.vehicle_last_waiting_time.values())
        total_network_wait = completed_wait + running_wait

        system_total_delay = total_network_wait + self.total_pending_delay
        scenario_demand = self.run_metadata.get("total_scenario_demand")
        if scenario_demand and isinstance(scenario_demand, (int, float)) and scenario_demand > 0:
            total_vehs_considered = float(scenario_demand)
        else:
            pending_end = self.pending_vehicles_history[-1] if self.pending_vehicles_history else 0
            total_vehs_considered = max(1, total_completed + len(self.vehicle_last_waiting_time) + pending_end)
        system_mean_delay = system_total_delay / total_vehs_considered
        completed_mean_wait = (completed_wait / total_completed) if total_completed > 0 else 0.0

        # Total waiting time / delay
        waiting_times = [r.waiting_time for r in self.step_records]
        if self.completed_waiting_times or self.vehicle_last_waiting_time:
            total_wait = total_network_wait
            mean_wait = completed_mean_wait if total_completed > 0 else (
                total_network_wait / max(1, len(self.vehicle_departure_times))
            )
        elif self.total_delay_seconds > 0.0:
            total_wait = self.total_delay_seconds
            unique_vehs = max(1, len(self.vehicle_departure_times) + total_completed)
            mean_wait = total_wait / unique_vehs
        else:
            total_wait = sum(waiting_times)
            mean_wait = total_wait / total_steps if total_steps > 0 else 0.0

        summary = {
            "scenario_name": self.scenario_name,
            "total_steps": total_steps,
            "total_completed_vehicles": total_completed,
            "mean_queue_length": round(sum(queue_lengths) / total_steps, 2),
            "max_queue_length": max(queue_lengths) if queue_lengths else 0,
            "mean_waiting_time": round(mean_wait, 2),
            "completed_mean_waiting_time": round(completed_mean_wait, 2),
            "system_mean_delay": round(system_mean_delay, 2),
            "system_total_delay": round(system_total_delay, 2),
            "pending_insertion_vehicles": pending_end,
            "total_pending_delay": round(self.total_pending_delay, 2),
            "completed_vehicles_waiting_time": round(completed_wait, 2),
            "running_vehicles_waiting_time": round(running_wait, 2),
            "total_network_waiting_time": round(total_network_wait, 2),
            "total_waiting_time": round(total_wait, 2),
            "total_queue_delay": round(self.total_delay_seconds, 2),
            "mean_travel_time": round(mean_travel, 2),
            "total_time_loss": round(total_loss, 2),
            "is_valid": self.is_valid,
            "measurement_errors": self.measurement_errors,
            "status": "OK" if self.is_valid else "INVALID",
        }

        # Merge run metadata if configured
        if self.run_metadata:
            summary.update(self.run_metadata)

        # Merge tripinfo ground truth if file exists and cache result
        if self._cached_tripinfo_summary is not None:
            summary.update(self._cached_tripinfo_summary)
        elif self.tripinfo_file and os.path.isfile(self.tripinfo_file):
            trip_stats = parse_tripinfo_output(self.tripinfo_file)
            if trip_stats:
                self._cached_tripinfo_summary = trip_stats
                summary.update(trip_stats)

        return summary

    def export_results(self, output_dir: str, prefix: str = "") -> Tuple[str, str]:
        """Exports metrics summary to JSON and step history to CSV in output_dir.

        Returns:
            Tuple of (summary_json_path, step_metrics_csv_path).
        """
        # Safety check: Prevent writing to data directory
        norm_output = os.path.normpath(os.path.abspath(output_dir))
        data_dir_abs = os.path.normpath(os.path.abspath("data"))
        if norm_output.startswith(data_dir_abs):
            raise ValueError(f"Cannot save results into data directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        tag = f"{prefix}_" if prefix else ""
        json_path = os.path.join(output_dir, f"{tag}summary.json")
        csv_path = os.path.join(output_dir, f"{tag}step_metrics.csv")

        # Write summary JSON
        summary = self.get_summary()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write step metrics CSV
        fieldnames = [
            "step",
            "sim_time",
            "running_vehicles",
            "queue_length",
            "waiting_time",
            "time_loss",
            "step_completed_vehicles",
            "cumulative_completed_vehicles",
            "pending_vehicles",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.step_records:
                writer.writerow(asdict(record))

        return json_path, csv_path


# Alias for compatibility
SimulationMetricsCollector = MetricsCollector
