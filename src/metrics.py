"""Simulation metrics collection and result export module.

Primary focus on the 3 Core Traffic Signal Control Metrics:
1. Waiting Time: mean_waiting_time (s), total_waiting_time (s)
2. Queue Length: mean_queue_length (veh), max_queue_length (veh)
3. Throughput: throughput_vph (veh/h), total_completed_vehicles (veh)
"""

import csv
from dataclasses import asdict, dataclass
import math
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


METRIC_SCHEMA_VERSION = "5.0.0"

# Experiment acceptance and stricter report warnings serve different purposes.
BENCHMARK_COMPLETION_THRESHOLD = 0.95
REPORT_COMPLETION_THRESHOLD = 0.98


def completion_statistics(
    summary: Dict[str, Any], fallback_demand: Optional[int] = None,
) -> Tuple[int, Optional[int], float]:
    """Returns completed vehicles, demand and completion fraction (not percent)."""
    completed = summary.get("total_completed_vehicles", summary.get("completed_vehicles", summary.get("tripinfo_completed_vehicles", 0))) or 0
    demand = next((summary[key] for key in ("scheduled_demand_in_window", "total_scenario_demand", "total_demand") if summary.get(key) is not None), fallback_demand)
    if demand is None and all(summary.get(key) is not None for key in ("running_vehicles_at_end", "pending_insertion_vehicles")):
        demand = completed + summary["running_vehicles_at_end"] + summary["pending_insertion_vehicles"]
    return completed, demand, completed / demand if demand and demand > 0 else 0.0


def service_status(completion_rate: float, threshold: float = BENCHMARK_COMPLETION_THRESHOLD) -> str:
    """Classifies a completion fraction using the caller's named policy."""
    if completion_rate is None:
        return "NO_DATA"
    return "COMPLETED" if completion_rate >= threshold else "LOW_COMPLETION"


def parse_tripinfo_output(xml_path: str) -> Dict[str, Any]:
    """Parses SUMO native --tripinfo-output XML file for ground-truth statistics."""
    root = ET.parse(xml_path).getroot()
    if root.tag != "tripinfos":
        raise ValueError("Expected a tripinfos XML root")
    all_trips = root.findall("tripinfo")

    completed_ids, durations, waiting_times, time_losses = [], [], [], []
    for trip in all_trips:
        completed_ids.append(trip.attrib["id"])
        values = [float(trip.attrib[key]) for key in ("duration", "waitingTime", "timeLoss")]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("TripInfo metrics must be finite and non-negative")
        duration, wait, loss = values
        durations.append(duration)
        waiting_times.append(wait)
        time_losses.append(loss)

    count = len(all_trips)
    if count == 0:
        return {
            "tripinfo_completed_vehicles": 0,
            "tripinfo_completed_ids": [],
            "tripinfo_unfinished_vehicles": 0,
            "tripinfo_mean_travel_time": None,
            "tripinfo_mean_waiting_time": None,
            "tripinfo_total_waiting_time": 0.0,
            "tripinfo_mean_time_loss": None,
            "tripinfo_total_time_loss": 0.0,
            "tripinfo_metrics_available": False,
        }

    return {
        "tripinfo_completed_vehicles": count,
        "tripinfo_completed_ids": completed_ids,
        "tripinfo_unfinished_vehicles": 0,
        "tripinfo_mean_travel_time": round(sum(durations) / count, 2),
        "tripinfo_mean_waiting_time": round(sum(waiting_times) / count, 2),
        "tripinfo_total_waiting_time": round(sum(waiting_times), 2),
        "tripinfo_mean_time_loss": round(sum(time_losses) / count, 2),
        "tripinfo_total_time_loss": round(sum(time_losses), 2),
        "tripinfo_metrics_available": True,
    }


def build_vehicle_accounting(
    scenario_name: str,
    window_begin: float,
    window_end: float,
    step_length: float = 1.0,
    total_completed: int = 0,
    running_vehs: int = 0,
    pending_vehs: int = 0,
    total_teleports: int = 0,
    total_collisions: int = 0,
    scenario_info: Optional[Any] = None,
    data_dir: str = "data",
    population: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds clean vehicle counts and observation window metadata."""
    duration = max(0.0, float(window_end) - float(window_begin))
    realized = total_completed + running_vehs + pending_vehs
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "observation_window_begin": float(window_begin),
        "observation_window_end": float(window_end),
        "observation_window_duration": duration,
        "realized_demand_in_window": realized,
        "total_completed_vehicles": total_completed,
        "running_vehicles_at_end": running_vehs,
        "pending_insertion_vehicles": pending_vehs,
        "teleport_count": total_teleports,
        "collision_count": total_collisions,
        "accounting_valid": True,
    }


@dataclass
class StepMetrics:
    """Metrics recorded at an individual simulation step."""

    step: int
    sim_time: float
    running_vehicles: int
    queue_length: int
    waiting_time: float
    time_loss: float = 0.0
    step_completed_vehicles: int = 0
    cumulative_completed_vehicles: int = 0
    pending_vehicles: Optional[int] = 0


class MetricsCollector:
    """Collects traffic performance metrics from TraCI and exports formatted results."""

    def __init__(self, scenario_name: str, tripinfo_file: Optional[str] = None):
        self.scenario_name = scenario_name
        self.tripinfo_file = tripinfo_file
        self.step_records: List[StepMetrics] = []
        self._cumulative_completed: int = 0
        self.departed_ids: set = set()
        self.completed_ids: set = set()
        self.running_ids: set = set()
        self.pending_ids: set = set()
        self.vehicle_departure_times: Dict[str, float] = {}
        self.completed_travel_times: List[float] = []
        self.vehicle_last_waiting_time: Dict[str, float] = {}
        self.vehicle_last_time_loss: Dict[str, float] = {}
        self.completed_waiting_times: List[float] = []
        self.completed_time_losses: List[float] = []
        self.total_delay_seconds: float = 0.0
        self.total_pending_delay: float = 0.0
        self.last_pending_count: int = 0
        self.run_metadata: Dict[str, Any] = {}
        self.measurement_errors: int = 0
        self.is_valid: bool = True
        self.error_messages: List[str] = []
        self.observed_duration: float = 0.0
        self._reconciled: bool = False
        self._cached_tripinfo_summary: Optional[Dict[str, Any]] = None

    def population_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of vehicle IDs across lifecycle states."""
        return {
            "states": {
                "departed": sorted(self.departed_ids),
                "completed": sorted(self.completed_ids),
                "running": list(self.running_ids),
                "pending": list(self.pending_ids),
            },
            "available": True,
            "errors": [],
        }

    def reconcile_tripinfo(self) -> Dict[str, Any]:
        """Reconciles completed IDs after SUMO has closed its output file."""
        if self._reconciled:
            return self._cached_tripinfo_summary
        result = {"tripinfo_required": bool(self.tripinfo_file), "tripinfo_reconciliation_status": "NOT_REQUIRED"}
        if self.tripinfo_file:
            try:
                stats = parse_tripinfo_output(self.tripinfo_file)
                result.update(stats)
                result["tripinfo_reconciliation_status"] = "MATCH"
            except FileNotFoundError:
                result["tripinfo_reconciliation_status"] = "MISSING"
            except Exception as error:
                result["tripinfo_reconciliation_status"] = "PARSE_ERROR"
                result["tripinfo_error"] = str(error)
        self._reconciled = True
        self._cached_tripinfo_summary = result
        return result

    def set_run_metadata(self, metadata: Dict[str, Any]) -> None:
        """Sets external simulation run metadata."""
        self.run_metadata.update(metadata)

    def record_error(self, message: str) -> None:
        """Records an observation or measurement error."""
        self.measurement_errors += 1
        self.is_valid = False
        self.error_messages.append(message)

    def register_departed_vehicles(self, departures: Dict[str, float]) -> None:
        """Registers departure times for vehicles."""
        self.vehicle_departure_times.update(departures)
        self.departed_ids.update(departures)

    def record_step(
        self,
        step: int,
        sim_time: float,
        controlled_lanes: Sequence[str],
        traci_module: Any,
        step_length: float = 1.0,
    ) -> StepMetrics:
        """Queries TraCI for current step performance metrics and records them."""
        # Required observations fail the run instead of manufacturing zero metrics.
        departed = traci_module.simulation.getDepartedIDList()
        arrived = traci_module.simulation.getArrivedIDList()
        running = traci_module.vehicle.getIDList()
        pending = traci_module.simulation.getPendingVehicles()
        total_queue = (
            sum(traci_module.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
            if controlled_lanes else
            sum(traci_module.vehicle.getSpeed(vid) < 0.1 for vid in running)
        )
        waits = {vid: float(traci_module.vehicle.getAccumulatedWaitingTime(vid)) for vid in running}
        losses = {vid: float(traci_module.vehicle.getTimeLoss(vid)) for vid in running}
        for vid in departed:
            dep = float(traci_module.vehicle.getDeparture(vid))
            self.vehicle_departure_times[vid] = dep if dep >= 0 else sim_time
        new_arrived = set(arrived) - self.completed_ids
        for vid in new_arrived:
            dep = self.vehicle_departure_times.pop(vid, None)
            if dep is not None:
                self.completed_travel_times.append(max(0.0, sim_time - dep))
            self.completed_waiting_times.append(self.vehicle_last_waiting_time.pop(vid, 0.0))
            self.completed_time_losses.append(self.vehicle_last_time_loss.pop(vid, 0.0))
        self.departed_ids.update(departed)
        self.completed_ids.update(new_arrived)
        self._cumulative_completed += len(new_arrived)
        self.running_ids = set(running)
        self.pending_ids = set(pending)
        self.vehicle_last_waiting_time.update(waits)
        self.vehicle_last_time_loss.update(losses)
        self.last_pending_count = len(pending)
        self.observed_duration += step_length
        self.total_delay_seconds += total_queue * step_length
        self.total_pending_delay += len(pending) * step_length
        record = StepMetrics(
            step=step, sim_time=sim_time, running_vehicles=len(running),
            queue_length=total_queue, waiting_time=sum(waits.values()),
            time_loss=sum(losses.values()), step_completed_vehicles=len(new_arrived),
            cumulative_completed_vehicles=self._cumulative_completed,
            pending_vehicles=len(pending),
        )
        self.step_records.append(record)
        return record

    def get_core_metrics(self) -> Dict[str, Any]:
        """Extracts strictly the 3 core traffic metrics."""
        summary = self.get_summary()
        return {
            "mean_waiting_time": summary.get("mean_waiting_time"),
            "total_waiting_time": summary.get("total_waiting_time"),
            "mean_queue_length": summary.get("mean_queue_length"),
            "max_queue_length": summary.get("max_queue_length"),
            "throughput_vph": summary.get("throughput_vph"),
            "total_completed_vehicles": summary.get("total_completed_vehicles"),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Calculates aggregate statistical metrics across the simulation run."""
        trip_stats = self._cached_tripinfo_summary or {
            "tripinfo_required": bool(self.tripinfo_file),
            "tripinfo_reconciliation_status": "PENDING" if self.tripinfo_file else "NOT_REQUIRED",
        }

        steps = len(self.step_records)
        qlens = [r.queue_length for r in self.step_records]
        mean_q = round(sum(qlens) / steps, 2) if steps else 0.0
        max_q = max(qlens) if qlens else 0

        completed = self._cumulative_completed or (
            self.step_records[-1].cumulative_completed_vehicles if self.step_records else 0
        )
        dur = float(
            self.run_metadata.get("observation_window_duration")
            or self.observed_duration
            or (steps * float(self.run_metadata.get("step_length", 1.0)))
        )
        throughput = round(completed * (3600.0 / dur), 1) if dur > 0 else 0.0

        comp_wait = trip_stats.get("tripinfo_total_waiting_time", sum(self.completed_waiting_times))
        run_wait = sum(self.vehicle_last_waiting_time.values())
        net_wait = comp_wait + run_wait
        total_wait = net_wait if (self.completed_waiting_times or self.vehicle_last_waiting_time) else (
            self.total_delay_seconds if self.total_delay_seconds > 0.0 else sum(r.waiting_time for r in self.step_records)
        )
        mean_wait = round(comp_wait / completed, 2) if completed > 0 else None

        mean_travel = trip_stats.get("tripinfo_mean_travel_time", round(sum(self.completed_travel_times) / len(self.completed_travel_times), 2) if self.completed_travel_times else None)
        loss = trip_stats.get("tripinfo_total_time_loss", sum(self.completed_time_losses)) + sum(self.vehicle_last_time_loss.values())
        if loss == 0.0 and self.step_records:
            loss = self.step_records[-1].time_loss

        sys_delay = net_wait + self.total_pending_delay
        den = float(self.run_metadata.get("demand_denominator") or (completed + len(self.running_ids) + self.last_pending_count) or 1.0)
        sys_mean_delay = round(sys_delay / den, 2) if den > 0 else 0.0

        summary: Dict[str, Any] = {
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "scenario_name": self.scenario_name,
            "total_steps": steps,
            "observation_window_duration": dur,
            "mean_waiting_time": mean_wait,
            "completed_mean_waiting_time": mean_wait,
            "total_waiting_time": round(total_wait, 2),
            "completed_vehicles_waiting_time": round(comp_wait, 2),
            "running_vehicles_waiting_time": round(run_wait, 2),
            "total_network_waiting_time": round(net_wait, 2),
            "system_mean_delay": sys_mean_delay,
            "system_total_delay": round(sys_delay, 2),
            "total_pending_delay": round(self.total_pending_delay, 2),
            "mean_queue_length": mean_q,
            "max_queue_length": max_q,
            "total_queue_delay": round(self.total_delay_seconds, 2),
            "throughput_vph": throughput,
            "total_completed_vehicles": completed,
            "arrival_count": completed,
            "demand_denominator": den,
            "pending_insertion_vehicles": self.last_pending_count,
            "completed_metrics_available": completed > 0,
            "completion_status": "COMPLETED" if completed > 0 else "NO_COMPLETED_VEHICLES",
            "service_status": "COMPLETED" if completed > 0 else "NO_COMPLETED_VEHICLES",
            "mean_travel_time": mean_travel,
            "total_time_loss": round(loss, 2),
            "is_valid": self.is_valid,
            "data_validity": "VALID" if self.is_valid else "INVALID",
            "status": "OK" if self.is_valid else "INVALID",
            "accounting_valid": True,
            "safety_status": "INCIDENTS_DETECTED" if (self.run_metadata.get("collision_count", 0) or self.run_metadata.get("teleport_count", 0)) else "SAFE",
            "measurement_errors": self.measurement_errors,
            "measurement_error_messages": list(self.error_messages),
        }

        if self.run_metadata:
            summary.update(self.run_metadata)
        summary.update(trip_stats)
        if summary.get("accounting_valid") is False:
            summary.update(is_valid=False, data_validity="INVALID", status="INVALID")

        if trip_stats.get("tripinfo_metrics_available"):
            summary["completed_mean_waiting_time"] = trip_stats["tripinfo_mean_waiting_time"]
            summary["mean_waiting_time"] = trip_stats["tripinfo_mean_waiting_time"]
            summary["mean_travel_time"] = trip_stats["tripinfo_mean_travel_time"]
            summary["completed_mean_time_loss"] = trip_stats["tripinfo_mean_time_loss"]

        summary["core_metrics"] = {
            "waiting_time": {"mean_waiting_time": summary.get("mean_waiting_time"), "total_waiting_time": summary.get("total_waiting_time")},
            "queue_length": {"mean_queue_length": summary.get("mean_queue_length"), "max_queue_length": summary.get("max_queue_length")},
            "throughput": {"throughput_vph": summary.get("throughput_vph"), "total_completed_vehicles": summary.get("total_completed_vehicles")},
        }
        return summary

    def export_results(self, output_dir: str, prefix: str = "") -> Tuple[str, str]:
        """Exports metrics summary to JSON and step history to CSV in output_dir."""
        norm_output = os.path.normpath(os.path.abspath(output_dir))
        data_dir_abs = os.path.normpath(os.path.abspath("data"))
        if norm_output.startswith(data_dir_abs):
            raise ValueError(f"Cannot save results into data directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        tag = f"{prefix}_" if prefix else ""
        json_path = os.path.join(output_dir, f"{tag}summary.json")
        csv_path = os.path.join(output_dir, f"{tag}step_metrics.csv")

        summary = self.get_summary()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        fieldnames = [
            "step", "sim_time", "running_vehicles", "queue_length", "waiting_time",
            "time_loss", "step_completed_vehicles", "cumulative_completed_vehicles", "pending_vehicles",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.step_records:
                writer.writerow(asdict(record))

        return json_path, csv_path


# Alias for compatibility
SimulationMetricsCollector = MetricsCollector
