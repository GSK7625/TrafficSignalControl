"""Simulation metrics collection and result export module."""

import csv
from dataclasses import asdict, dataclass
import json
import os
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


METRIC_SCHEMA_VERSION = "4.0.0"


def parse_tripinfo_output(xml_path: str) -> Dict[str, Any]:
    """Parses SUMO native --tripinfo-output XML file for ground-truth statistics."""
    if not os.path.isfile(xml_path):
        return {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if root.tag != "tripinfos":
        raise ValueError("Expected tripinfos root")
    all_trips = root.findall("tripinfo")
    trips = [t for t in all_trips if float(t.get("arrival", "0")) >= 0 and t.get("vaporized", "") not in ("true", "1")]
    ids = [t.get("id") for t in trips]
    if len(set(ids)) != len(ids) or any(not vid for vid in ids):
        raise ValueError("Missing or duplicate tripinfo IDs")
    for trip in all_trips:
        for field in ("duration", "waitingTime", "timeLoss"):
            value = float(trip.attrib[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid tripinfo {field}")
    if not trips:
        return {
            "tripinfo_completed_vehicles": 0,
            "tripinfo_completed_ids": [],
            "tripinfo_unfinished_vehicles": len(all_trips),
            "tripinfo_mean_travel_time": 0.0,
            "tripinfo_mean_waiting_time": 0.0,
            "tripinfo_total_waiting_time": 0.0,
            "tripinfo_mean_time_loss": 0.0,
            "tripinfo_total_time_loss": 0.0,
            "tripinfo_metrics_available": False,
        }

    durations = [float(t.get("duration", 0.0)) for t in trips]
    waiting_times = [float(t.get("waitingTime", 0.0)) for t in trips]
    time_losses = [float(t.get("timeLoss", 0.0)) for t in trips]
    count = len(trips)

    return {
        "tripinfo_completed_vehicles": count,
        "tripinfo_completed_ids": ids,
        "tripinfo_unfinished_vehicles": len(all_trips) - count,
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
    """Builds unified vehicle accounting metadata for an observation window [window_begin, window_end).

    Ensures CLI runner, benchmark runner, and RL environment share identical accounting,
    observation window demand, and system_mean_delay denominator definitions.
    """
    demand_meta = {}
    demand_error = None
    try:
        from src.data_inspector import get_window_demand
        demand_meta = get_window_demand(
            scenario_info if scenario_info is not None else scenario_name,
            window_begin=window_begin,
            window_end=window_end,
            data_dir=data_dir,
        )
        scheduled_in_window = demand_meta["scheduled_demand_in_window"]
        counting_mode = demand_meta["demand_counting_mode"]
        total_route_demand = demand_meta["total_route_demand"]
    except Exception as exc:
        demand_error = str(exc)
        scheduled_in_window = None
        counting_mode = "unknown"
        total_route_demand = getattr(scenario_info, "trips_or_vehicles_count", 0) if scenario_info else 0

    realized_demand = total_completed + running_vehs + pending_vehs
    accounting_valid = population is not None and population.get("available", False)
    accounting_errors = []
    if population is not None:
        states = population["states"]
        flat = [vid for ids in states.values() for vid in ids]
        if len(flat) != len(set(flat)):
            accounting_valid = False
            accounting_errors.append("Overlapping population IDs")
        scheduled_ids = demand_meta.get("scheduled_vehicle_ids")
        realized_demand = len(set(flat))
        if scheduled_ids is not None:
            accounted = set(flat) & set(scheduled_ids)
            realized_demand = len(accounted)
            unexpected = sorted(set(flat) - set(scheduled_ids))
            if unexpected:
                accounting_valid = False
                accounting_errors.append(f"IDs outside scheduled window; initial cohort unavailable: {unexpected}")
            missing = sorted(set(scheduled_ids) - accounted)
            if missing:
                accounting_valid = False
                accounting_errors.append(f"Unaccounted scheduled IDs: {missing}")
        accounting_errors.extend(population.get("errors", []))
    else:
        accounting_errors.append("ID population unavailable; counts only")
    if counting_mode != "exact":
        accounting_valid = False
        accounting_errors.append("Scheduled demand not exact")
    if demand_error:
        accounting_errors.append(demand_error)

    if scheduled_in_window is not None and scheduled_in_window > 0:
        demand_denominator = float(scheduled_in_window)
        denominator_source = "scheduled_in_window"
    elif realized_demand > 0:
        demand_denominator = float(realized_demand)
        denominator_source = "realized_in_window"
    else:
        demand_denominator = 0.0
        denominator_source = "none"

    conservation_error = (
        scheduled_in_window - realized_demand
        if scheduled_in_window is not None
        else None
    )

    w_b = float(window_begin)
    w_e = float(window_end)
    w_dur = max(0.0, w_e - w_b)

    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "observation_window_begin": w_b,
        "observation_window_end": w_e,
        "observation_window_duration": w_dur,
        "total_route_demand": total_route_demand,
        "total_scenario_demand": total_route_demand,  # compatibility alias
        "scheduled_demand_in_window": scheduled_in_window,
        "demand_counting_mode": counting_mode,
        "realized_demand_in_window": realized_demand,
        "running_vehicles_at_end": running_vehs,
        "pending_insertion_vehicles": pending_vehs,
        "teleport_count": total_teleports,
        "collision_count": total_collisions,
        "demand_denominator": demand_denominator,
        "demand_denominator_source": denominator_source,
        "conservation_error": conservation_error,
        "accounting_valid": accounting_valid and conservation_error == 0,
        "accounting_errors": accounting_errors,
        "population": population,
        "scheduled_before_window": demand_meta.get("scheduled_before_window"),
        "teleport_event_count": total_teleports,
        "incomplete_vehicles": running_vehs + pending_vehs + (len(population["states"]["teleport_buffer"]) if population else 0),
        "removed_population_available": population.get("removed_available", False) if population else False,
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
    pending_vehicles: Optional[int] = 0


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
        self.completed_ids = set()
        self.departed_ids = set()
        self.running_ids = set()
        self.pending_ids = set()
        self.teleport_buffer_ids = set()
        self.removed_ids = set()
        self.population_errors = []
        self.observed_duration = 0.0
        self.network_stopped_waiting = 0.0
        self.vehicle_integrated_wait = {}
        self._reconciled = False

    def population_snapshot(self) -> Dict[str, Any]:
        states = {"completed": sorted(self.completed_ids), "running": sorted(self.running_ids),
                  "pending": sorted(self.pending_ids), "teleport_buffer": sorted(self.teleport_buffer_ids),
                  "removed": sorted(self.removed_ids)}
        known = set().union(*(set(ids) for ids in states.values()))
        unknown = sorted(self.departed_ids - known)
        errors = list(self.population_errors)
        if unknown:
            errors.append(f"Unclassified departed IDs (removal API unavailable): {unknown}")
        flat = [vid for ids in states.values() for vid in ids]
        if len(flat) != len(set(flat)):
            errors.append("Overlapping final population states")
        return {"states": states, "unclassified_ids": unknown, "available": not errors,
                "removed_available": False, "errors": errors,
                "loaded_before_window": None,
                "loaded_before_window_reason": "No initial loaded-ID snapshot; loaded is not scheduled demand"}

    def reconcile_tripinfo(self) -> Dict[str, Any]:
        """Run once after XML flush; never treat merging one source as reconciliation."""
        if self._reconciled:
            return self._cached_tripinfo_summary or {}
        self._reconciled = True
        result = {"tripinfo_required": bool(self.tripinfo_file),
                  "tripinfo_reconciliation_status": "NOT_REQUIRED"}
        if self.tripinfo_file:
            try:
                if not os.path.isfile(self.tripinfo_file):
                    raise ValueError("Required tripinfo XML missing")
                stats = parse_tripinfo_output(self.tripinfo_file)
                result.update(stats)
                mismatches = []
                n = self._cumulative_completed
                dt = float(self.run_metadata.get("step_length", 1))
                tolerances = {"count": 0, "ids": 0, "waiting_total_seconds": n*dt+.01,
                              "time_loss_total_seconds": n*dt+.01, "travel_mean_seconds": dt+.01}
                result["reconciliation_tolerances"] = tolerances
                if stats["tripinfo_completed_vehicles"] != n:
                    mismatches.append(f"completed count: collector={n}, XML={stats['tripinfo_completed_vehicles']}, tolerance=0")
                if self.completed_ids and set(stats["tripinfo_completed_ids"]) != self.completed_ids:
                    mismatches.append("completed IDs differ (tolerance=0)")
                for values, field, tolerance in (
                    (self.completed_waiting_times, "tripinfo_total_waiting_time", tolerances["waiting_total_seconds"]),
                    (self.completed_time_losses, "tripinfo_total_time_loss", tolerances["time_loss_total_seconds"])):
                    if len(values) != n or abs(sum(values)-stats[field]) > tolerance:
                        mismatches.append(f"{field}: collector={sum(values)}, XML={stats[field]}, tolerance={tolerance}")
                if n:
                    travel = sum(self.completed_travel_times)/n if len(self.completed_travel_times) == n else None
                    if travel is None or stats["tripinfo_mean_travel_time"] is None or abs(travel-stats["tripinfo_mean_travel_time"]) > tolerances["travel_mean_seconds"]:
                        mismatches.append(f"completed travel mean: collector={travel}, XML={stats['tripinfo_mean_travel_time']}, tolerance={tolerances['travel_mean_seconds']}")
                result["tripinfo_reconciliation_status"] = "MISMATCH" if mismatches else "MATCH"
                for message in mismatches:
                    self.record_error(message)
            except (OSError, ET.ParseError, ValueError, KeyError, TypeError) as exc:
                result["tripinfo_reconciliation_status"] = "INVALID_XML"
                self.record_error(f"Tripinfo reconciliation failed: {exc}")
        self._cached_tripinfo_summary = result
        return result

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
        self.observed_duration += step_length
        # 1. Track newly departed vehicles
        try:
            departed_ids = traci_module.simulation.getDepartedIDList()
            self.departed_ids.update(departed_ids)
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

        arrived_ids = list(set(arrived_ids) - self.completed_ids)
        self.completed_ids.update(arrived_ids)
        try:
            starts = set(traci_module.simulation.getStartingTeleportIDList())
            ends = set(traci_module.simulation.getEndingTeleportIDList())
            self.teleport_buffer_ids.update(starts)
            self.teleport_buffer_ids.difference_update(ends)
            self.teleport_buffer_ids.difference_update(self.completed_ids)
        except Exception as exc:
            self.population_errors.append(f"Teleport ID telemetry unavailable: {exc}")
            self.record_error(self.population_errors[-1])
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
            wait = self.vehicle_integrated_wait.get(vid, self.vehicle_last_waiting_time.get(vid, 0.0))
            self.vehicle_last_waiting_time.pop(vid, None)
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
            self.running_ids = set(running_vehs)
            self.teleport_buffer_ids.difference_update(self.running_ids)
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

                try:
                    stopped = traci_module.vehicle.getWaitingTime(vid) > 0
                    delta = step_length if stopped else 0.0
                    self.network_stopped_waiting += delta
                    self.vehicle_integrated_wait[vid] = self.vehicle_integrated_wait.get(vid, 0.0) + delta
                except Exception as exc:
                    self.record_error(f"Network stopped waiting unavailable for {vid}: {exc}")
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
        pending_count: Optional[int] = None
        try:
            pending_ids = traci_module.simulation.getPendingVehicles()
            if pending_ids is None:
                raise ValueError("Pending IDs unavailable")
            self.pending_ids = set(pending_ids)
            pending_count = len(self.pending_ids)
        except Exception as e:
            self.measurement_errors += 1
            self.is_valid = False
            self.error_messages.append(f"Step {step} simulation getPendingVehicles read error: {e}")
            pending_count = None
            self.population_errors.append(f"Pending IDs unavailable: {e}")

        if pending_count is not None:
            self.pending_vehicles_history.append(pending_count)
            self.total_pending_delay += pending_count * step_length
        else:
            self.pending_vehicles_history.append(-1)

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
        trip_stats = self.reconcile_tripinfo() if self._reconciled else {"tripinfo_required": bool(self.tripinfo_file), "tripinfo_reconciliation_status": "NOT_REQUIRED"}
        # Summaries with a configured file are requested only after close by production exporters.
        if self.tripinfo_file and not self._reconciled:
            trip_stats = self.reconcile_tripinfo()
        total_steps = len(self.step_records)
        if total_steps == 0:
            return {
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "scenario_name": self.scenario_name,
                "total_steps": 0,
                "total_completed_vehicles": 0,
                "arrival_count": 0,
                "throughput_vph": 0.0,
                "mean_queue_length": 0.0,
                "max_queue_length": 0,
                "mean_waiting_time": None,
                "completed_mean_waiting_time": None,
                "completed_metrics_available": False,
                "completion_status": "NO_COMPLETED_VEHICLES",
                "system_mean_delay": 0.0,
                "system_total_delay": 0.0,
                "pending_insertion_vehicles": 0,
                "total_pending_delay": 0.0,
                "completed_vehicles_waiting_time": 0.0,
                "running_vehicles_waiting_time": 0.0,
                "total_network_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "total_queue_delay": 0.0,
                "mean_travel_time": None,
                "total_time_loss": 0.0,
                "observation_window_duration": 0.0,
                "data_validity": "INVALID",
                "measurement_error_messages": list(self.error_messages),
                **trip_stats,
                "is_valid": self.is_valid and (self.run_metadata.get("accounting_valid", True) is True),
                "measurement_errors": self.measurement_errors,
                "status": "OK" if self.is_valid and (self.run_metadata.get("accounting_valid", True) is True) else "INVALID",
            }

        queue_lengths = [r.queue_length for r in self.step_records]

        total_completed = (
            self._cumulative_completed
            if self._cumulative_completed > 0
            else (self.step_records[-1].cumulative_completed_vehicles if self.step_records else 0)
        )

        completed_metrics_available = (total_completed > 0)
        completion_status = "COMPLETED" if completed_metrics_available else "NO_COMPLETED_VEHICLES"

        mean_travel = (
            round(sum(self.completed_travel_times) / len(self.completed_travel_times), 2)
            if self.completed_travel_times and completed_metrics_available
            else None
        )

        # Aggregate total time loss: completed vehicles + remaining running vehicles
        total_loss = sum(self.completed_time_losses) + sum(self.vehicle_last_time_loss.values())
        if total_loss == 0.0 and self.step_records:
            total_loss = self.step_records[-1].time_loss

        # Metric Definitions & Accounting:
        # - completed_mean_wait (mean_waiting_time): average stopped time (speed < 0.1 m/s)
        #   for completed vehicles that exited the network. None if 0 completed vehicles.
        # - system_total_delay: total stopped delay inside network (completed + running) plus
        #   total pending delay (delay of vehicles queued to enter the network).
        # - system_mean_delay: system_total_delay averaged across demand denominator
        #   (scheduled_demand_in_window if known, else realized vehicles in window).
        completed_wait = sum(self.completed_waiting_times)
        running_wait = (sum(self.vehicle_integrated_wait.get(vid, 0.0) for vid in self.running_ids)
                        if self.observed_duration else sum(self.vehicle_last_waiting_time.values()))
        total_network_wait = completed_wait + running_wait

        if self.observed_duration > 0:
            total_network_wait = self.network_stopped_waiting
        system_total_delay = total_network_wait + self.total_pending_delay
        pending_end = self.pending_vehicles_history[-1] if self.pending_vehicles_history else 0
        pending_end_val = max(0, pending_end) if pending_end is not None else 0
        realized_vehs = total_completed + len(self.vehicle_last_waiting_time) + pending_end_val

        # Denominator contract for system_mean_delay:
        demand_den = self.run_metadata.get("demand_denominator")
        if demand_den is None:
            sched = self.run_metadata.get("scheduled_demand_in_window")
            if sched is not None and sched > 0:
                demand_den = float(sched)
            elif realized_vehs > 0:
                demand_den = float(realized_vehs)
            else:
                demand_den = 1.0

        system_mean_delay = (system_total_delay / demand_den) if demand_den > 0 else 0.0

        if completed_metrics_available:
            completed_mean_wait = round(completed_wait / total_completed, 2)
            mean_wait = completed_mean_wait
        else:
            completed_mean_wait = None
            mean_wait = None

        # Total waiting time / delay
        waiting_times = [r.waiting_time for r in self.step_records]
        if self.completed_waiting_times or self.vehicle_last_waiting_time:
            total_wait = total_network_wait
        elif self.total_delay_seconds > 0.0:
            total_wait = self.total_delay_seconds
        else:
            total_wait = sum(waiting_times)

        # Observation window duration & throughput
        window_duration = self.run_metadata.get("observation_window_duration")
        if window_duration is None or window_duration <= 0:
            window_duration = float(total_steps) * float(self.run_metadata.get("step_length", 1.0))
        if self.observed_duration > 0 and not self.run_metadata.get("observation_window_duration"):
            window_duration = self.observed_duration
        window_duration = float(window_duration)
        throughput_vph = round(total_completed * (3600.0 / window_duration), 1) if window_duration > 0 else 0.0

        summary = {
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "scenario_name": self.scenario_name,
            "total_steps": total_steps,
            "total_completed_vehicles": total_completed,
            "arrival_count": total_completed,
            "throughput_vph": throughput_vph,
            "mean_queue_length": round(sum(queue_lengths) / total_steps, 2),
            "max_queue_length": max(queue_lengths) if queue_lengths else 0,
            "mean_waiting_time": mean_wait,
            "completed_mean_waiting_time": completed_mean_wait,
            "completed_metrics_available": completed_metrics_available,
            "completion_status": completion_status,
            "system_mean_delay": round(system_mean_delay, 2),
            "system_total_delay": round(system_total_delay, 2),
            "demand_denominator": demand_den,
            "pending_insertion_vehicles": pending_end_val,
            "total_pending_delay": round(self.total_pending_delay, 2),
            "completed_vehicles_waiting_time": round(completed_wait, 2),
            "running_vehicles_waiting_time": round(running_wait, 2),
            "total_network_waiting_time": round(total_network_wait, 2),
            "total_waiting_time": round(total_wait, 2),
            "total_queue_delay": round(self.total_delay_seconds, 2),
            "mean_travel_time": mean_travel,
            "total_time_loss": round(total_loss, 2),
            "is_valid": self.is_valid and self.run_metadata.get("accounting_valid") is True,
            "measurement_errors": self.measurement_errors,
            "status": "OK" if self.is_valid and self.run_metadata.get("accounting_valid") is True else "INVALID",
        }

        # Merge run metadata if configured
        if self.run_metadata:
            summary.update(self.run_metadata)

        summary.update(trip_stats)
        if trip_stats.get("tripinfo_reconciliation_status") == "MATCH":
            summary["completed_mean_waiting_time"] = trip_stats["tripinfo_mean_waiting_time"]
            summary["mean_waiting_time"] = trip_stats["tripinfo_mean_waiting_time"]
            summary["mean_travel_time"] = trip_stats["tripinfo_mean_travel_time"]
            summary["completed_mean_time_loss"] = trip_stats["tripinfo_mean_time_loss"]
        summary["observation_window_duration"] = window_duration
        summary["pending_insertion_vehicles"] = None if pending_end == -1 else pending_end_val
        summary["metric_schema_version"] = METRIC_SCHEMA_VERSION
        summary["measurement_errors"] = self.measurement_errors
        summary["measurement_error_messages"] = list(self.error_messages)
        summary["is_valid"] = self.is_valid and (self.run_metadata.get("accounting_valid", True) is True)
        summary["data_validity"] = "VALID" if summary["is_valid"] else "INVALID"
        summary["status"] = "OK" if summary["is_valid"] else "INVALID"
        summary["safety_status"] = "INCIDENTS_DETECTED" if summary.get("collision_count", 0) or summary.get("teleport_count", 0) else "SAFE"
        summary["service_status"] = completion_status
        summary["completed_count_unit"] = "veh"
        summary["throughput_rate_unit"] = "veh/h"
        summary["network_waiting_source"] = "integrated_network_stopped_waiting" if self.observed_duration else "legacy_manual_records"

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
