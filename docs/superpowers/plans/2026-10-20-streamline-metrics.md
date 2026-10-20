# Streamline Metrics Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Streamline `src/metrics.py` (Option 2) by eliminating redundant schema keys/aliases, replacing unbounded in-memory list tracking with scalar accumulators, modularizing `get_summary()`, and updating tests.

**Architecture:** Decompose `MetricsCollector` state and aggregation logic into clean scalar accumulators and helper calculation methods. Remove dead/alias keys (`arrival_count`, `completed_mean_waiting_time`, `data_validity`, `total_network_waiting_time`, etc.) from output summary dictionary while preserving all core traffic signals metrics and external consumer interfaces. Bump `METRIC_SCHEMA_VERSION` to `7.0.0`.

**Tech Stack:** Python 3.11, TraCI / SUMO XML, pytest.

**Spec:** User selection of Option 2: Aggressive cleanup of `src/metrics.py` schema, tracking logic, and summary dictionary.

## Global Constraints

- Preserve all 3 Core Traffic Signal Control Metrics: Waiting Time, Queue Length, Throughput.
- Preserve canonical aggregation constants: `BENCHMARK_METRICS`, `COMPARISON_METRICS`, `BENCHMARK_COMPLETION_THRESHOLD`, `REPORT_COMPLETION_THRESHOLD`.
- Keep existing signatures and return types for `parse_tripinfo_output`, `build_vehicle_accounting`, `completion_statistics`, `service_status`, `export_results`.
- Maintain full compatibility with `runner.py`, `experiment_runner.py`, `compare_algorithms.py`, and `plot_comparison.py`.
- All pytest tests in the test suite must pass cleanly.

## Review Focus

1. `reconcile_tripinfo` must correctly match or report mismatch against `collector.completed_ids`.
2. Vehicle accounting errors must invalidate the summary (`is_valid=False`, `status="INVALID"`).
3. Zero-demand scenarios must produce `service_status="NO_DEMAND"` and `completion_status="NO_DEMAND"`.
4. Online tracking without tripinfo must still produce valid travel time, waiting time, and time loss averages.
5. `export_results` must still write valid `summary.json` and `step_metrics.csv` avoiding writes to the `data/` directory.

---

### Task 1: Refactor `MetricsCollector` Online Tracking & In-Memory State

**Files:**
- Modify: `src/metrics.py:163-275`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: TraCI API during simulation step (`simulation.getDepartedIDList`, `simulation.getArrivedIDList`, `vehicle.getIDList`, `simulation.getPendingVehicles`, `lane.getLastStepHaltingNumber`, `vehicle.getAccumulatedWaitingTime`, `vehicle.getTimeLoss`).
- Produces: `record_step(...) -> StepMetrics`, tracking cumulative completed vehicles, cumulative travel time, waiting time, and time loss scalars.

- [x] **Step 1: Write unit test verifying scalar accumulator calculations in `tests/test_metrics.py`**

Add a test verifying that online step recording without TripInfo produces exact non-zero averages for mean travel time and waiting time from scalar tracking:

```python
def test_online_tracking_accumulates_scalar_travel_and_waiting(mock_traci):
    collector = MetricsCollector("test_scen")
    mock_traci.simulation.getDepartedIDList.return_value = ["v1"]
    mock_traci.vehicle.getDeparture.return_value = 0.0
    mock_traci.vehicle.getIDList.return_value = ["v1"]
    mock_traci.vehicle.getAccumulatedWaitingTime.return_value = 5.0
    mock_traci.vehicle.getTimeLoss.return_value = 2.0
    mock_traci.simulation.getPendingVehicles.return_value = []
    mock_traci.lane.getLastStepHaltingNumber.return_value = 0

    collector.record_step(step=1, sim_time=1.0, controlled_lanes=[], traci_module=mock_traci)
    mock_traci.simulation.getDepartedIDList.return_value = []
    mock_traci.simulation.getArrivedIDList.return_value = ["v1"]
    mock_traci.vehicle.getIDList.return_value = []

    collector.record_step(step=2, sim_time=10.0, controlled_lanes=[], traci_module=mock_traci)
    summary = collector.get_summary()
    assert summary["total_completed_vehicles"] == 1
    assert summary["mean_travel_time"] == 10.0
    assert summary["mean_waiting_time"] == 5.0
```

- [x] **Step 2: Run test to check current behavior**

Run: `pytest tests/test_metrics.py -k test_online_tracking_accumulates_scalar_travel_and_waiting -v`
Expected: PASS (or FAIL if test setup needs adjustment).

- [x] **Step 3: Replace list attributes with scalar accumulators in `MetricsCollector.__init__` and `record_step`**

In `src/metrics.py`:
- Replace `self.completed_travel_times: List[float]`, `self.completed_waiting_times: List[float]`, `self.completed_time_losses: List[float]` with:
  ```python
  self.total_completed_travel_time: float = 0.0
  self.total_completed_travel_count: int = 0
  self.total_completed_waiting_time: float = 0.0
  self.total_completed_time_loss: float = 0.0
  ```
- In `record_step()`, update these scalar counters upon vehicle arrival instead of appending to unbounded lists.
- Pop arriving vehicle entries from `vehicle_departure_times`, `vehicle_last_waiting_time`, and `vehicle_last_time_loss`.

- [x] **Step 4: Run tests to verify**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "refactor(metrics): replace growing lists with scalar accumulators in MetricsCollector"
```

---

### Task 2: Streamline Summary Generation & Eliminate Redundant Schema Keys

**Files:**
- Modify: `src/metrics.py:19, 277-386`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: Internal accumulators, `self.run_metadata`, `self._cached_tripinfo_summary`.
- Produces: `get_summary() -> Dict[str, Any]` and `get_core_metrics() -> Dict[str, Any]` with clean schema.

- [x] **Step 1: Check existing assertions on redundant schema keys**

Verify that `tests/test_metrics.py:305` asserts `assert summary["data_validity"] == "INVALID"`.
Update `tests/test_metrics.py` to assert `assert summary["status"] == "INVALID"` and `assert summary["is_valid"] is False`.

- [x] **Step 2: Run tests to ensure test readiness**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 3: Update `src/metrics.py` schema version and modularize `get_summary()`**

1. Set `METRIC_SCHEMA_VERSION = "7.0.0"`.
2. In `parse_tripinfo_output()`, remove redundant unused `"tripinfo_unfinished_vehicles": 0`.
3. In `MetricsCollector`, break `get_summary()` into concise calculation sections:
   - Queue metrics: `mean_queue_length`, `max_queue_length`.
   - Waiting metrics: `mean_waiting_time`, `total_waiting_time`, `system_mean_delay`, `system_total_delay`.
   - Throughput & completion metrics: `throughput_vph`, `total_completed_vehicles`, `completion_rate_pct`.
   - Incident & validity metrics: `safety_status`, `is_valid`, `status`, `accounting_valid`.
4. Remove redundant keys from summary:
   - `completed_mean_waiting_time`
   - `arrival_count`
   - `completed_vehicles_waiting_time`
   - `running_vehicles_waiting_time`
   - `total_network_waiting_time`
   - `total_queue_delay`
   - `completed_mean_time_loss`
   - `data_validity`
5. Simplify `get_core_metrics()` to directly return the 6 core metrics as a flat dict without nested looping.

- [x] **Step 4: Run tests to verify**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "refactor(metrics): streamline summary schema and modularize get_summary"
```

---

### Task 3: Full Project Regression Verification

**Files:**
- Test: Entire test suite (`tests/`)

- [x] **Step 1: Run full pytest suite**

Run: `pytest -v`
Expected: 324/324 tests PASS (or 325 with new test).

- [x] **Step 2: Run git diff and verify lines reduced**

Run: `git diff --stat`
Expected: Net reduction in line count and cognitive complexity in `src/metrics.py`.

- [x] **Step 3: Commit and finalize**

```bash
git add .
git commit -m "chore(metrics): finalize metrics streamlining and verify regression suite"
```
