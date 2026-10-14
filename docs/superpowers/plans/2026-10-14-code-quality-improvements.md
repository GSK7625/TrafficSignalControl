# Code Quality Improvements & Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up unused imports, build a comprehensive and robust automated test suite (`pytest`) covering all core components of the 3-algorithm ATSC platform, and synchronize documentation and docstrings.

**Architecture:** 
- Modular refactoring of unused imports across 13 source files.
- Layered test suite in `tests/` isolating unit logic (`phase_manager`, `controllers`, `rl_components`, `config`, `metrics`) from TraCI live dependencies using mock TraCI and pure state objects.
- Documentation synchronization in `README.md` and outdated docstrings.

**Tech Stack:** Python 3.11+, pytest, dataclasses, numpy, PyYAML.

## Global Constraints
- Python 3.11+ compatibility.
- Pure Python unit tests must run without requiring an active SUMO GUI or external graphical display.
- Maintain existing public API signatures for all controllers and managers.

## Review Focus
1. `SafePhaseManager` yellow clearance must never skip intermediate yellow when transitioning green links to red.
2. `MaxPressureController` must not divide by zero if an incoming lane has no outgoing links.
3. `FixedTimeController` must strictly respect `effective_dur = base_dur + offset_dur` for offset synchronization.
4. `QLearningController` save and load must correctly deserialize tuple keys and restore discretizer settings.
5. `MetricsCollector` must safely handle zero completed vehicles and calculate non-negative averages.

---

### Task 1: Clean Up Unused Imports and Fix Outdated Docstrings

**Files:**
- Modify: `src/config.py`
- Modify: `src/controllers/base.py`
- Modify: `src/controllers/q_learning.py`
- Modify: `src/data_inspector.py`
- Modify: `src/evaluation/compare_algorithms.py`
- Modify: `src/evaluation/experiment_runner.py`
- Modify: `src/evaluation/plot_comparison.py`
- Modify: `src/experiments/plot_metrics.py`
- Modify: `src/experiments/plot_rl.py`
- Modify: `src/phase_manager.py`
- Modify: `src/rl/discretization.py`
- Modify: `src/rl/rewards.py`
- Modify: `src/runner.py`

**Interfaces:**
- Consumes: Existing module definitions
- Produces: Clean imports without unused variables/modules, updated docstrings for 3 algorithms.

- [ ] **Step 1: Remove unused imports in `src/config.py`, `src/controllers/base.py`, `src/controllers/q_learning.py`, `src/data_inspector.py`**
- [ ] **Step 2: Remove unused imports in `src/evaluation/*.py` and fix outdated docstrings referencing 4 algorithms**
- [ ] **Step 3: Remove unused imports in `src/experiments/*.py` and fix docstrings referencing 4 algorithms**
- [ ] **Step 4: Remove unused imports in `src/phase_manager.py`, `src/rl/*.py`, `src/runner.py`**
- [ ] **Step 5: Verify syntax compilation across all files**

Run: `.venv\Scripts\python.exe -m py_compile main.py src/config.py src/runner.py ...`

---

### Task 2: Build Unit Test Suite for SafePhaseManager and Controllers

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_phase_manager.py`
- Create: `tests/test_controllers.py`

**Interfaces:**
- Consumes: `SafePhaseManager`, `PhaseState`, `build_yellow_state`, `FixedTimeController`, `MaxPressureController`, `QLearningController`
- Produces: Tests covering phase transitions, min_green constraints, all-red clearance, pressure math, and Q-table Bellman updates.

- [ ] **Step 1: Write `tests/test_phase_manager.py`**
  - Test yellow state construction (`build_yellow_state`).
  - Test min green rejection in `request_phase()`.
  - Test transition sequence: GREEN -> YELLOW -> ALL_RED -> GREEN.
  - Test external synchronization (`sync_external_phase`).
- [ ] **Step 2: Write `tests/test_controllers.py`**
  - Test `FixedTimeController` cycle progression, custom phase sequence, and offset handling.
  - Test `MaxPressureController` pressure computation, queue-weighting, and tie-breaking.
  - Test `QLearningController` action selection, epsilon-decay, Bellman update, and save/load serialization.
- [ ] **Step 3: Run pytest on phase manager and controllers**

Run: `.venv\Scripts\pytest.exe tests/test_phase_manager.py tests/test_controllers.py -v`

---

### Task 3: Build Unit Test Suite for RL Components, Metrics & Config

**Files:**
- Create: `tests/test_rl_components.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_config_and_inspector.py`

**Interfaces:**
- Consumes: `StateDiscretizer`, `ObservationBuilder`, `NegativeAverageQueueReward`, `MetricsCollector`, `parse_tripinfo_output`, `SimulationConfig`, `inspect_scenario`
- Produces: Tests covering observation normalization, state binning, reward accumulation, summary statistics, and config validation.

- [ ] **Step 1: Write `tests/test_rl_components.py`**
  - Test `StateDiscretizer` binning logic (`serving_q`, `waiting_q`, `time_bin`).
  - Test `ObservationBuilder` vector generation and normalization bounds.
  - Test `NegativeAverageQueueReward` calculation.
- [ ] **Step 2: Write `tests/test_metrics.py`**
  - Test `MetricsCollector` step recording with mock TraCI.
  - Test summary calculation (mean queue, delay, travel time).
  - Test `parse_tripinfo_output` with sample XML data.
- [ ] **Step 3: Write `tests/test_config_and_inspector.py`**
  - Test `SimulationConfig` validation checks (negative values, missing datasets).
  - Test `inspect_scenario` parsing of `.net.xml` and `.rou.xml`.
- [ ] **Step 4: Run pytest across the full test suite**

Run: `.venv\Scripts\pytest.exe tests/ -v`

---

### Task 4: Documentation & Test Badge Synchronization

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Actual passed test count from Task 3
- Produces: Accurate test badge in README and clear training instructions before evaluation.

- [ ] **Step 1: Update test badge and section 10 in `README.md` to match the exact test count**
- [ ] **Step 2: Clarify Q-Learning evaluation requirement (training model first or using `--allow-untrained`)**
- [ ] **Step 3: Final verification run of `pytest` and CLI inspect commands**
