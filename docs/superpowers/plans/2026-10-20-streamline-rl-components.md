# Streamline RL Components Implementation Plan (Items 20-26)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamline reinforcement learning components across state discretization, reward calculation, topology caching, action masking, environment API, and training/evaluation loops (Items 20 to 26).

**Architecture:** 
1. `StateDiscretizer`: Simplify to canonical aggregate queue mode, eliminating directional group-by branches (`branch`, `phase`, `phase_group`) and auxiliary branch extraction methods.
2. `NegativePressureReward`: Eliminate unused switch penalty streak/chattering mechanisms; replace per-step `id(controlled_links)` cache with static intersection topology configured once; eliminate non-standard fallback pressure calculation branches.
3. `TrafficSignalRLEnv`: Streamline constructor to accept `SimulationConfig` directly or scenario name without duplicating safety timing parameters or discretizer threshold clumps; eliminate per-step boolean numpy mask allocation in `_observation_info`.
4. `train_q_learning.py` & `evaluate_rl.py`: Consume the canonical `(state, info)` from `reset()` and `(next_state, reward, terminated, truncated, next_info)` from `step()` without discarding the first tuple element.

**Tech Stack:** Python 3.11, SUMO TraCI, Gymnasium-style RL environment, pytest.

**Spec:** Refactoring items 20 through 26 as identified in user review:
- #20: `StateDiscretizer.group_by` -> keep only aggregate.
- #21: Switch penalty in reward -> remove unused streak, threshold, switch tracking.
- #22: `NegativePressureReward.record_step()` -> eliminate fallback heuristics; require proper link/topology inputs.
- #23: Cache topology -> configure static lane mapping once rather than checking `id(controlled_links)` per step.
- #24: `_observation_info()` -> remove redundant per-step `action_mask` allocation; keep `valid_actions` as source of truth.
- #25: Environment step/reset return tuple -> consume `state`/`next_state` directly in train/eval loops instead of `_` and `info["discrete_state"]`.
- #26: `TrafficSignalRLEnv.__init__` -> streamline constructor, eliminating parameter clumps and timing parameter duplication.

## Global Constraints

- Must preserve correct Tabular Q-Learning convergence and PressLight negative pressure formulation.
- State tuple output for `discretize()` must remain a 4-tuple: `(current_phase, serving_bin, waiting_bin, time_bin)`.
- Checkpoints configured with `aggregate` (the current standard across all models) must continue to load and evaluate identically.
- Backward compatibility with existing callers of `TrafficSignalRLEnv` via `**kwargs` should be preserved where feasible.
- All pytest tests must pass cleanly.

## Review Focus

1. Discretizer returns identical discrete state for aggregate queue observations.
2. Checkpoint restoration via `StateDiscretizer.from_config()` still works seamlessly with existing saved models.
3. Reward computation produces exact PressLight negative pressure matching prior mathematically sound outputs.
4. `env.reset()` and `env.step()` outputs cleanly drive both `train_q_learning.py` and `evaluate_rl.py`.
5. Existing test suite (325 tests) maintains 100% pass rate.

---

### Task 1: Streamline `StateDiscretizer` (#20)

**Files:**
- Modify: `src/rl/discretization.py`
- Test: `tests/test_rl_components.py:21-163`

**Interfaces:**
- Consumes: `IntersectionObservation` (current_phase, green_phases, controlled_links, in_lane_queues, time_in_phase, is_in_transition).
- Produces: `StateDiscretizer.discretize(obs) -> Tuple[int, int, int, int]`.

- [ ] **Step 1: Write test for simplified aggregate StateDiscretizer in `tests/test_rl_components.py`**

Confirm that `StateDiscretizer` produces standard 4-tuple state `(current_phase, serving_bin, waiting_bin, time_bin)` and gracefully accepts `group_by="aggregate"`.

- [ ] **Step 2: Simplify `src/rl/discretization.py`**

Remove `VALID_MODES`, `_extract_branch`, `_waiting_bins`, and the conditional branching for `branch`, `phase`, and `phase_group` in `discretize()`.
Keep `group_by="aggregate"` for config compatibility in `get_config()` and `from_config()`.

- [ ] **Step 3: Update obsolete group-by tests in `tests/test_rl_components.py`**

Remove `test_directional_waiting_bins_distinguishes_congested_directions` and `test_discretizer_group_by_modes` which specifically tested the deprecated experimental modes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rl_components.py -k Discretizer -v`
Expected: PASS

---

### Task 2: Streamline `NegativePressureReward` and Topology Caching (#21, #22, #23)

**Files:**
- Modify: `src/rl/rewards.py`
- Modify: `src/rl/environment.py:151-180`
- Test: `tests/test_rl_components.py:164-315`

**Interfaces:**
- Consumes: `in_lane_queues: Dict[str, float]`, `out_lane_queues: Dict[str, float]`, `pending_vehicles: int`, static `topology: Dict[str, Set[str]]`.
- Produces: `NegativePressureReward.record_step(...)`, `calculate_reward() -> float`.

- [ ] **Step 1: Write tests for streamlined `NegativePressureReward`**

Verify negative pressure calculation with static topology, absence of switch streak tracking, and single clear formula in `tests/test_rl_components.py`.

- [ ] **Step 2: Refactor `NegativePressureReward` in `src/rl/rewards.py`**

- Remove switch penalty tracking: delete `switch_penalty`, `consecutive_switch_penalty`, `consecutive_switch_threshold`, `record_switch()`, `consecutive_switches`, `interval_switch`, `current_switch_penalty`.
- Replace `_get_outs_per_in(controlled_links)` and `id(controlled_links)` caching with pre-computed `topology: Dict[str, Set[str]]` (or method `set_topology(controlled_links)`).
- Simplify `record_step()`: remove `elif out_q` and `else` fallback heuristics. Compute pressure strictly from `in_lane_queues`, `out_lane_queues`, and the mapped topology.
- Simplify `get_config()` to output `{"name": "NegativePressureReward", "scale": self.scale, "pending_weight": self.pending_weight}`.

- [ ] **Step 3: Update `TrafficSignalRLEnv` reward integration in `src/rl/environment.py`**

- Set topology once in `TrafficSignalRLEnv.__init__` on `self.reward_fn`.
- Remove `prev_switches` tracking and `self.reward_fn.record_switch(is_switch)` calls in `step()`.
- Remove `info["consecutive_switches"]` and `info["is_switch"]`.

- [ ] **Step 4: Update reward tests in `tests/test_rl_components.py`**

Update tests to remove tests for consecutive switch penalty streaks, and verify static topology and clean record_step behavior.

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/test_rl_components.py -k Reward -v`
Expected: PASS

---

### Task 3: Streamline Action Mask, Gym Return Tuples, and Env Constructor (#24, #25, #26)

**Files:**
- Modify: `src/rl/environment.py`
- Modify: `src/training/train_q_learning.py`
- Modify: `src/evaluation/evaluate_rl.py`
- Test: `tests/test_rl_components.py:316-430`
- Test: `tests/test_simplification.py`

**Interfaces:**
- `TrafficSignalRLEnv(config: Union[SimulationConfig, str], ...)`
- `env.reset(seed) -> Tuple[Tuple[int, ...], Dict[str, Any]]`
- `env.step(action) -> Tuple[Tuple[int, ...], float, bool, bool, Dict[str, Any]]`

- [ ] **Step 1: Simplify `_observation_info` in `src/rl/environment.py` (#24)**

Remove per-step `mask = np.zeros(self.num_actions, dtype=bool)` and `"action_mask"` from the dictionary returned by `_observation_info()`. Keep `valid_actions` as the canonical source. Retain `get_valid_action_mask()` as an on-demand helper method.

- [ ] **Step 2: Streamline `TrafficSignalRLEnv.__init__` (#26)**

Allow passing `config: Optional[SimulationConfig] = None` or `scenario_name: Optional[str] = None`.
Remove discretizer parameter clumps (`max_effective_green`, `serving_queue_thresholds`, `waiting_queue_thresholds`) - accept `state_discretizer: Optional[StateDiscretizer] = None`.
Retain keyword argument compatibility for any lingering callers by accepting `**kwargs` that forward to `get_scenario_config`.

- [ ] **Step 3: Standardize Gym tuple unpacking in `train_q_learning.py` & `evaluate_rl.py` (#25)**

- In `train_q_learning.py`:
  Replace `_, info = env.reset()` / `state = info["discrete_state"]` with `state, info = env.reset(seed=ep_seed)`.
  Replace `_, reward, terminated, truncated, next_info = env.step(action)` / `next_state = next_info["discrete_state"]` with `next_state, reward, terminated, truncated, next_info = env.step(action)`.
- In `evaluate_rl.py`:
  Similarly replace `_, info = env.reset()` and `_, reward, terminated, truncated, next_info = env.step(action)`.

- [ ] **Step 4: Update environment and training tests in `tests/test_rl_components.py`**

Update any test checking old signature parameters or mock return structures (such as `mock_env.reset.return_value`).

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/test_rl_components.py tests/test_simplification.py tests/test_training.py -v`
Expected: PASS

---

### Task 4: Complete Test Suite Verification and Regression Testing

- [ ] **Step 1: Run full pytest suite across entire repository**

Run: `pytest`
Expected: 100% tests passing (all 325+ tests).

- [ ] **Step 2: Run a smoke evaluation run using CLI or experiment runner**

Run: `python -m pytest tests/test_evaluation_runner.py tests/test_controllers.py`
Expected: PASS
