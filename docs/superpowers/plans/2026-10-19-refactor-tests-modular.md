# Modular Test Suite Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor and streamline the test suite by organizing tests into domain-specific modules, introducing a centralized `conftest.py`, eliminating boilerplate (`tempfile`, duplicate mocks, redundant inline imports), and retiring `test_audit_regressions.py` and `test_reuse_regressions.py` while maintaining 100% of the 159 tests and assertions.

**Architecture:** 
Establish a standard `pytest` layout with `tests/conftest.py` providing reusable fixtures (`mock_traci`, `sample_observation`, `dummy_scenario_info`). Reassign all tests from legacy audit/reuse regression files into their corresponding functional module files (`test_metrics.py`, `test_controllers.py`, `test_runner.py`, `test_phase_manager.py`, `test_rl_components.py`, `test_training.py`, `test_config_and_inspector.py`, `test_evaluation_runner.py`, `test_compare_algorithms.py`). Replace manual `tempfile` management with pytest's `tmp_path`, promote repeated inline imports to top-level, and parameterize repetitive validation checks.

**Tech Stack:** Python 3.11, pytest 9.1+, SUMO TraCI mocks, NumPy.

**Spec:** User selected "Cách 2" (Comprehensive modular refactoring of `tests/`).

## Global Constraints

- Total passing test count must remain at least 159 tests at all times.
- No regression or safety boundary check may be weakened or deleted.
- All temporary files must use pytest's `tmp_path` fixture (no manual `tempfile.NamedTemporaryFile` + `try...finally`).
- All imports must be cleanly placed at the top of each file, except where testing isolated import order behavior.

## Review Focus

- TraCI mock fidelity: Ensure TraCI method mock side-effects (`lane.getLastStepHaltingNumber`, `simulation.getCollisions`, `vehicle.getIDList`) behave identically across tests.
- Checkpoint file I/O: Ensure tests verifying invalid JSON or missing checkpoints still raise exact exception types (`ValueError`, `FileNotFoundError`, `TypeError`).
- Metrics calculation accuracy: Ensure `tripinfo_reconciliation_status` and summary statistics match previous results.
- Parameterized matrix consistency: Ensure seeds and parameter combinations retain the exact test item counts.

---

### Task 1: Create Centralized `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Produces fixtures:
  - `mock_traci()`: Configured TraCI/libsumo mock with standard default return values.
  - `sample_observation()`: Standard `IntersectionObservation` fixture with 2 phases and connected links.
  - `dummy_scenario_info()`: Mock `ScenarioInfo` with default TLS and phase definitions.

- [ ] **Step 1: Write `tests/conftest.py`**
Define standard fixtures for TraCI, `IntersectionObservation`, and scenario metadata.

- [ ] **Step 2: Run pytest to verify conftest loads without error**
Run: `pytest tests/test_evaluation_runner.py -v`
Expected: 2 passed

- [ ] **Step 3: Commit conftest setup**
Run: `git add tests/conftest.py`

---

### Task 2: Refactor `tests/test_config_and_inspector.py`

**Files:**
- Modify: `tests/test_config_and_inspector.py`

- [ ] **Step 1: Streamline temporary YAML tests with `tmp_path`**
Replace `tempfile.NamedTemporaryFile` in `test_invalid_yaml_format` with `tmp_path`.
- [ ] **Step 2: Parametrize invalid config validations**
Parametrize `step_length`, `decision_interval`, `yellow_duration` tests into a single parameterized test, absorbing `test_config_rejects_invalid_inputs_once` from `test_audit_regressions.py`.
- [ ] **Step 3: Run pytest on test_config_and_inspector**
Run: `pytest tests/test_config_and_inspector.py -v`
Expected: PASS

---

### Task 3: Refactor `tests/test_metrics.py`

**Files:**
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Streamline tripinfo tests using `tmp_path`**
Replace manual `tempfile.NamedTemporaryFile` and `os.remove` in `test_empty_tripinfo_xml` and `test_valid_tripinfo_xml` with `tmp_path`.
- [ ] **Step 2: Absorb metric & tripinfo regression tests**
Move from `test_audit_regressions.py` into `tests/test_metrics.py`:
  - `test_required_metric_failure_is_not_a_zero_queue`
  - `test_tripinfo_reconciliation_does_not_claim_false_match`
  - `test_early_summary_does_not_cache_missing_tripinfo`
  - `test_nonfinite_tripinfo_is_invalid`
  - `test_no_arrivals_has_unavailable_waiting_mean`
- [ ] **Step 3: Run pytest on test_metrics.py**
Run: `pytest tests/test_metrics.py -v`
Expected: PASS

---

### Task 4: Refactor `tests/test_phase_manager.py`

**Files:**
- Modify: `tests/test_phase_manager.py`

- [ ] **Step 1: Absorb valid actions boundary test**
Move `test_phase_manager_provides_valid_actions_at_boundaries` from `test_audit_regressions.py` into `TestSafePhaseManager` in `tests/test_phase_manager.py`.
- [ ] **Step 2: Run pytest on test_phase_manager.py**
Run: `pytest tests/test_phase_manager.py -v`
Expected: PASS

---

### Task 5: Refactor `tests/test_controllers.py`

**Files:**
- Modify: `tests/test_controllers.py`

- [ ] **Step 1: Use shared observation fixture and `tmp_path`**
Replace local `create_dummy_observation` usages with fixture or lightweight helper; replace `tempfile.TemporaryDirectory` with `tmp_path`.
- [ ] **Step 2: Absorb controller factory and checkpoint regression tests**
Move from `test_audit_regressions.py` & `test_reuse_regressions.py`:
  - `test_checkpoint_shape_is_checked_at_load`
  - `test_checkpoint_rejects_unknown_override_and_changed_state_encoding`
  - `test_q_cli_respects_the_environment_action_mask`
  - `test_empty_or_invalid_action_mask_is_not_expanded`
  - `test_load_checkpoint_restores_optional_target`
  - `test_factory_distinguishes_default_and_disabled_max_green`
  - `test_checkpoint_override_can_disable_max_green`
  - `test_legacy_checkpoint_uses_safety_and_discretizer_defaults`
  - `test_checkpoint_metadata_keeps_explicit_disabled_max_green`
  - `test_algorithm_aliases_use_same_factory_and_benchmark_path`
- [ ] **Step 3: Run pytest on test_controllers.py**
Run: `pytest tests/test_controllers.py -v`
Expected: PASS

---

### Task 6: Refactor `tests/test_runner.py`

**Files:**
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Absorb runner & simulation lifecycle regression tests**
Move from `test_audit_regressions.py`:
  - `test_observation_failure_is_not_a_zero_queue`
  - `test_original_mode_does_not_claim_control_of_tls`
  - `test_primary_collision_api_counts_vehicle_involvements`
  - `test_finalize_is_idempotent_and_closes_on_accounting_error`
  - `test_close_error_does_not_hide_simulation_error`
  - `test_runner_checks_q_action_count_before_simulation`
  - `test_controller_cannot_expand_configured_tls_scope`
- [ ] **Step 2: Run pytest on test_runner.py**
Run: `pytest tests/test_runner.py -v`
Expected: PASS

---

### Task 7: Refactor `tests/test_rl_components.py`

**Files:**
- Modify: `tests/test_rl_components.py`

- [ ] **Step 1: Use shared observation fixture**
Leverage `conftest.py`'s `sample_observation`.
- [ ] **Step 2: Absorb RL Environment regression tests**
Move from `test_audit_regressions.py` & `test_reuse_regressions.py`:
  - `test_env_reset_clears_summary_and_zero_limit_never_steps`
  - `test_env_rejects_out_of_range_actions`
  - `test_environment_rejects_misspelled_options`
  - `test_checkpoint_restores_reward_configuration`
  - `test_rl_environment_import_is_independent_of_import_order`
  - `test_rl_environment_uses_shared_all_red_default`
- [ ] **Step 3: Run pytest on test_rl_components.py**
Run: `pytest tests/test_rl_components.py -v`
Expected: PASS

---

### Task 8: Create `tests/test_training.py`

**Files:**
- Create: `tests/test_training.py`

- [ ] **Step 1: Create dedicated training test suite**
Move training and validation selector tests from `test_audit_regressions.py` and `test_reuse_regressions.py`:
  - `training_env` fixture
  - `test_training_honors_completion_threshold`
  - `test_validation_details_have_separate_seed_directories`
  - `test_selector_rejects_invalid_measurements`
  - `test_validation_keeps_legacy_top_level_completed_count`
  - `test_selector_rejects_missing_nonfinite_or_no_demand_results`
  - `test_selector_does_not_infer_perfect_completion_from_completed_count`
- [ ] **Step 2: Run pytest on test_training.py**
Run: `pytest tests/test_training.py -v`
Expected: PASS

---

### Task 9: Refactor `tests/test_evaluation_runner.py`

**Files:**
- Modify: `tests/test_evaluation_runner.py`

- [ ] **Step 1: Absorb evaluation & benchmark matrix regression tests**
Move from `test_audit_regressions.py` & `test_reuse_regressions.py`:
  - `test_invalid_summary_is_not_promoted_to_success`
  - `test_evaluation_requires_an_existing_checkpoint`
  - `test_fingerprint_changes_with_steps_and_model_contents`
  - `test_aggregation_preserves_failed_seeds_and_deduplicates_retries`
  - `test_experiments_loader_does_not_hide_bad_yaml`
  - `test_matrix_honors_yaml_timing_and_does_not_reuse_old_steps`
  - `test_zero_baseline_does_not_invent_a_relative_percentage`
  - `test_missing_sumo_version_is_unknown`
  - `test_zero_demand_run_is_not_a_failed_completion`
  - `test_nonfinite_rewards_do_not_break_strict_json_export`
  - `test_greedy_evaluation_does_not_change_learning_state`
  - `test_evaluation_setup_shares_checkpoint_target_and_disabled_timing`
  - `test_evaluation_setup_closes_env_on_action_count_mismatch`
  - `test_completion_statistics_uses_realized_demand_and_distinct_policies`
- [ ] **Step 2: Run pytest on test_evaluation_runner.py**
Run: `pytest tests/test_evaluation_runner.py -v`
Expected: PASS

---

### Task 10: Refactor `tests/test_compare_algorithms.py`

**Files:**
- Modify: `tests/test_compare_algorithms.py`

- [ ] **Step 1: Absorb reporting, plotting and dashboard regression tests**
Move from `test_audit_regressions.py` & `test_reuse_regressions.py`:
  - `test_report_handles_missing_metrics_and_zero_completion`
  - `test_no_baseline_does_not_generate_improvement_chart`
  - `test_phase_allocation_combines_seeds_before_plotting`
  - `test_comparison_plots_skip_failed_algorithms`
  - `test_dynamics_without_saved_run_details_does_not_crash`
  - `test_dashboard_handles_unavailable_waiting_metrics`
- [ ] **Step 2: Run pytest on test_compare_algorithms.py**
Run: `pytest tests/test_compare_algorithms.py -v`
Expected: PASS

---

### Task 11: Remove Legacy Files and Full Suite Verification

**Files:**
- Delete: `tests/test_audit_regressions.py`
- Delete: `tests/test_reuse_regressions.py`

- [ ] **Step 1: Remove legacy test files**
Remove `tests/test_audit_regressions.py` and `tests/test_reuse_regressions.py`.
- [ ] **Step 2: Run full test suite**
Run: `pytest -v`
Expected: 159+ passed (0 failures, 0 errors).
- [ ] **Step 3: Measure execution time and line count reduction**
Confirm performance and line reduction.
