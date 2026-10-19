# Kế Hoạch Tinh Gọn Mã Nguồn Dự Án (Mục 2, 3, 4, 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tinh gọn mã nguồn, chuẩn hóa cấu hình training thành dataclass, loại bỏ hàm đánh giá trùng lặp trong `evaluate_rl.py`, tự động hóa sinh báo cáo/biểu đồ sau benchmark và dọn dẹp repository.

**Architecture:** 
- Tách bạch vai trò các module: `evaluate_rl.py` chỉ làm helper quản lý session và greedy loop; `experiment_runner.py` là trung tâm điều phối toàn bộ thực nghiệm, tự động gọi báo cáo/chart.
- Chuẩn hóa quản lý tham số training bằng Dataclass `TrainingConfig`, tự động fallback về `DEFAULT_SAFETY_CONFIG`, loại bỏ 3 tầng parse dict lồng nhau trong `train_q_learning.py`.
- Chuẩn hóa cấu trúc thư mục git và dọn các entry legacy trong `.gitignore`.

**Tech Stack:** Python 3.11, Dataclasses, PyYAML, SUMO/TraCI/libsumo, Matplotlib, Pytest.

**Spec:** Yêu cầu người dùng triển khai các mục:
- Mục 2: Bỏ trùng lặp `evaluate_rl.py` (loại bỏ `evaluate_agent` thừa).
- Mục 3: Chuẩn hóa `TrainingConfig` dataclass trong `train_q_learning.py`.
- Mục 4: Tự động hóa báo cáo và biểu đồ trong `experiment_runner.py`.
- Mục 5: Dọn dẹp `.gitignore` và thêm `.gitkeep`.

## Global Constraints
- Không làm gãy 159 unit tests hiện tại.
- Bảo toàn tương thích ngược cho các hàm gọi `train_q_learning()` với kwargs hoặc file YAML.
- Không sửa đổi logic an toàn (min_green=5.0s, yellow=3.0s, all_red=1.0s, max_green=60.0s).

---

### Task 1: Dọn Dẹp .gitignore và Bổ Sung .gitkeep (Mục 5)

**Files:**
- Modify: `.gitignore`
- Create: `models/q_learning/.gitkeep`
- Create: `results/.gitkeep`
- Create: `results/figures/.gitkeep`
- Create: `results/evaluation/.gitkeep`
- Create: `results/training/.gitkeep`

**Interfaces:**
- N/A (Repository structure & git configuration)

- [x] **Step 1: Cập nhật `.gitignore` loại bỏ các quy tắc legacy**
  - Xóa `!results/figures_3algos/`, `!results/figures_3algos/**`, `!results/figures_4algos/`, `!results/figures_4algos/**`.
  - Đảm bảo ignore nội dung runtime trong `models/` và `results/` nhưng không ignore file `.gitkeep`.

- [x] **Step 2: Tạo các file `.gitkeep` cho các thư mục runtime**
  - Tạo `models/q_learning/.gitkeep`
  - Tạo `results/.gitkeep`
  - Tạo `results/figures/.gitkeep`
  - Tạo `results/evaluation/.gitkeep`
  - Tạo `results/training/.gitkeep`

- [x] **Step 3: Kiểm tra trạng thái git**
  - Chạy `git status` để kiểm tra các file `.gitkeep` được theo dõi đúng cách và không có file rác bị sót.

---

### Task 2: Loại Bỏ Mã Trùng Lặp trong `evaluate_rl.py` (Mục 2)

**Files:**
- Modify: `src/evaluation/evaluate_rl.py`
- Test: `tests/test_reuse_regressions.py`
- Test: `tests/test_audit_regressions.py`

**Interfaces:**
- Giữ nguyên các hàm cốt lõi phục vụ module khác:
  - `checkpoint_env_params(cfg: Dict[str, Any]) -> Dict[str, Any]`
  - `load_checkpoint_env_params(model_path: Optional[str]) -> Dict[str, Any]`
  - `create_evaluation_session(...) -> Tuple[QLearningController, TrafficSignalRLEnv]`
  - `run_greedy_eval_loop(agent, env, seed=None) -> Tuple[float, Dict[str, Any]]`
- Loại bỏ:
  - Hàm `evaluate_agent()` (dòng 123-196) và khối CLI `if __name__ == "__main__":` (dòng 198-230).

- [x] **Step 1: Viết test kiểm tra các interface cốt lõi của `evaluate_rl.py` vẫn nguyên vẹn**
  - Xác nhận `create_evaluation_session`, `run_greedy_eval_loop`, `load_checkpoint_env_params` hoạt động ổn định.

- [x] **Step 2: Xóa `evaluate_agent()` và CLI block trong `src/evaluation/evaluate_rl.py`**
  - Loại bỏ các dòng thừa, giữ file `evaluate_rl.py` tinh gọn dưới 120 dòng chỉ tập trung vào session & execution loop.

- [x] **Step 3: Chạy test kiểm tra toàn bộ suite**
  - Chạy: `pytest tests/test_reuse_regressions.py tests/test_audit_regressions.py -v`
  - Kết quả: PASS.

---

### Task 3: Chuẩn Hóa `TrainingConfig` Dataclass (Mục 3)

**Files:**
- Modify: `src/training/train_q_learning.py`
- Test: `tests/test_rl_components.py`
- Test: `tests/test_reuse_regressions.py`

**Interfaces:**
- Produces:
  - `@dataclass class TrainingConfig` với đầy đủ trường, type hints, giá trị mặc định từ `DEFAULT_SAFETY_CONFIG`.
  - `TrainingConfig.from_yaml(config_path: str, **overrides) -> TrainingConfig`
  - `train_q_learning(config: Optional[TrainingConfig] = None, **kwargs) -> Dict[str, Any]` (hỗ trợ cả truyền config object lẫn kwargs như cũ).

- [x] **Step 1: Viết test cho `TrainingConfig` và phương thức `.from_yaml()`**
  - Kiểm tra `TrainingConfig()` có các giá trị an toàn mặc định đúng chuẩn.
  - Kiểm tra `TrainingConfig.from_yaml("configs/q_learning.yaml")` nạp đúng hyperparameters và discretization.
  - Kiểm tra overrides qua kwargs hoạt động chính xác.

- [x] **Step 2: Định nghĩa `TrainingConfig` và refactor `train_q_learning()` trong `src/training/train_q_learning.py`**
  - Tạo `@dataclass class TrainingConfig`.
  - Thay thế khối 3 tầng gán dictionary `cfg_data` dài dòng bằng việc khởi tạo hoặc cập nhật instance `TrainingConfig`.
  - Kết nối `config.reward_scale`, `config.serving_thresholds`, `config.waiting_thresholds`, ... trực tiếp vào khởi tạo env và agent.

- [x] **Step 3: Chạy unit tests cho training và RL components**
  - Chạy: `pytest tests/test_rl_components.py tests/test_reuse_regressions.py -v`
  - Kết quả: PASS.

---

### Task 4: Tự Động Hóa Báo Cáo & Chart trong `experiment_runner.py` (Mục 4)

**Files:**
- Modify: `src/evaluation/experiment_runner.py`
- Test: `tests/test_evaluation_runner.py`
- Test: `tests/test_compare_algorithms.py`

**Interfaces:**
- Modifies `run_experiments`:
  - Thêm tham số: `generate_report: bool = True`, `generate_plots: bool = False`, `eval_dir: str = "results/evaluation"`, `figures_dir: str = "results/figures"`.
  - Khi hoàn tất tính toán tổng hợp, nếu `generate_report=True`, tự động gọi `generate_comparison_report`.
  - Nếu `generate_plots=True`, tự động gọi `generate_all_phase6_charts`.
- Modifies CLI `parse_args()` trong `experiment_runner.py`:
  - Thêm cờ `--report / --no-report` (mặc định: True).
  - Thêm cờ `--plots` (mặc định: False).

- [x] **Step 1: Viết test kiểm tra `run_experiments` tự động kích hoạt sinh báo cáo**
  - Mock `generate_comparison_report` và kiểm tra nó được gọi với đường dẫn summary chính xác khi `generate_report=True`.

- [x] **Step 2: Cập nhật `run_experiments` và CLI trong `src/evaluation/experiment_runner.py`**
  - Import `generate_comparison_report` từ `src.evaluation.compare_algorithms`.
  - Thêm logic tự động sinh báo cáo sau khi ghi file `benchmark_summary.json`.
  - Hỗ trợ cờ `--plots` để tùy chọn vẽ luôn biểu đồ.

- [x] **Step 3: Chạy test kiểm tra**
  - Chạy: `pytest tests/test_evaluation_runner.py tests/test_compare_algorithms.py -v`
  - Kết quả: PASS.

---

### Task 5: Kiểm Tra Toàn Diện Hệ Thống (Verification)

- [x] **Step 1: Chạy toàn bộ 159+ unit tests**
  - Lệnh: `pytest`
  - Kỳ vọng: 100% tests pass, không phát sinh regression.
- [x] **Step 2: Kiểm tra định dạng và code style**
  - Đảm bảo mã nguồn gọn gàng, không có imports thừa.
