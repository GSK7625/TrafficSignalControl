# Giai Đoạn 1: Tinh Gọn Config Path, CLI Overrides và Benchmark Runner

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai Giai đoạn 1 tinh gọn mã nguồn cho 3 mục độc lập, ít rủi ro:
1. (Mục 2) Hàm dùng chung giải quyết đường dẫn file YAML cấu hình `resolve_config_path()`.
2. (Mục 3) Gom các cờ dòng lệnh CLI thành dictionary ghi đè trong `main.py`.
3. (Mục 8) Hợp nhất parser và tinh gọn logic điều phối chạy đơn / ma trận trong `experiment_runner.py`.

**Architecture:** 
- Đưa logic tìm và kiểm tra file cấu hình YAML vào `resolve_config_path()` trong `src/config.py`, dùng chung cho `main.py`, `experiment_runner.py` và `train_q_learning.py`.
- Tinh gọn `main.main()` bằng cách lọc các giá trị CLI khác `None`, ánh xạ tên tham số và ghi đè trực tiếp lên dictionary cấu hình từ YAML.
- Hợp nhất các cờ số ít/số nhiều (`--scenario`/`--scenarios`, `--algorithm`/`--algorithms`, `--seed`/`--seeds`) trong `experiment_runner.py`, tái sử dụng helper tạo `ExperimentConfig` chung thay vì viết lặp ở 2 nhánh.

**Tech Stack:** Python 3.11, argparse, pytest.

**Spec:** Yêu cầu người dùng triển khai Giai đoạn 1 từ 8 mục tinh gọn (các mục 2, 3, 8).

## Global Constraints

- Toàn bộ 325 bài test hiện tại trong bộ kiểm thử phải tiếp tục PASS 100%.
- Không làm thay đổi signature công khai của `run_single_experiment`, `run_full_benchmark_matrix`, `load_yaml_config`, `load_experiments_yaml`.
- Giữ nguyên toàn bộ hành vi CLI hiện có: các cờ lệnh cũ (cả số ít và số nhiều) phải tiếp tục hoạt động tương thích ngược.
- Giữ nguyên định dạng thông báo lỗi khi thiếu cấu hình hoặc không tìm thấy model.

## Review Focus

1. `resolve_config_path` phải raise `FileNotFoundError` với thông báo tiếng Việt chuẩn khi file không tồn tại (để khớp với assertion trong `test_main_cli.py`).
2. Giá trị `None` truyền qua CLI không được ghi đè giá trị hợp lệ trong YAML (chỉ ghi đè khi cờ được truyền thực sự).
3. CLI `main.py` khi thiếu trường `algorithm` trong cả CLI và YAML phải raise `ValueError` kèm thông báo đúng như cũ.
4. `experiment_runner.main()` khi chạy đơn thất bại vẫn phải ghi thông báo lỗi ra `sys.stderr` và trả về mã thoát khác 0 (đáp ứng `test_cli_single_experiment_failure_exits_nonzero`).
5. Hỗ trợ đầy đủ định dạng danh sách phân tách bằng dấu phẩy qua cả `--scenario` lẫn `--scenarios`, `--algorithm` lẫn `--algorithms`, `--seed` lẫn `--seeds`.

---

### Task 1: Hàm dùng chung giải quyết đường dẫn config (`resolve_config_path`)

**Files:**
- Modify: `src/config.py:27-40`
- Modify: `main.py:84-101`
- Modify: `src/evaluation/experiment_runner.py:58-70`
- Modify: `src/training/train_q_learning.py:208-217`
- Test: `tests/test_config_and_inspector.py`
- Test: `tests/test_main_cli.py`

**Interfaces:**
- Consumes: `get_master_config_path()`, `os.path.isfile`, `os.path.abspath`.
- Produces: `resolve_config_path(config_path: Optional[str] = None) -> str`.

- [ ] **Step 1: Viết test cho `resolve_config_path` trong `tests/test_config_and_inspector.py`**

```python
def test_resolve_config_path_default_and_custom(tmp_path):
    from src.config import resolve_config_path
    # 1. Custom file exists
    custom = tmp_path / "custom.yaml"
    custom.write_text("simulation: {}", encoding="utf-8")
    assert resolve_config_path(str(custom)) == os.path.abspath(str(custom))

    # 2. Custom file does not exist -> raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Bắt buộc phải có file cấu hình YAML"):
        resolve_config_path("non_existent.yaml")

    # 3. Default path resolves to master config when present
    resolved = resolve_config_path("configs/config.yaml")
    assert os.path.isfile(resolved)
```

- [ ] **Step 2: Chạy test để kiểm tra lỗi**

Run: `.venv\Scripts\pytest tests/test_config_and_inspector.py -k test_resolve_config_path`
Expected: FAIL with `cannot import name 'resolve_config_path'`

- [ ] **Step 3: Cài đặt `resolve_config_path` trong `src/config.py`**

Thêm hàm `resolve_config_path(config_path: Optional[str] = None) -> str` vào `src/config.py`:
- Nếu `config_path` là None hoặc rỗng, mặc định là `"configs/config.yaml"`.
- Nếu file tồn tại, trả về `os.path.abspath(config_path)`.
- Nếu là default path `"configs/config.yaml"` hoặc None: tìm qua `get_master_config_path()`. Nếu thấy thì trả về; nếu không thấy, raise `FileNotFoundError` với thông báo hướng dẫn tạo file.
- Nếu là custom path không tồn tại: raise `FileNotFoundError` thông báo không tìm thấy file đã chỉ định.

- [ ] **Step 4: Cập nhật các caller sử dụng `resolve_config_path`**

- Trong `main.py`: Thay thế đoạn if-else dài kiểm tra `config_path` bằng:
  ```python
  config_path = resolve_config_path(args.config)
  ```
- Trong `src/evaluation/experiment_runner.py`: Cập nhật `load_experiments_yaml(config_path: str = "configs/config.yaml")`:
  ```python
  config_path = resolve_config_path(config_path)
  raw = load_yaml_config(config_path)
  ```
- Trong `src/training/train_q_learning.py`: Cập nhật `TrainingConfig.from_yaml`:
  ```python
  config_path = resolve_config_path(config_path)
  yaml_cfg = load_yaml_config(config_path)
  ```

- [ ] **Step 5: Chạy lại toàn bộ test suite để xác minh**

Run: `.venv\Scripts\pytest tests/test_config_and_inspector.py tests/test_main_cli.py tests/test_simplification.py`
Expected: PASS

---

### Task 2: Tinh gọn việc chọn CLI / YAML trong `main.py` bằng dictionary ghi đè

**Files:**
- Modify: `main.py:27-142`
- Test: `tests/test_main_cli.py`

**Interfaces:**
- Consumes: `args = parse_args()`, `load_yaml_config(config_path)`.
- Produces: `exp_config = ExperimentConfig(...)`.

- [ ] **Step 1: Viết test kiểm tra dictionary override trong `tests/test_main_cli.py`**

Kiểm tra việc kết hợp giữa YAML defaults và nhiều cờ CLI đồng thời (bao gồm cờ boolean `--gui`, tham số số `-s`, tham số chuỗi `-d`, `-a`):
```python
def test_main_cli_overrides_dictionary(monkeypatch, tmp_path):
    import main
    # Kiểm tra dictionary overrides tôn trọng YAML khi CLI là None
    ...
```

- [ ] **Step 2: Cài đặt logic dictionary override trong `main.py`**

- Gom CLI overrides:
  ```python
  CLI_KEY_MAP = {"dataset": "scenario_name", "model": "model_path"}
  cli_overrides = {
      CLI_KEY_MAP.get(k, k): v
      for k, v in vars(args).items()
      if v is not None and k != "config"
  }
  ```
- Xây dựng `base_cfg` từ `sim_cfg` và `ql_cfg`:
  ```python
  base_cfg = {
      "scenario_name": sim_cfg.get("dataset", DEFAULT_DATASET),
      "algorithm": sim_cfg.get("algorithm"),
      "gui": sim_cfg.get("gui", False),
      "steps": sim_cfg.get("steps", None),
      "model_path": sim_cfg.get("model_path", ql_cfg.get("model_path", None)),
      "seed": int(sim_cfg.get("seed", RANDOM_SEED)),
      "output_dir": str(sim_cfg.get("output_dir", OUTPUT_DIR)),
      "step_length": float(sim_cfg.get("step_length", 1.0)),
      "use_tripinfo": bool(sim_cfg.get("use_tripinfo", True)),
  }
  final_cfg = {**base_cfg, **cli_overrides}
  ```
- Kiểm tra tính hợp lệ của `algorithm`, tính `gui_delay`, trích xuất `timing_overrides` từ `safety_cfg`.
- Khởi tạo `ExperimentConfig(**final_cfg, gui_delay=gui_delay, timing_overrides=timing_overrides)`.

- [ ] **Step 3: Chạy test xác minh**

Run: `.venv\Scripts\pytest tests/test_main_cli.py`
Expected: PASS

---

### Task 3: Tinh gọn parser và điều phối trong `src/evaluation/experiment_runner.py`

**Files:**
- Modify: `src/evaluation/experiment_runner.py:430-482`
- Test: `tests/test_evaluation_runner.py`

**Interfaces:**
- Consumes: CLI args (`--scenario`, `--scenarios`, `--algorithm`, `--algorithms`, `--seed`, `--seeds`, etc.).
- Produces: `main(argv: Optional[Sequence[str]] = None) -> int`.

- [ ] **Step 1: Viết test kiểm tra tính năng parser đồng nhất cho cả số ít và số nhiều**

Kiểm tra:
- `--scenario cologne1,ingolstadt1` hoặc `--scenarios cologne1,ingolstadt1`
- `--algorithm ft,mp` hoặc `--algorithms ft,mp`
- `--seed 42,99` hoặc `--seeds 42,99`

- [ ] **Step 2: Cài đặt parser và luồng điều phối tinh gọn**

1. Hợp nhất khai báo cờ trong parser:
   ```python
   parser.add_argument("--scenario", "--scenarios", dest="scenarios", type=str, default=None)
   parser.add_argument("--algorithm", "--algorithms", dest="algorithms", type=str, default=None)
   parser.add_argument("--seed", "--seeds", dest="seeds", type=str, default=None)
   ```
2. Helper phân tích chuỗi danh sách gọn gàng:
   ```python
   def _parse_cli_items(raw_value: Optional[str], default_val: Any, cast=str) -> Optional[List[Any]]:
       if raw_value is None:
           return default_val
       parts = [p.strip() for p in raw_value.split(",") if p.strip()]
       return [cast(p) for p in parts] if parts else default_val
   ```
3. Nhận diện chế độ chạy:
   - `is_multi = args.run_all or (args.scenarios is not None and "," in args.scenarios) or (args.algorithms is not None and "," in args.algorithms) or (args.seeds is not None and "," in args.seeds) or (args.report is False) or args.plots or args.fresh` (hoặc kiểm tra kích thước danh sách).
   - Nếu chạy đơn (1 scenario, 1 algorithm, 1 seed, không có cờ benchmark matrix): chạy trực tiếp `run_single_experiment(cfg)`, in lỗi ra `sys.stderr` nếu thất bại.
   - Nếu chạy ma trận: gọi `run_full_benchmark_matrix(...)`.
   - Cả 2 nhánh dùng chung helper `_create_experiment_config(...)` để tránh lặp lại việc đọc `simulation_options(sim_yaml)`.

- [ ] **Step 3: Chạy test xác minh**

Run: `.venv\Scripts\pytest tests/test_evaluation_runner.py`
Expected: PASS

---

### Task 4: Chạy toàn bộ test suite kiểm tra hồi quy (Regression Test)

- [ ] **Step 1: Chạy toàn bộ 325 tests của dự án**

Run: `.venv\Scripts\pytest`
Expected: 325 passed in < 10s.
