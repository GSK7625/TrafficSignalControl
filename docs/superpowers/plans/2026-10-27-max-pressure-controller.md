# Max Pressure Traffic Signal Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai thuật toán điều khiển đèn tín hiệu giao thông Max Pressure (Pravin Varaiya 2013) theo đúng cơ sở lý thuyết chuẩn, tích hợp vào hệ thống mô phỏng SUMO/TraCI, hỗ trợ kiểm thử đơn vị, và đánh giá so sánh hiệu năng trực tiếp với Fixed-Time trên 2 kịch bản `cologne1` và `ingolstadt1`.

**Architecture:** 
- `MaxPressureController` trong `src/controllers/max_pressure.py`: Tính áp lực chuyển động $w(u, d) = x_u - x_d$ (trong đó $x_u$ là số xe trên làn vào upstream $u$, $x_d$ là số xe trên làn ra downstream $d$). Áp lực của pha xanh $p$ là $W(p) = \sum_{(u,d) \in \mathcal{M}(p)} w(u, d)$. Pha có áp lực lớn nhất được chọn kích hoạt.
- Tích hợp ràng buộc vật lý & an toàn giao thông:
  - Thời gian xanh tối thiểu $t_{\min\_green} = 10\text{s}$ tránh đảo pha giật cục (flickering).
  - Pha vàng dọn đường an toàn $t_{\text{yellow}} = 3\text{s}$ (sử dụng `build_yellow_state`).
  - Thời gian xanh tối đa $t_{\max\_green} = 60\text{s}$ chống nghẽn và chống bỏ đói (starvation) các luồng phụ.
- `SimulationRunner` trong `src/simulation_runner.py`: Tương thích đa hình với bất kỳ controller nào tuân theo controller protocol (`step`, `reset`, `get_current_state`), tự động nạp ánh xạ movement links nếu dùng `MaxPressureController`.
- `src/evaluate.py` và `main.py`: Mở rộng CLI với cờ `--controller {fixed_time,max_pressure}` và `--compare` để chạy thử nghiệm độc lập hoặc so sánh đối đầu giữa Fixed-Time và Max-Pressure.

**Tech Stack:** Python 3.11+, Eclipse SUMO 1.28.0+, TraCI, sumolib, Pytest, Matplotlib.

**Spec:** Yêu cầu người dùng triển khai thuật toán Max Pressure theo thiết kế chuẩn lý thuyết giao thông, tương thích với cấu trúc hiện tại của dự án.

## Global Constraints

- **Theoretical Formulation:** Áp lực pha xanh $p$ được tính theo công thức:
  $$W(p) = \sum_{(u, d) \in \mathcal{M}(p)} (x_u - x_d)$$
  với $\mathcal{M}(p)$ là tập các liên kết chuyển động (movement pairs $u \to d$) được cấp quyền đi (tín hiệu xanh 'G'/'g') trong pha $p$.
- **Safety Constraints:**
  - $t_{\min\_green} = 10.0\text{s}$ (không chuyển pha trước khi chạy đủ thời gian xanh tối thiểu).
  - $t_{\text{yellow}} = 3.0\text{s}$ (chuyển qua pha vàng an toàn trước khi vào pha xanh mới).
  - $t_{\max\_green} = 60.0\text{s}$ (bắt buộc chuyển sang pha kế tiếp có áp lực cao nhất nếu pha hiện tại đã chạy chạm ngưỡng tối đa).
- **Interface Compatibility:** Giữ nguyên chữ ký `step(step_duration: float = 1.0) -> Tuple[str, bool]` để tương thích hoàn toàn với `SimulationRunner`.
- **Test Independence:** `MaxPressureController` hỗ trợ callback cung cấp số lượng xe `lane_vehicle_fn: Optional[Callable[[str], int]]` để có thể unit-test độc lập 100% mà không bắt buộc khởi động TraCI server. Khi chạy thực tế với SUMO, mặc định truy vấn `traci.lane.getLastStepVehicleNumber(lane_id)`.

## Review Focus

1. **Khớp ánh xạ Phase-to-Movement:** Đảm bảo với mỗi pha xanh, các ký tự tín hiệu `'G'` hoặc `'g'` tại vị trí linkIndex tương ứng chính xác với cặp làn `(from_lane, to_lane)`.
2. **Xử lý pha có áp lực âm hoặc bằng 0:** Khi toàn bộ nút giao vắng xe ($W(p) \le 0$), bộ điều khiển duy trì pha hiện tại hoặc chuyển nhịp nhàng, không gây loop vô hạn hay crash.
3. **Pha vàng an toàn chuyển tiếp:** Khi áp lực quyết định chuyển từ pha $A \to B$, đèn tín hiệu phải qua pha vàng $3\text{s}$ được sinh tự động bởi `build_yellow_state(phase_A, phase_B)`.
4. **Không làm vỡ 46 test hiện tại:** Đảm bảo tất cả test suite trong `tests/` của Fixed-Time và hệ thống hiện hành đều giữ nguyên trạng thái PASS.
5. **Hiệu năng Max-Pressure so với Fixed-Time:** Trên kịch bản bất đối xứng như `cologne1`, Max-Pressure phải cải thiện rõ rệt hàng đợi và thời gian chờ so với Fixed-Time 30s.

---

### Task 1: Cài đặt và Unit Test `MaxPressureController`

**Files:**
- Create: `src/controllers/max_pressure.py`
- Modify: `src/controllers/__init__.py`
- Create: `tests/test_max_pressure.py`

**Interfaces:**
- Consumes: `build_yellow_state` từ `src.controllers.fixed_time`, `MovementInfo` và `PhaseInfo` từ `src.data_inspector`.
- Produces: `MaxPressureController` class với:
  - `__init__(green_phases, movements, min_green=10.0, max_green=60.0, yellow_time=3.0, tls_id=None, lane_vehicle_fn=None)`
  - `calculate_phase_pressures() -> Dict[int, float]`
  - `select_next_phase() -> int`
  - `step(step_duration: float = 1.0) -> Tuple[str, bool]`
  - `reset() -> None`
  - `get_current_state() -> str`

- [ ] **Step 1: Viết test suite cho `MaxPressureController`**
  Tạo `tests/test_max_pressure.py` kiểm tra:
  - Khởi tạo controller với phases và movements.
  - Tính toán áp lực chính xác cho từng phase dựa trên mock `lane_vehicle_fn`.
  - Giữ pha xanh ít nhất `min_green` giây trước khi chuyển đổi.
  - Kích hoạt pha vàng `yellow_time` giây khi chuyển pha.
  - Tự động cưỡng bức chuyển pha khi chạm ngưỡng `max_green` giây.
  - Reset trạng thái về ban đầu.

- [ ] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_max_pressure.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'src.controllers.max_pressure').

- [ ] **Step 3: Triển khai `MaxPressureController` trong `src/controllers/max_pressure.py`**
  - Định nghĩa cấu trúc ánh xạ giữa phase index và các movements được cấp quyền (`G`/`g`).
  - Cài đặt `calculate_phase_pressures()` tính $\sum_{(u, d)} (x_u - x_d)$.
  - Cài đặt bộ máy trạng thái (state machine) quản lý: `GREEN`, `YELLOW`, `time_in_phase`, `current_phase_idx`, `next_phase_idx`.
  - Xuất `MaxPressureController` trong `src/controllers/__init__.py`.

- [ ] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_max_pressure.py -v`
  Expected: All tests PASS.

- [ ] **Step 5: Commit**
  `git add src/controllers/max_pressure.py src/controllers/__init__.py tests/test_max_pressure.py`
  `git commit -m "feat: implement MaxPressureController with pressure calculation and safety constraints"`

---

### Task 2: Cập nhật `SimulationRunner` hỗ trợ `MaxPressureController`

**Files:**
- Modify: `src/simulation_runner.py`
- Modify: `tests/test_simulation_runner.py`

**Interfaces:**
- Consumes: `MaxPressureController` từ `src.controllers.max_pressure`, `inspect_scenario` từ `src.data_inspector`.
- Produces: `SimulationRunner` có thể khởi tạo tự động hoặc nhận `controller_type="max_pressure"` hoặc instance `MaxPressureController`.

- [ ] **Step 1: Viết test cho `SimulationRunner` với Max-Pressure**
  Trong `tests/test_simulation_runner.py`, thêm test khởi tạo `SimulationRunner` với `controller_type="max_pressure"` hoặc truyền controller `MaxPressureController`, kiểm tra runner nạp đúng movements và thực hiện step hợp lệ.

- [ ] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_simulation_runner.py -k "max_pressure" -v`
  Expected: FAIL.

- [ ] **Step 3: Cập nhật `src/simulation_runner.py`**
  - Bổ sung tham số `controller_type: str = "fixed_time"` và nạp metadata `movements` từ `ScenarioInfo.get_movements(tls_id)`.
  - Nếu `controller_type == "max_pressure"`, tự động khởi tạo `MaxPressureController` với danh sách pha xanh và movements tương ứng.
  - Cập nhật type annotation cho `controller` thành `Union[FixedTimeController, MaxPressureController, Any]`.

- [ ] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_simulation_runner.py -v`
  Expected: All tests PASS.

- [ ] **Step 5: Commit**
  `git add src/simulation_runner.py tests/test_simulation_runner.py`
  `git commit -m "feat: support MaxPressureController in SimulationRunner"`

---

### Task 3: Tích hợp Đánh giá và So sánh Đa thuật toán trong `src/evaluate.py`

**Files:**
- Modify: `src/evaluate.py`
- Modify: `tests/test_evaluation_flow.py`

**Interfaces:**
- Consumes: `SimulationRunner`, `MetricsCollector`.
- Produces:
  - `evaluate_scenario(..., controller_type="max_pressure")`
  - `compare_controllers(scenarios, controllers=["fixed_time", "max_pressure"], ...)`
  - Xuất bảng so sánh Markdown và biểu đồ so sánh 2 thuật toán trên 5 chỉ số giao thông.

- [ ] **Step 1: Viết test cho module đánh giá Max-Pressure và hàm so sánh**
  Thêm test kiểm tra `evaluate_scenario` với `controller_type="max_pressure"` và test hàm `compare_controllers` xuất cấu trúc kết quả hợp lệ.

- [ ] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_evaluation_flow.py -k "compare or max_pressure" -v`
  Expected: FAIL.

- [ ] **Step 3: Cập nhật `src/evaluate.py`**
  - Thêm tham số `controller_type: str = "fixed_time"` vào `evaluate_scenario`.
  - Cài đặt `evaluate_comparison(scenarios, max_steps, output_dir)` chạy cả Fixed-Time và Max-Pressure, sinh file `results/comparison_evaluation.json`, biểu đồ trực quan `results/algorithm_comparison.png` và báo cáo so sánh chi tiết `results/COMPARISON_REPORT.md`.

- [ ] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_evaluation_flow.py -v`
  Expected: All tests PASS.

- [ ] **Step 5: Commit**
  `git add src/evaluate.py tests/test_evaluation_flow.py`
  `git commit -m "feat: add multi-controller evaluation and comparison in evaluate module"`

---

### Task 4: Mở rộng CLI `main.py` và Kiểm thử Toàn diện (End-to-End)

**Files:**
- Modify: `main.py`
- Create: `tests/test_cli_max_pressure.py`

**Interfaces:**
- CLI arguments:
  - `--controller {fixed_time, max_pressure}`
  - `--compare` (chạy so sánh trực tiếp 2 thuật toán trên các bản đồ được chỉ định)
  - `--scenario <name>` và `--all`

- [ ] **Step 1: Viết test cho CLI arguments mới**
  Kiểm tra cú pháp parse `--controller max_pressure` và `--compare` trong `tests/test_cli_max_pressure.py`.

- [ ] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_cli_max_pressure.py -v`
  Expected: FAIL.

- [ ] **Step 3: Cập nhật `main.py`**
  - Bổ sung `--controller` (default: `fixed_time`, choices: `["fixed_time", "max_pressure"]`).
  - Bổ sung `--compare` (flag chạy đánh giá so sánh cả 2 thuật toán và xuất báo cáo đối sánh).
  - Tích hợp gọi `evaluate_scenario`, `evaluate_multiple` và `evaluate_comparison`.

- [ ] **Step 4: Chạy toàn bộ test suite để đảm bảo không lỗi hồi quy**
  Run: `pytest -v`
  Expected: Toàn bộ tests (cũ + mới) đều PASS.

- [ ] **Step 5: Chạy thử nghiệm thực tế với SUMO và kiểm tra kết quả**
  Run: `python main.py --scenario cologne1 --controller max_pressure --max-steps 1000`
  Verify: Chạy thành công, xuất ra 5 metrics hoàn chỉnh.

- [ ] **Step 6: Commit**
  `git add main.py tests/test_cli_max_pressure.py`
  `git commit -m "feat: add CLI options for Max Pressure controller and comparison mode"`
