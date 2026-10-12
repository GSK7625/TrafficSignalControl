# SUMO Simulation Platform & RESCO Dataset Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng nền tảng mô phỏng giao thông sử dụng Python và TraCI trên nền tảng RESCO dataset (trọng tâm là `cologne1`, `ingolstadt1`, mở rộng `cologne3`), kèm bộ kiểm tra dữ liệu, bộ quản lý pha đèn an toàn (Safe Phase Manager), bộ thu thập chỉ số mô phỏng (Metrics Collector) xuất vào `results/`, và hệ thống kiểm thử tự động.

**Architecture:** Thiết kế module hóa chuẩn Python hướng đối tượng với 5 tầng rõ ràng:
1. `src/data_inspector.py`: Phân tích và kiểm tra tính toàn vẹn của dataset RESCO (`.sumocfg`, `.net.xml`, `.rou.xml`, pha đèn, làn điều khiển).
2. `src/config.py`: Quản lý cấu hình kịch bản mô phỏng, tham số bước thời gian, chuyển mạch pha, đường dẫn tệp và tùy chọn SUMO / SUMO-GUI.
3. `src/phase_manager.py`: Bộ điều khiển pha đèn an toàn (Safe Phase Manager) quản lý min green time, tự động chèn pha vàng (yellow clearance) để tránh tai nạn / phanh gấp, đảm bảo dùng chung an toàn cho 4 thuật toán sau này.
4. `src/metrics.py`: Thu thập trực tiếp từ TraCI các chỉ số: waiting time, queue length, completed vehicles, travel time, time loss; hỗ trợ xuất tệp JSON & CSV vào `results/`.
5. `src/runner.py` / `main.py`: Trình điều khiển vòng đời mô phỏng TraCI (khởi động, bước mô phỏng, đồng bộ dữ liệu, đóng kết nối an toàn) với giao diện CLI thân thiện.

**Tech Stack:** Python 3.11+, SUMO 1.28.0+, TraCI, sumolib, unittest.

**Spec:** Yêu cầu người dùng "Giai đoạn 2 — Kiểm tra dataset RESCO và xây dựng nền tảng SUMO" (Giai đoạn 1: Kiểm tra dữ liệu 1-7, Giai đoạn 2: Xây dựng nền tảng mô phỏng 1-8, các giới hạn và tiêu chí hoàn thành).

## Global Constraints

- Không sửa đổi bất kỳ tệp dữ liệu gốc nào trong thư mục `data/`.
- Chỉ sử dụng RESCO dataset có sẵn trong `data/`, không dùng framework ngoài (không dùng thư viện ngoài `resco_benchmark` hay `sumo-rl`).
- Không thêm các thư viện bên ngoài không cần thiết (tận dụng thư viện có sẵn: Python standard library, traci, sumolib).
- Mọi kết quả đo đạc và báo cáo xuất ra phải được lưu vào thư mục `results/`, tuyệt đối không ghi vào `data/`.
- Kiểm thử tự động phải chạy độc lập và vượt qua 100% các bài test (sử dụng `unittest`).
- Phải đảm bảo khả năng chạy được cả chế độ headless (sumo) và giao diện trực quan (sumo-gui) qua cờ cấu hình.

## Review Focus

1. **Thư mục làm việc của SUMO khi nạp `.sumocfg`**: Các đường dẫn trong `.sumocfg` của RESCO (ví dụ `<net-file value="cologne1.net.xml"/>`) là đường dẫn tương đối so với thư mục chứa tệp cấu hình đó. Nếu khởi động TraCI từ thư mục gốc dự án mà không chỉ rõ working directory hoặc đường dẫn tuyệt đối, SUMO có thể không tìm thấy file mạng. Cần kiểm tra kỹ cơ chế xác định đường dẫn.
2. **Kịch bản thiếu pha vàng trong `.net.xml` (`cologne1`)**: Trong `cologne1`, mạng lưới SUMO gốc chỉ định nghĩa các pha xanh/đỏ mà không có pha vàng trung gian. Safe Phase Manager phải có khả năng tự động tạo pha vàng an toàn (chuyển các đèn từ xanh sang vàng trước khi sang đỏ) với độ dài cấu hình được (mặc định 3s).
3. **Thời gian mô phỏng thực tế của kịch bản**: `cologne1` bắt đầu từ giây 25200 đến 28800 (buổi sáng); `ingolstadt1` bắt đầu từ giây 57600 đến 61200 (buổi chiều). Vòng lặp mô phỏng phải tôn trọng `begin` và `end` thời gian thực của từng scenario, tránh giả định thời gian bắt đầu luôn là 0.
4. **Theo dõi xe hoàn thành hành trình**: Các xe đã hoàn thành và rời khỏi mạng lưới biến mất khỏi danh sách xe hiện hành. Chỉ số `completed_vehicles`, `travel_time` và `time_loss` tích lũy phải được lưu vết thông qua `traci.simulation.getArrivedIDList()` hoặc sự kiện rời mạng để tránh mất mát dữ liệu.
5. **Đóng TraCI an toàn khi xảy ra ngoại lệ**: Nếu tiến trình mô phỏng bị gián đoạn hoặc gặp lỗi, kết nối TraCI phải được đóng đúng cách (`traci.close()`) trong khối `finally` để tránh rò rỉ socket và tiến trình sumo chạy ngầm trên Windows.

---

### Task 1: Dataset Inspection & Verification Module

**Files:**
- Create: `src/data_inspector.py`
- Test: `tests/test_data_inspector.py`

**Interfaces:**
- Produces:
  * `ScenarioInfo`: dataclass chứa `name`, `cfg_path`, `net_path`, `rou_path`, `begin_time`, `end_time`, `duration`, `tls_count`, `tls_ids`, `phases_per_tls`, `controlled_lanes_per_tls`, `trips_or_vehicles_count`.
  * `inspect_scenario(scenario_name: str, data_dir: str = "data") -> ScenarioInfo`
  * `inspect_all_scenarios(data_dir: str = "data") -> Dict[str, ScenarioInfo]`

- [ ] **Step 1: Write the failing test**
  Viết `tests/test_data_inspector.py` kiểm tra việc phân tích cấu trúc của `cologne1`, `ingolstadt1`, `cologne3`. Xác nhận đọc đúng `begin_time`, `end_time`, số nút giao đèn, danh sách ID đèn, và số làn được điều khiển.

- [ ] **Step 2: Run test to verify it fails**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_data_inspector.py`
  Kết quả mong đợi: FAIL do `src.data_inspector` chưa tồn tại.

- [ ] **Step 3: Implement `src/data_inspector.py`**
  Thực thi hàm đọc XML cho `.sumocfg`, `.net.xml`, `.rou.xml`. Phân tích logic `tlLogic`, đếm pha, trích xuất chuỗi tín hiệu, đếm kết nối làn (`connection`), đếm số lượng trips/vehicles/flows.

- [ ] **Step 4: Run test to verify it passes**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_data_inspector.py`
  Kết quả mong đợi: PASS.

---

### Task 2: Configuration Module

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  * `SimulationConfig`: dataclass chứa `dataset_name`, `gui`, `step_length`, `yellow_duration`, `min_green`, `max_steps`, `seed`, `output_dir`.
  * `get_scenario_config(dataset_name: str, **kwargs) -> SimulationConfig`

- [ ] **Step 1: Write the failing test**
  Viết `tests/test_config.py` kiểm thử tạo cấu hình mặc định cho `cologne1`, `ingolstadt1`, ghi đè các tham số (`gui=True`, `yellow_duration=4.0`), kiểm tra xác thực tên dataset hợp lệ (ném `ValueError` nếu dataset không tồn tại).

- [ ] **Step 2: Run test to verify it fails**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_config.py`
  Kết quả mong đợi: FAIL do `src.config` chưa tồn tại.

- [ ] **Step 3: Implement `src/config.py`**
  Xây dựng dataclass `SimulationConfig` với các giá trị mặc định chuẩn mực cho nghiên cứu tín hiệu đèn giao thông và phương thức phân giải đường dẫn tệp SUMO tuyệt đối an toàn.

- [ ] **Step 4: Run test to verify it passes**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_config.py`
  Kết quả mong đợi: PASS.

---

### Task 3: Safe Phase Manager

**Files:**
- Create: `src/phase_manager.py`
- Test: `tests/test_phase_manager.py`

**Interfaces:**
- Consumes: `SimulationConfig` từ `src/config.py`
- Produces:
  * `SafePhaseManager`:
    - `__init__(tls_id: str, green_phases: List[str], yellow_duration: float = 3.0, min_green: float = 5.0)`
    - `request_phase(target_phase_idx: int) -> bool`
    - `update(step_duration: float = 1.0) -> None`
    - `get_current_phase() -> int`
    - `is_in_transition() -> bool`
    - `build_yellow_state(current_state: str, target_state: str) -> str`

- [ ] **Step 1: Write the failing test**
  Viết `tests/test_phase_manager.py` kiểm thử:
  - Chuyển pha an toàn: từ pha xanh 0 sang pha xanh 1 phải trải qua pha vàng có thời gian `yellow_duration`.
  - Ràng buộc `min_green`: không cho phép chuyển pha khi thời gian xanh chưa đạt `min_green`.
  - Sinh chuỗi đèn vàng chính xác (`'G'`/`'g'` chuyển thành `'y'`, các vị trí khác giữ nguyên hoặc chuyển đỏ).
  - Không chuyển pha nếu `target_phase == current_phase`.

- [ ] **Step 2: Run test to verify it fails**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_phase_manager.py`
  Kết quả mong đợi: FAIL do `src.phase_manager` chưa tồn tại.

- [ ] **Step 3: Implement `src/phase_manager.py`**
  Hiện thực hóa bộ máy trạng thái (State Machine: `GREEN` -> `YELLOW` -> `GREEN`) kèm cơ chế tính toán chuỗi đèn vàng động cho mọi cấu hình mạng SUMO.

- [ ] **Step 4: Run test to verify it passes**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_phase_manager.py`
  Kết quả mong đợi: PASS.

---

### Task 4: Metrics Collector

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  * `StepMetrics`: dataclass chứa `step`, `time`, `waiting_time`, `queue_length`, `running_vehicles`, `completed_vehicles_step`, `time_loss_step`.
  * `MetricsCollector`:
    - `record_step(time: float, tls_ids: List[str], traci_module) -> StepMetrics`
    - `get_summary() -> Dict[str, Any]`
    - `export_results(output_dir: str, prefix: str = "") -> Tuple[str, str]` (trả về đường dẫn JSON và CSV)

- [ ] **Step 1: Write the failing test**
  Viết `tests/test_metrics.py` sử dụng mock hoặc dữ liệu giả lập để kiểm tra:
  - Tính toán tổng thời gian chờ, độ dài hàng đợi (halting vehicles).
  - Theo dõi xe hoàn thành và thời gian hành trình từ departure tới arrival.
  - Xuất file `.json` và `.csv` vào thư mục tạm và xác nhận cấu trúc dữ liệu hợp lệ.

- [ ] **Step 2: Run test to verify it fails**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_metrics.py`
  Kết quả mong đợi: FAIL do `src.metrics` chưa tồn tại.

- [ ] **Step 3: Implement `src/metrics.py`**
  Thu thập số liệu từ TraCI:
  - `traci.lane.getLastStepHaltingNumber(lane)` cho độ dài hàng đợi.
  - `traci.lane.getWaitingTime(lane)` / `traci.vehicle.getWaitingTime(veh)` cho thời gian chờ.
  - `traci.vehicle.getTimeLoss(veh)` cho độ trễ / time loss.
  - `traci.simulation.getArrivedIDList()` cho danh sách xe hoàn thành.
  - Xuất báo cáo JSON tổng hợp và CSV chi tiết theo từng bước mô phỏng.

- [ ] **Step 4: Run test to verify it passes**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_metrics.py`
  Kết quả mong đợi: PASS.

---

### Task 5: Simulation Runner & CLI

**Files:**
- Create: `src/runner.py`
- Create: `main.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `SimulationConfig`, `SafePhaseManager`, `MetricsCollector`, `inspect_scenario`
- Produces:
  * `SimulationRunner`:
    - `__init__(config: SimulationConfig)`
    - `start() -> None`
    - `step() -> StepMetrics`
    - `run(num_steps: Optional[int] = None) -> Dict[str, Any]`
    - `close() -> None`
  * CLI arguments: `--dataset`, `--gui`, `--steps`, `--yellow-dur`, `--min-green`, `--output-dir`.

- [ ] **Step 1: Write the failing test**
  Viết `tests/test_runner.py` kiểm thử khởi tạo runner, kiểm tra cấu hình lệnh SUMO (`sumo` vs `sumo-gui`), kiểm tra cơ chế dọn dẹp TraCI an toàn.

- [ ] **Step 2: Run test to verify it fails**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_runner.py`
  Kết quả mong đợi: FAIL do `src.runner` chưa tồn tại.

- [ ] **Step 3: Implement `src/runner.py` and `main.py`**
  Hiện thực hóa `SimulationRunner` tích hợp TraCI, nạp file `.sumocfg` an toàn, quản lý vòng đời đèn tín hiệu và thu thập chỉ số. Cung cấp file thực thi `main.py` cho người dùng.

- [ ] **Step 4: Run test to verify it passes**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_runner.py`
  Kết quả mong đợi: PASS.

---

### Task 6: End-to-End Simulation Tests (Cologne1, Ingolstadt1, Cologne3)

**Files:**
- Create: `tests/test_integration_sumo.py`

- [ ] **Step 1: Write integration test**
  Viết test chạy thực tế một đoạn mô phỏng ngắn (50-100 bước) trên cả `cologne1` và `ingolstadt1` (và kiểm tra mở rộng trên `cologne3`) qua TraCI thật (headless).
  Kiểm tra rằng:
  - TraCI kết nối thành công, không gặp lỗi.
  - Các pha đèn được đọc và điều khiển an toàn.
  - Chỉ số thời gian chờ, hàng đợi và phương tiện được thu thập.
  - File kết quả JSON và CSV được tạo ra trong `results/test_runs/`.

- [ ] **Step 2: Run integration test and verify it passes**
  Chạy: `& ".\.venv\Scripts\python.exe" -m unittest tests/test_integration_sumo.py`
  Kết quả mong đợi: PASS.

---

### Task 7: Comprehensive Documentation & Final Reporting

**Files:**
- Create: `docs/USER_GUIDE.md`
- Create: `docs/DATASET_INSPECTION_REPORT.md`

- [ ] **Step 1: Write dataset inspection report**
  Lập bảng chi tiết 10 dataset RESCO, đặc điểm kịch bản `cologne1`, `ingolstadt1`, `cologne3`, thời gian mô phỏng, số lượng nút giao, các pha đèn hợp lệ, số làn điều khiển.

- [ ] **Step 2: Write user guide**
  Hướng dẫn cụ thể cách chạy nền tảng mô phỏng qua CLI, chạy với GUI, chạy headless, vị trí các tệp kết quả trong `results/`, cách chạy kiểm thử tự động.

- [ ] **Step 3: Run full test suite verification**
  Chạy toàn bộ test suite:
  `& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py"`
  Xác nhận 100% tests vượt qua.
