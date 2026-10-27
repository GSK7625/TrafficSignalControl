# Fixed Time Traffic Signal Evaluation with SUMO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống điều khiển tín hiệu đèn giao thông chu kỳ cố định (Fixed Time - 30 giây mỗi pha xanh) và đánh giá hiệu năng dựa trên 5 chỉ số giao thông chuẩn (Waiting Time, Travel Time, Queue Length, Time Loss, Throughput) trên 2 kịch bản 1 nút giao (`cologne1` và `ingolstadt1`) sử dụng SUMO và TraCI.

**Architecture:** Kiến trúc mô-đun hóa tách biệt rõ ràng giữa logic điều khiển (`FixedTimeController`), mô hình chuyển pha an toàn (`PhaseManager`), lớp thu thập chỉ số (`MetricsCollector`), bộ chạy mô phỏng TraCI (`SimulationRunner`), và bộ điều phối thực nghiệm đa kịch bản (`EvaluationRunner`). Dữ liệu chỉ số được kiểm chứng song song qua cả TraCI real-time và SUMO native `--tripinfo-output` XML để đảm bảo độ chính xác tuyệt đối.

**Tech Stack:** Python 3.10+, Eclipse SUMO 1.28.0+, TraCI, sumolib, Pytest, Matplotlib.

**Spec:** User prompt yêu cầu đánh giá thuật toán Fixed Time (giới hạn mỗi pha đèn 30 giây) mô phỏng bằng SUMO với 2 map cơ bản 1 nút giao (`cologne1` và `ingolstadt1`), tính toán 5 metrics: Average Waiting Time, Travel Time, Queue Length, Time Loss, Throughput.

## Global Constraints

- **Green Phase Duration:** Mỗi pha xanh chính kéo dài đúng 30 giây (`green_time = 30.0s`).
- **Yellow Clearance:** Mỗi lần chuyển giữa 2 pha xanh đối xung phải có pha vàng an toàn kéo dài 3 giây (`yellow_time = 3.0s`).
- **Simulation Interface:** Kết nối và điều khiển mô phỏng qua TraCI (`traci.start` / `traci.trafficlight.setRedYellowGreenState`).
- **Metrics Accuracy:** Đo thời gian chờ và di chuyển từ cả TraCI per-step và file Tripinfo XML của SUMO.
- **Datasets:** Sử dụng 2 bộ dữ liệu có sẵn tại `data/cologne1` và `data/ingolstadt1`.

## Review Focus

1. **Pha vàng chuyển tiếp (Yellow clearance):** Đảm bảo không xảy ra chuyển pha tức thời (xanh sang xanh đối nghịch), tránh gây va chạm mô phỏng.
2. **Xử lý xe chưa hoàn thành hành trình:** Tách biệt rõ xe đã hoàn thành (arrived) để tính Travel Time/Time Loss và xe đang lưu thông (running) để tránh sai lệch thống kê.
3. **Độ dài hàng đợi (Queue Length):** Chỉ tính các xe có vận tốc $v < 0.1\text{ m/s}$ trên các làn vào (incoming lanes) của nút giao tại mỗi bước mô phỏng.
4. **Đóng cổng TraCI an toàn:** Luôn bảo đảm `traci.close()` được thực thi ngay cả khi có ngoại lệ (exception) xảy ra.
5. **Tính nhất quán cấu hình pha:** Tự động phát hiện và ánh xạ đúng danh sách pha xanh từ file `.net.xml` cho từng nút giao (`cluster_357187_359543` và `cluster_274083968_cluster_1200364014_1200364088`).

---

### Task 1: Khảo sát cấu trúc mạng & Ánh xạ pha giao thông

**Files:**
- Create: `src/data_inspector.py`
- Test: `tests/test_data_inspector.py`

**Interfaces:**
- Consumes: SUMO net files tại `data/cologne1/cologne1.net.xml` và `data/ingolstadt1/ingolstadt1.net.xml`.
- Produces: `inspect_scenario(scenario_name: str) -> Dict[str, Any]` trả về thông tin TLS ID, danh sách các pha xanh (green phases), các pha vàng (yellow phases) và các làn vào (incoming lanes).

- [x] **Step 1: Viết test cho data_inspector**
  Kiểm tra hàm `inspect_scenario` đọc đúng TLS ID của `cologne1` (`cluster_357187_359543`) và `ingolstadt1` (`cluster_274083968_cluster_1200364014_1200364088`), trích xuất được danh sách pha xanh.

- [x] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_data_inspector.py -v`
  Expected: FAIL (ModuleNotFoundError / function not defined).

- [x] **Step 3: Cài đặt `src/data_inspector.py`**
  Sử dụng `sumolib` và XML parser trích xuất cấu trúc tín hiệu đèn và các làn vào của nút giao.

- [x] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_data_inspector.py -v`
  Expected: PASS.

- [x] **Step 5: Commit**
  `git add src/data_inspector.py tests/test_data_inspector.py`
  `git commit -m "feat: add data_inspector for SUMO network and TLS phase analysis"`

---

### Task 2: Xây dựng Bộ điều khiển Fixed Time (FixedTimeController)

**Files:**
- Create: `src/controllers/fixed_time.py`
- Test: `tests/test_fixed_time.py`

**Interfaces:**
- Consumes: Cấu hình `green_time: float = 30.0`, `yellow_time: float = 3.0`, danh sách pha xanh `green_phases: List[str]`.
- Produces: 
  - `FixedTimeController.step(step_duration: float = 1.0) -> Tuple[str, bool]` trả về chuỗi tín hiệu đèn hiện tại (state string) và cờ báo hiệu chuyển pha.
  - `FixedTimeController.get_current_phase() -> int`
  - `FixedTimeController.is_in_yellow() -> bool`

- [x] **Step 1: Viết test cho FixedTimeController**
  Kiểm tra: Duy trì pha xanh đúng 30 bước (với `step_duration=1.0s`), sau đó chuyển sang pha vàng 3 bước, rồi chuyển sang pha xanh tiếp theo đúng 30 bước theo chu kỳ tuần hoàn.

- [x] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_fixed_time.py -v`
  Expected: FAIL.

- [x] **Step 3: Cài đặt `FixedTimeController` trong `src/controllers/fixed_time.py`**
  Quản lý trạng thái: `current_phase_index`, `time_in_phase`, `is_yellow`, logic sinh pha vàng trung gian giữa 2 pha xanh đối xung.

- [x] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_fixed_time.py -v`
  Expected: PASS.

- [x] **Step 5: Commit**
  `git add src/controllers/fixed_time.py tests/test_fixed_time.py`
  `git commit -m "feat: implement FixedTimeController with 30s green and 3s yellow phases"`

---

### Task 3: Xây dựng Module Thu thập & Tính toán 5 Metrics

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: Dữ liệu TraCI từng step (vận tốc xe, trạng thái dừng chờ) và file TripInfo XML.
- Produces:
  - `MetricsCollector.record_step(step_time: float, halted_vehicles: int, queue_length: float, running_vehicles: int)`
  - `MetricsCollector.parse_tripinfo(xml_path: str) -> Dict[str, float]`
  - `MetricsCollector.get_summary() -> Dict[str, Any]` trả về 5 metrics chuẩn:
    1. `average_waiting_time` (s)
    2. `average_travel_time` (s)
    3. `mean_queue_length` (veh) & `max_queue_length` (veh)
    4. `average_time_loss` (s)
    5. `throughput` (completed vehicles)

- [x] **Step 1: Viết test cho MetricsCollector**
  Kiểm tra tính toán trung bình hàng đợi, đọc file TripInfo mẫu XML và trích xuất đúng `waitingTime`, `timeLoss`, `duration`, và số xe hoàn thành (`throughput`).

- [x] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_metrics.py -v`
  Expected: FAIL.

- [x] **Step 3: Cài đặt `MetricsCollector` trong `src/metrics.py`**
  Triển khai các phương thức ghi nhận từng bước, phân tích tệp XML tripinfo, và tổng hợp thống kê.

- [x] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_metrics.py -v`
  Expected: PASS.

- [x] **Step 5: Commit**
  `git add src/metrics.py tests/test_metrics.py`
  `git commit -m "feat: implement MetricsCollector for 5 traffic performance metrics"`

---

### Task 4: Xây dựng Bộ chạy mô phỏng TraCI (SimulationRunner)

**Files:**
- Create: `src/simulation_runner.py`
- Test: `tests/test_simulation_runner.py`

**Interfaces:**
- Consumes: Đường dẫn cấu hình SUMO (`.sumocfg`), `FixedTimeController`.
- Produces: `SimulationRunner.run(gui: bool = False, max_steps: int = 3600) -> SimulationResult` điều phối vòng lặp mô phỏng và trả về kết quả số bước, thống kê hàng đợi.

- [x] **Step 1: Viết test cho SimulationRunner (sử dụng mock TraCI hoặc kịch bản mô phỏng ngắn 100 bước)**
  Kiểm tra khởi động SUMO với `--tripinfo-output`, chuyển giao trạng thái đèn tín hiệu tại mỗi bước, đo hàng đợi và đóng TraCI an toàn.

- [x] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_simulation_runner.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.simulation_runner'`).

- [x] **Step 3: Cài đặt `SimulationRunner` trong `src/simulation_runner.py`**
  Tích hợp TraCI, kết nối controller Fixed Time, đo hàng đợi trên các làn vào của nút giao, gọi `--tripinfo-output`, đảm bảo `traci.close()` trong `finally`.

- [x] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_simulation_runner.py -v`
  Expected: PASS.

- [x] **Step 5: Commit**
  `git add src/simulation_runner.py tests/test_simulation_runner.py`
  `git commit -m "feat: implement SimulationRunner orchestrating TraCI and FixedTime control"`

---

### Task 5: Xây dựng Kịch bản Đánh giá Đa bản đồ & Xuất Báo cáo So sánh

**Files:**
- Create: `src/evaluate.py`
- Create: `main.py`
- Test: `tests/test_evaluation_flow.py`

**Interfaces:**
- Consumes: Kịch bản `cologne1` và `ingolstadt1`.
- Produces:
  - Bảng so sánh 5 metrics định dạng bảng Terminal và Markdown.
  - File JSON kết quả: `results/fixed_time_evaluation.json`.
  - Biểu đồ so sánh cột trực quan hóa: `results/fixed_time_metrics.png`.

- [x] **Step 1: Viết test kiểm tra luồng tổng hợp và vẽ biểu đồ so sánh**
  Kiểm tra hàm tạo bảng và xuất file kết quả.

- [x] **Step 2: Chạy test để xác nhận FAIL**
  Run: `pytest tests/test_evaluation_flow.py -v`
  Expected: FAIL.

- [x] **Step 3: Cài đặt `src/evaluate.py` và `main.py`**
  Hỗ trợ chạy tự động 2 map với cờ `--gui` (xem trực quan) hoặc headless, xuất bảng tổng hợp và biểu đồ.

- [x] **Step 4: Chạy test để xác nhận PASS**
  Run: `pytest tests/test_evaluation_flow.py -v`
  Expected: PASS.

- [x] **Step 5: Commit**
  `git add src/evaluate.py main.py tests/test_evaluation_flow.py`
  `git commit -m "feat: implement multi-map evaluation runner and report generator"`

---

### Task 6: Chạy Thực nghiệm Thực tế trên cả 2 Map & Phân tích Đánh giá

**Files:**
- Output: `results/fixed_time_evaluation.json`
- Output: `results/fixed_time_metrics.png`
- Output: `results/REPORT.md`

- [x] **Step 1: Chạy mô phỏng thực tế kịch bản `cologne1`**
  Run: `python main.py --scenario cologne1`
  Thu thập kết quả 5 metrics.

- [x] **Step 2: Chạy mô phỏng thực tế kịch bản `ingolstadt1`**
  Run: `python main.py --scenario ingolstadt1`
  Thu thập kết quả 5 metrics.

- [x] **Step 3: Chạy kịch bản so sánh toàn diện và sinh biểu đồ**
  Run: `python main.py --all`
  Tạo biểu đồ `results/fixed_time_metrics.png` và file tổng hợp `results/REPORT.md`.

- [x] **Step 4: Phân tích số liệu thực nghiệm**
  Đánh giá hiệu quả của chu kỳ cố định 30 giây đối với lưu lượng thực tế của Cologne 1 (đối xứng/bất đối xứng) và Ingolstadt 1 (nguy cơ tràn hàng đợi hoặc thời gian chết pha).
