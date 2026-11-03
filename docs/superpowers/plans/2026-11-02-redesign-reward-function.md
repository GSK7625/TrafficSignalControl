# Kế Hoạch Triển Khai: Tái Thiết Kế Hàm Phần Thưởng Delay-Aware Pressure (DAP-Reward)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai hàm phần thưởng Delay-Aware Pressure (DAP) kết hợp áp suất nút giao, thời gian chờ lũy kế trên các làn vào và chi phí phạt đổi đèn, giải quyết dứt điểm hiện tượng bỏ đói đường nhánh và nâng cao toàn diện 5 chỉ số benchmark (Waiting Time, Travel Time, Time Loss, Queue, Throughput).

**Architecture:** Mở rộng `simulation_runner.py` để trích xuất `step_waiting_time` từ TraCI qua `traci.lane.getWaitingTime` mỗi giây; nâng cấp phương thức `observe_step` trong cả `DQNController` và `QLearningController` để tính toán hàm thưởng đa mục tiêu chuẩn hóa theo số làn vào; tối ưu hóa bộ đếm thời gian đỏ `phase_red_timers` để duy trì chính xác giám sát chống đói pha.

**Tech Stack:** Python 3.10+, PyTorch, SUMO (TraCI / libsumo), pytest.

**Spec:** [`docs/reward_redesign_specification.md`](file:///c:/Users/dotru/Documents/TrafficSignalControl/docs/reward_redesign_specification.md)

## Global Constraints
- Tuân thủ nghiêm ngặt quy tắc `AGENT.md`: giữ code đơn giản, trực tiếp, giải thích rõ công thức toán học và vị trí code tương ứng.
- Đảm bảo 100% tính tương thích ngược với các test case hiện có; không phá vỡ giao diện của các bộ điều khiển Fixed-Time và Max-Pressure.
- Giữ nguyên tập 5 test seeds độc lập `[42, 101, 2024, 3407, 9999]` cho khâu đánh giá cuối cùng.

## Review Focus
- Sự ổn định của hàm thưởng khi không có backend mô phỏng (trong unit tests hoặc mocking): tự động fallback an toàn về giá trị xấp xỉ `step_queue * step_duration`.
- Chuẩn hóa quy mô thời gian chờ theo số làn vào ($N_{\text{lanes}}$) để giá trị reward không bị thổi phồng lệch giữa nút 3 làn và nút 8 làn.
- Chi phí phạt chuyển pha ($c_{\text{switch}}$) chỉ được tính đúng 1 lần duy nhất tại mốc thời gian ra quyết định SWITCH, không bị cộng dồn lặp lại ở các giây đèn vàng/đỏ tiếp theo.
- Bộ đếm thời gian đỏ `phase_red_timers` không bị reset về 0 khi có xe giảm tốc từ từ tiếp cận vạch dừng.
- Đảm bảo tính toán discount factor $\gamma_{\text{step}}^k$ của SMDP vẫn chính xác tuyệt đối trên từng bước thời gian 1s.

---

### Task 1: Mở rộng `simulation_runner.py` truyền `step_waiting_time`
**Files:**
- Modify: `src/simulation_runner.py:170-200`
- Test: `tests/test_metrics_and_runner.py`

- [ ] Viết unit test trong `tests/test_metrics_and_runner.py` kiểm tra `simulation_runner` trích xuất `step_waiting_time` và truyền cho controller.
- [ ] Chạy test để xác nhận test chạy đúng kỳ vọng.
- [ ] Chỉnh sửa `src/simulation_runner.py`: sau mỗi `traci.simulationStep()`, tính tổng thời gian chờ từ `traci.lane.getWaitingTime(lane_id)` của `incoming_lanes` và truyền vào `controller.observe_step(step_queue=step_queue, step_waiting_time=step_waiting_time)`.
- [ ] Chạy lại `pytest tests/test_metrics_and_runner.py` để xác nhận thành công.

---

### Task 2: Nâng cấp hàm thưởng DAP và hiệu chỉnh Anti-starvation trong `DQNController`
**Files:**
- Modify: `src/controllers/dqn.py:520-575`
- Test: `tests/test_dqn.py`

- [ ] Viết unit test trong `tests/test_dqn.py` kiểm chứng:
  1. Công thức tính step_reward mới khi có `step_waiting_time`.
  2. Xử lý phạt chuyển pha $c_{\text{switch}}$ khi $a \ne p_{\text{curr}}$.
  3. Bộ đếm `phase_red_timers` không bị reset đột ngột khi vận tốc xe tiếp cận vạch dừng dao động.
- [ ] Chạy pytest kiểm tra test thất bại trước khi cài đặt.
- [ ] Cập nhật `DQNController`:
  - Thêm thuộc tính `beta_wait: float = 0.05`, `switch_cost: float = 1.0`.
  - Cập nhật `observe_step(step_queue, step_pressure, step_waiting_time)` áp dụng công thức DAP-Reward.
  - Sửa logic tích lũy `phase_red_timers` trong `step()` để chỉ reset khi pha đó chuyển sang xanh.
- [ ] Chạy lại `pytest tests/test_dqn.py` xác nhận toàn bộ test passed.

---

### Task 3: Nâng cấp hàm thưởng DAP và hiệu chỉnh Anti-starvation trong `QLearningController`
**Files:**
- Modify: `src/controllers/q_learning.py:410-460`
- Test: `tests/test_q_learning.py`

- [ ] Viết unit test trong `tests/test_q_learning.py` kiểm chứng công thức DAP-Reward và cơ chế phạt chuyển pha cho Q-Learning dạng bảng.
- [ ] Cập nhật `QLearningController` đồng bộ cấu trúc tham số `beta_wait`, `switch_cost` và công thức DAP-Reward.
- [ ] Hiệu chỉnh logic `phase_red_timers` đồng bộ với DQNController.
- [ ] Chạy lại `pytest tests/test_q_learning.py` xác nhận 100% test passed.

---

### Task 4: Kiểm thử hồi quy toàn diện
**Files:**
- Test: toàn bộ `tests/`

- [ ] Chạy lệnh `pytest` trên toàn bộ 48+ bài kiểm thử tự động, bảo đảm không có bất kỳ regression lỗi nào.

---

### Task 5: Huấn luyện lại và Đánh giá Benchmark 5 Test Seeds
**Files:**
- Script: `src/train_dqn.py`, `src/train_q_learning.py`, `src/evaluate_*.py`, `src/consolidate_results.py`
- Output: `models/`, `results/`, `docs/benchmark_evaluation_report.md`

- [ ] Huấn luyện lại Tabular Q-Learning trên `ingolstadt1` và `cologne1` (30 episodes).
- [ ] Huấn luyện lại Deep Q-Network trên `ingolstadt1` và `cologne1` (30 episodes).
- [ ] Chạy đánh giá độc lập 5 test seeds qua `evaluate_q_learning` và `evaluate_dqn`.
- [ ] Chạy `consolidate_results` và xuất biểu đồ `plot_benchmark`.
- [ ] Cập nhật số liệu so sánh chi tiết vào `docs/benchmark_evaluation_report.md`.
