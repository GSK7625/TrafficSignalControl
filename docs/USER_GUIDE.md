# HƯỚNG DẪN SỬ DỤNG NỀN TẢNG MÔ PHỎNG SUMO (GIAI ĐOẠN 2)

Hệ thống cung cấp Simulation Runner bằng Python và TraCI để chạy, kiểm tra và thu thập chỉ số mô phỏng giao thông trên dataset RESCO.

---

## 1. Yêu cầu môi trường

- **Hệ điều hành:** Windows 10/11
- **Python:** 3.11+
- **SUMO:** Phiên bản 1.28.0+ (đã cấu hình biến môi trường `SUMO_HOME` trỏ tới thư mục cài đặt SUMO, ví dụ `C:\Program Files (x86)\SUMO\`)
- **Các thư viện Python:**
  - `traci` (giao tiếp TraCI với SUMO)
  - `sumolib` (tiện ích mạng đường SUMO)
  - `unittest` (kiểm thử tích hợp sẵn)

---

## 2. Cấu trúc mã nguồn

```text
TrafficSignalControl/
├── data/                             # Thư mục dataset RESCO gốc (chỉ đọc)
├── results/                          # Thư mục lưu trữ kết quả mô phỏng (tách biệt)
├── docs/                             # Tài liệu kỹ thuật
│   ├── DATASET_INSPECTION_REPORT.md  # Báo cáo kiểm tra chi tiết dataset RESCO
│   └── USER_GUIDE.md                 # Hướng dẫn sử dụng nền tảng mô phỏng
├── src/                              # Mã nguồn chính của nền tảng
│   ├── __init__.py
│   ├── config.py                     # Quản lý cấu hình SimulationConfig
│   ├── data_inspector.py             # Trình kiểm tra & phân tích dữ liệu XML SUMO
│   ├── metrics.py                    # Thu thập & xuất chỉ số (waiting time, queue, ...)
│   ├── phase_manager.py              # Bộ quản lý pha đèn an toàn (SafePhaseManager)
│   └── runner.py                     # SimulationRunner điều khiển vòng đời TraCI
├── tests/                            # Bộ kiểm thử tự động
│   ├── __init__.py
│   ├── test_config.py                # Kiểm thử cấu hình
│   ├── test_data_inspector.py        # Kiểm thử bộ phân tích dữ liệu
│   ├── test_integration_sumo.py      # Kiểm thử tích hợp chạy SUMO thật qua TraCI
│   ├── test_metrics.py               # Kiểm thử bộ thu thập chỉ số
│   ├── test_phase_manager.py         # Kiểm thử quản lý pha an toàn
│   └── test_runner.py                # Kiểm thử runner
└── main.py                           # Điểm vào dòng lệnh (CLI)
```

---

## 3. Các lệnh sử dụng chính

### 3.1. Kiểm tra thông tin dataset

Kiểm tra thông tin chi tiết của một kịch bản cụ thể:
```powershell
python main.py --inspect --dataset cologne1
python main.py --inspect --dataset ingolstadt1
```

Kiểm tra và in bảng tổng hợp của toàn bộ 10 dataset RESCO:
```powershell
python main.py --inspect-all
```

---

### 3.2. Chạy mô phỏng (Simulation)

#### a) Chạy chế độ Headless (mặc định — chạy ngầm với hiệu năng cao)
Chạy kịch bản `cologne1` 100 bước:
```powershell
python main.py --dataset cologne1 --steps 100
```

Chạy kịch bản `ingolstadt1` 200 bước:
```powershell
python main.py --dataset ingolstadt1 --steps 200
```

Chạy toàn bộ kịch bản (mặc định 3600 giây):
```powershell
python main.py --dataset cologne1
```

#### b) Chạy với giao diện đồ họa trực quan (SUMO-GUI)
Thêm cờ `--gui` vào bất kỳ lệnh nào:
```powershell
python main.py --dataset cologne1 --gui --steps 300
python main.py --dataset ingolstadt1 --gui --steps 300
python main.py --dataset cologne3 --gui --steps 300
```
*Ghi chú: Khi bật GUI, SUMO sẽ tự động mở cửa sổ đồ họa, chạy mô phỏng và đóng cửa sổ khi kết thúc.*

#### c) Tùy biến chế độ điều khiển và thuật toán tối ưu (Giai đoạn 4)
- `--controller`: Thuật toán điều khiển tín hiệu đèn:
  - `none` (mặc định): Hoạt động theo cờ `--mode`.
  - `fixed-time`: Bộ điều khiển Fixed-Time theo chu kỳ và thời lượng cố định định trước.
  - `max-pressure`: Bộ điều khiển Max-Pressure tối ưu hóa áp lực hàng đợi giữa làn vào và làn ra.
- `--fixed-time-green`: Thời gian đèn xanh cho Fixed-Time (giây), mặc định `30.0`.
- `--pressure-metric`: Tiêu chí đo áp lực Max-Pressure: `queue` (số xe dừng chờ < 0.1 m/s, mặc định) hoặc `vehicle_count` (tổng số xe trên làn).
- `--mode`: Chế độ điều khiển tín hiệu đèn khi `--controller none`:
  - `original` (mặc định cho baseline): SUMO tự thực thi chương trình đèn gốc được định nghĩa sẵn trong mạng (`.net.xml`), không bị TraCI can thiệp hay ép chu kỳ cố định.
  - `cyclic`: Bộ điều khiển xoay vòng các pha xanh định kỳ 30s.
  - `agent`: Chế độ dành cho thuật toán hoặc bộ điều khiển bên ngoài (khi nhận `actions={}` sẽ giữ nguyên pha an toàn mà không tự ý đổi).
- `--yellow-dur`: Thời gian pha vàng giải tỏa (giây), mặc định `3.0` (chuẩn benchmark RESCO).
- `--all-red-dur`: Thời gian toàn đỏ giải tỏa (giây), mặc định `0.0` (chuẩn benchmark RESCO; có thể cấu hình lên `1.0`-`2.0` cho mô phỏng an toàn giao thông thực tế).
- `--min-green`: Thời gian đèn xanh tối thiểu (giây), mặc định `5.0`.
- `--tripinfo`: Tùy chọn sinh và trích xuất chỉ số đối chiếu chuẩn tuyệt đối từ tệp XML `--tripinfo-output` của SUMO.
- `--step-len`: Độ dài mỗi bước mô phỏng (giây), mặc định `1.0`.
- `--output-dir`: Thư mục lưu kết quả, mặc định `results`.

Ví dụ thực thi các thuật toán:
```powershell
# 1. Chạy thuật toán Fixed-Time (chu kỳ xanh 30s) trên Cologne1:
python main.py --dataset cologne1 --controller fixed-time --fixed-time-green 30.0 --tripinfo

# 2. Chạy thuật toán Max-Pressure trên Ingolstadt1:
python main.py --dataset ingolstadt1 --controller max-pressure --min-green 5.0 --tripinfo

# 3. Chạy baseline nguyên bản của SUMO trên Cologne1:
python main.py --dataset cologne1 --mode original --tripinfo

# 4. Chạy toàn bộ thực nghiệm tự động so sánh Fixed-Time, Max-Pressure, Original và sinh biểu đồ:
python -m src.experiments.run_experiments
```

---

## 4. Định dạng kết quả xuất ra (`results/`)

Mỗi lần chạy sẽ tạo một thư mục con tương ứng với kịch bản và timestamp, ví dụ:
`results/cologne1_20261012_174444/`

Bao gồm 2 tệp:
1. `summary.json`: Báo cáo thống kê tổng hợp (tách bạch rõ ràng giữa thời gian chờ, độ trễ hàng đợi làn vào `total_queue_delay`, và số liệu ground-truth `tripinfo_*` nếu bật `--tripinfo`):
   ```json
   {
     "scenario_name": "cologne1",
     "total_steps": 3600,
     "total_completed_vehicles": 1998,
     "mean_queue_length": 6.43,
     "max_queue_length": 24,
     "mean_waiting_time": 13.32,
     "total_waiting_time": 26613.36,
     "total_queue_delay": 23148.0,
     "mean_travel_time": 56.27,
     "total_time_loss": 67048.9,
     "tripinfo_completed_vehicles": 1998,
     "tripinfo_mean_travel_time": 56.27,
     "tripinfo_mean_waiting_time": 13.32,
     "controller_name": "MaxPressure"
   }
   ```
2. `step_metrics.csv`: Dữ liệu chuỗi thời gian chi tiết theo từng bước mô phỏng:
   - `step`: Thứ tự bước (1, 2, ...).
   - `sim_time`: Thời gian mô phỏng thực tế của kịch bản (ví dụ 25201.0s).
   - `running_vehicles`: Số phương tiện đang lưu thông trên mạng.
   - `queue_length`: Tổng số xe đang dừng chờ (tốc độ < 0.1 m/s) trên các làn vào của nút giao.
   - `waiting_time`: Tổng thời gian chờ tích lũy của các xe đang chạy.
   - `time_loss`: Tổng thời gian hao phí (trễ) do tắc nghẽn.
   - `step_completed_vehicles`: Số xe về đích trong bước đó.
   - `cumulative_completed_vehicles`: Số xe về đích cộng dồn.

Thư mục `results/experiments/` chứa kết quả tổng hợp của toàn bộ các thử nghiệm đối chuẩn:
- `benchmark_summary.json`: Bảng dữ liệu JSON gộp tất cả thuật toán và kịch bản.
- `benchmark_report.md`: Báo cáo khoa học markdown tổng kết số liệu 9 chỉ số bắt buộc.
- `charts/`: 5 biểu đồ so sánh (`waiting_time_comparison.png`, `queue_length_comparison.png`, `throughput_comparison.png`, `time_loss_comparison.png`, `combined_metrics_dashboard.png`).

---

## 5. Chạy bộ kiểm thử tự động (Unit & Integration Tests)

Chạy toàn bộ 106 bài kiểm thử tự động bằng pytest:
```powershell
pytest -v
```

Hoặc qua module unittest:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Chạy riêng kiểm thử RL:
```powershell
pytest tests/test_rl_components.py tests/test_rl_env.py tests/test_q_learning.py tests/test_dqn.py -v
```

---

## 6. Huấn luyện và Đánh giá Học tăng cường (Giai đoạn 5: Q-Learning & DQN)

### 6.1. Huấn luyện Q-Learning (Tabular Q-Learning)
```powershell
# Huấn luyện trên Ingolstadt1 (15 episodes, seed 42):
python -m src.training.train_q_learning --dataset ingolstadt1 --episodes 15 --seed 42

# Huấn luyện trên Cologne1 (15 episodes, seed 42):
python -m src.training.train_q_learning --dataset cologne1 --episodes 15 --seed 42
```

### 6.2. Huấn luyện Deep Q-Network (DQN)
```powershell
# Huấn luyện trên Ingolstadt1 (15 episodes, seed 42):
python -m src.training.train_dqn --dataset ingolstadt1 --episodes 15 --seed 42

# Huấn luyện trên Cologne1 (15 episodes, seed 42):
python -m src.training.train_dqn --dataset cologne1 --episodes 15 --seed 42
```

### 6.3. Đánh giá độc lập (Evaluation Mode)
Chạy mô hình ở chế độ suy luận hoàn toàn không tiếp tục cập nhật trọng số và tắt exploration ngẫu nhiên ($\epsilon = 0$):
```powershell
# Đánh giá Q-Learning:
python -m src.evaluation.evaluate_rl --dataset cologne1 --algo q_learning --model models/q_learning/q_learning_cologne1_best.json
python -m src.evaluation.evaluate_rl --dataset ingolstadt1 --algo q_learning --model models/q_learning/q_learning_ingolstadt1_best.json

# Đánh giá DQN:
python -m src.evaluation.evaluate_rl --dataset cologne1 --algo dqn --model models/dqn/dqn_cologne1_best.pt
python -m src.evaluation.evaluate_rl --dataset ingolstadt1 --algo dqn --model models/dqn/dqn_ingolstadt1_best.pt
```

### 6.4. Tổng hợp báo cáo so sánh và sinh biểu đồ
```powershell
# Sinh bảng đối chiếu và báo cáo khoa học 4 thuật toán:
python -m src.evaluation.compare_algorithms

# Sinh toàn bộ 16 biểu đồ tiến trình học và so sánh hiệu năng:
python -m src.experiments.plot_rl
```

### 6.5. Chạy quan sát mô phỏng trực quan (SUMO-GUI) với Q-Learning và DQN
```powershell
python main.py --dataset cologne1 --controller q-learning --gui --steps 300
python main.py --dataset cologne1 --controller dqn --gui --steps 300
```

---

## 7. Giai đoạn 6: Tự động hóa Thực nghiệm Đa Seed và Đối chuẩn Toàn diện (Phase 6 Benchmark)

### 7.1. Chạy một thực nghiệm đánh giá đơn lẻ
Thực thi kiểm chuẩn một thuật toán trên một kịch bản với một seed cụ thể:
```powershell
# Chạy Fixed-Time trên Cologne1 (Seed 42):
python -m src.evaluation.experiment_runner --scenario cologne1 --algorithm Fixed-Time --seed 42

# Chạy Max-Pressure trên Ingolstadt1 (Seed 101):
python -m src.evaluation.experiment_runner --scenario ingolstadt1 --algorithm Max-Pressure --seed 101

# Chạy Q-Learning trên Cologne1 (Seed 42):
python -m src.evaluation.experiment_runner --scenario cologne1 --algorithm Q-Learning --seed 42

# Chạy DQN trên Cologne1 (Seed 42):
python -m src.evaluation.experiment_runner --scenario cologne1 --algorithm DQN --seed 42
```

### 7.2. Chạy toàn bộ ma trận thực nghiệm chuẩn hóa (Full Benchmark Matrix)
Tự động chạy toàn bộ 30 lượt mô phỏng độc lập (full 3600 giây, đa seed 42, 101, 2024 cho cả 5 thuật toán trên Cologne1 và Ingolstadt1), kiểm tra tính hợp lệ và xuất bảng tổng hợp:
```powershell
python -m src.evaluation.experiment_runner --run-all
```
- Dữ liệu từng lượt chạy lưu tại: `results/evaluation/runs/<run_id>/`
- Dữ liệu bảng tổng hợp các lượt chạy: `results/evaluation/benchmark_runs.csv`
- Báo cáo phân tích thống kê (mean, std, min, max, % cải thiện): `results/evaluation/benchmark_summary.json`

### 7.3. Tạo lại toàn bộ 30 biểu đồ học thuật xuất bản (Publication Charts)
Sinh đồng loạt tất cả các biểu đồ Nhóm A (so sánh thuật toán có error bars), Nhóm B (diễn biến động thời gian) và Nhóm C (tiến trình học tăng cường):
```powershell
python -m src.evaluation.plot_comparison
```
Toàn bộ biểu đồ định dạng PNG độ phân giải cao 300 DPI được lưu tại thư mục: `results/figures/`.

### 7.4. Chạy kiểm thử tự động toàn diện (Unit & Integration Tests)
```powershell
pytest -v
```
Toàn bộ 115/115 bài kiểm thử tự động của dự án sẽ được kiểm chứng tự động.

