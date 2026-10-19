# TrafficSignalControl

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/Eclipse%20SUMO-1.28.0+-00599C?style=flat)](https://eclipse.dev/sumo/)
[![Tests](https://img.shields.io/badge/pytest-282%20passed-brightgreen?style=flat&logo=pytest)](tests/)
[![Schema](https://img.shields.io/badge/schema-6.0.0-blue?style=flat)](results/)

Nền tảng mô phỏng và đánh giá đối chuẩn (Benchmark) các thuật toán điều khiển tín hiệu đèn giao thông thích ứng (**Adaptive Traffic Signal Control - ATSC**) trên môi trường **Eclipse SUMO** (hỗ trợ TraCI và libsumo) với các kịch bản chuẩn từ bộ dữ liệu **RESCO**.

---

## 🚀 Tính năng nổi bật

- **3 thuật toán cốt lõi**: **Fixed-Time** (chu kỳ cố định 30s), **Max-Pressure** (heuristic giải tỏa áp lực theo làn), và **Tabular Q-Learning** (học tăng cường dạng bảng).
- **Bộ quản lý an toàn pha đèn (SafePhaseManager)**: Tự động chèn pha vàng dọn nút và pha đỏ toàn phần, đảm bảo ràng buộc `min_green >= 5s` và `max_green <= 60s`.
- **Hỗ trợ linh hoạt libsumo & SUMO-GUI**: Tự động dùng `libsumo` in-process tốc độ cao khi chạy headless, hoặc chuyển sang `SUMO-GUI` (`--gui`) để quan sát trực quan theo thời gian thực.
- **Tập dữ liệu chuẩn RESCO**: Tích hợp sẵn 3 kịch bản trọng tâm: `cologne1` (1 nút, 8 pha), `ingolstadt1` (1 nút, 3 pha) và `cologne3` (tuyến 3 nút liên tiếp).
- **Bộ đo lường & đối chuẩn chuẩn mực**: Phân tích ground-truth từ SUMO tripinfo, tính toán đa chỉ số (Waiting Time, Time Loss, Queue, Throughput) và xuất báo cáo/biểu đồ so sánh học thuật.

---

## 📁 Cấu trúc thư mục

```text
TrafficSignalControl/
├── configs/          # Cấu hình thử nghiệm & siêu tham số (experiments.yaml, q_learning.yaml)
├── data/             # Mạng lưới & luồng xe SUMO RESCO (cologne1, cologne3, ingolstadt1)
├── models/           # Checkpoints mô hình RL (*_best.json, *_final.json)
├── results/          # Kết quả đánh giá, log huấn luyện và biểu đồ (figures/)
├── src/              # Mã nguồn chính (controllers, rl, evaluation, runner, phase_manager)
├── tests/            # Bộ kiểm thử tự động (282 unit tests chuẩn hóa & mo-đun hóa)
├── main.py           # Điểm vào dòng lệnh (CLI)
└── requirements.txt  # Thư viện phụ thuộc
```

---

## 🛠️ Cài đặt & Chuẩn bị môi trường

### 1. Yêu cầu hệ thống
- **Python**: 3.11+
- **Eclipse SUMO**: 1.20.0+ (khuyến nghị 1.28.0+) và cấu hình biến môi trường `SUMO_HOME`.

### 2. Cài đặt dependencies
```powershell
# Kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Trên Linux/macOS: source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 💻 Hướng dẫn sử dụng CLI

Dự án thực thi hoàn toàn thông qua giao diện dòng lệnh (CLI). Bắt buộc chỉ định thuật toán điều khiển qua cờ `--algorithm` (`-a`):

```powershell
# 1. Chạy thuật toán Max-Pressure (kịch bản mặc định cologne1, headless libsumo tốc độ cao)
python main.py -a max-pressure

# 2. Chạy thuật toán Tabular Q-Learning (tự động nạp checkpoint tốt nhất sau khi train, hoặc dùng cờ -m <path>)
python main.py -a q-learning
# hoặc chỉ định trực tiếp file model đã huấn luyện:
python main.py -a q-learning -m models/q_learning/q_learning_cologne1_best.json

# 3. Chạy thuật toán Fixed-Time (chu kỳ cố định)
python main.py -a fixed-time

# 4. Xem mô phỏng trực quan với giao diện SUMO-GUI
python main.py -a max-pressure --gui

# 5. Đổi kịch bản giao thông (cologne1, ingolstadt1, cologne3) và giới hạn bước chạy
python main.py -d ingolstadt1 -a max-pressure --gui -s 500
```

### 3. Huấn luyện Agent RL (Tabular Q-Learning)
```powershell
python -m src.training.train_q_learning --dataset cologne1 --config configs/q_learning.yaml --episodes 15
```
*Checkpoint tốt nhất sẽ được lưu tự động tại `models/q_learning/q_learning_<dataset>_best.json`.*

### 4. Chạy Benchmark đối chuẩn & Tạo báo cáo
```powershell
# Chạy toàn bộ ma trận benchmark (2 kịch bản x 3 thuật toán x 3 seeds)
python -m src.evaluation.experiment_runner --run-all

# Tổng hợp kết quả so sánh dạng Markdown / JSON
python -m src.evaluation.compare_algorithms

# Tạo dashboard biểu đồ 300 DPI công bố học thuật
python -m src.evaluation.plot_comparison --dashboard-only
```

---

## 🧪 Kiểm thử tự động

Dự án trang bị bộ kiểm thử tự động toàn diện với **282 unit tests** bao quát 100% logic nghiệp vụ cốt lõi, quản lý an toàn và tích hợp mô phỏng SUMO:

```powershell
# Chạy toàn bộ 282 unit tests (tốc độ thực thi ~4 giây)
pytest -q
```

### Phân bổ các mo-đun kiểm thử:
- `tests/test_controllers.py` (46 tests): Kiểm thử các bộ điều khiển Fixed-Time, Max-Pressure, Tabular Q-Learning.
- `tests/test_phase_manager.py` (25 tests): Kiểm thử quản lý chuyển pha an toàn SafePhaseManager (min_green, max_green, yellow, all_red).
- `tests/test_rl_components.py` (19 tests): Rời rạc hóa StateDiscretizer, hàm thưởng Negative Pressure, môi trường TrafficSignalRLEnv.
- `tests/test_runner.py` (19 tests): Vòng lặp SimulationRunner, động cơ TraCI & libsumo, trích xuất pha xanh và thu thập quan sát.
- `tests/test_metrics.py` (27 tests): Thu thập chỉ số (MetricsCollector), reconciliation TripInfo, tính toán hàng đợi, độ trễ, hoàn thành.
- `tests/test_training.py` (37 tests): Pipeline huấn luyện Q-Learning, TrainingConfig, viability check, validation checkpoint selector.
- `tests/test_evaluation_runner.py` (32 tests): Điều phối thực nghiệm đa thuật toán, fingerprint resume caching, chạy song song / hàng loạt.
- `tests/test_compare_algorithms.py` (12 tests): Tổng hợp dữ liệu kết quả, đối chuẩn thuật toán, xuất báo cáo Markdown / JSON.
- `tests/test_config_and_inspector.py` (32 tests): Cấu hình mô phỏng, thanh tra cấu trúc mạng và luồng xe RESCO.
- `tests/test_simplification.py` (24 tests): Kiểm thử hồi quy logic cửa sổ demand `[begin, end)`, khớp ID TripInfo, timing overrides.
- `tests/test_main_cli.py` (6 tests): Điểm vào CLI `main.py`, parse tham số và xử lý ngoại lệ.


## Độ tin cậy kết quả

- Schema kết quả mới: `6.0.0`; nhu cầu xe được đếm trong cửa sổ `[begin, end)`, phân biệt `NO_DEMAND` với không có xe về đích.
- Đánh giá Q-Learning yêu cầu checkpoint tồn tại và đúng số action. `cologne3` chưa có checkpoint đi kèm; hãy huấn luyện trước khi đánh giá.
- TripInfo bắt buộc phải khớp chính xác tập ID xe hoàn thành. Thiếu, lỗi XML, trùng hoặc khác ID làm kết quả không hợp lệ. Lượt chẩn đoán không bật TripInfo không được dùng cho benchmark hoặc chọn checkpoint tốt nhất.
- Resume so khớp cấu hình hiệu lực, schema kết quả và nội dung checkpoint; fingerprint hiện chưa bao gồm nội dung dataset hoặc version SUMO. Khi thay dữ liệu/SUMO, dùng `--fresh`. Kết quả schema cũ cần chạy lại; giữ artifacts lịch sử để tham khảo.
- TraCI và libsumo được hỗ trợ. `libsumo` là lựa chọn bổ sung cho headless; nếu không cài, runner dùng TraCI.
- Bộ đếm demand hiện hỗ trợ route dạng `trip` và `vehicle` với thời điểm khởi hành bằng số; chưa hỗ trợ `flow` hoặc departure động.

## API sau khi tinh gọn

Các lệnh CLI ở trên được giữ nguyên. Mô phỏng Q-Learning cần checkpoint đã huấn luyện, không tự tạo policy chưa học khi thiếu model.

Training ưu tiên một đối tượng cấu hình:

```python
from src.training.train_q_learning import TrainingConfig, train_q_learning

config = TrainingConfig.from_yaml("configs/q_learning.yaml", num_episodes=15)
result = train_q_learning(config)
```

Các lời gọi bằng từ khóa như `train_q_learning(scenario_name="cologne1", num_episodes=15)` vẫn được hỗ trợ. Với lời gọi cũ truyền nhiều tham số theo vị trí, chuyển sang `TrainingConfig`. Không trộn config object với các từ khóa ghi đè. Thứ tự ưu tiên: mặc định < YAML < override được truyền tường minh; CLI bỏ trống không ghi đè YAML. `max_green=None` (hoặc CLI training `--max-green none`) tắt giới hạn xanh tối đa.

Benchmark dùng một dict cho những thông số thời gian muốn ghi đè:

```python
from src.evaluation.experiment_runner import ExperimentConfig, run_single_experiment

config = ExperimentConfig("cologne1", "Q-Learning", timing_overrides={"max_green": None})
record = run_single_experiment(config)
```

Thay các field timing cũ của `ExperimentConfig` bằng `timing_overrides`. Khi không ghi đè, Q-Learning kế thừa checkpoint, baseline dùng mặc định dự án. Thay `control_mode="cyclic"` bằng `FixedTimeController` với mode `agent`. Các tiện ích mở rộng, checkpoint cũ, hai backend, ba dataset và mọi nhóm biểu đồ vẫn được giữ.

Biểu đồ hỗ trợ dữ liệu nằm ngoài thư mục mặc định:

```powershell
python -m src.evaluation.plot_comparison --summary-file results/custom/benchmark_summary.json --runs-file results/custom/benchmark_runs.json --output-dir results/custom/figures
```
