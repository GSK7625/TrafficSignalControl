# TrafficSignalControl

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/Eclipse%20SUMO-1.28.0+-00599C?style=flat)](https://eclipse.dev/sumo/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen?style=flat&logo=pytest)](tests/)

Nền tảng mô phỏng và đánh giá đối chuẩn (Benchmark) các thuật toán điều khiển tín hiệu đèn giao thông thích ứng (**Adaptive Traffic Signal Control - ATSC**) trên môi trường **Eclipse SUMO** (hỗ trợ TraCI và libsumo) với các kịch bản chuẩn từ bộ dữ liệu **RESCO**.

---

## 🚀 Tính năng nổi bật

- **3 thuật toán cốt lõi**: **Fixed-Time** (chu kỳ cố định 30s), **Max-Pressure** (heuristic giải tỏa áp lực theo làn), và **Tabular Q-Learning** (học tăng cường dạng bảng).
- **Bộ quản lý an toàn pha đèn (SafePhaseManager)**: Tự động chèn pha vàng dọn nút và pha đỏ toàn phần, đảm bảo ràng buộc `min_green >= 5s` và `max_green <= 60s`.
- **Hỗ trợ linh hoạt libsumo & SUMO-GUI**: Tự động dùng `libsumo` in-process tốc độ cao khi chạy headless, hoặc chuyển sang `SUMO-GUI` (`--gui`, `--gui-delay`) để quan sát trực quan theo thời gian thực.
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
├── tests/            # Bộ kiểm thử tự động
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

# 2. Chạy thuật toán Tabular Q-Learning (tự động nạp checkpoint tốt nhất)
python main.py -a q-learning

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

Chạy bộ kiểm thử tự động (unit tests và regression tests) bao quát toàn bộ logic điều khiển, quản lý an toàn và tích hợp mô phỏng:

```powershell
pytest -q
```

## Độ tin cậy kết quả

- Schema kết quả mới: `5.0.0`; nhu cầu xe được đếm đúng cửa sổ chạy.
- Đánh giá Q-Learning yêu cầu checkpoint tồn tại và đúng số action. `cologne3` chưa có checkpoint đi kèm; hãy huấn luyện trước khi đánh giá.
- Dữ liệu bắt buộc bị lỗi làm thất bại lượt chạy; dữ liệu thiếu không được tự thay bằng số 0. Báo cáo phân biệt dữ liệu hợp lệ, an toàn và tỷ lệ hoàn thành.
- Resume chỉ dùng lại kết quả có cùng cấu hình, dữ liệu và nội dung checkpoint. Kết quả cũ thiếu fingerprint cần chạy lại; giữ artifacts lịch sử để tham khảo.
- TraCI và libsumo được hỗ trợ. `libsumo` là lựa chọn bổ sung cho headless; nếu không cài, runner dùng TraCI.
- Bộ đếm demand hiện hỗ trợ route dạng `trip` và `vehicle` với thời điểm khởi hành bằng số; chưa hỗ trợ `flow` hoặc departure động.
