# TrafficSignalControl: SUMO Reinforcement Learning & Baseline Benchmark

Hệ thống điều khiển tín hiệu đèn giao thông thích ứng (Adaptive Traffic Signal Control - ATSC) trên nền tảng mô phỏng **Eclipse SUMO** với các kịch bản chuẩn thuộc tập dữ liệu **RESCO** (Reinforcement Learning Benchmarks for Traffic Signal Control).

Hệ thống tích hợp 4 thuật toán điều khiển tín hiệu cùng baseline mặc định của SUMO:
1. **Fixed-Time Controller**: Điều khiển chu kỳ pha cố định chuẩn hóa.
2. **Max-Pressure Controller**: Điều khiển dựa trên áp lực chênh lệch hàng đợi vào/ra của mạng lưới.
3. **Tabular Q-Learning**: Học tăng cường bảng rời rạc hóa không gian trạng thái hàng đợi và pha đèn.
4. **Deep Q-Network (DQN)**: Mạng neural sâu xấp xỉ Q-function với Replay Buffer, Target Network và Action Masking an toàn.
5. **Original SUMO**: Kế hoạch tín hiệu gốc được cấu hình trong kịch bản SUMO.

---

## 1. Yêu cầu hệ thống và Cài đặt

### Yêu cầu:
- **Hệ điều hành:** Windows 10/11 hoặc Linux
- **Python:** 3.11+
- **SUMO:** Phiên bản 1.28.0+ (yêu cầu cấu hình biến môi trường `SUMO_HOME`, ví dụ: `C:\Program Files (x86)\SUMO`)

### Thiết lập môi trường ảo và cài đặt thư viện:
```powershell
# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

---

## 2. Hướng dẫn sử dụng CLI

### 2.1. Kiểm tra thông tin kịch bản (Dataset Inspection)
```powershell
# Kiểm tra một kịch bản cụ thể
python main.py --inspect --dataset ingolstadt1
python main.py --inspect --dataset cologne1

# Phân tích tổng thể toàn bộ 10 kịch bản RESCO
python main.py --inspect-all
```

### 2.2. Chạy mô phỏng điều khiển giao thông
Mỗi thuật toán hỗ trợ chạy thông qua `main.py` với nhịp điều khiển `--decision-interval` chuẩn hóa (mặc định 5.0s cho RL và 1.0s cho baseline):
```powershell
# Chạy Fixed-Time trên kịch bản Cologne1
python main.py --dataset cologne1 --algorithm fixed-time

# Chạy Max-Pressure trên kịch bản Ingolstadt1
python main.py --dataset ingolstadt1 --algorithm max-pressure

# Chạy mô hình DQN đã huấn luyện với giao diện trực quan (GUI)
python main.py --dataset ingolstadt1 --algorithm dqn --model models/dqn/dqn_ingolstadt1_best.pt --gui
```

### 2.3. Huấn luyện Agent Học tăng cường (RL Training)
Huấn luyện hỗ trợ nạp cấu hình YAML và đánh giá Greedy Validation định kỳ trên seed kiểm định cố định để chọn checkpoint tốt nhất:
```powershell
# Huấn luyện DQN với file cấu hình
python -m src.training.train_dqn --dataset ingolstadt1 --config configs/dqn.yaml --episodes 15

# Huấn luyện Tabular Q-Learning
python -m src.training.train_q_learning --dataset ingolstadt1 --config configs/q_learning.yaml --episodes 15
```

### 2.4. Thực thi Benchmark chuẩn hóa (Phase 6 Evaluation)
Chạy ma trận thực nghiệm đánh giá đối chiếu đa thuật toán, đa kịch bản với kiểm tra toàn vẹn Fingerprint SHA-256 và thống kê mẫu $ddof=1$:
```powershell
# Chạy benchmark hoàn chỉnh toàn bộ ma trận (30 runs, tự động resume)
python -m src.evaluation.experiment_runner --run-all

# Chạy thử nghiệm một thuật toán đơn lẻ
python -m src.evaluation.experiment_runner --scenario ingolstadt1 --algorithm DQN --seed 42
```

---

## 3. Kiểm thử tự động (Test Suite)

Chạy toàn bộ bộ kiểm thử đơn vị và tích hợp:
```powershell
pytest -q
```
Tất cả các kiểm thử bao gồm kiểm tra tính đúng thuật toán, an toàn pha đèn, xử lý ngoại lệ TraCI, bộ đệm Replay Buffer, và thống kê thực nghiệm đều được thực thi tự động.
