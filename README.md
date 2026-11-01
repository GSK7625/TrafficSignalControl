# TrafficSignalControl: Đối Chuẩn 4 Thuật Toán Điều Khiển Đèn Giao Thông (SUMO)

Dự án nghiên cứu, triển khai và đối chuẩn thực nghiệm 4 thuật toán điều khiển tín hiệu đèn giao thông trên nền tảng mô phỏng **SUMO (Simulation of Urban MObility)**:
1. **Fixed-Time (FT)**: Điều khiển thời gian cố định (Baseline chu kỳ 30s).
2. **Max-Pressure (MP)**: Thuật toán thích ứng phân tán dựa trên chênh lệch áp lực hàng đợi các nhánh vào/ra.
3. **Tabular Q-Learning (QL)**: Học tăng cường dạng bảng với không gian trạng thái rời rạc hóa.
4. **Deep Q-Network (DQN)**: Học tăng cường sâu kết hợp mạng nơ-ron (PyTorch) xấp xỉ hàm giá trị Q liên tục.

Mọi thuật toán đều được kiểm thử trên 2 bản đồ thực tế với cấu trúc khác biệt:
- **Cologne1**: Nút giao 4 pha tại Cologne, lưu lượng cao (~1.970 xe/giờ).
- **Ingolstadt1**: Nút giao 3 pha tại Ingolstadt, lưu lượng trung bình (~1.716 xe/giờ).

Tất cả các thuật toán dùng chung cơ chế an toàn dọn nút giao: pha vàng ($Y=3$s) và pha đỏ toàn phần ($R_{\text{all}}=1$s).

---

## 1. Cài Đặt Môi Trường

Yêu cầu máy tính đã cài đặt **SUMO** (đã thiết lập biến môi trường `SUMO_HOME`) và **Python 3.10+**.

```powershell
# Kích hoạt môi trường ảo (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 2. Huấn Luyện Mô Hình RL (Q-Learning & DQN)

### 2.1. Huấn luyện Tabular Q-Learning (30 episodes)
```powershell
# Huấn luyện trên Cologne1
python -m src.train_q_learning --scenario cologne1 --episodes 30

# Huấn luyện trên Ingolstadt1
python -m src.train_q_learning --scenario ingolstadt1 --episodes 30
```
Mô hình checkpoint tốt nhất được lưu tại: `models/q_learning/<scenario>_best.json`.

### 2.2. Huấn luyện Deep Q-Network (DQN - 30 episodes)
```powershell
# Huấn luyện trên Cologne1
python -m src.train_dqn --scenario cologne1 --episodes 30

# Huấn luyện trên Ingolstadt1
python -m src.train_dqn --scenario ingolstadt1 --episodes 30
```
Mô hình trọng số PyTorch được lưu tại: `models/dqn/<scenario>_best.pt`.

---

## 3. Chạy Đánh Giá Độc Lập (Evaluation qua 5 Seeds)

Tập hạt giống đánh giá chuẩn: `[42, 101, 2024, 3407, 9999]`, mỗi lượt chạy trọn vẹn 3.600 giây (1 giờ mô phỏng).

```powershell
# 1. Đánh giá Fixed-Time
python -m src.evaluate_fixed_time

# 2. Đánh giá Max-Pressure
python -m src.evaluate_max_pressure

# 3. Đánh giá Tabular Q-Learning
python -m src.evaluate_q_learning

# 4. Đánh giá Deep Q-Network (DQN)
python -m src.evaluate_dqn
```

---

## 4. Tổng Hợp Kết Quả & Vẽ Biểu Đồ

```powershell
# Tổng hợp bảng so sánh số liệu Mean ± Std và phần trăm cải thiện
python -m src.consolidate_results

# Vẽ biểu đồ trực quan hóa (lưu tại results/plots/)
python -m src.plot_benchmark
```

Báo cáo phân tích lý thuyết và thực nghiệm chi tiết xem tại: [`docs/benchmark_evaluation_report.md`](docs/benchmark_evaluation_report.md).

---

## 5. Chạy Kiểm Thử Tự Động (Unit Tests)

```powershell
pytest
```
Toàn bộ 22 bài kiểm thử (bao gồm đối chiếu tính tay, kiểm tra biên rỗng, kiểm thử an toàn chuyển pha và từng bộ điều khiển) sẽ được thực thi.

---

## 6. Cấu Trúc Thư Mục Dự Án

```text
TrafficSignalControl/
├── AGENT.md                       # Nguyên tắc phát triển và chuẩn mực code
├── docs/                          # Tài liệu đặc tả kỹ thuật và báo cáo benchmark
│   ├── benchmark_evaluation_report.md
│   ├── evaluation_specification.md
│   ├── phase_specification.md
│   └── q_learning_specification.md
├── data/                          # Mạng lưới SUMO (cologne1, ingolstadt1)
├── models/                        # Trọng số mô hình đã huấn luyện
│   ├── dqn/                       # Trọng số PyTorch (.pt)
│   └── q_learning/                # Bảng Q-table JSON (.json)
├── results/                       # Kết quả thực nghiệm và biểu đồ
│   ├── dqn/
│   ├── fixed_time/
│   ├── max_pressure/
│   ├── plots/
│   └── q_learning/
├── src/                           # Mã nguồn chính
│   ├── controllers/               # Các bộ điều khiển tín hiệu đèn
│   ├── simulation_runner.py       # Vòng lặp mô phỏng tích hợp TraCI
│   ├── metrics.py                 # Bộ thu thập và tính toán 5 chỉ số
│   ├── train_q_learning.py        # Huấn luyện Q-Learning
│   ├── train_dqn.py               # Huấn luyện DQN
│   ├── evaluate_*.py              # Đánh giá 5 seeds độc lập
│   ├── consolidate_results.py     # Bảng tổng hợp Mean ± Std
│   └── plot_benchmark.py          # Vẽ biểu đồ kết quả
└── tests/                         # Bộ kiểm thử tự động (pytest)
```
