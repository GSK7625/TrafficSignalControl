# TrafficSignalControl: Đối Chuẩn 4 Thuật Toán Điều Khiển Đèn Giao Thông (SUMO)

Dự án nghiên cứu, triển khai và đối chuẩn thực nghiệm 4 thuật toán điều khiển tín hiệu đèn giao thông trên nền tảng mô phỏng **SUMO (Simulation of Urban MObility)**:
1. **Fixed-Time (FT)**: Điều khiển chu kỳ cố định 30 giây (Baseline).
2. **Max-Pressure (MP)**: Điều khiển thích ứng tối ưu hóa chênh lệch áp lực hàng đợi các nhánh vào/ra.
3. **Tabular Q-Learning (QL)**: Học tăng cường dạng bảng với không gian trạng thái rời rạc thu gọn.
4. **Deep Q-Network (DQN)**: Học tăng cường sâu (PyTorch) với mạng neural xấp xỉ hàm giá trị Q.

Mọi thuật toán đều được kiểm thử trên 2 mạng lưới thực tế:
- **Cologne1**: Nút giao 8 pha, lưu lượng cao (nhu cầu 2.015 xe/giờ).
- **Ingolstadt1**: Nút giao 3 pha, lưu lượng trung bình (nhu cầu 1.716 xe/giờ).

---

## 1. Cài Đặt

Yêu cầu: **SUMO 1.20+** (đã thiết lập `SUMO_HOME`) và **Python 3.10+**.

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 2. Huấn Luyện Mô Hình RL (Q-Learning & DQN)

Huấn luyện mô hình với hàm thưởng **Delay-Aware Pressure (DAP)**:

```powershell
# Tabular Q-Learning
python -m src.train_q_learning --scenario cologne1 --episodes 100
python -m src.train_q_learning --scenario ingolstadt1 --episodes 100

# Deep Q-Network (DQN)
python -m src.train_dqn --scenario cologne1 --episodes 100
python -m src.train_dqn --scenario ingolstadt1 --episodes 100
```
- Checkpoint Q-Learning: `models/q_learning/<scenario>_best.json`
- Checkpoint DQN: `models/dqn/<scenario>_best.pt`

---

## 3. Đánh Giá Đối Chuẩn (5 Test Seeds)

Tập hạt giống đánh giá chuẩn: `[42, 101, 2024, 3407, 9999]`, mỗi lượt chạy 3.600 giây (1 giờ mô phỏng).

```powershell
# Đánh giá từng thuật toán
python -m src.evaluate_fixed_time
python -m src.evaluate_max_pressure
python -m src.evaluate_q_learning
python -m src.evaluate_dqn

# Tổng hợp bảng số liệu Mean ± Std và phần trăm cải thiện
python -m src.consolidate_results

# Vẽ biểu đồ so sánh (lưu tại results/plots/)
python -m src.plot_benchmark
```

---

## 4. Tóm Tắt Kết Quả Benchmark

| Bản đồ | Thuật toán | Waiting Time (s) | Travel Time (s) | Mean Queue (xe) | Throughput (xe) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cologne1** (8 pha) | Fixed-Time | 74.35 ± 0.48 | 112.30 ± 0.64 | 38.75 ± 0.33 | 1970.0 ± 0.0 |
| | Max-Pressure | 52.85 ± 3.10 (-28.9%) | 91.87 ± 2.97 (-18.2%) | 27.36 ± 1.80 (-29.4%) | 1988.4 ± 6.8 |
| | Q-Learning | 34.20 ± 2.93 (-54.0%) | 70.51 ± 3.29 (-37.2%) | 18.05 ± 1.56 (-53.4%) | 1982.2 ± 7.5 |
| | **DQN (Tốt nhất)** | **24.51 ± 2.61 (-67.0%)** | **59.93 ± 3.38 (-46.6%)** | **12.79 ± 1.59 (-67.0%)** | **1994.2 ± 4.0** |
| **Ingolstadt1** (3 pha) | Fixed-Time | 18.63 ± 0.85 | 49.12 ± 1.16 | 4.90 ± 0.05 | 1698.0 ± 2.8 |
| | Max-Pressure | 12.92 ± 0.97 (-30.7%) | 42.82 ± 1.47 (-12.8%) | **2.23 ± 0.08 (-54.5%)** | 1697.2 ± 3.5 |
| | **Q-Learning (Tốt nhất)**| **9.39 ± 1.18 (-49.6%)** | **39.06 ± 1.78 (-20.5%)** | 2.37 ± 0.18 (-51.6%) | **1701.0 ± 2.4** |
| | DQN | 16.05 ± 2.91 (-13.9%) | 46.21 ± 3.49 (-5.9%) | 2.40 ± 0.24 (-51.0%) | 1689.6 ± 7.8 |

Báo cáo phân tích chi tiết: [`docs/benchmark_evaluation_report.md`](docs/benchmark_evaluation_report.md).

---

## 5. Chạy Kiểm Thử (Unit Tests)

```powershell
pytest
```
Toàn bộ 51 bài kiểm thử tự động (kiểm tra tính toán lý thuyết, biên rỗng, an toàn chuyển pha và 4 bộ điều khiển) sẽ được thực thi.

---

## 6. Cấu Trúc Dự Án

```text
TrafficSignalControl/
├── AGENT.md                       # Nguyên tắc phát triển và chuẩn mực code
├── docs/                          # Tài liệu kỹ thuật và báo cáo đối chuẩn
│   ├── benchmark_evaluation_report.md
│   ├── evaluation_specification.md
│   ├── phase_specification.md
│   ├── q_learning_specification.md
│   └── reward_redesign_specification.md
├── data/                          # Mạng lưới SUMO (cologne1, ingolstadt1)
├── models/                        # Checkpoint mô hình (dqn/*.pt, q_learning/*.json)
├── results/                       # Kết quả thực nghiệm và biểu đồ (results/plots/)
├── src/                           # Mã nguồn chính (controllers, runners, metrics)
└── tests/                         # Bộ kiểm thử pytest (51 tests)
```
