# TrafficSignalControl: So Sánh 3 Thuật Toán Điều Khiển Đèn Giao Thông (SUMO)

Dự án đối chuẩn 3 thuật toán điều khiển tín hiệu đèn giao thông trên SUMO:
1. **Fixed-Time** (Cố định - Baseline)
2. **Max-Pressure** (Thích ứng theo áp suất hàng đợi)
3. **Q-Learning** (Học tăng cường đa mục tiêu)

---

## 1. Kích hoạt môi trường

```powershell
# Kích hoạt virtualenv trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Hoặc dùng trực tiếp qua đường dẫn:
# .venv\Scripts\python ...
```

---

## 2. Bước 1: Huấn luyện Q-Learning đến khi ổn định

Chạy huấn luyện tác tử Q-Learning với thời gian xanh tối thiểu `min_green = 20.0s` (khuyến nghị 35–45 episodes để Q-table hội tụ và epsilon giảm về sàn `0.05`):

```powershell
python -m src.training.train_q_learning --scenario cologne1 --episodes 35 --min-green 20.0 --max-steps 3600
```

- Mô hình tốt nhất (Best checkpoint) sẽ được lưu tại:  
  `models/q_learning/q_learning_cologne1_best.json`

---

## 3. Bước 2: Chạy so sánh 3 thuật toán trên nhiều Seed ngẫu nhiên

Đánh giá cả 3 thuật toán trên cùng một tập seed ngẫu nhiên (`42 101 2024 777 999`):

```powershell
python -m src.benchmark `
  --scenario cologne1 `
  --seeds 42 101 2024 777 999 `
  --model models/q_learning/q_learning_cologne1_best.json `
  --max-steps 3600 `
  --output-dir results/benchmark
```

> **Ghi chú chạy thử nhanh (Quick test):**
> ```powershell
> python -m src.benchmark --scenario cologne1 --seeds 42 101 --max-steps 600
> ```

---

## 4. Bước 3: Xem kết quả so sánh

Kết quả được tự động tổng hợp thành báo cáo và bảng số liệu:
- **Báo cáo Markdown:** `results/benchmark/benchmark_cologne1.md`
- **File số liệu JSON:** `results/benchmark/benchmark_cologne1.json`

Các chỉ số được tính toán $\text{Mean} \pm \text{Std}$ và $\%$ cải thiện so với Fixed-Time gồm:
- Thời gian chờ trung bình (Average Waiting Time)
- Thời gian di chuyển trung bình (Average Travel Time)
- Độ dài hàng đợi trung bình (Mean Queue Length)
- Tổn thất thời gian trung bình (Average Time Loss)
- Thông lượng xe hoàn thành (Throughput)

---

## 5. Chạy kiểm thử tự động (Unit Tests)

```powershell
pytest
```
