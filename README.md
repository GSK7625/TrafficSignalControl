# TrafficSignalControl

Hệ thống mô phỏng và đánh giá thuật toán điều khiển đèn tín hiệu giao thông trên **SUMO**, so sánh giữa **Fixed-Time**, **Max-Pressure** và **Q-Learning (Tabular)** trên các dataset RESCO (`cologne1`, `ingolstadt1`, `cologne3`).

---

## 1. Yêu cầu & Cài đặt

- **Yêu cầu:** Python 3.11+ và [SUMO](https://eclipse.dev/sumo/) (hỗ trợ `libsumo` hoặc TraCI).

```powershell
# Tạo môi trường ảo và cài đặt thư viện phụ thuộc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. Hướng dẫn sử dụng

### Chạy mô phỏng đơn lẻ (`main.py`)

Cấu hình mặc định được nạp từ `configs/config.yaml`:

```powershell
# Chạy mặc định (Max-Pressure, cologne1)
python main.py

# Chạy với thuật toán, dataset và số bước xác định
python main.py -a fixed-time -d ingolstadt1 -s 3600

# Chạy có giao diện trực quan (GUI cần TraCI)
python main.py -a max-pressure --gui
```

### Chạy Benchmark / Đánh giá

Chạy thực nghiệm so sánh đa thuật toán và đa seed:

```powershell
# Chạy đánh giá toàn bộ baseline theo cấu hình benchmark
python -m src.evaluation.experiment_runner --run-all --fresh

# Vẽ biểu đồ so sánh kết quả (dashboard 3 chỉ số cốt lõi)
python -m src.evaluation.plot_comparison --dashboard-only
```

> **Kết quả:** Lưu tại thư mục `results/evaluation/` (bao gồm file JSON, CSV và báo cáo tự động `benchmark_comparison.md`).

### Huấn luyện & Đánh giá Q-Learning

```powershell
# 1. Huấn luyện mô hình Q-Learning cho kịch bản
python -m src.training.train_q_learning --dataset cologne1 --episodes 60

# 2. Đánh giá benchmark bao gồm cả Q-Learning sau khi huấn luyện
python -m src.evaluation.experiment_runner --run-all --algorithms Fixed-Time,Max-Pressure,Q-Learning --fresh
```

---

## 3. Cấu hình (`configs/config.yaml`)

Toàn bộ tham số được quản lý tập trung trong file `configs/config.yaml`:

| Nhóm | Mô tả chính |
|---|---|
| `safety` | Ràng buộc thời gian an toàn: min/max xanh, thời gian vàng, all-red, max đỏ (`SafePhaseManager`). |
| `simulation` | Thiết lập khi chạy đơn qua CLI: thuật toán, dataset, seed, số bước, GUI. |
| `q_learning` | Siêu tham số (LR, gamma, epsilon), rời rạc hóa trạng thái (discretization) và hàm reward. |
| `benchmark` | Danh sách kịch bản, thuật toán và seed phục vụ chạy thực nghiệm đối chuẩn. |

---

## 4. Kiểm thử

```powershell
# Chạy toàn bộ test suite
python -m pytest -q --basetemp=.pytest_cache/test-tmp
```

---

## 5. Cấu trúc thư mục

```text
TrafficSignalControl/
├── configs/             # Cấu hình tập trung (config.yaml)
├── data/                # Dataset RESCO (cologne1, cologne3, ingolstadt1)
├── models/              # Checkpoint Q-Learning (.json)
├── results/             # Kết quả đánh giá, biểu đồ và báo cáo
├── src/
│   ├── controllers/     # Fixed-Time, Max-Pressure, Q-Learning
│   ├── evaluation/      # Benchmark runner và vẽ biểu đồ kết quả
│   ├── rl/              # Rời rạc hóa trạng thái, reward và môi trường RL
│   ├── training/        # Huấn luyện Q-Learning và chọn checkpoint
│   ├── metrics.py       # Thu thập chỉ số hiệu năng và đối soát TripInfo
│   ├── phase_manager.py # Kiểm soát an toàn và chuyển pha (SafePhaseManager)
│   └── runner.py        # Vòng lặp mô phỏng với SUMO/libsumo
├── tests/               # Unit & integration tests
├── main.py              # Điểm vào dòng lệnh (CLI)
└── requirements.txt     # Danh sách thư viện Python
```
