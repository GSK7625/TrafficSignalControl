# `configs/` — Hệ thống Cấu hình Thử nghiệm & Siêu tham số

Thư mục [`configs/`](file:///configs) chứa các tệp cấu hình YAML chuẩn hóa dùng để thiết lập môi trường mô phỏng, tham số an toàn, ma trận đánh giá đối chuẩn và siêu tham số cho thuật toán Học tăng cường.

---

## 1. Danh mục các tệp cấu hình

| Tệp cấu hình | Mục đích | Áp dụng cho |
| :--- | :--- | :--- |
| [`configs/experiments.yaml`](file:///configs/experiments.yaml) | Cấu hình ma trận benchmark đối chuẩn đa kịch bản, đa thuật toán và tham số mô phỏng chuẩn. | `src.evaluation.experiment_runner` |
| [`configs/q_learning.yaml`](file:///configs/q_learning.yaml) | Cấu hình siêu tham số học tăng cường, hàm phần thưởng và các ngưỡng chia bin rời rạc hóa trạng thái. | `src.training.train_q_learning`, `main.py` |

---

## 2. Chi tiết cấu hình Benchmark (`configs/experiments.yaml`)

```yaml
# Benchmark Experiments Configuration
scenarios:
  - cologne1        # Kịch bản mạng Cologne 1 nút giao
  - ingolstadt1     # Kịch bản mạng Ingolstadt 1 nút giao

algorithms:
  - Fixed-Time      # Baseline chu kỳ đèn cố định 30s
  - Max-Pressure    # Thuật toán Heuristic tối ưu áp lực hàng đợi
  - Q-Learning      # Thuật toán Học tăng cường dạng bảng

simulation:
  steps: 3600               # Tổng số bước mô phỏng (tương đương 3600 giây = 1 giờ thực tế)
  step_length: 1.0          # Bước thời gian vi mô của SUMO (giây)
  seed: 42                  # Seed mặc định
  use_tripinfo: true        # Bật trích xuất dữ liệu ground-truth từ tripinfo XML
  yellow_duration: 3.0      # Thời gian đèn vàng dọn nút (giây)
  all_red_duration: 1.0     # Thời gian đỏ toàn bộ dọn nút (giây)
  min_green: 5.0            # Thời gian đèn xanh tối thiểu (giây)
  max_green: 60.0           # Thời gian đèn xanh tối đa liên tục cho 1 pha (giây)
```

### Các trường dữ liệu chính:
- **`scenarios`**: Danh sách tên các bộ dữ liệu trong thư mục `data/` được đưa vào ma trận thử nghiệm.
- **`algorithms`**: Các thuật toán tham gia đối chuẩn.
- **`simulation.steps`**: Số bước mô phỏng (chuẩn 3600s tương ứng cửa sổ phân tích luồng xe RESCO).
- **`simulation.use_tripinfo`**: Kích hoạt cờ SUMO `--tripinfo-output` để đo lường chính xác từng xe rời mạng lưới.
- **`simulation.min_green` & `max_green`**: Ràng buộc an toàn vật lý và dọn nút cho nút giao.

---

## 3. Chi tiết cấu hình Tabular Q-Learning (`configs/q_learning.yaml`)

```yaml
# Tabular Q-Learning Configuration
algorithm: QLearning
decision_interval: 5.0      # Chu kỳ ra quyết định của agent RL (giây)
min_green: 5.0              # Thời gian đèn xanh tối thiểu (giây)
max_green: 60.0             # Thời gian đèn xanh tối đa liên tục cho 1 pha (giây)
yellow_duration: 3.0        # Thời gian pha vàng chuyển tiếp (giây)
all_red_duration: 1.0       # Thời gian toàn đỏ dọn nút (giây)

hyperparameters:
  learning_rate: 0.1         # Tốc độ học alpha trong cập nhật Bellman
  gamma: 0.95                # Hệ số chiết khấu phần thưởng tương lai
  epsilon_start: 1.0         # Giá trị khám phá epsilon ban đầu
  epsilon_min: 0.05          # Giá trị epsilon tối thiểu
  epsilon_decay: 0.85        # Hệ số phân rã epsilon sau mỗi episode
  num_episodes: 15           # Số lượng episode huấn luyện
  seed: 42                   # Seed khởi tạo giá trị ngẫu nhiên

discretization:
  max_effective_green: 25.0  # Ngưỡng trần thời gian xanh hiệu dụng
  serving_queue_thresholds: [1.0, 5.0]        # Ngưỡng chia bins hàng đợi luồng đang phục vụ
  waiting_queue_thresholds: [1.0, 6.0, 15.0]  # Ngưỡng chia bins hàng đợi luồng đang chờ

reward:
  name: NegativePressureReward # Hàm phần thưởng áp lực (hoặc NegativeAverageQueueReward)
  scale: 1.0                 # Hệ số co giãn phần thưởng
  pending_weight: 0.0        # Trọng số phạt bổ sung cho phương tiện chưa gia nhập mạng
```

### Hướng dẫn tinh chỉnh rời rạc hóa (Discretization Tuning):
- **`serving_queue_thresholds` `[1.0, 5.0]`**: Chia hàng đợi luồng xanh thành 3 mức:
  - `0`: Trống ($< 1$ xe)
  - `1`: Vừa phải ($1 \le \text{queue} < 5$ xe)
  - `2`: Đông ($\ge 5$ xe)
- **`waiting_queue_thresholds` `[1.0, 6.0, 15.0]`**: Chia hàng đợi các luồng đỏ thành 4 mức (`0`, `1`, `2`, `3`), giúp agent cảm nhận được mức độ ùn tắc tích lũy ở các hướng chờ.

---

## 4. Thứ tự ưu tiên cấu hình (Precedence Rules)

Khi khởi chạy chương trình, các tham số được nạp và áp dụng theo thứ tự ưu tiên giảm dần:
1. **Tham số dòng lệnh CLI** (Ví dụ: `--min-green 10.0`, `--decision-interval 5.0`): Có mức ưu tiên cao nhất, ghi đè toàn bộ cấu hình khác.
2. **Tham số trong Checkpoint mô hình** (nếu đang chạy suy luận với file checkpoint `.json`).
3. **Cấu hình trong tệp YAML** (`configs/q_learning.yaml` hoặc `configs/experiments.yaml`).
4. **Giá trị an toàn mặc định của hệ thống** (định nghĩa tại `DEFAULT_SAFETY_CONFIG` trong [`src/config.py`](file:///src/config.py)).
