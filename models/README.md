# `models/` — Checkpoints Mô hình Học Tăng Cường (RL)

Thư mục [`models/`](file:///models) chứa các checkpoint trọng số mô hình đã được huấn luyện, lưu trữ dưới định dạng JSON phục vụ suy luận, kiểm thử và đánh giá đối chuẩn độc lập.

---

## 1. Cấu trúc thư mục

```text
models/
└── q_learning/                                # Checkpoints Tabular Q-Learning
    ├── q_learning_cologne1_best.json          # Checkpoint tốt nhất trên kịch bản cologne1
    ├── q_learning_cologne1_final.json         # Checkpoint tập cuối cùng trên kịch bản cologne1
    ├── q_learning_ingolstadt1_best.json       # Checkpoint tốt nhất trên kịch bản ingolstadt1
    └── q_learning_ingolstadt1_final.json      # Checkpoint tập cuối cùng trên kịch bản ingolstadt1
```

---

## 2. Tiêu chuẩn lưu Checkpoint & Cổng khả thi (Viability Gate)

Trong quá trình huấn luyện bằng [`src/training/train_q_learning.py`](file:///src/training/train_q_learning.py):
1. **Validation định kỳ (`--val-interval 3`)**: Sau mỗi khoảng episode nhất định, agent được chuyển sang chế độ thuần Greedy ($\varepsilon = 0$) và đánh giá trên các seed độc lập (`--val-seeds 101 2024`).
2. **Cổng khả thi (Viability Gate — `min_completion >= 0.95`)**: Checkpoint chỉ được cân nhắc lưu thành `*_best.json` nếu tỷ lệ hoàn thành hành trình trung bình đạt tối thiểu **95%** nhu cầu giao thông. Điều này ngăn chặn việc lưu những tác tử tối ưu phần thưởng cục bộ nhưng gây nghẽn mạng lưới nghiêm trọng.
3. **Mô hình kết thúc (`*_final.json`)**: Luôn được lưu lại sau khi hoàn thành toàn bộ số tập huấn luyện (`num_episodes`).

---

## 3. Cấu trúc dữ liệu của tệp Checkpoint JSON

Tệp checkpoint được mã hóa dạng JSON tương thích cao, bao gồm đầy đủ metadata môi trường và bảng giá trị $Q(s, a)$:

```json
{
  "algorithm": "QLearning",
  "scenario_name": "cologne1",
  "target_tls_id": "gneJ0",
  "num_actions": 8,
  "env_params": {
    "min_green": 5.0,
    "max_green": 60.0,
    "yellow_duration": 3.0,
    "all_red_duration": 1.0,
    "decision_interval": 5.0
  },
  "training_params": {
    "learning_rate": 0.1,
    "gamma": 0.95,
    "epsilon": 0.05,
    "total_episodes": 15
  },
  "q_table": {
    "(0, 0, 1, 2)": [0.12, -0.45, 1.25, 0.0, -0.1, 0.3, 0.05, -0.2],
    "(1, 2, 0, 1)": [-0.8, 2.15, 0.4, -0.3, 0.1, 0.0, -0.15, 0.5]
  }
}
```

- **`q_table`**: Ánh xạ từ khóa trạng thái rời rạc (tuple chuỗi) sang mảng giá trị Q tương ứng với các pha đèn xanh (actions).
- **`env_params`**: Các tham số an toàn chuẩn hóa mà mô hình yêu cầu để bảo đảm tính tương thích khi nạp vào bộ điều khiển.

---

## 4. Cách sử dụng Checkpoint trong lệnh CLI

Để nạp và chạy kiểm thử một checkpoint cụ thể:

```powershell
# Chạy suy luận với checkpoint trên SUMO dòng lệnh
python main.py --dataset cologne1 --algorithm q-learning --model models/q_learning/q_learning_cologne1_best.json --tripinfo

# Chạy trực quan checkpoint với SUMO-GUI
python main.py --dataset cologne1 --algorithm q-learning --model models/q_learning/q_learning_cologne1_best.json --gui --gui-delay 50

# Đánh giá độc lập checkpoint
python -m src.evaluation.evaluate_rl --dataset cologne1 --algo q_learning --model models/q_learning/q_learning_cologne1_best.json --seed 42
```
