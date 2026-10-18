# `docs/` — Trung Tâm Tài Liệu Dự Án (Documentation Portal)

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật của dự án **TrafficSignalControl: SUMO Reinforcement Learning & Baseline Benchmark**.

---

## 1. Sơ đồ Chỉ mục Tài liệu

Dự án được cấu trúc với hệ thống tài liệu phân tầng, giúp lập trình viên và nhà nghiên cứu dễ dàng tra cứu:

| Khu vực | Tài liệu hướng dẫn | Nội dung chính |
| :--- | :--- | :--- |
| **Dự án chung** | [`README.md`](file:///README.md) | Tổng quan hệ thống, tính năng, hướng dẫn cài đặt và sử dụng toàn diện. |
| **Mã nguồn** | [`src/README.md`](file:///src/README.md) | Kiến trúc chi tiết các module controllers, rl, evaluation, runner và phase_manager. |
| **Cấu hình** | [`configs/README.md`](file:///configs/README.md) | Đặc tả các tham số trong `experiments.yaml` và `q_learning.yaml`. |
| **Dữ liệu** | [`data/README.md`](file:///data/README.md) | Danh mục kịch bản RESCO, luồng xe ground-truth và cấu trúc file mạng SUMO. |
| **Bản quyền Dữ liệu** | [`data/SOURCES.md`](file:///data/SOURCES.md) | Nguồn gốc commit SHA, giấy phép phân phối và quy trình tiền xử lý dữ liệu. |
| **Checkpoints** | [`models/README.md`](file:///models/README.md) | Định dạng tệp JSON Q-table, tiêu chuẩn lọc mô hình Viability Gate. |
| **Kết quả & Biểu đồ** | [`results/README.md`](file:///results/README.md) | Bảng chỉ số đối chuẩn, schema file CSV/JSON và danh mục biểu đồ 300 DPI. |
| **Kiểm thử tự động** | [`tests/README.md`](file:///tests/README.md) | Hướng dẫn chạy 80 unit tests và chiến lược giả lập (mocking) không cần SUMO. |

---

## 2. Tóm tắt Lý thuyết & Phương pháp Luận

### 2.1. Fixed-Time Control (Baseline Cố định)
Phương pháp kinh điển chia chu kỳ đèn thành các khoảng thời gian cố định (mặc định 30.0s mỗi pha). Hệ thống đảm bảo việc chuyển pha diễn ra tuần hoàn kèm theo pha vàng (3.0s) và toàn đỏ (1.0s) để dọn sạch nút giao.

### 2.2. Max-Pressure Control (Heuristic dựa trên Áp lực)
Dựa trên lý thuyết của Varaiya (2013), Max-Pressure chọn pha xanh $p^*$ có áp lực giải phóng lớn nhất:
$$p^* = \arg\max_{p} \sum_{(l, m) \in \text{Links}(p)} \left( Q_l - Q_m \right)$$
Trong đó $Q_l$ là số xe chờ ở làn vào (upstream), $Q_m$ là số xe ở làn ra (downstream). Nhờ đó, thuật toán hạn chế tối đa nguy cơ dồn ứ hạ lưu (spillback).

### 2.3. Tabular Q-Learning (Học tăng cường dạng bảng)
Agent tối ưu hóa hàm giá trị $Q(s, a)$ thông qua kinh nghiệm tương tác với môi trường theo phương trình Bellman:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
Không gian trạng thái liên tục được ánh xạ sang không gian rời rạc bao gồm: chỉ số pha hiện tại, nhóm thời gian xanh hiệu dụng, và các mức độ dài hàng đợi luồng phục vụ / luồng chờ.

---

## 3. Lệnh Nhanh Thường Dùng (Cheatsheet)

```powershell
# Chạy toàn bộ 80 unit tests
pytest -q

# Kiểm tra thông tin các kịch bản
python main.py --inspect-all

# Chạy mô phỏng Max-Pressure với giao diện SUMO-GUI
python main.py --dataset cologne1 --algorithm max-pressure --gui --gui-delay 50

# Huấn luyện Tabular Q-Learning
python -m src.training.train_q_learning --dataset cologne1 --episodes 15

# Chạy ma trận đối chuẩn đa seed (18 runs)
python -m src.evaluation.experiment_runner --run-all

# Tổng hợp kết quả và vẽ biểu đồ khoa học
python -m src.evaluation.compare_algorithms
python -m src.evaluation.plot_comparison --dashboard-only
```
