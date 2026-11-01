# Đặc Tả Phương Pháp Đánh Giá Thuật Toán Điều Khiển Đèn Giao Thông (Evaluation Specification)

Tài liệu này xác định các quy ước, công thức, định nghĩa và điều kiện đánh giá nhất quán cho ba thuật toán: **Fixed-Time (FT)**, **Max-Pressure (MP)**, và **Tabular Q-Learning** trên hai bản đồ chuẩn `cologne1` và `ingolstadt1`.

---

## 1. Không Gian & Thời Gian Đánh Giá

| Bản đồ | Thời điểm bắt đầu ($T_{\text{start}}$) | Thời điểm kết thúc ($T_{\text{end}}$) | Tổng thời lượng ($\Delta T$) | Bước mô phỏng ($\Delta t$) | Tần suất đo hàng đợi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cologne1` | 25.200 s | 28.800 s | 3.600 s (1 giờ) | 1,0 s | Mỗi bước (1,0 s) |
| `ingolstadt1` | 57.600 s | 61.200 s | 3.600 s (1 giờ) | 1,0 s | Mỗi bước (1,0 s) |

---

## 2. Định Nghĩa & Công Thức 5 Chỉ Số Hiệu Năng (Core Metrics)

### 2.1. Average Travel Time ($\bar{T}_{\text{travel}}$)
- **Định nghĩa**: Thời gian hành trình trung bình tính từ khi xe thực tế xuất hiện trên mạng (`depart`) cho đến khi rời mạng khi đến đích (`arrival`).
- **Công thức**:
  $$\bar{T}_{\text{travel}} = \frac{1}{|V_{\text{completed}}|} \sum_{v \in V_{\text{completed}}} (t_{\text{arrival}}(v) - t_{\text{depart}}(v))$$
- **Đơn vị**: Giây (s).
- **Tập xe**: Chỉ tính trên tập xe đã hoàn thành lộ trình ($V_{\text{completed}}$). Báo cáo kèm số lượng và tỷ lệ xe chưa hoàn thành.
- **Nguồn dữ liệu**: Thuộc tính `duration` trong file XML `--tripinfo-output`.

### 2.2. Average Waiting Time ($\bar{T}_{\text{wait}}$)
- **Định nghĩa**: Tổng thời gian xe ở trạng thái dừng chờ (vận tốc $v < 0,1\text{ m/s}$) tích lũy trong suốt hành trình.
- **Công thức**:
  $$\bar{T}_{\text{wait}} = \frac{1}{|V_{\text{completed}}|} \sum_{v \in V_{\text{completed}}} \text{waitingTime}(v)$$
- **Đơn vị**: Giây (s).
- **Tập xe**: $V_{\text{completed}}$.
- **Nguồn dữ liệu**: Thuộc tính `waitingTime` trong file XML `--tripinfo-output`.

### 2.3. Average Time Loss ($\bar{T}_{\text{loss}}$)
- **Định nghĩa**: Thời gian bị mất do phải di chuyển chậm hơn tốc độ mong muốn tối đa (gồm thời gian dừng đèn đỏ, giảm tốc, tăng tốc, và đi chậm sau đuôi xe khác).
- **Công thức**:
  $$\bar{T}_{\text{loss}} = \frac{1}{|V_{\text{completed}}|} \sum_{v \in V_{\text{completed}}} \text{timeLoss}(v)$$
- **Đơn vị**: Giây (s).
- **Tập xe**: $V_{\text{completed}}$.
- **Nguồn dữ liệu**: Thuộc tính `timeLoss` trong file XML `--tripinfo-output`.

### 2.4. Mean & Max Queue Length ($\bar{Q}, Q_{\text{max}}$)
- **Định nghĩa**: Giá trị trung bình theo thời gian và giá trị cực đại của tổng số lượng xe dừng chờ ($v < 0,1\text{ m/s}$) trên tất cả các làn vào được điều khiển của nút giao.
- **Công thức**:
  Tại mỗi bước mô phỏng $k \in \{1, 2, \dots, K\}$:
  $$Q_k = \sum_{l \in L_{\text{incoming}}} \text{halting\_vehicles}(l, k)$$
  Giá trị trung bình và cực đại:
  $$\bar{Q} = \frac{1}{K} \sum_{k=1}^K Q_k, \quad Q_{\text{max}} = \max_{k \in \{1, \dots, K\}} Q_k$$
- **Đơn vị**: Số xe (phương tiện).
- **Phạm vi không gian**: Tập làn vào kết nối trực tiếp tới nút giao ($L_{\text{incoming}}$):
  - `cologne1`: 8 làn vào.
  - `ingolstadt1`: 7 làn vào.
  - Mỗi làn chỉ tính đúng một lần trong tổng.
- **Nguồn dữ liệu**: Thu thập trực tiếp từng bước qua TraCI: `traci.lane.getLastStepHaltingNumber(lane_id)`.

### 2.5. Throughput ($TP$)
- **Định nghĩa**: Tổng số lượng xe hoàn thành toàn bộ hành trình trong cửa sổ đánh giá.
- **Công thức**:
  $$TP = |V_{\text{completed}}|$$
- **Đơn vị**: Số xe (phương tiện).
- **Nguồn dữ liệu**: Số phần tử `<tripinfo>` có trong file XML `--tripinfo-output`.

---

## 3. Phân Biệt Các Đại Lượng Thời Gian Của SUMO

- **`departDelay`**: Thời gian trễ xuất phát (chênh lệch giữa giờ muốn xuất phát trong file route và giờ thực tế chèn được vào mạng). Diễn ra **ngoài mạng**.
- **`duration`**: Tổng thời gian lưu thông **trong mạng** ($t_{\text{arrival}} - t_{\text{depart}}$).
- **`waitingTime`**: Tổng thời gian xe đứng yên ($v < 0,1\text{ m/s}$) **trong mạng**. Là một phần con của `duration`.
- **`timeLoss`**: Tổng thời gian chậm hơn điều kiện lý tưởng. Bao hàm cả `waitingTime` và độ trễ do chạy chậm.
- **Quy tắc**: Không cộng tùy tiện $\text{waitingTime} + \text{timeLoss}$ hoặc $\text{duration} + \text{timeLoss}$ vì bị tính trùng lặp thời gian. Luôn đảm bảo:
  $$\text{waitingTime} \le \text{timeLoss} \le \text{duration}$$

---

## 4. Xử Lý Các Trường Hợp Biên & Ngoại Lệ

1. **Xe chưa hoàn thành / kẹt trên mạng**:
   - Ghi nhận vào số lượng xe chưa về đích $N_{\text{unfinished}} = N_{\text{total\_demand}} - N_{\text{completed}}$.
   - Báo cáo kèm tỷ lệ hoàn thành $\text{Completion Rate} = \frac{N_{\text{completed}}}{N_{\text{total\_demand}}} \times 100\%$.
2. **Teleport**: Tắt tính năng teleport (`--time-to-teleport -1`) để bảo toàn bản chất tắc nghẽn giao thông thực tế.
3. **Va chạm / Phanh khẩn cấp**: Đặt cờ `--collision.action warn`, đếm số cảnh báo va chạm và phanh khẩn cấp từ log.
4. **Trường hợp bế tắc hoàn toàn ($|V_{\text{completed}}| = 0$)**:
   - $TP = 0$.
   - Các đại lượng trung bình trả về `0.0` kèm cờ cảnh báo `Gridlock Detected`, tránh lỗi `ZeroDivisionError`.

---

## 5. Phân Chia Hạt Giống Ngẫu Nhiên (Random Seeds)

- **Test Seeds (Dùng chung cho FT, MP và Test Q-Learning)**:
  $$\text{Test Seeds} = [42, 101, 2024, 3407, 9999]$$
- **Train Seeds cho Q-Learning**: Các seed riêng biệt $[1, 2, 3, \dots, 20]$.
- **Validation Seeds cho Q-Learning**: $[11, 22, 33]$ (dùng để chọn model checkpoint tốt nhất).
- **Nguyên tắc**: Tuyệt đối không dùng Test Seeds để huấn luyện hoặc tinh chỉnh tham số của Q-Learning.

---

## 6. Điều Kiện Giữ Cố Định & Nhất Quán Giữa Các Thuật Toán

1. Cùng kịch bản dữ liệu gốc (`.net.xml`, `.rou.xml`, `.sumocfg`).
2. Cùng khung giờ mô phỏng (3.600 bước, 1 giờ).
3. Cùng thời gian dọn nút giao an toàn: Vàng $t_{\text{yellow}} = 3,0\text{ s}$, Đỏ toàn phần $t_{\text{all\_red}} = 1,0\text{ s}$.
4. Cùng bước mô phỏng $\Delta t = 1,0\text{ s}$.
5. Cùng 1 module `MetricsCollector` để tính 5 chỉ số từ cùng một nguồn dữ liệu.
6. Cùng 5 hạt giống đánh giá (Test Seeds).
