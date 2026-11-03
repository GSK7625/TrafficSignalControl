# Đặc Tả Phương Pháp Đánh Giá Hiệu Năng (Evaluation Specification)

Tài liệu xác định các quy ước, công thức tính chỉ số và điều kiện đánh giá nhất quán cho 4 thuật toán: **Fixed-Time (FT)**, **Max-Pressure (MP)**, **Tabular Q-Learning (QL)**, và **Deep Q-Network (DQN)** trên hai mạng lưới `cologne1` và `ingolstadt1`.

---

## 1. Không Gian & Thời Gian Đánh Giá

| Mạng lưới | Thời điểm bắt đầu ($T_{\text{start}}$) | Thời điểm kết thúc ($T_{\text{end}}$) | Thời lượng ($\Delta T$) | Bước mô phỏng ($\Delta t$) | Tần suất đo hàng đợi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cologne1` | 25.200 s | 28.800 s | 3.600 s (1 giờ) | 1,0 s | Mỗi bước (1,0 s) |
| `ingolstadt1` | 57.600 s | 61.200 s | 3.600 s (1 giờ) | 1,0 s | Mỗi bước (1,0 s) |

---

## 2. Định Nghĩa & Công Thức 5 Chỉ Số Cốt Lõi

1. **Average Travel Time ($\bar{T}_{\text{travel}}$)**: Thời gian hành trình trung bình trong mạng của các xe đã hoàn thành:
   $$\bar{T}_{\text{travel}} = \frac{1}{|V_{\text{completed}}|} \sum_{v \in V_{\text{completed}}} (t_{\text{arrival}}(v) - t_{\text{depart}}(v)) \quad \text{[giây]}$$
   *(Nguồn: thuộc tính `duration` trong file XML `--tripinfo-output`)*

2. **Average Waiting Time ($\bar{T}_{\text{wait}}$)**: Thời gian xe đứng yên tích lũy ($v < 0,1$ m/s):
   $$\bar{T}_{\text{wait}} = \frac{1}{|V_{\text{completed}}|} \sum_{v \in V_{\text{completed}}} \text{waitingTime}(v) \quad \text{[giây]}$$
   *(Nguồn: thuộc tính `waitingTime` trong file XML `--tripinfo-output`)*

3. **Average Time Loss ($\bar{T}_{\text{loss}}$)**: Thời gian tổn thất so với tốc độ tối đa lý tưởng:
   $$\bar{T}_{\text{loss}} = \frac{1}{|V_{\text{completed}}|} \sum_{v \in V_{\text{completed}}} \text{timeLoss}(v) \quad \text{[giây]}$$
   *(Nguồn: thuộc tính `timeLoss` trong file XML `--tripinfo-output`)*

4. **Mean & Max Queue Length ($\bar{Q}, Q_{\text{max}}$)**: Chiều dài hàng đợi dừng chờ trên các làn vào nút giao:
   $$Q_k = \sum_{l \in L_{\text{incoming}}} \text{halting\_vehicles}(l, k), \quad \bar{Q} = \frac{1}{K} \sum_{k=1}^K Q_k, \quad Q_{\text{max}} = \max_{k} Q_k \quad \text{[xe]}$$
   *(Nguồn: hàm TraCI `traci.lane.getLastStepHaltingNumber(lane_id)` thu thập mỗi giây)*

5. **Throughput ($TP$)**: Số lượng xe hoàn thành toàn bộ hành trình trong cửa sổ đánh giá:
   $$TP = |V_{\text{completed}}| \quad \text{[xe]}$$
   *(Nguồn: số lượng phần tử `<tripinfo>` trong file XML xuất ra)*

---

## 3. Quy Ước Thời Gian & Quan Hệ Giữa Các Đại Lượng

Các đại lượng thời gian trong SUMO tuân thủ bất đẳng thức:
$$\text{waitingTime} \le \text{timeLoss} \le \text{duration}$$
- `departDelay`: Chậm trễ xuất phát ngoài mạng (không tính vào thời gian lưu thông).
- `duration`: Tổng thời gian lưu thông trong mạng từ `depart` đến `arrival`.
- Không cộng gộp các đại lượng thời gian để tránh tính trùng lặp.

---

## 4. Xử Lý Trường Hợp Biên & Ngoại Lệ

1. **Xe chưa hoàn thành:** Ghi nhận số lượng $N_{\text{chưa xong}} = N_{\text{vào mạng}} - TP$. Phân tích đồng thời cùng chỉ số hành trình.
2. **Tắt Teleport:** Thiết lập `--time-to-teleport -1` để phản ánh đúng hiện tượng ùn tắc tự nhiên.
3. **Cảnh báo va chạm:** Thiết lập `--collision.action warn` để ghi nhận sự cố an toàn vào log.
4. **Bế tắc hoàn toàn ($|V_{\text{completed}}| = 0$):** $TP = 0$, các đại lượng trung bình trả về `None` (kèm cờ `Gridlock Detected`) để tránh chia cho 0.

---

## 5. Phân Chia Hạt Giống Ngẫu Nhiên (Seeds)

- **Test Seeds (Dùng chung cho cả 4 thuật toán):** `[42, 101, 2024, 3407, 9999]`
- **Train Seeds (Huấn luyện QL & DQN):** $[1, 2, \dots, 20]$
- **Validation Seeds (Chọn lọc checkpoint):** $[21, 22, 23]$
- *Quy tắc:* Tách biệt 100% giữa các tập hạt giống, không dùng Validation/Test seeds để huấn luyện.

---

## 6. Điều Kiện Nhất Quán Giữa Các Thuật Toán

1. Cùng kịch bản dữ liệu mạng lưới và nhu cầu giao thông (`.net.xml`, `.rou.xml`, `.sumocfg`).
2. Cùng thời lượng mô phỏng 3.600 giây với bước thời gian $\Delta t = 1,0$ giây.
3. Cùng cơ chế an toàn dọn nút giao: đèn vàng $t_{\text{yellow}} = 3,0$s và đỏ toàn phần $t_{\text{all\_red}} = 1,0$s.
4. Cùng module `MetricsCollector` để tính toán 5 chỉ số độc lập từ cùng nguồn dữ liệu.
