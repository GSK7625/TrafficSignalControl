# Đặc Tả Kỹ Thuật: Hàm Phần Thưởng Delay-Aware Pressure (DAP-Reward)

Tài liệu đặc tả cơ sở lý thuyết, công thức toán học và cấu hình triển khai của hàm phần thưởng **Delay-Aware Pressure (DAP)** dùng cho **Tabular Q-Learning** và **Deep Q-Network (DQN)**.

---

## 1. Đặt Vấn Đề

Hàm thưởng áp suất thuần túy ($r_k = -P_k / \Delta t$) có 3 hạn chế thực nghiệm lớn:
1. **Không nhận biết độ trễ thời gian (No Delay Awareness):** Áp suất chỉ đếm lượng xe tức thời, không phân biệt xe mới đến hay đã dừng chờ lâu. Tại nút bất đối xứng (như `ingolstadt1`), trục chính đông xe lấn át đường nhánh, khiến mô hình từ chối chuyển sang nhánh phụ, gây đói pha nghiêm trọng.
2. **Không tính chi phí chuyển pha (No Switch Penalty):** Việc đổi đèn tốn 4s dọn nút giao (3s vàng + 1s đỏ toàn phần). Thiếu phạt chuyển pha dẫn đến hiện tượng nhảy pha liên tục gây giảm thông lượng.
3. **Lệch pha với mục tiêu đánh giá:** Mục tiêu benchmark chính là cực tiểu hóa `Average Waiting Time` và tối đa hóa `Throughput` của phương tiện.

---

## 2. Công Thức Toán Học Của DAP-Reward

Phần thưởng tức thời tại mỗi bước mô phỏng $k$ ($1,0$s):

$$\mathcal{R}_k = - \left( \frac{P_k}{\Delta t_{\text{decision}}} + \beta \cdot \frac{W_k}{\Delta t_{\text{decision}} \cdot N_{\text{lanes}}} \right) - c_{\text{switch}} \cdot \mathbb{I}(a \ne p_{\text{curr}})$$

### Các thành phần:
1. **Áp suất phân bổ theo bậc kết nối ($P_k$):**
   $$P_k = \sum_{m} \left| \frac{x_u}{\text{deg}_u} - \frac{x_d}{\text{deg}_d} \right|$$
   - Giữ nguyên ưu điểm điều phối cân bằng mạng lưới của PressLight / Max-Pressure. Trị tuyệt đối ngăn triệt tiêu áp suất và xử phạt khi làn ra bị tắc nghẽn ($w_m < 0$).
2. **Tổng thời gian chờ thực tế trên các làn vào ($W_k$):**
   $$W_k = \sum_{l \in \mathcal{L}_{\text{in}}} \text{waiting\_time}(l)$$
   - Thu thập trực tiếp qua TraCI `traci.lane.getWaitingTime(lane_id)`.
   - Chuẩn hóa theo số làn vào $N_{\text{lanes}} = |\mathcal{L}_{\text{in}}|$ và điều tiết bằng hệ số $\beta = 0,05$. Khi có xe chờ lâu, số hạng này sẽ tăng vọt và buộc mô hình phải phục vụ nhánh phụ.
3. **Chi phí phạt chuyển pha ($c_{\text{switch}}$):**
   - Áp dụng $c_{\text{switch}} = 1,0$ chỉ tại bước ra quyết định đổi pha ($a \ne p_{\text{curr}}$) để bù đắp 4s tổn thất dọn nút.

---

## 3. Tích Lũy Bán Markov (SMDP Intra-Option Discounting)

Do thời lượng hành động biến đổi (`KEEP`: $5$s, `SWITCH`: $24$s), phần thưởng tích lũy được chiết khấu theo từng giây:

$$R_{\text{SMDP}} = \sum_{k=0}^{K-1} \gamma_{\text{step}}^k \mathcal{R}_k \quad \text{với } \gamma_{\text{step}} = 0,95^{1/5} \approx 0,9898$$

- Với `KEEP` ($5$s): $\gamma_{\text{eff}} = 0,95^{5/5} = 0,95$.
- Với `SWITCH` ($24$s): $\gamma_{\text{eff}} = 0,95^{24/5} \approx 0,782$.

---

## 4. Hiệu Chỉnh Giám Sát Chống Đói Pha Linh Hoạt

Để phối hợp đồng bộ với DAP-Reward:
- Bộ đếm thời gian đỏ `phase_red_timers[p]` của pha $p$ **chỉ được reset về 0 khi pha đó được bật xanh**.
- Khi pha $p$ đang đỏ, bộ đếm tăng đều nếu làn của pha đó có xe dừng chờ hoặc tồn tại xe, ngăn việc reset sai lệch do dao động vận tốc xe đang giảm tốc.
