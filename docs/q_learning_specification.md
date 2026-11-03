# Đặc Tả Thuật Toán Học Tăng Cường (Q-Learning & DQN Specification)

Tài liệu xác định không gian trạng thái, không gian hành động chọn pha trực tiếp, hàm phần thưởng **Delay-Aware Pressure (DAP)**, quy tắc cập nhật Bellman bán Markov (SMDP) và cơ chế Action Masking cho hai thuật toán học tăng cường (**Tabular Q-Learning** và **Deep Q-Network**).

---

## 1. Không Gian Trạng Thái (State Space)

### 1.1. Tabular Q-Learning (Rời rạc thu gọn: 4 phần tử)
$$s = (p_{\text{curr}}, t_{\text{bin}}, q_{\text{serving}}, q_{\text{competing}})$$
1. $p_{\text{curr}}$: Chỉ số pha xanh hiện hành ($0 \dots N-1$).
2. $t_{\text{bin}} \in \{0, 1\}$: Thời gian xanh đã phục vụ ($0$: $< 35$s; $1$: $\ge 35$s).
3. $q_{\text{serving}} \in \{0, 1\}$: Hàng đợi pha hiện tại ($0$: $< 3$ xe; $1$: $\ge 3$ xe).
4. $q_{\text{competing}} \in \{0, 1\}$: Hàng đợi lớn nhất của các pha đối thủ ($0$: $< 6$ xe; $1$: $\ge 6$ xe).

- **Q-Table Size:** 
  - `ingolstadt1`: $|S| = 3 \times 2 \times 2 \times 2 = 24$ states $\times 3$ actions = **72 giá trị Q**.
  - `cologne1`: $|S| = 8 \times 2 \times 2 \times 2 = 64$ states $\times 8$ actions = **512 giá trị Q**.

### 1.2. Deep Q-Network (Liên tục chuẩn hóa $[0, 1]$)
Vector trạng thái đầu vào mạng MLP gồm:
- One-hot encoding của pha hiện tại ($N$ phần tử).
- Áp suất chuyển động chuẩn hóa của từng movement: $w_m / 20.0$.
- Hàng đợi chuẩn hóa của từng pha: $q_p / 20.0$.

---

## 2. Không Gian Hành Động Chọn Pha Trực Tiếp (Action Space)

Không gian hành động là tập chỉ số các pha xanh hợp lệ:
$$A = \{0, 1, \dots, N_{\text{phases}} - 1\}$$
- **$a = p_{\text{curr}}$ (`KEEP`):** Tiếp tục duy trì pha xanh hiện tại thêm chu kỳ quyết định $\Delta t_{\text{decision}} = 5,0$s.
- **$a \ne p_{\text{curr}}$ (`SWITCH`):** Chuyển trực tiếp sang pha mục tiêu $a$ qua chuỗi an toàn 3s vàng + 1s đỏ toàn phần + tối thiểu 20s xanh ban đầu.

---

## 3. Hàm Phần Thưởng Delay-Aware Pressure (DAP-Reward)

Phần thưởng tức thời tại mỗi giây $k$:
$$r_k = - \left( \frac{P_k}{\Delta t_{\text{decision}}} + \beta \cdot \frac{W_k}{\Delta t_{\text{decision}} \cdot N_{\text{lanes}}} \right) - c_{\text{switch}} \cdot \mathbb{I}(a \ne p_{\text{curr}})$$

- **Áp suất phân bổ bậc kết nối ($P_k$):** $P_k = \sum_{m} \left| \frac{x_u}{\text{deg}_u} - \frac{x_d}{\text{deg}_d} \right|$ (lấy cảm hứng từ PressLight; dùng trị tuyệt đối ngăn triệt tiêu và phạt nghẽn làn ra).
- **Thời gian chờ thực tế ($W_k$):** Tổng thời gian chờ tích lũy trên các làn vào (TraCI `getWaitingTime`), chuẩn hóa theo $N_{\text{lanes}}$ với $\beta = 0,05$.
- **Chi phí đổi pha ($c_{\text{switch}}$):** Phạt $1,0$ tại mốc ra quyết định SWITCH để bù đắp 4s tổn thất dọn nút.

### Tích lũy phần thưởng SMDP (Intra-Option Discounting)
$$R_{\text{SMDP}} = \sum_{k=0}^{K-1} \gamma_{\text{step}}^k r_k \quad \text{với } \gamma_{\text{step}} = \gamma^{1 / \Delta t_{\text{decision}}} = 0,95^{1/5} \approx 0,9898$$

---

## 4. Công Thức Cập Nhật Bellman SMDP & Huấn Luyện

### 4.1. Tabular Q-Learning
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{\text{SMDP}} + \gamma_{\text{eff}} \max_{a' \in A_{\text{valid}}(s')} Q(s', a') - Q(s, a) \right]$$
- $\alpha = 0,1$ (Tốc độ học).
- Chiết khấu hiệu dụng: $\gamma_{\text{eff}} = \gamma^{\Delta t / 5}$ (`KEEP` 5s $\implies \gamma_{\text{eff}} = 0,95$; `SWITCH` 24s $\implies \gamma_{\text{eff}} \approx 0,782$).
- $\epsilon$-Greedy: $\epsilon_e = \max(0,05, 1,0 \times 0,90^e)$ qua các episodes.

### 4.2. Deep Q-Network (DQN)
- **Kiến trúc MLP:** `state_dim` $\rightarrow$ `Linear(64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(action_dim)`.
- **Hàm mất mát:** Smooth L1 (Huber Loss) tối ưu bằng Adam ($\text{lr} = 10^{-3}$).
- **Kỹ thuật ổn định:** Replay Buffer ($10.000$ transitions), mini-batch $64$, đồng bộ Target Network định kỳ mỗi $100$ bước cập nhật.

---

## 5. Ràng Buộc An Toàn & Action Masking

Mọi quyết định chọn pha phải tuân thủ tập hành động hợp lệ $A_{\text{valid}}$:
1. **$t_{\text{in\_phase}} < 20$s (`min_green`):** Bắt buộc giữ pha $A_{\text{valid}} = \{p_{\text{curr}}\}$.
2. **$t_{\text{in\_phase}} \ge 45$s (`max_green`):** Bắt buộc đổi pha $A_{\text{valid}} = \{p \mid p \ne p_{\text{curr}}\}$.
3. **Chống đói pha linh hoạt (`max_red = 90`s):** Bộ đếm thời gian đỏ chỉ tích lũy khi làn đường pha đó có xe chờ ($q > 0$). Khi $t_{\text{red}} \ge 90$s, $A_{\text{valid}}$ bị thu hẹp chỉ gồm các pha bị đói.
4. **Khoảng tự do $[20\text{s}, 45\text{s})$:** Cho phép chọn bất kỳ pha nào $A_{\text{valid}} = \{0, \dots, N-1\}$.

---

## 6. Giao Thức Đánh Giá & Chọn Checkpoint

- **Train Seeds:** $[1, 2, \dots, 20]$ (luân phiên 100 episodes).
- **Validation Seeds:** $[21, 22, 23]$ (đánh giá định kỳ mỗi 5 episodes để lưu `*_best` checkpoint dựa trên $\bar{T}_{\text{wait}}$ nhỏ nhất).
- **Test Seeds:** `[42, 101, 2024, 3407, 9999]` (kiểm thử khách quan cuối cùng).
