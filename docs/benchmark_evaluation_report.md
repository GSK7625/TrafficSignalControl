# Báo Cáo Đối Chuẩn 4 Thuật Toán Điều Khiển Đèn Giao Thông (FT, MP, QL, DQN)

**Mạng mô phỏng:** Cologne1 (8 pha, 2.015 xe) & Ingolstadt1 (3 pha, 1.716 xe)  
**Tập Test Seeds (5 seeds độc lập):** `[42, 101, 2024, 3407, 9999]` (3.600s/seed, 40 lượt đánh giá)  
**Mô hình RL:** Huấn luyện 100 episodes với hàm thưởng **Delay-Aware Pressure (DAP)**

---

## 1. Bảng So Sánh Hiệu Năng Thực Nghiệm (Mean ± Std)

> [!NOTE]
> Chỉ số hành trình (**Waiting Time**, **Travel Time**, **Time Loss**) được tính trên tập xe hoàn thành ($V_{\text{completed}}$). Cần đối chiếu kèm **Throughput** và **Xe chưa xong** ($t=3.600$s) để phản ánh đầy đủ hiệu năng giải tỏa mạng lưới. Định luật bảo toàn lượng xe đạt $100\%$: $N_{\text{vào mạng}} = N_{\text{hoàn thành}} + N_{\text{chưa xong}}$.

### 1.1. Kịch bản Cologne1 (Nút 8 pha, lưu lượng cao: 2.015 xe)

| Thuật toán | Waiting Time (s) | Travel Time (s) | Time Loss (s) | Mean Queue (xe) | Max Queue (xe) | Throughput (xe) | Xe chưa xong |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | 74.35 ± 0.48 | 112.30 ± 0.64 | 89.41 ± 0.64 | 38.75 ± 0.33 | 88.0 ± 1.9 | 1970.0 ± 0.0 | 42.2 |
| **Max-Pressure** | 52.85 ± 3.10 | 91.87 ± 2.97 | 69.09 ± 2.97 | 27.36 ± 1.80 | 89.8 ± 4.1 | 1988.4 ± 6.8 | 22.8 |
| **Q-Learning (DAP)** | 34.20 ± 2.93 | 70.51 ± 3.29 | 47.69 ± 3.31 | 18.05 ± 1.56 | 55.0 ± 7.3 | 1982.2 ± 7.5 | 29.0 |
| **DQN (DAP)** | **24.51 ± 2.61** | **59.93 ± 3.38** | **37.17 ± 3.40** | **12.79 ± 1.59** | **45.6 ± 7.5** | **1994.2 ± 4.0** | **19.6** |

**So với Baseline Fixed-Time:**
- **Max-Pressure:** Chờ **-28.92%** | Hành trình **-18.19%** | Hàng đợi **-29.39%** | Thông lượng **+0.93%**
- **Q-Learning:** Chờ **-54.00%** | Hành trình **-37.21%** | Hàng đợi **-53.42%** | Thông lượng **+0.62%**
- **DQN:** Chờ **-67.03%** | Hành trình **-46.63%** | Hàng đợi **-66.99%** | Thông lượng **+1.23%**

---

### 1.2. Kịch bản Ingolstadt1 (Nút 3 pha bất đối xứng, lưu lượng vừa: 1.716 xe)

| Thuật toán | Waiting Time (s) | Travel Time (s) | Time Loss (s) | Mean Queue (xe) | Max Queue (xe) | Throughput (xe) | Xe chưa xong |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | 18.63 ± 0.85 | 49.12 ± 1.16 | 28.28 ± 1.13 | 4.90 ± 0.05 | 27.4 ± 0.9 | 1698.0 ± 2.8 | 17.2 |
| **Max-Pressure** | 12.92 ± 0.97 | 42.82 ± 1.47 | 22.00 ± 1.48 | **2.23 ± 0.08** | **15.0 ± 3.9** | 1697.2 ± 3.5 | 17.8 |
| **Q-Learning (DAP)** | **9.39 ± 1.18** | **39.06 ± 1.78** | **18.23 ± 1.77** | 2.37 ± 0.18 | 24.8 ± 4.2 | **1701.0 ± 2.4** | **14.0** |
| **DQN (DAP)** | 16.05 ± 2.91 | 46.21 ± 3.49 | 25.47 ± 3.54 | 2.40 ± 0.24 | 22.6 ± 1.8 | 1689.6 ± 7.8 | 20.8 |

**So với Baseline Fixed-Time:**
- **Max-Pressure:** Chờ **-30.65%** | Hành trình **-12.83%** | Hàng đợi **-54.49%** | Thông lượng **-0.05%**
- **Q-Learning:** Chờ **-49.60%** | Hành trình **-20.48%** | Hàng đợi **-51.63%** | Thông lượng **+0.18%**
- **DQN:** Chờ **-13.85%** | Hành trình **-5.92%** | Hàng đợi **-51.02%** | Thông lượng **-0.49%**

---

## 2. Đối Chiếu Lý Thuyết Giữa 4 Thuật Toán

| Đặc tính | Fixed-Time (FT) | Max-Pressure (MP) | Tabular Q-Learning (QL) | Deep Q-Network (DQN) |
| :--- | :--- | :--- | :--- | :--- |
| **Quyết định** | Chu kỳ tĩnh 30s | Tham lam: $\arg\max_i w(P_i)$ | RL tối ưu: $\arg\max_a Q(s, a)$ | RL xấp xỉ: $\arg\max_a Q(s, a; \theta)$ |
| **Hành động ($A$)** | Không có | Chọn pha $0 \dots N-1$ | Chọn pha trực tiếp $0 \dots N-1$ | Chọn pha trực tiếp $0 \dots N-1$ |
| **Trạng thái ($S$)** | Không quan sát | Số xe vào/ra ($x_u, x_d$) | Tuple rời rạc 4 phần tử | Vector chuẩn hóa $[0, 1]$ |
| **Hàm thưởng** | Không áp dụng | Không áp dụng | DAP-Reward ($P + \beta W - c_{\text{switch}}$) | DAP-Reward ($P + \beta W - c_{\text{switch}}$) |
| **Chống đói pha** | Không cần | Watchdog ($t_{\text{red}} \ge 90$s khi $q > 0$) | Action Masking ($t_{\text{red}} \ge 90$s) | Action Masking ($t_{\text{red}} \ge 90$s) |
| **Tầm nhìn (Horizon)**| Tĩnh | Tức thời ($\gamma = 0$) | Dài hạn (SMDP $\gamma = 0.95$) | Dài hạn (SMDP $\gamma = 0.95$) |
| **Độ trễ suy luận** | $< 0.001$ ms | $\sim 0.05$ ms | $\sim 0.01$ ms (hash lookup) | $\sim 0.40$ ms (MLP forward) |

---

## 3. Các Cải Tiến Kiến Trúc Cốt Lõi

1. **Hàm thưởng DAP (Delay-Aware Pressure):**
   $$r_k = -\left( \frac{P_k}{\Delta t_{\text{decision}}} + \beta \frac{W_k}{\Delta t_{\text{decision}} \cdot N_{\text{lanes}}} \right) - c_{\text{switch}} \mathbb{I}(a \ne p_{\text{curr}})$$
   - Kết hợp áp suất nút giao $P_k = \sum_m |w_m|$, tổng thời gian chờ $W_k$ (hệ số $\beta = 0.05$), và chi phí chuyển pha ($c_{\text{switch}} = 1.0$).
   - Khắc phục triệt để hiện tượng bỏ đói nhánh phụ và triệt tiêu dao động đổi pha liên tục.

2. **Direct Phase Selection & Action Masking:**
   - Cho phép nhảy trực tiếp tới pha có nhu cầu cao ($a \in \{0, \dots, N-1\}$) qua chuỗi an toàn: 3s vàng + 1s đỏ toàn phần.
   - Cưỡng chế an toàn giao thông: Giữ pha tối thiểu $20$s (`min_green`), đổi pha bắt buộc sau $45$s (`max_green`), giải tỏa nhánh đói sau $90$s (`max_red`).

3. **Cập nhật bán Markov (SMDP Intra-Option Discounting):**
   - Chiết khấu hiệu dụng $\gamma_{\text{eff}} = \gamma^{\Delta t / 5}$ với $\gamma_{\text{step}} = 0.95^{1/5} \approx 0.9898$.
   - Đồng bộ hóa chuyển tiếp cuối của hành động `KEEP` vào bộ đệm và Q-table chính xác.

---

## 4. Kết Luận & Xếp Hạng

1. **Cologne1 (8 pha, mật độ cao):** **DQN > Q-Learning > Max-Pressure > Fixed-Time**
   - Mạng neural trong DQN biểu diễn hiệu quả không gian trạng thái liên tục phức tạp của 8 pha, giảm **67.0%** thời gian chờ so với Fixed-Time.
2. **Ingolstadt1 (3 pha, bất đối xứng):** **Q-Learning > Max-Pressure > DQN > Fixed-Time**
   - Không gian trạng thái nhỏ gọn ($24$ states) giúp Tabular Q-Learning hội tụ tối ưu, giảm **49.6%** thời gian chờ (đạt 9.39s), vượt trội hơn cả Max-Pressure và DQN.
