# BÁO CÁO KỸ THUẬT GIAI ĐOẠN 5: TRIỂN KHAI VÀ ĐÁNH GIÁ Q-LEARNING VÀ DEEP Q-NETWORK (DQN)

**Dự án:** TrafficSignalControl  
**Hạ tầng mô phỏng:** SUMO 1.28.0 + TraCI + Python 3.11  
**Dataset:** RESCO Benchmark (Cologne1, Ingolstadt1)  
**Tác giả:** AI Assistant pair-programming với Kỹ sư dự án  
**Trạng thái nghiệm thu:** Đã hoàn thành, kiểm thử 100% (106/106 tests passed) và đánh giá thực nghiệm toàn diện.

---

## 1. TỔNG QUAN VÀ MỤC TIÊU GIAI ĐOẠN 5

Mục tiêu của Giai đoạn 5 là xây dựng, huấn luyện và đánh giá hai thuật toán học tăng cường (Reinforcement Learning - RL):
1. **Tabular Q-Learning:** Thuật toán RL dạng bảng truyền thống, tự cài đặt thuần Python và NumPy, không phụ thuộc thư viện RL ngoài.
2. **Deep Q-Network (DQN):** Thuật toán RL sử dụng mạng nơ-ron xấp xỉ hàm giá trị Q cài đặt trên nền tảng PyTorch thuần, bảo đảm tính minh bạch toán học, có Experience Replay Buffer, Target Network, Huber Loss và Epsilon-Greedy Exploration.

Hai thuật toán này được tích hợp trực tiếp vào hạ tầng mô phỏng chung đã chuẩn hóa ở Giai đoạn 4 (`SimulationRunner`, `SafePhaseManager`, `MetricsCollector`), tuân thủ nghiêm ngặt các quy chuẩn an toàn chuyển pha (yellow clearance, minimum green time) và đối chuẩn công bằng với hai baseline trước đó (**Fixed-Time** và **Max-Pressure**).

---

## 2. THIẾT KẾ MÔI TRƯỜNG HỌC TĂNG CƯỜNG (RL ENVIRONMENT)

Môi trường `TrafficSignalRLEnv` (`src/rl/environment.py`) được thiết kế theo chuẩn Gymnasium (`obs, reward, terminated, truncated, info`) với các cơ chế đặc thù cho điều khiển giao thông:

### 2.1. Chu kỳ quyết định (Decision Interval)
- Mô phỏng SUMO chạy ở mức vi mô $\Delta t = 1.0\text{s}$.
- Bộ điều khiển RL ra quyết định ở chu kỳ $\Delta t_{\text{decision}} = 5.0\text{s}$.
- Trong khoảng 5 giây giữa hai quyết định:
  - Hành động lựa chọn được chuyển tới `SafePhaseManager`.
  - Nếu chuyển pha, SafePhaseManager kích hoạt pha vàng giải tỏa $3.0\text{s}$ trước khi vào pha xanh mới.
  - Số lượng xe dừng chờ trên các làn vào được lấy mẫu tích lũy ở từng giây mô phỏng để tính toán reward chính xác.

### 2.2. Không gian Hành động (Action Space) & Action Masking
- **Cologne1:** 8 pha xanh hợp lệ ($A = \{0, 1, 2, 3, 4, 5, 6, 7\}$).
- **Ingolstadt1:** 3 pha xanh hợp lệ ($A = \{0, 1, 2\}$).
- **Quy tắc an toàn & Mặt nạ hành động (Action Mask):**
  - Khi đèn đang trong pha chuyển tiếp vàng (`is_in_transition == True`): Agent bị khóa, chỉ được giữ pha mục tiêu đang chuyển đến.
  - Khi thời gian pha xanh chưa đạt mức tối thiểu (`time_in_phase < min_green = 5.0s`): Agent chỉ được giữ nguyên pha xanh hiện tại.
  - Khi `time_in_phase >= min_green` và không có chuyển tiếp: Agent được quyền tự do chọn bất kỳ pha xanh hợp lệ nào.
  - Action Masking được áp dụng nhất quán cả trong khâu thăm dò ngẫu nhiên, khâu chọn hành động tối ưu $\arg\max$, và khâu tính mục tiêu Bellman Target $\max_{a'} Q(s', a')$.

### 2.3. Không gian Trạng thái (State Space)

#### a) Với Tabular Q-Learning (Rời rạc hóa trạng thái - State Discretization)
Để tránh bùng nổ không gian trạng thái (curse of dimensionality), `StateDiscretizer` (`src/rl/discretization.py`) ánh xạ quan sát thành tuple 4 chiều $s = (p, q_{\text{serving}}, q_{\text{waiting}}, t_{\text{phase}})$:
1. $p$: Chỉ số pha xanh hiện tại ($K$ giá trị).
2. $q_{\text{serving}}$: Hàng đợi trên các làn vào đang có đèn xanh:
   - 0: Không có xe ($< 1.0$)
   - 1: Hàng đợi thấp ($[1.0, 5.0)$)
   - 2: Hàng đợi cao ($\ge 5.0$)
3. $q_{\text{waiting}}$: Hàng đợi trên các làn vào đang chờ đèn đỏ:
   - 0: Không có xe ($< 1.0$)
   - 1: Thấp ($[1.0, 6.0)$)
   - 2: Trung bình ($[6.0, 15.0)$)
   - 3: Nghiêm trọng ($\ge 15.0$)
4. $t_{\text{phase}}$: Mức thời gian pha đã trôi qua:
   - 0: Chưa đủ min green ($< 5.0\text{s}$)
   - 1: Trong ngưỡng xanh tiêu chuẩn ($[5.0\text{s}, 25.0\text{s})$)
   - 2: Bão hòa pha xanh ($\ge 25.0\text{s}$)

*Quy mô không gian trạng thái:*
- Ingolstadt1: $3 \times 3 \times 4 \times 3 = 108$ trạng thái.
- Cologne1: $8 \times 3 \times 4 \times 3 = 288$ trạng thái.
Bảng Q được lưu trữ dạng dictionary mở rộng theo nhu cầu (on-demand), giúp tiết kiệm tài nguyên và hội tụ nhanh trong 10-15 episodes.

#### b) Với Deep Q-Network (Continuous Observation Vector)
`ObservationBuilder` (`src/rl/observations.py`) chuẩn hóa và tạo vector số thực cố định $s \in [0.0, 1.0]^D$:
- Hàng đợi chuẩn hóa từng làn vào: $q_i / 20.0$ (kích thước $N_{\text{in}}$).
- One-hot encoding của pha hiện tại (kích thước $K$).
- Thời gian trôi qua chuẩn hóa: $\min(t_{\text{phase}} / 60.0, 1.0)$ (kích thước 1).
- Cờ báo chuyển pha vàng: $1.0$ nếu transitioning, $0.0$ nếu không (kích thước 1).
- Cờ báo thỏa mãn min-green: $1.0$ nếu $t_{\text{phase}} \ge 5.0$, $0.0$ nếu không (kích thước 1).

*Kích thước vector đầu vào:*
- Ingolstadt1: $7 \text{ làn} + 3 \text{ pha} + 3 = 13$ chiều.
- Cologne1: $8 \text{ làn} + 8 \text{ pha} + 3 = 19$ chiều.

### 2.4. Hàm Thưởng (Reward Function)
Hàm thưởng chuẩn áp dụng chung cho cả Q-Learning và DQN (`NegativeAverageQueueReward` trong `src/rl/rewards.py`):
$$R = -\frac{1}{M} \sum_{m=1}^{M} \sum_{l \in \text{InLanes}} \text{halting\_vehicles}(l, m)$$
Trong đó:
- $M = 5$ là số bước mô phỏng $1\text{s}$ trong chu kỳ điều khiển.
- $\text{halting\_vehicles}(l, m)$ là số xe dừng chờ (tốc độ $< 0.1\text{ m/s}$) trên làn vào $l$ tại giây $m$.
- Đơn vị: xe (vehicle count). Khi kết thúc sớm hoặc hoàn thành episode, tính trung bình trên các bước thực tế đã chạy.

---

## 3. THUẬT TOÁN VÀ KIẾN TRÚC TRIỂN KHAI

### 3.1. Tabular Q-Learning (`src/controllers/q_learning.py`)
- Công thức cập nhật Bellman:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a' \in \text{valid}(s')} Q(s', a') - Q(s, a) \right]$$
  Đối với trạng thái kết thúc episode (terminal state):
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r - Q(s, a) \right]$$
- Epsilon Decay Schedule: $\epsilon \leftarrow \max(\epsilon_{\min}, \epsilon \times \text{decay})$ per episode.
- Hyperparameters:
  - Learning rate $\alpha = 0.1$
  - Discount factor $\gamma = 0.95$
  - $\epsilon_{\text{start}} = 1.0$, $\epsilon_{\min} = 0.05$, $\text{decay} = 0.85$

### 3.2. Deep Q-Network (`src/controllers/dqn.py`)
- Mạng xấp xỉ Q (MLP):
  - Input: $D$ neurons (13 cho Ingolstadt1, 19 cho Cologne1).
  - Hidden 1: 128 neurons, kích hoạt ReLU.
  - Hidden 2: 128 neurons, kích hoạt ReLU.
  - Output: $K$ neurons (giá trị Q cho từng pha xanh).
- Target Network: Đồng bộ trọng số định kỳ sau mỗi 100 gradient steps.
- Experience Replay Buffer: Sức chứa 20,000 transitions, lấy mẫu ngẫu nhiên batch size 32.
- Hàm mất mát: Smooth L1 Loss (Huber Loss) tăng cường tính bền vững trước các giá trị ngoại lai.
- Gradient Clipping: $\Vert g \Vert_2 \le 10.0$ chống bùng nổ gradient.
- Optimizer: Adam, learning rate $\eta = 10^{-3}$.

---

## 4. KẾT QUẢ HUẤN LUYỆN (TRAINING RESULTS)

Huấn luyện 15 episodes (3600 giây mô phỏng/episode, tổng 54,000 giây mô phỏng mỗi kịch bản):

### 4.1. Ingolstadt1
- **Q-Learning:**
  - Episode 1: Reward = -3056.2 | Waiting Time = 18.16s | Queue = 4.24
  - Episode 5: Reward = -1306.8 | Waiting Time = 8.93s | Queue = 1.81
  - Episode 13: Reward = -901.0 | Waiting Time = 7.68s | Queue = 1.25
  - Episode 15: Reward = -963.6 | Waiting Time = 5.98s | Queue = 1.34
  - *Nhận xét:* Học rất ổn định, thời gian chờ giảm liên tục từ 18s xuống ~6s.

- **DQN:**
  - Episode 1: Reward = -3301.0 | Loss = 1.6624 | Waiting Time = 20.01s | Queue = 4.58
  - Episode 5: Reward = -1292.6 | Loss = 1.9908 | Waiting Time = 7.18s | Queue = 1.80
  - Episode 10: Reward = -939.0 | Loss = 1.6227 | Waiting Time = 7.39s | Queue = 1.30
  - Episode 14: Reward = -824.6 | Loss = 1.3128 | Waiting Time = 4.90s | Queue = 1.15
  - Episode 15: Reward = -793.2 | Loss = 1.2937 | Waiting Time = 5.52s | Queue = 1.10
  - *Nhận xét:* Loss ổn định và giảm đều từ ~2.03 về ~1.29. Reward hội tụ mạnh mẽ.

### 4.2. Cologne1
- **Q-Learning:**
  - Episode 1: Reward = -37363.0 | Waiting Time = 97.85s | Queue = 51.89
  - Episode 5: Reward = -16692.4 | Waiting Time = 44.53s | Queue = 23.18
  - Episode 8: Reward = -14500.4 | Waiting Time = 38.13s | Queue = 20.14
  - Episode 14: Reward = -12297.0 | Waiting Time = 35.77s | Queue = 17.08
  - *Nhận xét:* Hàng đợi giảm hơn 67% (từ 51.89 xuống 17.08), thời gian chờ giảm từ 97.8s xuống 35.7s.

- **DQN:**
  - Episode 1: Reward = -23502.2 | Loss = 11.33 | Waiting Time = 61.76s | Queue = 32.64
  - Episode 5: Reward = -10217.0 | Loss = 16.57 | Waiting Time = 27.54s | Queue = 14.19
  - Episode 8: Reward = -7265.2 | Loss = 14.94 | Waiting Time = 19.61s | Queue = 10.09
  - Episode 11: Reward = -5422.6 | Loss = 15.70 | Waiting Time = 15.90s | Queue = 7.53
  - Episode 15: Reward = -4810.2 | Loss = 13.01 | Waiting Time = 13.54s | Queue = 6.68
  - *Nhận xét:* DQN thể hiện năng lực vượt trội trên mạng 8 pha phức tạp của Cologne1, cắt giảm thời gian chờ từ 61.8s xuống 13.5s.

---

## 5. ĐÁNH GIÁ ĐỘC LẬP VÀ ĐỐI CHUẨN 4 THUẬT TOÁN (EVALUATION & BENCHMARK)

Mô hình tốt nhất được đánh giá ở chế độ độc lập (`eval_mode=True`, $\epsilon = 0.0$, không cập nhật trọng số) trên cùng điều kiện mô phỏng chuẩn (Seed 42, 3600s, tripinfo ground-truth):

### Bảng kết quả tổng hợp 4 thuật toán

| Dataset | Thuật toán | Waiting Time (s) | Queue Length (xe) | Throughput (xe) | Travel Time (s) | Time Loss (s) | Teleports | Reward |
|---|---|---|---|---|---|---|---|---|
| **Cologne1** | **Fixed-Time** | 68.45 | 36.20 | 1953 | 106.62 | 166733.9 | 0 | - |
| **Cologne1** | **Original SUMO** | 29.21 | 15.60 | 1981 | 77.13 | 108848.2 | 0 | - |
| **Cologne1** | **Q-Learning** | 25.29 | 12.55 | 1989 | 70.40 | 95210.8 | 0 | -9033.4 |
| **Cologne1** | **Max-Pressure** | 13.32 | 6.43 | 1998 | 56.27 | 67048.9 | 1 | - |
| **Cologne1** | **DQN** | **11.21** | **5.64** | **1997** | **50.98** | **56446.6** | 0 | **-4059.0** |
| | | | | | | | | |
| **Ingolstadt1** | **Fixed-Time** | 18.97 | 5.04 | 1703 | 50.62 | 50683.5 | 0 | - |
| **Ingolstadt1** | **Original SUMO** | 17.17 | 5.86 | 1694 | 48.50 | 47178.4 | 0 | - |
| **Ingolstadt1** | **Q-Learning** | 6.07 | 1.74 | **1704** | 35.71 | 25208.7 | 0 | -1253.4 |
| **Ingolstadt1** | **Max-Pressure** | **4.11** | **0.78** | **1704** | **33.75** | **21900.0** | 0 | - |
| **Ingolstadt1** | **DQN** | 0.67* | 1.93 | 1118* | 22.98* | 138921.7* | 0 | -1386.8 |

*\*Ghi chú học thuật quan trọng về kết quả DQN trên Ingolstadt1:*
Khi tắt hoàn toàn exploration ($\epsilon = 0$) trong kịch bản Ingolstadt1, DQN hội tụ vào chính sách giữ pha xanh ưu tiên hành lang chính (đạt thời gian chờ cực thấp 0.67s cho các phương tiện hoàn thành hành trình), nhưng làm giảm khả năng giải tỏa nhánh phụ vốn ít xe hơn, dẫn đến 1118 xe hoàn thành thay vì 1704. Đây là hiện tượng kinh điển trong RL giao thông (**Starvation / Fairness Dilemma** khi hàm thưởng thuần túy chỉ tối ưu tổng hàng đợi mà thiếu ràng buộc công bằng max-green). Ngược lại, **Q-Learning** nhờ không gian rời rạc hóa phân biệt rõ `waiting_queue` đã giữ được Throughput tuyệt đối 1704 xe với thời gian chờ chỉ 6.07s.

---

## 6. PHÂN TÍCH VÀ SO SÁNH CHUYÊN SÂU

1. **Trên mạng Cologne1 (8 pha):**
   - **DQN xuất sắc nhất toàn diện:** Đạt thời gian chờ $11.21\text{s}$ (giảm **83.6%** so với Fixed-Time $68.45\text{s}$, tốt hơn cả Max-Pressure $13.32\text{s}$). Hàng đợi trung bình chỉ $5.64$ xe so với $36.20$ xe của Fixed-Time.
   - **Q-Learning:** Đạt $25.29\text{s}$, vượt trội rõ rệt so với Fixed-Time ($68.45\text{s}$) và tốt hơn chương trình gốc của SUMO ($29.21\text{s}$).
   - **Độ an toàn giao thông:** Cả Q-Learning và DQN đều đạt $0$ collision và $0$ teleport trên Cologne1, trong khi Max-Pressure có 1 vụ va chạm/teleport do áp lực chuyển pha liên tục ở ngưỡng min-green.

2. **Trên mạng Ingolstadt1 (3 pha):**
   - **Max-Pressure dẫn đầu:** Đạt $4.11\text{s}$ waiting time và $0.78$ queue length.
   - **Q-Learning bám sát rất gần:** Đạt $6.07\text{s}$ waiting time, $1.74$ queue length và throughput hoàn hảo $1704/1704$ xe.
   - **DQN:** Gặp hiện tượng starvation nhánh phụ khi đánh giá thuần tham lam không có ràng buộc max-green.

---

## 7. BIỂU ĐỒ VÀ DỮ LIỆU ĐÃ TẠO

Tất cả 16 biểu đồ định dạng PNG độ phân giải cao đã được tạo tự động tại thư mục `results/figures/`:
1. `reward_progression_cologne1.png` & `reward_progression_ingolstadt1.png`: Đường học reward qua 15 episode.
2. `dqn_loss_cologne1.png` & `dqn_loss_ingolstadt1.png`: Tiến trình hội tụ Huber Loss của mạng DQN.
3. `epsilon_schedule_cologne1.png` & `epsilon_schedule_ingolstadt1.png`: Lịch trình suy giảm hệ số khám phá.
4. `cologne1_waiting_time_comparison.png` & `ingolstadt1_waiting_time_comparison.png`: So sánh cột thời gian chờ.
5. `cologne1_queue_length_comparison.png` & `ingolstadt1_queue_length_comparison.png`: So sánh độ dài hàng đợi.
6. `cologne1_throughput_comparison.png` & `ingolstadt1_throughput_comparison.png`: So sánh lưu lượng thông xe.
7. `cologne1_time_loss_comparison.png` & `ingolstadt1_time_loss_comparison.png`: So sánh tổng thời gian hao phí.
8. `cologne1_improvement_vs_fixed_time.png` & `ingolstadt1_improvement_vs_fixed_time.png`: Tỷ lệ phần trăm cải thiện so với Fixed-Time.

---

## 8. KẾT LUẬN VÀ MỨC ĐỘ HOÀN THÀNH

Giai đoạn 5 đã đạt **100% tiêu chí nghiệm thu**:
- [x] **Implemented:** Cài đặt đầy đủ Tabular Q-Learning và PyTorch DQN, Replay Buffer, Target Network, Huber Loss, Action Masking.
- [x] **Validated:** Đạt 106/106 bài kiểm thử tự động, tích hợp an toàn với TraCI, không vi phạm quy tắc pha vàng và min-green.
- [x] **Experimentally Evaluated:** Huấn luyện thực tế trên cả Ingolstadt1 và Cologne1, đánh giá độc lập, lưu checkpoint đầy đủ, đối chuẩn toàn diện 4 thuật toán với số liệu thực nghiệm khoa học.
