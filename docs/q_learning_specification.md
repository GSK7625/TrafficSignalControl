# Đặc Tả Kỹ Thuật Tabular Q-Learning Cho Điều Khiển Đèn Giao Thông (Q-Learning Specification)

Tài liệu này xác định đầy đủ không gian trạng thái, hành động, hàm phần thưởng, quy tắc cập nhật Bellman, và quy trình thực nghiệm cho thuật toán **Tabular Q-Learning** trên hai bản đồ `cologne1` và `ingolstadt1`.

---

## 1. Không Gian Trạng Thái Rời Rạc Thu Gọn (Compact State Space)

Trạng thái tại thời điểm ra quyết định $s$ là một bộ tuple 4 phần tử:
$$s = (p_{\text{curr}}, t_{\text{bin}}, q_{\text{serving}}, q_{\text{competing}})$$

1. **$p_{\text{curr}}$ (Pha xanh hiện tại)**:
   - `ingolstadt1`: $p_{\text{curr}} \in \{0, 1, 2\}$ (3 giá trị).
   - `cologne1`: $p_{\text{curr}} \in \{0, 1, 2, 3, 4, 5, 6, 7\}$ (8 giá trị).
2. **$t_{\text{bin}}$ (Mức thời gian xanh đã phục vụ)**:
   - $t_{\text{bin}} = 0$: $20\text{s} \le t_{\text{in\_phase}} < 35\text{s}$ (vừa qua thời gian xanh tối thiểu).
   - $t_{\text{bin}} = 1$: $35\text{s} \le t_{\text{in\_phase}} < 45\text{s}$ (sắp chạm ngưỡng xanh tối đa).
3. **$q_{\text{serving}}$ (Mức hàng đợi của pha hiện tại)**:
   - $q_{\text{serving}} = 0$: $q_{\text{curr}} < 3\text{ xe}$ (ít xe dừng).
   - $q_{\text{serving}} = 1$: $q_{\text{curr}} \ge 3\text{ xe}$ (còn nhiều xe dừng chờ).
4. **$q_{\text{competing}}$ (Mức hàng đợi của các pha đối thủ)**:
   - $q_{\text{competing}} = 0$: $\max_{p \ne p_{\text{curr}}} q_p < 6\text{ xe}$ (nhánh đối thủ thông thoáng).
   - $q_{\text{competing}} = 1$: $\max_{p \ne p_{\text{curr}}} q_p \ge 6\text{ xe}$ (nhánh đối thủ đang dồn ứ).

### Kích thước Q-Table:
- `ingolstadt1`: $|S| = 3 \times 2 \times 2 \times 2 = \mathbf{24}\text{ states} \implies 24 \times 2 = \mathbf{48}\text{ Q-values}$.
- `cologne1`: $|S| = 8 \times 2 \times 2 \times 2 = \mathbf{64}\text{ states} \implies 64 \times 2 = \mathbf{128}\text{ Q-values}$.

---

## 2. Không Gian Hành Động (Action Space)

Tập hành động nhị phân $A = \{0, 1\}$:
- **$a = 0$ (`KEEP`)**: Giữ nguyên pha xanh hiện tại thêm một khoảng quyết định $\Delta t_{\text{decision}} = 5\text{s}$.
- **$a = 1$ (`SWITCH`)**: Yêu cầu kết thúc pha xanh hiện tại và chuyển tuần tự sang pha xanh kế tiếp $(p_{\text{curr}} + 1) \pmod{N_{\text{phases}}}$.
  - Kích hoạt chuỗi dọn nút giao an toàn: $3\text{s}$ vàng $\rightarrow$ $1\text{s}$ đỏ toàn phần $\rightarrow$ bật pha xanh mới.

---

## 3. Hàm Phần Thưởng (Reward Function)

Phần thưởng $r$ là giá trị âm của hàng đợi trung bình trên tất cả các làn vào trong khoảng thời gian vừa qua:
$$r = - \bar{Q}_{\text{interval}} = - \frac{1}{K} \sum_{k=1}^K \sum_{l \in L_{\text{incoming}}} \text{halting\_vehicles}(l, k)$$

- **Đơn vị**: `- số xe` (giá trị thực dao động trong khoảng $[-50, 0]$).
- Tối đa hóa reward $r$ đồng nghĩa với việc tối thiểu hóa tổng số xe dừng chờ, kéo giảm thời gian chờ của mạng lưới.

---

## 4. Công Thức Cập Nhật Bellman & $\epsilon$-Greedy

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a' \in A_{\text{valid}}(s')} Q(s', a') - Q(s, a) \right]$$

- $\alpha = 0,1$ (Tốc độ học - Learning Rate).
- $\gamma = 0,95$ (Hệ số chiết khấu - Discount Factor).
- Chính sách $\epsilon$-greedy:
  - Xác suất $\epsilon$: Chọn ngẫu nhiên $a \in A_{\text{valid}}(s)$.
  - Xác suất $1 - \epsilon$: Chọn $a = \arg\max_{a \in A_{\text{valid}}(s)} Q(s, a)$.
  - Lịch giảm $\epsilon$: $\epsilon_e = \max(0,05, 1,0 \times 0,90^e)$ qua 30 episodes.

---

## 5. Xử Lý Ràng Buộc An Toàn & Hành Động Không Hợp Lệ

1. Khi $t_{\text{in\_phase}} < 20\text{s}$: Chỉ cho phép `KEEP` (Action Masking).
2. Khi $t_{\text{in\_phase}} \ge 45\text{s}$: Chỉ cho phép `SWITCH` (Force Switch).
3. Khi có pha bị đói đèn đỏ $\ge 90\text{s}$: Bắt buộc `SWITCH`.
4. Khi tính $\max_{a'} Q(s', a')$: Chỉ lấy max trên tập hành động hợp lệ $A_{\text{valid}}(s')$.

---

## 6. Phân Chia Tập Dữ Liệu & Quy Trình Thực Nghiệm

- **Train Seeds**: $[1, 2, 3, \dots, 20]$ (luân phiên 30 episodes).
- **Validation Seeds**: $[11, 22, 33]$ (đánh giá sau mỗi 5 episodes để chọn checkpoint tốt nhất).
- **Test Seeds**: $[42, 101, 2024, 3407, 9999]$ (chạy độc lập ở giai đoạn đối chuẩn cuối cùng).
- **Tiêu chí chọn model**:
  $$\text{Best Checkpoint} = \arg\min_{\text{checkpoint}} \bar{T}_{\text{wait}}(S_{\text{val}})$$
- **Thái độ khoa học**: Không khẳng định trước Q-learning sẽ thắng FT hay MP; kết quả sẽ được đối chiếu khách quan bằng số liệu thực nghiệm.
