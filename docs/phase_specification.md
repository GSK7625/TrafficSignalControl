# Đặc Tả Pha Phục Vụ & Chuyển Pha An Toàn (Phase Specification)

Tài liệu xác định ánh xạ liên kết (links), tập pha xanh hợp lệ, quy tắc tạo chuỗi đèn vàng/đỏ dọn nút giao an toàn và các ràng buộc thời gian dùng chung cho cả 4 thuật toán (**Fixed-Time**, **Max-Pressure**, **Tabular Q-Learning**, **Deep Q-Network**) trên `ingolstadt1` và `cologne1`.

---

## 1. Ký Hiệu Đèn Tín Hiệu Trong SUMO

| Ký hiệu | Ý nghĩa | Hành vi phương tiện |
| :---: | :--- | :--- |
| `G` | Green Major (Xanh ưu tiên) | Được đi, quyền ưu tiên tuyệt đối không phải nhường đường. |
| `g` | Green Minor (Xanh thứ cấp) | Được đi nhưng phải nhường đường cho luồng ngược chiều (rẽ trái/quay đầu). |
| `y` | Yellow (Đèn vàng) | Dừng trước vạch trừ khi đã quá gần không thể phanh an toàn ($t=3$s). |
| `r` | Red (Đèn đỏ) | Dừng tuyệt đối trước vạch dừng. |
| `s` | Stop-Green (Dừng quan sát) | Phải dừng hẳn, quan sát an toàn mới được di chuyển (nhập làn rẽ phải). |

---

## 2. Bản Đồ `ingolstadt1`

- **Bộ điều khiển:** `cluster_274083968_cluster_1200364014_1200364088` (8 liên kết, 3 pha xanh).

### 2.1. Danh mục liên kết (Links 0 đến 7)
- **Links 0, 1:** Trục chính đi thẳng (`201963537#1` $\rightarrow$ `104010475#0`).
- **Link 2:** Trục chính rẽ trái (`201963537#1` $\rightarrow$ `-164051413`).
- **Link 3:** Đường nhánh rẽ phải (`164051413` $\rightarrow$ `124812857#0`).
- **Link 4:** Đường nhánh rẽ trái (`164051413` $\rightarrow$ `104010475#0`).
- **Link 5:** Hướng đối rẽ phải (`104010354` $\rightarrow$ `-164051413`).
- **Links 6, 7:** Hướng đối đi thẳng (`104010354` $\rightarrow$ `124812857#0`).

### 2.2. Tập 3 pha xanh hợp lệ
- **Pha 0 (`GGgGrGGG`):** Trục chính hai chiều thông suốt (rẽ trái Link 2 nhường đường `g`).
- **Pha 1 (`GGGrrrrr`):** Bảo hộ rẽ trái trục chính (Links 0, 1, 2 đều xanh `G`).
- **Pha 2 (`rrrGGGrr`):** Đường nhánh phục vụ (Links 3, 4, 5 xanh `G`).
*(Bao phủ 100% cả 8/8 liên kết).*

---

## 3. Bản Đồ `cologne1`

- **Bộ điều khiển:** `cluster_357187_359543` (20 liên kết, 8 pha xanh hợp lệ).
- Gồm hai trục vuông góc: **Trục 1** (Hướng A và C) và **Trục 2** (Hướng B và D).

### 2.1. Phân bổ 20 liên kết theo hướng
- **Links 0–4 (Hướng A):** Rẽ phải (0), Đi thẳng (1, 2), Rẽ trái (3), Quay đầu (4).
- **Links 5–9 (Hướng B):** Rẽ phải (5), Đi thẳng (6, 7), Rẽ trái (8), Quay đầu (9).
- **Links 10–14 (Hướng C):** Rẽ phải (10), Đi thẳng (11, 12), Rẽ trái (13), Quay đầu (14).
- **Links 15–19 (Hướng D):** Rẽ phải (15), Đi thẳng (16, 17), Rẽ trái (18), Quay đầu (19).

### 2.2. Tập 8 pha xanh hợp lệ
- **Pha 0 (`srrrrrrrrrGGGGGrrrrr`):** Trục 1 - Hướng C xanh toàn bộ; hướng A rẽ phải `s`.
- **Pha 1 (`GGGGGrrrrrsrrrrrrrrr`):** Trục 1 - Hướng A xanh toàn bộ; hướng C rẽ phải `s`.
- **Pha 2 (`GGGggrrrrrGGGggrrrrr`):** Trục 1 - Hai chiều thông suốt (A và C cùng đi thẳng `G`, rẽ trái `g`).
- **Pha 3 (`rrrGGrrrrrrrrGGrrrrr`):** Trục 1 - Bảo hộ rẽ trái hai chiều (A và C rẽ trái `G`).
- **Pha 4 (`rrrrrGGGGGrrrrrsrrrr`):** Trục 2 - Hướng B xanh toàn bộ; hướng D rẽ phải `s`.
- **Pha 5 (`rrrrrsrrrrrrrrrGGGGG`):** Trục 2 - Hướng D xanh toàn bộ; hướng B rẽ phải `s`.
- **Pha 6 (`rrrrrGGGggrrrrrGGGgg`):** Trục 2 - Hai chiều thông suốt (B và D cùng đi thẳng `G`, rẽ trái `g`).
- **Pha 7 (`rrrrrrrrGGrrrrrrrrGG`):** Trục 2 - Bảo hộ rẽ trái hai chiều (B và D rẽ trái `G`).
*(Bao phủ 100% cả 20/20 liên kết).*

---

## 4. Chuỗi Chuyển Pha Dọn Nút An Toàn (Clearance Rules)

Tuyệt đối cấm chuyển trực tiếp giữa hai pha xanh có xung đột:
$$\text{Pha Xanh Hiện Tại} \xrightarrow{t_{\text{yellow}} = 3,0\text{s}} \text{Pha Vàng Dọn Nút} \xrightarrow{t_{\text{all\_red}} = 1,0\text{s}} \text{Pha Đỏ Toàn Phần} \xrightarrow{} \text{Pha Xanh Mục Tiêu}$$

1. **Trạng thái đèn vàng (`build_yellow_state`):**
   - Vị trí $i$ đang xanh (`G/g/s`) mà pha mục tiêu chuyển sang đỏ (`r`) $\rightarrow$ chuyển thành `'y'`.
   - Vị trí $i$ tiếp tục xanh ở pha mục tiêu $\rightarrow$ giữ nguyên ký tự xanh.
   - Vị trí $i$ đang đỏ (`r`) $\rightarrow$ giữ `'r'`.
2. **Trạng thái đỏ toàn phần (`build_all_red_state`):**
   - Vị trí $i$ tiếp tục xanh ở pha mục tiêu $\rightarrow$ giữ nguyên ký tự xanh.
   - Toàn bộ các vị trí còn lại $\rightarrow$ chuyển thành `'r'` để giải phóng giao lộ.

---

## 5. Ràng Buộc Thời Gian & Quy Tắc Chọn Pha

| Thuật toán | Xanh tối thiểu ($t_{\text{min}}$) | Xanh tối đa ($t_{\text{max}}$) | Xanh cố định ($t_{\text{green}}$) | Đèn vàng ($t_{\text{yellow}}$) | Đỏ dọn nút ($t_{\text{all\_red}}$) | Giới hạn chống đói ($t_{\text{max\_red}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time (FT)** | — | — | Đúng $30,0\text{ s}$ | $3,0\text{ s}$ | $1,0\text{ s}$ | — |
| **Max-Pressure (MP)**| $20,0\text{ s}$ | $45,0\text{ s}$ | Động | $3,0\text{ s}$ | $1,0\text{ s}$ | $90,0\text{ s}$ (khi $q > 0$) |
| **Q-Learning (QL)** | $20,0\text{ s}$ | $45,0\text{ s}$ | Động | $3,0\text{ s}$ | $1,0\text{ s}$ | $90,0\text{ s}$ (Action Masking) |
| **Deep Q-Network (DQN)** | $20,0\text{ s}$ | $45,0\text{ s}$ | Động | $3,0\text{ s}$ | $1,0\text{ s}$ | $90,0\text{ s}$ (Action Masking) |

### Quy tắc phá vỡ thế hòa (Tie-Breaking)
Khi nhiều pha có cùng giá trị đánh giá cực đại:
1. Giữ pha hiện tại nếu nằm trong tập tốt nhất (tiết kiệm thời gian dọn nút 4s).
2. Chọn pha có thời gian chờ đỏ (`phase_red_time`) lớn nhất.
3. Nếu bằng nhau, chọn pha có chỉ số index nhỏ nhất (tiền định 100%).

---

## 6. Lệnh Kiểm Tra Trực Quan Với SUMO-GUI

```powershell
& "C:\Program Files (x86)\SUMO\bin\sumo-gui.EXE" -c data/cologne1/cologne1.sumocfg
```
*(Chỉnh thanh trượt Delay 50–100ms và nhấn Play để quan sát chuyển pha an toàn).*
