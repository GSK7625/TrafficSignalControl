# Đặc Tả Pha Phục Vụ Và Quy Tắc Chuyển Pha (Phase Specification)

Tài liệu này xác định ánh xạ chi tiết các liên kết (links/movements), ý nghĩa các ký hiệu đèn, tập pha xanh hợp lệ và quy tắc chuyển pha dọn nút giao an toàn dùng chung cho cả ba thuật toán: **Fixed-Time (FT)**, **Max-Pressure (MP)**, và **Tabular Q-Learning** trên hai bản đồ `cologne1` và `ingolstadt1`.

---

## 1. Ý Nghĩa Các Ký Hiệu Đèn Tín Hiệu Trong SUMO

| Ký hiệu | Tên gọi | Quyền ưu tiên | Hành vi của phương tiện |
| :---: | :--- | :--- | :--- |
| `G` | Green Major (Xanh ưu tiên) | Quyền ưu tiên cao nhất | Xe được phép đi thẳng hoặc rẽ mà không phải nhường đường cho luồng ngược chiều. |
| `g` | Green Minor (Xanh thứ cấp) | Không ưu tiên (nhường đường) | Xe được phép đi (thường là rẽ trái hoặc quay đầu), nhưng bắt buộc phải nhường đường cho xe đi thẳng đối diện. |
| `y` | Yellow (Đèn vàng) | Chuyển tiếp dọn nút giao | Xe phải dừng lại trước vạch dừng trừ khi đã quá gần vạch dừng và không thể phanh an toàn. |
| `r` | Red (Đèn đỏ) | Cấm tuyệt đối | Cấm phương tiện vượt qua vạch dừng. |
| `s` | Stop-Green (Dừng quan sát) | Phải dừng hoàn toàn | Xe phải dừng hẳn trước vạch dừng, quan sát luồng ưu tiên, khi an toàn mới được di chuyển tiếp (tương tự biển STOP). |

---

## 2. Bản Đồ `ingolstadt1`

### 2.1. Ánh xạ 8 liên kết (Links 0 đến 7)
- **Bộ điều khiển**: `cluster_274083968_cluster_1200364014_1200364088` (1 bộ điều khiển, 8 links).

| Link Index | Làn vào (`from_lane`) | Làn ra (`to_lane`) | Hướng (`dir`) | Mô tả chuyển động |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `201963537#1_1` | `104010475#0_1` | `s` (thẳng) | Trục chính: Đi thẳng làn trong |
| **1** | `201963537#1_2` | `104010475#0_2` | `s` (thẳng) | Trục chính: Đi thẳng làn giữa |
| **2** | `201963537#1_3` | `-164051413_1` | `l` (trái) | Trục chính: Rẽ trái sang đường nhánh |
| **3** | `164051413_1` | `124812857#0_1` | `r` (phải) | Đường nhánh: Rẽ phải nhập vào trục chính |
| **4** | `164051413_2` | `104010475#0_2` | `l` (trái) | Đường nhánh: Rẽ trái cắt ngang trục chính |
| **5** | `104010354_1` | `-164051413_1` | `r` (phải) | Hướng đối: Rẽ phải sang đường nhánh |
| **6** | `104010354_1` | `124812857#0_2` | `s` (thẳng) | Hướng đối: Đi thẳng làn trong |
| **7** | `104010354_2` | `124812857#0_3` | `s` (thẳng) | Hướng đối: Đi thẳng làn ngoài |

### 2.2. Tập 3 pha xanh hợp lệ
- **Pha xanh 0 (`GGgGrGGG`)**: Trục chính hai chiều thông suốt (Links 0, 1, 3, 5, 6, 7 là `G`; link 2 rẽ trái nhường đường `g`).
- **Pha xanh 1 (Index 2 trong XML: `GGGrrrrr`)**: Pha bảo hộ rẽ trái trục chính (Links 0, 1 đi thẳng `G`, link 2 rẽ trái bảo hộ `G`).
- **Pha xanh 2 (Index 4 trong XML: `rrrGGGrr`)**: Pha đường nhánh cắt ngang (Link 3 rẽ phải `G`, link 4 rẽ trái `G`, link 5 rẽ phải `G`).

### 2.3. Kiểm tra độ bao phủ
- Cả 8/8 liên kết (100% các hướng) đều được phục vụ ít nhất 1 pha xanh:
  - Links 0, 1: Xanh ở Pha 0, Pha 1.
  - Link 2: Xanh ở Pha 0 (nhường đường `g`), Pha 1 (bảo hộ `G`).
  - Links 3, 5: Xanh ở Pha 0, Pha 2.
  - Link 4: Xanh ở Pha 2.
  - Links 6, 7: Xanh ở Pha 0.

---

## 3. Bản Đồ `cologne1`

### 3.1. Ánh xạ 20 liên kết (Links 0 đến 19)
- **Bộ điều khiển**: `cluster_357187_359543` (1 bộ điều khiển, 20 links).
- Gồm hai trục vuông góc: Trục 1 (A: `-32038056#3` đối xứng C: `28198821#3`), Trục 2 (B: `23429231#1` đối xứng D: `27115123#3`).

| Link Index | Làn vào | Làn ra | Hướng | Mô tả chuyển động |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `-32038056#3_0` | `32038051#0_0` | `r` (phải) | Trục 1 - Hướng A: Rẽ phải |
| **1** | `-32038056#3_0` | `-28198821#4_0`| `s` (thẳng) | Trục 1 - Hướng A: Đi thẳng làn ngoài |
| **2** | `-32038056#3_1` | `-28198821#4_1`| `s` (thẳng) | Trục 1 - Hướng A: Đi thẳng làn trong |
| **3** | `-32038056#3_1` | `32324544#0_1` | `l` (trái) | Trục 1 - Hướng A: Rẽ trái |
| **4** | `-32038056#3_1` | `32038056#0_1` | `t` (quay) | Trục 1 - Hướng A: Quay đầu |
| **5** | `23429231#1_0` | `32038056#0_0` | `r` (phải) | Trục 2 - Hướng B: Rẽ phải |
| **6** | `23429231#1_0` | `32038051#0_0` | `s` (thẳng) | Trục 2 - Hướng B: Đi thẳng làn ngoài |
| **7** | `23429231#1_1` | `32038051#0_1` | `s` (thẳng) | Trục 2 - Hướng B: Đi thẳng làn trong |
| **8** | `23429231#1_1` | `-28198821#4_1`| `l` (trái) | Trục 2 - Hướng B: Rẽ trái |
| **9** | `23429231#1_1` | `32324544#0_1` | `t` (quay) | Trục 2 - Hướng B: Quay đầu |
| **10** | `28198821#3_0` | `32324544#0_0` | `r` (phải) | Trục 1 - Hướng C: Rẽ phải |
| **11** | `28198821#3_0` | `32038056#0_0` | `s` (thẳng) | Trục 1 - Hướng C: Đi thẳng làn ngoài |
| **12** | `28198821#3_1` | `32038056#0_1` | `s` (thẳng) | Trục 1 - Hướng C: Đi thẳng làn trong |
| **13** | `28198821#3_1` | `32038051#0_1` | `l` (trái) | Trục 1 - Hướng C: Rẽ trái |
| **14** | `28198821#3_1` | `-28198821#4_1`| `t` (quay) | Trục 1 - Hướng C: Quay đầu |
| **15** | `27115123#3_0` | `-28198821#4_0`| `r` (phải) | Trục 2 - Hướng D: Rẽ phải |
| **16** | `27115123#3_0` | `32324544#0_0` | `s` (thẳng) | Trục 2 - Hướng D: Đi thẳng làn ngoài |
| **17** | `27115123#3_1` | `32324544#0_1` | `s` (thẳng) | Trục 2 - Hướng D: Đi thẳng làn trong |
| **18** | `27115123#3_1` | `32038056#0_1` | `l` (trái) | Trục 2 - Hướng D: Rẽ trái |
| **19** | `27115123#3_1` | `32038051#0_1` | `t` (quay) | Trục 2 - Hướng D: Quay đầu |

### 3.2. Tập pha xanh hợp lệ
Trong `cologne1.net.xml`, cả 8 pha gốc đều là các pha xanh hợp lệ (không có chứa đèn vàng `y`):
- **Pha 0 (`srrrrrrrrrGGGGGrrrrr`)**: Trục 1 hướng C xanh toàn bộ, hướng A rẽ phải `s`.
- **Pha 1 (`GGGGGrrrrrsrrrrrrrrr`)**: Trục 1 hướng A xanh toàn bộ, hướng C rẽ phải `s`.
- **Pha 2 (`GGGggrrrrrGGGggrrrrr`)**: Trục 1 hai chiều thông suốt (A và C cùng đi thẳng/rẽ phải ưu tiên `G`, rẽ trái/quay đầu nhường đường `g`).
- **Pha 3 (`rrrGGrrrrrrrrGGrrrrr`)**: Trục 1 bảo hộ rẽ trái hai chiều (A và C rẽ trái `G`).
- **Pha 4 (`rrrrrGGGGGrrrrrsrrrr`)**: Trục 2 hướng B xanh toàn bộ, hướng D rẽ phải `s`.
- **Pha 5 (`rrrrrsrrrrrrrrrGGGGG`)**: Trục 2 hướng D xanh toàn bộ, hướng B rẽ phải `s`.
- **Pha 6 (`rrrrrGGGggrrrrrGGGgg`)**: Trục 2 hai chiều thông suốt (B và D cùng đi thẳng/rẽ phải ưu tiên `G`, rẽ trái/quay đầu nhường đường `g`).
- **Pha 7 (`rrrrrrrrGGrrrrrrrrGG`)**: Trục 2 bảo hộ rẽ trái hai chiều (B và D rẽ trái `G`).

### 3.3. Kiểm tra độ bao phủ
- Toàn bộ 20/20 liên kết (100% các hướng) đều được phục vụ trong các pha trên.
- Ngay cả khi xét tập thu gọn 2 pha chính (Pha 2 và Pha 6), toàn bộ 20 liên kết cũng đã được bao phủ đầy đủ.

---

## 4. Quy Tắc Chuyển Pha Dọn Nút Giao An Toàn (Clearance Rules)

### 4.1. Quy tắc cấm chuyển trực tiếp
Tuyệt đối **không được chuyển trực tiếp giữa hai pha xanh có luồng xung đột**. Khi chuyển từ pha xanh hiện tại $P_{\text{curr}}$ sang pha xanh mục tiêu $P_{\text{target}}$ ($P_{\text{target}} \ne P_{\text{curr}}$):

$$\text{Pha Xanh Hiện Tại} \xrightarrow{t_{\text{yellow}} = 3,0\text{s}} \text{Pha Vàng Dọn Nút} \xrightarrow{t_{\text{all\_red}} = 1,0\text{s}} \text{Pha Đỏ Toàn Phần} \xrightarrow{} \text{Pha Xanh Mục Tiêu}$$

### 4.2. Công thức tạo trạng thái đèn vàng
Tại vị trí ký tự thứ $i$ của chuỗi tín hiệu:
$$\text{State}_{\text{yellow}}[i] = \begin{cases}
c, & \text{nếu } c \in \{\text{'G'}, \text{'g'}, \text{'s'}\} \text{ và } \text{State}_{\text{target}}[i] \in \{\text{'G'}, \text{'g'}, \text{'s'}\} \text{ (tiếp tục xanh)} \\
\text{'y'}, & \text{nếu } c \in \{\text{'G'}, \text{'g'}, \text{'s'}\} \text{ và } \text{State}_{\text{target}}[i] = \text{'r'} \text{ (sắp đỏ)} \\
\text{'r'}, & \text{nếu } c = \text{'r'} \text{ (tiếp tục đỏ)}
\end{cases}$$

### 4.3. Công thức tạo trạng thái đỏ toàn phần (All-Red)
Toàn bộ chuỗi tín hiệu hiển thị `'r'`:
$$\text{State}_{\text{all\_red}} = \text{'r'} \times N_{\text{links}}$$

---

## 5. Ràng Buộc Thời Gian & Quy Tắc Chọn Pha

| Thuật toán | Giới hạn xanh tối thiểu ($t_{\text{min}}$) | Giới hạn xanh tối đa ($t_{\text{max}}$) | Thời gian xanh cố định ($t_{\text{green}}$) | Thời gian vàng ($t_{\text{yellow}}$) | Thời gian đỏ toàn phần ($t_{\text{all\_red}}$) | Giới hạn chống đói pha ($t_{\text{max\_red}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time (FT)** | Không áp dụng | Không áp dụng | Đúng $30,0\text{ s}$ | $3,0\text{ s}$ | $1,0\text{ s}$ | Không áp dụng |
| **Max-Pressure (MP)** | $20,0\text{ s}$ | $45,0\text{ s}$ | Thích ứng động | $3,0\text{ s}$ | $1,0\text{ s}$ | $90,0\text{ s}$ |
| **Tabular Q-Learning**| $20,0\text{ s}$ | $45,0\text{ s}$ | Thích ứng động | $3,0\text{ s}$ | $1,0\text{ s}$ | $90,0\text{ s}$ |

### 5.1. Quy tắc hòa (Tie-Breaking Rule)
Khi có 2 hoặc nhiều pha đối thủ có cùng giá trị áp suất tối đa (MP) hoặc cùng giá trị Q tối đa (Q-learning):
1. **Ưu tiên 1**: Nếu pha xanh hiện tại cũng nằm trong tập các pha đạt giá trị tối đa $\rightarrow$ Giữ nguyên pha hiện tại (tránh chuyển pha lãng phí thời gian dọn nút giao).
2. **Ưu tiên 2**: Chọn pha đối thủ có thời gian chờ đèn đỏ (`phase_red_time`) lớn nhất (ưu tiên giải phóng hướng chờ lâu hơn).
3. **Ưu tiên 3**: Nếu thời gian chờ đỏ bằng nhau, chọn pha có chỉ số index nhỏ hơn (đảm bảo tính tiền định và tái lập 100%).

---

## 6. Hướng Dẫn Kiểm Tra Trực Quan Bằng SUMO-GUI

1. Khởi động SUMO-GUI:
   ```powershell
   & "C:\Program Files (x86)\SUMO\bin\sumo-gui.EXE" -c data/cologne1/cologne1.sumocfg
   ```
2. Phóng to nút giao `cluster_357187_359543`.
3. Nhấp chuột phải vào nút giao $\rightarrow$ Chọn **Show Phases** hoặc mở cửa sổ theo dõi tín hiệu.
4. Đặt độ trễ mô phỏng (Delay) khoảng $50\text{ms} - 100\text{ms}$ tại thanh trượt trên thanh công cụ.
5. Nhấn nút **Play** (tam giác xanh) để quan sát luồng xe và trạng thái đèn.
