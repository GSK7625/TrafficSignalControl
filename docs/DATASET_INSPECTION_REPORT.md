# BÁO CÁO KIỂM TRA TOÀN DIỆN DATASET RESCO (GIAI ĐOẠN 1)

**Dự án:** Tối ưu hóa điều khiển tín hiệu đèn giao thông bằng học tăng cường và các thuật toán kinh điển  
**Thời gian thực hiện:** Tháng 10/2026  
**Môi trường:** Python 3.11, SUMO 1.28.0, TraCI  

---

## 1. Tổng quan cấu trúc thư mục `data/`

Toàn bộ 10 dataset chuẩn từ benchmark RESCO ([Pi-Star-Lab/RESCO](https://github.com/Pi-Star-Lab/RESCO), commit `f1ed9a1`) đã được lưu trữ nguyên bản và toàn vẹn trong thư mục `data/`.

Cấu trúc thư mục:
```text
data/
├── SOURCES.md                     # Tài liệu nguồn gốc và giấy phép bản quyền
├── cologne1/                      # Dataset chính: 1 nút giao (Cologne TAPAS)
├── ingolstadt1/                   # Dataset chính: 1 nút giao (Ingolstadt InTAS)
├── cologne3/                      # Dataset mở rộng: 3 nút giao (Cologne TAPAS)
├── cologne8/                      # Dataset mở rộng: 8 nút giao
├── ingolstadt7/                   # Dataset mở rộng: 7 nút giao
├── ingolstadt21/                  # Dataset mở rộng: 21 nút giao
├── arterial4x4/                   # Dataset lưới động mạch: 16 nút giao (1,400 tuyến xe)
├── grid4x4/                       # Dataset lưới đồng nhất: 16 nút giao (1,400 tuyến xe)
├── saltlake2_400sX200w/           # Dữ liệu thực nghiệm Salt Lake City (2 nút giao)
└── saltlake2_stateXuniversity/    # Dữ liệu thực nghiệm Salt Lake City (2 nút giao)
```

**Nguyên tắc tuân thủ:**
- Không sửa đổi bất kỳ tệp dữ liệu gốc nào (`.net.xml`, `.rou.xml`, `.sumocfg`).
- Không thêm hoặc xóa tệp trong thư mục `data/`.
- Mọi kết quả mô phỏng được tách biệt và lưu vào `results/`.

---

## 2. Bảng tổng hợp chi tiết 10 Dataset RESCO

| STT | Dataset | Tệp `.sumocfg` | Cửa sổ thời gian mô phỏng (Giây) | Số nút giao (TLS) | Số làn điều khiển | Nhu cầu xe / Chuyến | Đặc điểm kịch bản |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **`cologne1`** | `cologne1.sumocfg` | **25,200 – 28,800** (3,600s / 1h) | **1** | **8** | **2,015 trips** | Cao điểm sáng Cologne (07:00–08:00), 1 nút giao lớn |
| **2** | **`ingolstadt1`** | `ingolstadt1.sumocfg` | **57,600 – 61,200** (3,600s / 1h) | **1** | **7** | **1,716 trips** | Cao điểm chiều Ingolstadt (16:00–17:00), luồng xe thực tế |
| **3** | **`cologne3`** | `cologne3.sumocfg` | **25,200 – 28,800** (3,600s / 1h) | **3** | **19** | **4,494 vehs** | Hành lang 3 nút giao liên tiếp Cologne |
| 4 | `cologne8` | `cologne8.sumocfg` | 25,200 – 28,800 (3,600s) | 8 | 33 | 2,046 vehs | Mạng lưới đô thị Cologne 8 nút giao |
| 5 | `ingolstadt7` | `ingolstadt7.sumocfg` | 57,600 – 61,200 (3,600s) | 7 | 56 | 3,031 trips | Phân vùng 7 nút giao Ingolstadt |
| 6 | `ingolstadt21` | `ingolstadt21.sumocfg` | 57,600 – 61,200 (3,600s) | 21 | 165 | 4,283 trips | Toàn bộ mạng lưới trung tâm Ingolstadt |
| 7 | `arterial4x4` | `arterial4x4.sumocfg` | 0 – 3,600 (3,600s) | 16 | 96 | 2,484 vehs | Lưới động mạch nhân tạo 4x4, đa kịch bản |
| 8 | `grid4x4` | `grid4x4.sumocfg` | 0 – 3,600 (3,600s) | 16 | 192 | 1,473 vehs | Lưới đối xứng nhân tạo 4x4 |
| 9 | `saltlake2_400sX200w` | `saltlake2_400sX200w.sumocfg` | 0 – 3,600 (3,600s) | 2 | 31 | Cảm biến CSV | Nút giao thực tế Salt Lake City 2023 |
| 10 | `saltlake2_stateXuniversity` | `saltlake2_stateXuniversity.sumocfg` | 0 – 3,600 (3,600s) | 2 | 29 | Cảm biến CSV | Nút giao thực tế Salt Lake City 2023 |

---

## 3. Phân tích chuyên sâu các Scenario trọng tâm

### 3.1. Cologne1 (`data/cologne1/`)
- **Tệp cấu hình:** `cologne1.sumocfg`
- **Mạng đường:** `cologne1.net.xml` (38.8 KB)
- **Tuyến đường:** `cologne1.rou.xml` (184.3 KB, chứa 2,015 `<trip>`, khởi hành từ `25205.0s` đến `28799.0s`)
- **Nút giao điều khiển:** `cluster_357187_359543`
  - Số liên kết được điều khiển (links): 20
  - Số làn vào (controlled incoming lanes): 8 làn
  - Số pha tín hiệu gốc trong XML: 8 pha (mỗi pha 5 giây mặc định)
  - Chuỗi tín hiệu:
    - Phase 0: `srrrrrrrrrGGGGGrrrrr`
    - Phase 1: `GGGGGrrrrrsrrrrrrrrr`
    - Phase 2: `GGGggrrrrrGGGggrrrrr`
    - Phase 3: `rrrGGrrrrrrrrGGrrrrr`
    - Phase 4: `rrrrrGGGGGrrrrrsrrrr`
    - Phase 5: `rrrrrsrrrrrrrrrGGGGG`
    - Phase 6: `rrrrrGGGggrrrrrGGGgg`
    - Phase 7: `rrrrrrrrGGrrrrrrrrGG`
- **Phát hiện quan trọng:** Trong mạng lưới SUMO gốc của Cologne1, không có pha vàng (`y`) trung gian được định nghĩa trước. Nếu thuật toán chuyển pha trực tiếp giữa hai pha xanh, SUMO sẽ cảnh báo và xe có thể phanh đột ngột. Do đó, **bộ SafePhaseManager đã được thiết kế để tự động tạo pha vàng an toàn** (`build_yellow_state`) trước mỗi lần chuyển pha.

### 3.2. Ingolstadt1 (`data/ingolstadt1/`)
- **Tệp cấu hình:** `ingolstadt1.sumocfg`
- **Mạng đường:** `ingolstadt1.net.xml` (33.6 KB)
- **Tuyến đường:** `ingolstadt1.rou.xml` (173.8 KB, chứa 45 `<vType>` và 1,716 `<trip>`, khởi hành từ `57600.2s` đến `61198.0s`)
- **Nút giao điều khiển:** `cluster_274083968_cluster_1200364014_1200364088`
  - Số liên kết được điều khiển (links): 8
  - Số làn vào: 7 làn
  - Số pha tín hiệu gốc trong XML: 6 pha
    - Phase 0: `GGgGrGGG` (38s) — Pha xanh chính luồng Bắc - Nam
    - Phase 1: `yygyryyy` (3s)  — Pha vàng giải tỏa
    - Phase 2: `GGGrrrrr` (6s)  — Pha xanh rẽ
    - Phase 3: `yyyrrrrr` (3s)  — Pha vàng giải tỏa
    - Phase 4: `rrrGGGrr` (37s) — Pha xanh nhánh Đông - Tây
    - Phase 5: `rrryyyrr` (3s)  — Pha vàng giải tỏa
- **Phân loại pha:** Tách biệt rõ ràng 3 pha xanh khả dụng (0, 2, 4) và 3 pha vàng chuyển tiếp (1, 3, 5). Bộ điều khiển SafePhaseManager đã tự động trích xuất đúng 3 pha xanh này để thuật toán lựa chọn.

### 3.3. Cologne3 (`data/cologne3/`)
- **Tệp cấu hình:** `cologne3.sumocfg`
- **Mạng đường:** `cologne3.net.xml` (150.6 KB)
- **Tuyến đường:** `cologne3.rou.xml` (776.9 KB, chứa 4,494 phương tiện)
- **Thời gian:** 25,200s – 28,800s (3,600s)
- **Nút giao điều khiển:** 3 nút giao gồm:
  - `360082`: 6 pha (3 pha xanh, 3 pha vàng), 11 links, 5 làn vào.
  - `360086`: 8 pha (4 pha xanh, 4 pha vàng), 18 links, 6 làn vào.
  - `cluster_2415878664_254486231_359566_359576`: 8 pha (4 pha xanh, 4 pha vàng), 20 links, 8 làn vào.

---

## 4. Kết quả kiểm tra khởi động với SUMO

1. **Chế độ kiểm tra độc lập (CLI):**
   - Đã chạy `sumo -c data/cologne1/cologne1.sumocfg`: Khởi động thành công, mô phỏng diễn ra chuẩn xác.
   - Đã chạy `sumo -c data/ingolstadt1/ingolstadt1.sumocfg`: Khởi động thành công, mã thoát code 0.
   - Đã chạy `sumo -c data/cologne3/cologne3.sumocfg`: Khởi động thành công, mã thoát code 0.

2. **Chế độ kết nối TraCI qua Python:**
   - Đã kiểm tra kết nối `traci.start(...)` và điều khiển từng bước `traci.simulationStep()` trên cả 3 kịch bản.
   - Tất cả tham số mạng, thời gian, danh sách xe, độ dài hàng đợi và pha đèn đều được đọc thành công 100%.
