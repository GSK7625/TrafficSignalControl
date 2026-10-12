# RESCO Benchmark Datasets - Data Provenance & Documentation

Tài liệu này ghi nhận nguồn gốc, giấy phép và thông tin chi tiết về các bộ dữ liệu giao thông được thu thập từ repository RESCO phục vụ nghiên cứu và phát triển các thuật toán điều khiển tín hiệu đèn giao thông (Fixed-Time, Max-Pressure, Q-Learning, DQN).

---

## 1. Nguồn dữ liệu chính thức

- **Repository**: [Pi-Star-Lab/RESCO](https://github.com/Pi-Star-Lab/RESCO)
- **Đường dẫn gốc**: `resco_benchmark/environments/`
- **Commit SHA**: `f1ed9a174f8de41fc9d8689373b836bc882570dc`
- **Thời gian commit**: 2025-10-28T15:13:08Z
- **Phạm vi trích xuất**: Chỉ trích xuất toàn bộ dữ liệu mô phỏng SUMO và tài liệu bản quyền liên quan. Không tích hợp hay giữ lại mã nguồn thuật toán, mô hình agent, runner hoặc framework thử nghiệm của RESCO.

---

## 2. Danh sách các bộ dữ liệu và Giấy phép bản quyền

Tất cả 10 dataset chính thức từ RESCO đã được tải đầy đủ và lưu trực tiếp trong thư mục `data/`:

| STT | Tên Dataset | Giấy phép (License) | Mô tả & Loại kịch bản | Tệp cấu hình SUMO |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `arterial4x4` | CC BY-NC-SA 4.0 | Mạng lưới trục động mạch 4x4 (16 nút giao) | `arterial4x4.sumocfg` |
| 2 | `cologne1` | CC BY-NC-SA 3.0 | Mạng lưới Cologne (Cologne TAPAS) - 1 nút giao | `cologne1.sumocfg` |
| 3 | `cologne3` | CC BY-NC-SA 3.0 | Mạng lưới Cologne (Cologne TAPAS) - 3 nút giao | `cologne3.sumocfg` |
| 4 | `cologne8` | CC BY-NC-SA 3.0 | Mạng lưới Cologne (Cologne TAPAS) - 8 nút giao | `cologne8.sumocfg` |
| 5 | `grid4x4` | CC BY-NC-SA 4.0 | Mạng lưới dạng lưới 4x4 đồng nhất (16 nút giao) | `grid4x4.sumocfg` |
| 6 | `ingolstadt1` | GNU GPL v3.0 | Mạng lưới Ingolstadt (InTAS) - 1 nút giao | `ingolstadt1.sumocfg` |
| 7 | `ingolstadt7` | GNU GPL v3.0 | Mạng lưới Ingolstadt (InTAS) - 7 nút giao | `ingolstadt7.sumocfg` |
| 8 | `ingolstadt21` | GNU GPL v3.0 | Mạng lưới Ingolstadt (InTAS) - 21 nút giao | `ingolstadt21.sumocfg` |
| 9 | `saltlake2_400sX200w` | GNU GPL v3.0 | Salt Lake City - Nút giao 400S x 200W (Dữ liệu thực tế 2023) | `saltlake2_400sX200w.sumocfg` |
| 10 | `saltlake2_stateXuniversity` | GNU GPL v3.0 | Salt Lake City - State St x University (Dữ liệu thực tế 2023) | `saltlake2_stateXuniversity.sumocfg` |

*Ghi chú: Bản quyền cụ thể của từng dataset được lưu kèm tại tệp `LICENSE` bên trong từng thư mục.*

---

## 3. Thao tác chuẩn bị và giải nén dữ liệu (Data Preparation)

1. **arterial4x4**:
   - Tệp gốc `arterial4x4.zip` chứa 1,400 kịch bản lưu lượng tuyến xe (`arterial4x4_1.rou.xml` đến `arterial4x4_1400.rou.xml`).
   - Đã giải nén toàn bộ 1,400 tệp `.rou.xml` trực tiếp vào `data/arterial4x4/` để tệp `arterial4x4.sumocfg` (tham chiếu `arterial4x4_1.rou.xml`) và các cấu hình chạy thử nghiệm có thể định vị tệp ngay lập tức mà không cần tạo thư mục lồng nhau.
   - Giữ nguyên tệp `arterial4x4.zip` gốc để đảm bảo tính nguyên bản của dataset.

2. **grid4x4**:
   - Tệp gốc `grid4x4.zip` chứa 1,400 kịch bản lưu lượng tuyến xe (`grid4x4_1.rou.xml` đến `grid4x4_1400.rou.xml`).
   - Đã giải nén toàn bộ 1,400 tệp `.rou.xml` trực tiếp vào `data/grid4x4/` để tệp `grid4x4.sumocfg` (tham chiếu `grid4x4_1.rou.xml`) hoạt động chính xác.
   - Giữ nguyên tệp `grid4x4.zip` gốc.

3. **saltlake2_400sX200w & saltlake2_stateXuniversity**:
   - Giữ nguyên toàn bộ cấu trúc thư mục con đo đếm thực nghiệm hàng ngày theo trạm cảm biến (`7241/`, `7242/` cho `saltlake2_400sX200w`; `7142/`, `7243/` cho `saltlake2_stateXuniversity`), mỗi thư mục gồm 365 tệp CSV dữ liệu lưu lượng cả năm 2023 (`2023-01-01.csv` đến `2023-12-31.csv`).
   - Giữ nguyên các tệp cấu hình mạng đường (`.net.xml`), dòng xe tổng hợp (`.flo.xml`) và tệp cấu hình (`.sumocfg`).

4. **cologne1, cologne3, cologne8, ingolstadt1, ingolstadt7, ingolstadt21**:
   - Giữ nguyên cấu trúc các tệp mạng (`.net.xml`), tuyến đường (`.rou.xml`), cấu hình mô phỏng (`.sumocfg`) và tệp bản quyền (`LICENSE`).

---

## 4. Cấu trúc thư mục dữ liệu

```text
data/
├── SOURCES.md
├── arterial4x4/
│   ├── arterial4x4.net.xml
│   ├── arterial4x4.sumocfg
│   ├── arterial4x4.zip
│   ├── LICENSE
│   └── arterial4x4_*.rou.xml (1,400 tệp)
├── cologne1/
│   ├── cologne1.net.xml
│   ├── cologne1.rou.xml
│   ├── cologne1.sumocfg
│   └── LICENSE
├── cologne3/
│   ├── cologne3.net.xml
│   ├── cologne3.rou.xml
│   ├── cologne3.sumocfg
│   └── LICENSE
├── cologne8/
│   ├── cologne8.net.xml
│   ├── cologne8.rou.xml
│   ├── cologne8.sumocfg
│   └── LICENSE
├── grid4x4/
│   ├── grid4x4.net.xml
│   ├── grid4x4.sumocfg
│   ├── grid4x4.zip
│   ├── LICENSE
│   └── grid4x4_*.rou.xml (1,400 tệp)
├── ingolstadt1/
│   ├── ingolstadt1.net.xml
│   ├── ingolstadt1.rou.xml
│   ├── ingolstadt1.sumocfg
│   └── LICENSE
├── ingolstadt7/
│   ├── ingolstadt7.net.xml
│   ├── ingolstadt7.rou.xml
│   ├── ingolstadt7.sumocfg
│   └── LICENSE
├── ingolstadt21/
│   ├── ingolstadt21.net.xml
│   ├── ingolstadt21.rou.xml
│   ├── ingolstadt21.sumocfg
│   └── LICENSE
├── saltlake2_400sX200w/
│   ├── 7241/ (365 tệp CSV 2023-01-01.csv -> 2023-12-31.csv)
│   ├── 7242/ (365 tệp CSV 2023-01-01.csv -> 2023-12-31.csv)
│   ├── saltlake2_400sX200w.flo.xml
│   ├── saltlake2_400sX200w.net.xml
│   ├── saltlake2_400sX200w.sumocfg
│   └── LICENSE
└── saltlake2_stateXuniversity/
    ├── 7142/ (365 tệp CSV 2023-01-01.csv -> 2023-12-31.csv)
    ├── 7243/ (365 tệp CSV 2023-01-01.csv -> 2023-12-31.csv)
    ├── saltlake2_stateXuniversity.flo.xml
    ├── saltlake2_stateXuniversity.net.xml
    ├── saltlake2_stateXuniversity.sumocfg
    └── LICENSE
```

---

## 5. Kết quả kiểm tra tính toàn vẹn (Integrity Verification)

- **Số lượng dataset**: 10/10 dataset hợp lệ.
- **Tổng số tệp**: 4,301 tệp (bao gồm `SOURCES.md`).
- **Tổng dung lượng**: ~914 MB (bao gồm tệp giải nén tuyến đường và tệp zip gốc).
- **Kiểm tra XML**: Đã phân tích cú pháp toàn bộ 2,828 tệp `.xml` và `.sumocfg`, 100% hợp lệ, không có lỗi cấu trúc.
- **Kiểm tra tham chiếu `.sumocfg`**: 10/10 kịch bản có đầy đủ tệp mạng (`.net.xml`) và tuyến đường (`.rou.xml` hoặc `.flo.xml`) được tham chiếu.
