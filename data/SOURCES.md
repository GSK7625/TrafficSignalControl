# RESCO Benchmark Datasets - Data Provenance & Documentation

Tài liệu này ghi nhận nguồn gốc, giấy phép và thông tin chi tiết về các bộ dữ liệu giao thông được sử dụng trong dự án, thu thập từ repository RESCO phục vụ nghiên cứu và phát triển các thuật toán điều khiển tín hiệu đèn giao thông (Fixed-Time, Max-Pressure, Tabular Q-Learning).

---

## 1. Nguồn dữ liệu chính thức

- **Repository**: [Pi-Star-Lab/RESCO](https://github.com/Pi-Star-Lab/RESCO)
- **Đường dẫn gốc**: `resco_benchmark/environments/`
- **Commit SHA**: `f1ed9a174f8de41fc9d8689373b836bc882570dc`
- **Thời gian commit**: 2025-10-28T15:13:08Z
- **Phạm vi trích xuất**: Chỉ trích xuất dữ liệu mô phỏng SUMO và tài liệu bản quyền liên quan. Không tích hợp mã nguồn thuật toán, mô hình agent, runner hoặc framework thử nghiệm của RESCO.

---

## 2. Danh sách các bộ dữ liệu và Giấy phép bản quyền

Để tối ưu hóa hiệu năng tính toán, giảm thời gian giao tiếp TraCI và tiết kiệm tài nguyên bộ nhớ cho máy tính, dự án giữ lại **4 kịch bản trọng tâm** (đại diện cho cấu hình đơn nút giao, đa nút giao nhỏ và mạng lưới dạng lưới 4x4):

| STT | Tên Dataset | Giấy phép (License) | Mô tả & Loại kịch bản | Tệp cấu hình SUMO |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cologne1` | CC BY-NC-SA 3.0 | Mạng lưới Cologne (Cologne TAPAS) - 1 nút giao | `cologne1.sumocfg` |
| 2 | `cologne3` | CC BY-NC-SA 3.0 | Mạng lưới Cologne (Cologne TAPAS) - 3 nút giao | `cologne3.sumocfg` |
| 3 | `ingolstadt1` | GNU GPL v3.0 | Mạng lưới Ingolstadt (InTAS) - 1 nút giao | `ingolstadt1.sumocfg` |
| 4 | `grid4x4` | CC BY-NC-SA 4.0 | Mạng lưới dạng lưới 4x4 đồng nhất (16 nút giao) | `grid4x4.sumocfg` |

*Ghi chú: Bản quyền chi tiết của từng dataset được lưu kèm tại tệp `LICENSE` bên trong từng thư mục. Các kịch bản nặng hoặc dư thừa dữ liệu cảm biến (`arterial4x4`, `cologne8`, `ingolstadt7`, `ingolstadt21`, `saltlake2_400sX200w`, `saltlake2_stateXuniversity`) đã được lược bỏ nhằm bảo đảm hệ thống vận hành mượt mà trên môi trường máy có tài nguyên phần cứng giới hạn.*

---

## 3. Thao tác chuẩn bị và cấu trúc dữ liệu (Data Preparation)

1. **cologne1 & ingolstadt1** (Kịch bản đơn nút giao - Core Benchmarks):
   - Cấu trúc gọn nhẹ gồm các tệp mạng (`.net.xml`), tuyến đường (`.rou.xml`), cấu hình mô phỏng (`.sumocfg`) và tệp bản quyền (`LICENSE`).
   - Phục vụ làm kịch bản chuẩn cho huấn luyện RL (Tabular Q-Learning) và đánh giá đối chuẩn đa thuật toán.

2. **cologne3** (Kịch bản đa nút giao nhỏ - 3 nút giao):
   - Mạng lưới gồm 3 cụm đèn tín hiệu thực tế thuộc Cologne TAPAS.
   - Sử dụng kiểm thử khả năng điều khiển mở rộng trên mạng lưới đa nút giao liên tiếp.

3. **grid4x4** (Kịch bản dạng lưới bàn cờ - 16 nút giao):
   - Tệp `grid4x4.sumocfg` tham chiếu `grid4x4_1.rou.xml` kết hợp mạng lưới `grid4x4.net.xml`.
   - Giữ nguyên các tệp tuyến đường biến thể (`grid4x4_*.rou.xml`) và tệp `grid4x4.zip` gốc để phục vụ nhu cầu nghiên cứu kịch bản phân bố lưu lượng đa dạng.

---

## 4. Cấu trúc thư mục dữ liệu hiện tại

```text
data/
├── SOURCES.md
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
├── grid4x4/
│   ├── grid4x4.net.xml
│   ├── grid4x4.sumocfg
│   ├── grid4x4.zip
│   ├── LICENSE
│   └── grid4x4_*.rou.xml (1,400 tệp)
└── ingolstadt1/
    ├── ingolstadt1.net.xml
    ├── ingolstadt1.rou.xml
    ├── ingolstadt1.sumocfg
    └── LICENSE
```

---

## 5. Kết quả kiểm tra tính toàn vẹn (Integrity Verification)

- **Số lượng dataset hoạt động**: 4/4 dataset hợp lệ (`cologne1`, `cologne3`, `grid4x4`, `ingolstadt1`).
- **Tổng số tệp**: 1,417 tệp (bao gồm `SOURCES.md`).
- **Tổng dung lượng**: ~254.9 MB.
- **Kiểm tra XML/SUMO**: 100% các tệp `.xml` và `.sumocfg` (1,411 tệp) đều hợp lệ cú pháp, tham chiếu chính xác các tệp mạng và luồng giao thông.
- **Kiểm thử tự động**: Tương thích hoàn toàn với bộ kiểm thử tự động của dự án (`pytest`).
