# `data/` — Tập Dữ liệu Chuẩn RESCO SUMO

Thư mục [`data/`](file:///data) lưu trữ các kịch bản mô phỏng giao thông chuẩn hóa được trích xuất từ bộ benchmark quốc tế **RESCO** ([Reinforcement Learning Benchmarks for Traffic Signal Control](https://github.com/Pi-Star-Lab/RESCO)), phục vụ huấn luyện và đối chuẩn khoa học.

---

## 1. Danh sách kịch bản trọng tâm

Dự án tinh gọn và duy trì **3 kịch bản cốt lõi** đại diện cho các đặc trưng mạng lưới khác nhau:

| Tên Kịch bản | Số nút giao (TLS) | Số pha xanh (Actions) | Cửa sổ quan sát chuẩn | Nhu cầu xe trong cửa sổ | Đặc điểm mạng lưới | Giấy phép |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| [`cologne1`](file:///data/cologne1) | 1 | 8 pha | [25200.0, 28800.0) | 2,015 xe | Nút giao đơn thực tế tại TP. Cologne (Cologne TAPAS) | CC BY-NC-SA 3.0 |
| [`ingolstadt1`](file:///data/ingolstadt1) | 1 | 3 pha | [57600.0, 61200.0) | 1,716 xe | Nút giao đơn thực tế tại TP. Ingolstadt (InTAS) | GNU GPL v3.0 |
| [`cologne3`](file:///data/cologne3) | 3 | 3, 4, 4 pha | [25200.0, 28800.0) | 4,494 xe | Chuỗi 3 nút giao liên tiếp trên trục đường chính | CC BY-NC-SA 3.0 |

> [!NOTE]
> Thông tin chi tiết về nguồn gốc commit SHA, bản quyền gốc và quy trình tiền xử lý được ghi nhận đầy đủ tại [`data/SOURCES.md`](file:///data/SOURCES.md).

---

## 2. Cấu trúc tệp tin trong mỗi kịch bản

Mỗi thư mục kịch bản chứa đầy đủ các tệp định nghĩa mô phỏng của Eclipse SUMO:

```text
data/<scenario_name>/
├── <scenario_name>.net.xml     # Mạng lưới đường bộ (các nút giao, làn đường, kết nối, tín hiệu đèn)
├── <scenario_name>.rou.xml     # Nhu cầu phương tiện, lịch trình xuất phát và lộ trình di chuyển
├── <scenario_name>.sumocfg     # Tệp cấu hình tổng thể nạp vào engine SUMO / libsumo
└── LICENSE                     # Bản quyền phân phối của kịch bản gốc
```

---

## 3. Lệnh kiểm tra thông số kịch bản (Dataset Inspection)

Bộ công cụ [`src/data_inspector.py`](file:///src/data_inspector.py) tích hợp sẵn trong CLI cho phép kiểm tra nhanh thông số mạng lưới mà không cần mở SUMO:

```powershell
# Kiểm tra kịch bản cologne1
python main.py --inspect --dataset cologne1

# Kiểm tra kịch bản ingolstadt1
python main.py --inspect --dataset ingolstadt1

# Quét và hiển thị bảng tổng quan toàn bộ kịch bản
python main.py --inspect-all
```

**Ví dụ kết quả trích xuất:**
- Danh sách Traffic Light IDs (`tls_ids`).
- Chi tiết số lượng pha đèn và trạng thái chuỗi tín hiệu (`state string`) cho từng pha.
- Cửa sổ thời gian mô phỏng khuyến nghị (`begin_time`, `end_time`).
- Tổng số lượng xe xuất phát trong cửa sổ quan sát (`window_demand`).
