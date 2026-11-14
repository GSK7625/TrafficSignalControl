# Nguồn Dữ Liệu (Benchmark Datasets)

Dữ liệu mô phỏng giao thông được trích xuất từ repository [RESCO](https://github.com/Pi-Star-Lab/RESCO) (`resco_benchmark/environments/`).

## Danh Sách Kịch Bản

| Kịch bản | Số nút giao | Nguồn gốc | Giấy phép | Tệp cấu hình |
| :--- | :---: | :--- | :--- | :--- |
| `cologne1` | 1 | Cologne TAPAS | CC BY-NC-SA 3.0 | `data/cologne1/cologne1.sumocfg` |
| `cologne3` | 3 | Cologne TAPAS | CC BY-NC-SA 3.0 | `data/cologne3/cologne3.sumocfg` |
| `ingolstadt1` | 1 | Ingolstadt InTAS | GNU GPL v3.0 | `data/ingolstadt1/ingolstadt1.sumocfg` |

## Cấu Trúc Mỗi Kịch Bản

Mỗi thư mục kịch bản chứa 4 tệp:
- `*.net.xml`: Mạng lưới giao thông và các nút giao.
- `*.rou.xml`: Tuyến đường và nhu cầu lưu lượng phương tiện.
- `*.sumocfg`: Cấu hình khởi chạy mô phỏng SUMO.
- `LICENSE`: Giấy phép bản quyền của dữ liệu.
