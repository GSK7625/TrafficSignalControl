# TrafficSignalControl-ML

Đồ án học tập điều khiển tín hiệu giao thông bằng SUMO và Python: FT, MP,
Q-Learning và DQN. Hiện mới hoàn thành **Giai đoạn 1 — khảo sát**;
chưa có môi trường điều khiển hoặc thuật toán.

Đọc [AGENTS.md](AGENTS.md) trước khi thay đổi dự án.
Nguồn và giấy phép dữ liệu: [data/SOURCES.md](data/SOURCES.md).

## Môi trường đã kiểm tra

- Windows, Python 3.11.9; dùng `.venv\Scripts\python.exe` để chạy đúng môi trường.
- SUMO và SUMO-GUI 1.28.0; `sumolib`/`traci` 1.28.0.
- `sumo` và `sumo-gui` phải có trong `PATH`.
- Máy khảo sát có `SUMO_HOME=C:\Program Files (x86)\SUMO\` và
  `PYTHONPATH=C:\Program Files (x86)\SUMO\tools`. Vì vậy, import mặc định lấy
  TraCI/sumolib từ bộ cài SUMO, dù `.venv` cũng có hai package.
  Script ghi đường dẫn import thực tế vào `survey.json`.

Không cần cài thêm thư viện để chạy khảo sát trên môi trường đã kiểm tra.
Nếu thiết lập máy khác: tạo virtualenv, cài `requirements.txt` và cài SUMO riêng.

## Chạy khảo sát (PowerShell, tại thư mục dự án)

```powershell
# Chỉ đọc XML, không khởi chạy mô phỏng
.\.venv\Scripts\python.exe scripts/inspect_scenario.py cologne1

# Chạy đủ một giờ mô phỏng theo cấu hình gốc, không điều khiển đèn
.\.venv\Scripts\python.exe scripts/inspect_scenario.py cologne1 --run
.\.venv\Scripts\python.exe scripts/inspect_scenario.py ingolstadt1 --run

# Quan sát GUI; --delay chỉ làm chậm hiển thị, đơn vị mili giây
.\.venv\Scripts\python.exe scripts/inspect_scenario.py cologne1 --gui --delay 15
.\.venv\Scripts\python.exe scripts/inspect_scenario.py ingolstadt1 --gui --delay 15

# Smoke test 100 bước từ thời điểm begin gốc, không đổi tệp cấu hình
.\.venv\Scripts\python.exe scripts/inspect_scenario.py cologne1 --run --steps 100
```

GUI tự chạy và tự đóng khi kết thúc. Để mở chương trình gốc và tự thao tác:

```powershell
sumo-gui -c data/cologne1/cologne1.sumocfg
sumo-gui -c data/ingolstadt1/ingolstadt1.sumocfg
```

Mỗi lần khảo sát tạo thư mục riêng `results/stage1/<scenario>_<mode>_<UTC>/`:

- `survey.json`: cấu hình, demand, pha, links, phiên bản, seed thực tế, SHA-256 đầu vào;
  có thông tin TraCI và số xe khi chạy với `--run`/`--gui`.
- `phase_observations.csv`: trạng thái TLS quan sát ban đầu và sau mỗi bước.
- `sumo.log`, `warnings.log`: giữ cả cảnh báo và lỗi của chương trình gốc.
- `summary.xml`, `tripinfo.xml`: đầu ra SUMO, gồm cả xe chưa hoàn thành/chưa vào mạng.

`results/` được Git bỏ qua. Chỉ đọc XML sẽ chỉ có `survey.json`.
Script không ghi vào `data/`, không gọi API đặt pha và không đổi seed, demand,
begin/end hay simulation step. `--steps` chỉ kết thúc sớm lượt smoke test.

## Đọc và trình bày kết quả

Xem [báo cáo Giai đoạn 1 và đề xuất Giai đoạn 2](docs/stage1_survey.md),
gồm bảng pha, làn, toàn bộ 28 controlled links và giới hạn đã phát hiện.

**Lưu ý:** Cologne gốc thiếu pha vàng và có cảnh báo phanh/dừng khẩn cấp.
Chạy được chương trình gốc chưa chứng minh được chuyển pha an toàn.
Chương trình gốc cũng không phải baseline FT-30 của đồ án.

Giai đoạn 2 chỉ bắt đầu sau khi người dùng duyệt thiết kế state/action/reward,
quy tắc thời gian và chuyển pha trong báo cáo.
