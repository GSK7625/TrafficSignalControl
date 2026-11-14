# Giai đoạn 1 — Khảo sát TrafficSignalControl-ML

Ngày khảo sát: 14/11/2026 (Asia/Saigon). Phạm vi: `cologne1`, `ingolstadt1`.
Không triển khai FT/MP/QL/DQN, không khảo sát sâu `cologne3`, không sửa dữ liệu gốc.
Các mục thiết kế ở cuối tài liệu là **đề xuất chưa được phê duyệt**, chưa có mã Giai đoạn 2.

## 1. Repository và môi trường

- Git branch `main`, HEAD `ede72c7693d373bcd8625998d47b2aa9710b6a52`.
  Trước nhiệm vụ: `AGENTS.md` đã được staged (`A  AGENTS.md`); giữ nguyên.
- Ban đầu có `AGENTS.md`, `requirements.txt`, `.gitignore`, `.venv`, dữ liệu và giấy phép;
  chưa có README, `src/`, `scripts/`, tests, models hay kết quả cũ.
- Đã đọc toàn bộ AGENTS, hai cấu hình, mạng, nhu cầu, SOURCES và thông tin giấy phép liên quan.
  Nguồn theo SOURCES: RESCO, Cologne TAPAS (CC BY-NC-SA 3.0), Ingolstadt InTAS (GPL v3).
  Chưa kiểm tra lịch sử/upstream commit của bản sao dữ liệu.

| Thành phần | Kết quả thực tế |
| --- | --- |
| Python thường / virtualenv | 3.11.9 / 3.11.9, Windows 64 bit |
| Python dùng chạy | `.venv\Scripts\python.exe` |
| SUMO / SUMO-GUI | 1.28.0 / 1.28.0 |
| TraCI server | API 22, `SUMO 1.28.0` |
| pip distributions traci / sumolib | 1.28.0 / 1.28.0 |
| numpy / matplotlib / pytest | 2.4.6 / 3.11.2 / 9.2.0 |
| Kiểm tra phụ thuộc | `python -m pip check`: No broken requirements found |
| SUMO executables | `C:\Program Files (x86)\SUMO\bin\` |
| SUMO_HOME | `C:\Program Files (x86)\SUMO\` |
| PYTHONPATH | `C:\Program Files (x86)\SUMO\tools` |
| Import thực tế | `traci` và `sumolib` từ `SUMO\tools`, do PYTHONPATH ưu tiên trước site-packages |
| Tham số thực tế qua TraCI | seed `23423`, random `false`, step-length `1 s`, route-steps `200 s` |
| Xử lý lỗi mặc định được giữ nguyên | collision.action `teleport`, time-to-teleport `300 s`, xml-validation `local` |

Không cài thêm package, không thay môi trường hoặc requirements. Hai package có trong
virtualenv không có nghĩa Python đang import bản đó; xem `environment.*_path` trong JSON.
Kiểm tra tương thích thực tế bằng kết nối TraCI và chạy đủ cửa sổ mô phỏng.

## 2. Cấu hình và nhu cầu

| Thuộc tính | cologne1 | ingolstadt1 |
| --- | --- | --- |
| begin → end (giây SUMO) | 25200 → 28800 | 57600 → 61200 |
| Giờ trong ngày của dữ liệu | 07:00 → 08:00 | 16:00 → 17:00 |
| Độ dài cửa sổ | 3600 s | 3600 s |
| Network format | 1.9 | 1.9 |
| External / internal edges | 10 / 28 | 11 / 13 |
| TLS count | 1 | 1 |
| Program / type / offset | `0` / static / 0 | `0` / static / 0 |
| Controlled link indices / lane-to-lane connections | 20 / 20 | 8 / 8 |
| Controlled lanes trả về / số làn không trùng | 20 / 8 | 8 / 7 |
| Số `<trip>` | 2015 | 1716 |
| Số ID trip không trùng | 2015 | 1716 |
| Depart sớm nhất → muộn nhất (s) | 25205 → 28799 | 57600.2 → 61198 |
| Trips có depart trong [begin, end) | 2015 | 1716 |
| `<vehicle>` / `<flow>` / `<route>` khai báo | 0 / 0 / 0 | 0 / 0 / 0 |
| vType khai báo / thực sự dùng | 1 / 1 | 45 / 8 |

TLS Cologne: `cluster_357187_359543`.
TLS Ingolstadt: `cluster_274083968_cluster_1200364014_1200364088`.

Đầu vào là **trip với điểm đi–đến**, không phải danh sách route edges đầy đủ. SUMO có
routing device để tìm đường; log ghi DijkstraRouter. OD và lịch depart bất biến khi
giữ nguyên XML. Cologne có `speedDev=0.1`, length=4.3 m, minGap=1.5 m cho `pkw`.
Ingolstadt có 17 trip bus, 1699 trip passenger; tên `random_016`/`random_017`
không tự chứng minh có sinh thêm demand ngẫu nhiên. Phân loại theo type:

| Type | Số trip Ingolstadt |
| --- | ---: |
| default_016 | 647 |
| default_015 | 7 |
| random_016 | 52 |
| bus | 17 |
| default_017 | 907 |
| random_017 | 80 |
| default_018 | 4 |
| random_018 | 2 |

Các OD `from == to` có sẵn (Cologne 4 trip, Ingolstadt 1 trip); giữ nguyên, không
coi là dữ liệu cần tự sửa. Toàn bộ OD counts và vType attributes nằm trong JSON.
Chưa chạy nhiều seed; không tuyên bố các seed sẽ tạo các nhu cầu giao thông khác nhau.

## 3. Controlled lanes và controlled links

Đã gọi `getControlledLanes()` và `getControlledLinks()` rồi đối chiếu **tất cả**
tuple `(linkIndex, incoming_lane, outgoing_lane, via)` với `<connection>` trong XML.
Đã đối chiếu cả program ID, thứ tự pha, state và duration qua `getAllProgramLogics()`.
Tất cả khớp trong hai lượt headless và hai lượt GUI đầy đủ đầu tiên.

Các làn vào Cologne: `-32038056#3_0`, `-32038056#3_1`, `23429231#1_0`,
`23429231#1_1`, `28198821#3_0`, `28198821#3_1`, `27115123#3_0`, `27115123#3_1`.

Các làn vào được TLS điều khiển ở Ingolstadt: `201963537#1_1`, `201963537#1_2`,
`201963537#1_3`, `164051413_1`, `164051413_2`, `104010354_1`, `104010354_2`.
`incLanes` của junction còn liệt kê ba làn `_0` trên ba cạnh vào; không tự thêm
chúng vào tập controlled lanes. Phạm vi đo queue phải được chốt ở Giai đoạn 2.

`getControlledLanes()` lặp làn theo movement. Phải dùng danh sách không trùng khi
cộng queue; một link index có thể chứa nhiều connection trong mạng khác, nên mã
khảo sát duyệt từng group của `getControlledLinks()`.

### Cologne — toàn bộ 20 links

| Index | Incoming lane | Outgoing lane | Dir | Via suffix |
| ---: | --- | --- | --- | --- |
| 0 | -32038056#3_0 | 32038051#0_0 | r | 0_0 |
| 1 | -32038056#3_0 | -28198821#4_0 | s | 1_0 |
| 2 | -32038056#3_1 | -28198821#4_1 | s | 1_1 |
| 3 | -32038056#3_1 | 32324544#0_1 | l | 3_0 |
| 4 | -32038056#3_1 | 32038056#0_1 | t | 4_0 |
| 5 | 23429231#1_0 | 32038056#0_0 | r | 5_0 |
| 6 | 23429231#1_0 | 32038051#0_0 | s | 6_0 |
| 7 | 23429231#1_1 | 32038051#0_1 | s | 6_1 |
| 8 | 23429231#1_1 | -28198821#4_1 | l | 8_0 |
| 9 | 23429231#1_1 | 32324544#0_1 | t | 9_0 |
| 10 | 28198821#3_0 | 32324544#0_0 | r | 10_0 |
| 11 | 28198821#3_0 | 32038056#0_0 | s | 11_0 |
| 12 | 28198821#3_1 | 32038056#0_1 | s | 11_1 |
| 13 | 28198821#3_1 | 32038051#0_1 | l | 13_0 |
| 14 | 28198821#3_1 | -28198821#4_1 | t | 14_0 |
| 15 | 27115123#3_0 | -28198821#4_0 | r | 15_0 |
| 16 | 27115123#3_0 | 32324544#0_0 | s | 16_0 |
| 17 | 27115123#3_1 | 32324544#0_1 | s | 16_1 |
| 18 | 27115123#3_1 | 32038056#0_1 | l | 18_0 |
| 19 | 27115123#3_1 | 32038051#0_1 | t | 19_0 |

Via đầy đủ = `:cluster_357187_359543_` + suffix. `dir`: r phải, s thẳng,
l trái, t quay đầu. Đây là hướng movement, khác với ký tự signal state `s`.

### Ingolstadt — toàn bộ 8 links

| Index | Incoming lane | Outgoing lane | Dir | Via suffix |
| ---: | --- | --- | --- | --- |
| 0 | 201963537#1_1 | 104010475#0_1 | s | 0_0 |
| 1 | 201963537#1_2 | 104010475#0_2 | s | 0_1 |
| 2 | 201963537#1_3 | -164051413_1 | l | 2_0 |
| 3 | 164051413_1 | 124812857#0_1 | r | 3_0 |
| 4 | 164051413_2 | 104010475#0_2 | l | 4_0 |
| 5 | 104010354_1 | -164051413_1 | r | 5_0 |
| 6 | 104010354_1 | 124812857#0_2 | s | 6_0 |
| 7 | 104010354_2 | 124812857#0_3 | s | 6_1 |

Via đầy đủ = `:cluster_274083968_cluster_1200364014_1200364088_` + suffix.
MP về sau cần dùng đúng outgoing lane trong bảng, không chỉ tổng queue đầu vào.

## 4. Chương trình đèn gốc và vấn đề chuyển pha

Ký tự thứ 0 (bên trái) của state ứng với link index 0. `G` xanh có ưu tiên,
`g` xanh phải nhường, `s` mũi tên phải buộc dừng/nhường, `r` đỏ, `y` vàng.
Không biến `g` thành `G`, không coi `s` là đỏ hoặc xanh ưu tiên.
Đối chiếu [tài liệu signal states của SUMO](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions).

### Cologne

| Phase index | Duration (s) | State |
| ---: | ---: | --- |
| 0 | 5 | `srrrrrrrrrGGGGGrrrrr` |
| 1 | 5 | `GGGGGrrrrrsrrrrrrrrr` |
| 2 | 5 | `GGGggrrrrrGGGggrrrrr` |
| 3 | 5 | `rrrGGrrrrrrrrGGrrrrr` |
| 4 | 5 | `rrrrrGGGGGrrrrrsrrrr` |
| 5 | 5 | `rrrrrsrrrrrrrrrGGGGG` |
| 6 | 5 | `rrrrrGGGggrrrrrGGGgg` |
| 7 | 5 | `rrrrrrrrGGrrrrrrrrGG` |

Chu kỳ gốc: 40 s, thứ tự 0→1→…→7→0. Tất cả có `minDur=5`, `maxDur=50`,
nhưng type **static** và duration thực tế là 5 s; không tự diễn giải thành điều khiển actuated.
Không có `y`, `u` hay pha toàn đỏ. Tổng thời gian mỗi loại tín hiệu trong chu kỳ:

| Link indices | G (s) | g (s) | s (s) | r (s) | y (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0,5,10,15 | 10 | 0 | 5 | 25 | 0 |
| 1,2,6,7,11,12,16,17 | 10 | 0 | 0 | 30 | 0 |
| 3,4,8,9,13,14,18,19 | 10 | 5 | 0 | 25 | 0 |

SUMO cảnh báo **6 dòng Missing yellow phase**, khi chuyển tới pha 1,3,4,5,7,0
(lần lượt nêu tl-index 11,0,3,6,5,8). Đây là thông báo kiểm tra của SUMO,
không phải danh sách đầy đủ mọi movement cần được kiểm chứng.
Ví dụ 0→1 tắt xanh links 10–14 và mở xanh 0–4 mà không có pha vàng riêng.
Vì vậy chưa chốt cả 8 pha làm action hợp lệ cho môi trường mới.

### Ingolstadt

| Phase index | Duration (s) | State | Vai trò trong chu kỳ gốc |
| ---: | ---: | --- | --- |
| 0 | 38 | `GGgGrGGG` | xanh |
| 1 | 3 | `yygyryyy` | chuyển tiếp, vẫn có g ở link 2 |
| 2 | 6 | `GGGrrrrr` | xanh |
| 3 | 3 | `yyyrrrrr` | chuyển tiếp |
| 4 | 37 | `rrrGGGrr` | xanh |
| 5 | 3 | `rrryyyrr` | chuyển tiếp |

Chu kỳ 90 s = 81 s ở các pha xanh + 9 s ở các pha chuyển tiếp.
Không có pha toàn đỏ. Các pha xanh **ứng viên**: 0,2,4; không đưa pha 1,3,5
thành action độc lập. Pha 1 được thiết kế cho đường chuyển 0→2, không mặc nhiên
dùng được cho 0→4: link 2 vẫn `g` trong pha này.

| Link indices | G (s) | g (s) | r (s) | y (s) |
| --- | ---: | ---: | ---: | ---: |
| 0,1 | 44 | 0 | 40 | 6 |
| 2 | 6 | 41 | 40 | 3 |
| 3,5 | 75 | 0 | 9 | 6 |
| 4 | 37 | 0 | 50 | 3 |
| 6,7 | 38 | 0 | 49 | 3 |

### Giới hạn của kiểm chứng hiện tại

- Đã giữ lại toàn bộ `junction/request` (`foes`, `response`, `cont`) trong JSON.
  Chưa xây dựng bảng xung đột theo movement hoặc kiểm chứng tất cả cặp chuyển pha.
- Chỉ số request của junction không được mặc nhiên coi là TLS link index;
  Giai đoạn 2 phải nối qua connection/via/internal lane trước khi đối chiếu.
- `state` đọc trái→phải; bitsets `foes/response` đọc phải→trái, theo
  [định dạng network của SUMO](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#requests).
- Cologne có `s` trong program dù junction XML ghi type `traffic_light`;
  đã quan sát SUMO chấp nhận dữ liệu, chưa xác minh ý đồ tác giả khi tạo chương trình.
- Không có va chạm trong một lượt chạy không chứng minh an toàn cho mọi hành động,
  mọi seed hoặc phương tiện đang nằm trong vùng giao cắt.

## 5. Kết quả chạy thực tế và bằng chứng

Không override seed, begin/end, routing, collision policy hoặc step-length.
Chỉ thêm các tùy chọn xuất log/summary/tripinfo và tùy chọn hiển thị GUI.
Đây là **original-program inspection**, không phải kết quả FT-30 hay đánh giá thuật toán.

Số liệu bảng dưới thuộc **hai lượt headless đầu tiên có ID ghi bên dưới**.
Không thay số liệu lượt này bằng lượt khác khi kiểm tra lại.

| Chỉ số kiểm tra | Cologne | Ingolstadt |
| --- | ---: | ---: |
| Simulation steps | 3600 | 3600 |
| Departed/thực sự vào mạng | 2014 | 1715 |
| Arrived/hoàn thành | 1979 | 1694 |
| Còn trong mạng tại end | 35 | 21 |
| Pending/chưa được chèn vào mạng | 1 | 1 |
| Collision vehicle events qua TraCI | 0 | 0 |
| Teleport starts qua TraCI | 0 | 0 |
| Emergency braking theo SUMO log | 9 | 0 |
| Emergency stops theo SUMO log | 5 | 0 |
| Dòng warning | 20 | 0 |

Đối chiếu bảo toàn số xe:
`N_input = N_arrived + N_active + N_pending`:
2015 = 1979 + 35 + 1; 1716 = 1694 + 21 + 1.
Trong code tương ứng: `survey['demand']['trip_count']`,
`run['arrived']`, `run['active_at_end']`, `run['pending_at_end']`.
`departed = arrived + active` trong hai lượt không có teleport/removal này.
`tripinfo_counts` đối chiếu độc lập cũng cho các số tương ứng.

Đã mở và quan sát trực tiếp **cả hai GUI**: mạng, xe, thanh đèn và thời gian chạy.
Hai lượt GUI đầy đủ ban đầu cho đúng số xe như headless.
GUI mặc định chờ hộp thoại kết thúc; đã đóng thủ công và bổ sung `--quit-on-end`
để các lượt sau tự thoát. Tùy chọn này không thay mô hình giao thông.

Các lượt đầy đủ đầu tiên (mốc UTC trong tên thư mục):

| Run | Thư mục dưới `results/stage1/` |
| --- | --- |
| Cologne headless | `cologne1_sumo_20261114T131432499111Z` |
| Ingolstadt headless | `ingolstadt1_sumo_20261114T131445296396Z` |
| Cologne GUI quan sát trực tiếp | `cologne1_gui_20261114T131454516829Z` |
| Ingolstadt GUI quan sát trực tiếp | `ingolstadt1_gui_20261114T131551063168Z` |

Mỗi thư mục có JSON, hash script, hash dữ liệu trước/sau, Git HEAD/status, options,
log, tripinfo, summary và CSV pha. Các lượt kiểm tra lại có thư mục mới,
không ghi đè lượt cũ. Tệp output được Git ignore; lưu bản sao nếu cần mang sang máy khác.

**Sai khác khi kiểm tra lặp lại:** lượt
`cologne1_sumo_20261114T132158419101Z` (khởi chạy cùng lúc với lượt Ingolstadt)
ghi 1978 arrived, 36 active, 1 pending, 10 emergency braking và 7 emergency stops;
không có collision/teleport. Seed vẫn `23423`, random=false và hash dữ liệu không đổi.
Lượt Cologne chạy riêng sau đó `cologne1_sumo_20261114T132220426916Z` trở lại
1979 arrived, 35 active, 1 pending, 9 braking và 5 stops như lượt đầu.
Chưa xác định nguyên nhân; việc chạy đồng thời chỉ là bối cảnh quan sát,
**chưa chứng minh quan hệ nhân quả**. Giữ toàn bộ log, không loại lượt sai khác.
Chưa thể cam kết tái lập chính xác mọi quỹ đạo/số xe chỉ bằng seed hiện tại.

**Đọc log pha đúng thời điểm:** CSV ghi `observed_time_s` từ TraCI, không phải log
sự kiện do bộ điều khiển đặt pha. Ví dụ Cologne initial time=25200, phase=0,
next_switch=25205; phase=1 được quan sát sau bước tại time=25206, next_switch=25210.
Không kết luận pha đầu dài 6 s bằng cách lấy hiệu hai snapshot. Các next_switch kế
tiếp cách nhau 5 s. Giai đoạn 2 phải log mốc đặt pha/đầu-cuối interval rõ ràng để
kiểm thử đúng 30 s xanh FT, tránh lệch một bước khi đọc trạng thái sau simulationStep.

Giữ cả xe chưa hoàn thành/chưa xuất phát qua hai tùy chọn `tripinfo-output.write-*`.
Không dùng con số trung bình tổng quát trong `sumo.log` như thể chỉ tính xe đến đích:
mẫu số có thể bao gồm xe chưa hoàn thành khi bật write-unfinished. Giai đoạn 1
chưa triển khai bộ metrics so sánh. Xem [TripInfo SUMO](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).

## 6. Mã khảo sát: đầu vào → xử lý → đầu ra

Tệp `scripts/inspect_scenario.py`, không có class hoặc framework:

1. `main()`: nhận scenario và cờ CLI → tạo thư mục kết quả riêng, ghi môi trường,
   Git version/status → gọi khảo sát, ghi `survey.json`.
2. `inspect_xml(scenario)`: `.sumocfg` → đường dẫn network/demand → đọc TLS,
   phase, connections, request bitsets và trip counts → dictionary có đơn vị rõ ràng.
3. `inspect_tls(connection, survey)`: API TraCI → so tuple links và pha với XML →
   danh sách lane gốc/lane không trùng, trạng thái xác minh; khác nhau thì báo lỗi.
4. `run_original(...)`: SUMO/GUI + config gốc → `simulationStep()` từng bước,
   chỉ dùng getters, cộng số xe/sự kiện → phase CSV, log, tripinfo, summary và run counts.
5. `hash_inputs(scenario)`: bytes của 4 tệp gốc → SHA-256 → so trước/sau, khác thì lỗi.

Nguyên lý tính: `duration_seconds = end_seconds - begin_seconds`;
`cycle_seconds = sum(duration)`; `departed/arrived` là tổng số xe khởi hành/đến
trong từng simulation step. Công thức này không phải reward hay cập nhật RL.
Giá trị trung gian quan sát trực tiếp trong JSON, CSV và XML output.
Lệnh tái lập xem README. Chưa tạo `src/`, không có lệnh train/evaluate.

Kiểm chứng cuối: `py_compile` đạt, `pip check` không báo lỗi phụ thuộc,
`git diff --check` đạt. Kiểm tra độc lập toàn bộ 9 lượt đầy đủ xác nhận 3600 bước,
conservation xe, tripinfo counts, seed/step-length và input hashes; kiểm tra các bảng
trong báo cáo khớp 28 connections và 14 pha XML. Kết quả kiểm tra được lưu tại
`results/stage1/verification.json`. Sai khác Cologne đã nêu vẫn chưa được giải thích.
Smoke test `--run --steps 100` sau cùng: time 25200→25300, 59 departed,
19 arrived, 40 active, 0 pending; đây là lượt cắt ngắn, chưa load/chạy toàn bộ demand.

## 7. Kế hoạch chi tiết Giai đoạn 2 — chờ phê duyệt

### 7.1. Các lựa chọn cần chốt trước khi code

| Vấn đề | Lựa chọn | Đề xuất để duyệt |
| --- | --- | --- |
| Action Cologne | Giữ cả 8 pha gốc sau xác minh; hoặc chọn tập con được chứng minh phù hợp | Khảo sát movement/foes trước, ưu tiên giữ ngữ nghĩa G/g/s; chưa tự chọn tập con |
| Action Ingolstadt | 0,2,4 là ứng viên; xác minh conflict/priority trước | ID action liên tiếp → ánh xạ tường minh sang phase index; không suy từ chẵn/lẻ |
| Chuyển pha | Chỉ dùng đường chuyển gốc đã kiểm chứng; hoặc tạo chuyển tiếp riêng cho từng cặp | Ưu tiên bảng chuyển tiếp từng cặp, không sửa XML gốc; xử lý G/g/s rõ ràng |
| Vàng/đỏ | Dùng giá trị gốc ở nơi phù hợp; hoặc thời gian chung sau phân tích clearance | Thử nghiệm ứng viên vàng 3 s, toàn đỏ 1 s; **chưa xác nhận đủ an toàn**, phải tăng nếu kiểm chứng yêu cầu |
| Khoảng quyết định | Cố định theo giây, hoặc giữ xanh cố định + chuyển tiếp làm thời gian thay đổi | Ưu tiên 30 s xanh, vàng/đỏ riêng; elapsed_seconds biến thiên → SMDP |
| Queue quan sát | Chỉ controlled lanes không trùng, hoặc toàn bộ làn vào junction | Ưu tiên controlled lanes cho bản đầu; khai báo phạm vi chính xác, không gọi là queue toàn mạng |
| Reward | Âm queue tại cuối bước, hoặc âm queue tích lũy theo thời gian | Ưu tiên tích lũy, gồm cả vàng/đỏ; công bố đơn vị vehicle·s |
| Episode | Cửa sổ cấu hình cố định; hoặc drain đến hết xe | Giữ begin/end gốc, báo cáo cả active/pending, không tự kéo dài |

30 s ở đây là **đề xuất cấu hình dùng chung** để chuẩn bị FT-30; chưa lập lịch FT.
Nếu ưu tiên MDP bước cố định, cần duyệt thiết kế khác có bộ đếm phase/transition
và action constraints tường minh; không gọi khoảng 30/34 s là bước cố định.

### 7.2. Trình tự triển khai sau khi người dùng duyệt

1. **Chốt bản thiết kế**: action candidates, G/g/s, thời gian vàng/đỏ, minimum green,
   queue scope, reward, discount, terminal và cấu trúc giá trị trả về.
   Chỉ chốt phase nào hợp lệ sau khi kiểm chứng movement, không theo vị trí XML.
   Điều tra thêm sai khác Cologne cùng seed bằng các lượt lặp tuần tự/đồng thời,
   so cấu hình và các tripinfo đầu tiên khác nhau; không tự đổi routing/demand để che sai khác.
2. **Lập bảng an toàn**: connection→via/internal lane→request/foes/response;
   phân biệt request index và TLS index; xét green ưu tiên, permissive green,
   right-turn-stop và xe đang trong junction. Kiểm tra mọi cặp action có thể chọn,
   cả giữ nguyên pha; chỉ tạo chương trình chuyển tiếp trong bộ nhớ hoặc file mới
   ngoài `data/`. Không sửa network/route/sumocfg gốc.
3. **Tạo `src/traffic_signal.py`**: hàm đọc mapping, kiểm tra action, lập/chạy chuỗi
   vàng→toàn đỏ→xanh đã duyệt. Cặp chưa chứng minh thì từ chối. Giữ nguyên pha
   không cần chèn vàng/đỏ. Log target, state string, start/end, lý do chuyển và elapsed.
4. **Tạo `src/simulation.py`**: một lớp nhỏ với `reset(seed)`, `step(action)`,
   `get_state()`, `get_reward()`, `close()`. `reset` đóng lượt trước rồi mở SUMO mới;
   mỗi lượt ghi scenario/config/seed/version. Dùng đúng begin/end và mạng/demand gốc.
   `step` thực hiện trọn transition+green, lấy dữ liệu mỗi simulation step;
   gặp end thì cắt interval và ghi elapsed thực tế. `close` dùng được khi đã đóng.
5. **Tạo `src/observations.py`**: đọc queue lane vào/ra không trùng, phase hiện tại,
   thời gian xanh và transition nếu xuất hiện trong observation. Trả dữ liệu thô;
   chưa rời rạc hóa QL hoặc chuẩn hóa DQN trong Giai đoạn 2.
6. **Tạo `src/metrics.py`**: thu queue từng giây, departed/arrived/pending,
   collisions/teleports; đọc tripinfo để tổng hợp trip metrics với mẫu số rõ ràng.
   Mọi bộ điều khiển về sau dùng cùng cách đo và cùng quy tắc chuyển pha.
7. **Tạo script chọn pha thủ công và kiểm tra nhỏ**: thử đủ cặp chuyển tiếp bằng
   SUMO-GUI, assertions cho state length/action/timing; reset cùng scenario ít nhất
   hai lần; đếm thời gian vàng/đỏ, green đủ thời lượng và interval bị cắt ở end.
   Đối chiếu queue không trùng, reward tích lũy và conservation xe bằng ví dụ tính tay.
8. **Cổng hoàn thành**: hai kịch bản chọn pha thủ công được, không có đường chuyển
   chưa kiểm chứng; log giải thích được mọi mốc thời gian; kiểm tra lifecycle/action/
   transition/reward/metrics đạt. Không bắt đầu FT hoặc huấn luyện trong giai đoạn này.

### 7.3. Công thức đề xuất, chưa triển khai

Với `incoming_lanes_unique`, queue ở thời điểm t:

`queue_total(t) = Σ_i getLastStepHaltingNumber(i)` (xe dừng, ngưỡng SUMO < 0.1 m/s).

`queue_integral = Σ_k queue_total(k) * delta_t_k` (vehicle·s).
`reward = -queue_integral`; tính cả mọi giây vàng/đỏ và đoạn cuối chưa đủ 30 s.
`average_queue = queue_integral / measured_seconds` (xe, tổng trên tập làn đã khai báo,
không phải trung bình chia thêm số làn). Đây là xấp xỉ tích phân theo bước mô phỏng.

Ví dụ minh họa thiết kế, **không phải số đo**: queue=2 xe trong 3 s vàng,
1 xe trong 1 s đỏ, 3 xe trong 30 s xanh → integral=2×3+1×1+3×30=97 vehicle·s,
reward=-97; elapsed_seconds=34. Giữ cùng pha có thể chỉ elapsed=30.

Với khoảng quyết định biến thiên, đề xuất dùng `gamma_ref` cho `reference_seconds=30`:
`discount = gamma_ref ** (elapsed_seconds / reference_seconds)`.
QL/DQN về sau dùng `target = reward + (1-done)*discount*next_q` theo cùng quy ước.
Nếu chọn discounted return đúng theo từng simulation step, reward cũng cần trọng số
discount trong interval; đây là lựa chọn khác phải duyệt, không âm thầm đổi công thức.

Observation thô cần queue theo hướng/approach và current phase; lưu outgoing queues
cho MP. QL về sau có thể dùng 4 nhóm (Cologne) / 3 nhóm (Ingolstadt), 3 mức queue:
tối đa 3^4×8=648 hoặc 3^3×3=81 state nếu chỉ có nhóm queue+phase, chưa tính timer
hay biến khác. Ngưỡng bins phải dựa trên dữ liệu và duyệt ở Giai đoạn 5.
Đây là state tổng hợp, chưa chứng minh tính Markov đầy đủ; không hứa kích thước này
đủ biểu diễn mọi ràng buộc thời gian hoặc tắc nghẽn đầu ra.

Trip metrics sau này: trên tập xe hoàn thành `C`, average waiting/travel/time loss
lần lượt là tổng `waitingTime`/`duration`/`timeLoss` chia `|C|` (s); nếu `C` rỗng
thì báo không có giá trị. Throughput=`|C|` trong cửa sổ đo. Kèm tổng demand,
departed, active, pending và `departDelay` (mẫu số công bố riêng) để thể hiện thiên lệch
của chỉ số chỉ tính xe đến đích. Chưa tính Average Queue Length của hai lượt khảo sát.

## 8. Đã xác minh / chưa xác minh

**Đã xác minh:** môi trường hoạt động; XML đọc được và SUMO chạy hết hai cửa sổ gốc;
TLS IDs, links/via, làn và pha khớp TraCI; số lượng và lịch demand; không có vàng
ở Cologne; GUI hiển thị được xe/đèn; lưu cảnh báo và số xe chưa hoàn thành;
SHA-256 trước/sau các lượt khảo sát giữ nguyên, không thay dữ liệu hay Git staged AGENTS.

**Chưa xác minh:** tính an toàn mọi cặp action; thời gian clearance phù hợp; ý đồ `s`
trong Cologne; tương đương hành vi trên phiên bản SUMO khác; ảnh hưởng nhiều seed;
nguyên nhân sai khác một lượt Cologne cùng seed khi kiểm tra lặp lại;
nguồn gốc upstream commit; chất lượng hoặc độ hội tụ bất kỳ thuật toán nào.

**Bước tiếp theo duy nhất:** người dùng duyệt thiết kế ở mục 7 rồi mới triển khai
Giai đoạn 2. Nếu việc kiểm chứng an toàn dẫn tới thay đổi action/timing/reward,
trình bày thay đổi và xin duyệt lại trước khi triển khai phần phụ thuộc.
