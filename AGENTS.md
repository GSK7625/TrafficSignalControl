# AGENTS.md — TrafficSignalControl-ML

## 1. Mục tiêu của dự án

Dự án học tập điều khiển tín hiệu đèn giao thông bằng SUMO và Python.

- **Thuật toán trọng tâm:** Q-Learning (bảng Q) và Deep Q-Network (DQN).
- **Thuật toán đối chứng (baseline):** Fixed-Time (FT) và Max-Pressure (MP).
- **Kịch bản chính:** `data/cologne1` và `data/ingolstadt1`, mỗi kịch bản một bộ điều khiển tín hiệu.
- **Kịch bản mở rộng:** `data/cologne3` (ba bộ điều khiển), chỉ thực hiện sau khi hoàn tất hai kịch bản chính và khi được yêu cầu.
- **Ưu tiên:** đúng nguyên lý > dễ hiểu, dễ thuyết trình > ít mã > tối ưu tốc độ.

Đây là **đồ án học tập**, không phải sản phẩm thương mại. Mọi thay đổi phải giúp người mới học có thể đọc mã, theo dõi dữ liệu và giải thích thuật toán bằng cách chỉ trực tiếp vào các hàm/dòng mã.

## 2. Quy trình làm việc bắt buộc của coding agent

**Trước khi sửa bất kỳ tệp nào:**

1. Đọc `AGENTS.md`, `README.md` (nếu có), các tài liệu và mã liên quan.
2. Kiểm tra cây thư mục, trạng thái Git (`git status`), mã hiện có và kết quả trước đó; không đoán nội dung.
3. Nêu ngắn gọn: mục tiêu bước hiện tại, tệp sẽ đọc/sửa/tạo, giả định kỹ thuật, cách kiểm tra và điều kiện hoàn thành.
4. Với thay đổi về state, action, reward, chu kỳ đèn, pha vàng/đỏ, công thức, cách đánh giá hoặc cấu trúc lớn: trình bày các lựa chọn và **xin xác nhận trước khi triển khai**.
5. Chỉ sửa những gì thuộc giai đoạn đã giao. Không tự động làm nhiều giai đoạn cùng lúc.

**Sau khi sửa:**

1. Chạy kiểm tra phù hợp nếu môi trường cho phép; không tuyên bố đã chạy khi chưa chạy.
2. Tóm tắt tệp đã sửa, chức năng mới, lệnh chạy, kết quả thực tế và hạn chế còn lại.
3. Giải thích luồng thuật toán theo thứ tự **đầu vào → xử lý → đầu ra**, gắn với **tệp, hàm và đoạn mã cụ thể**.
4. Ghi rõ công thức toán học, biến số tương ứng trong code và cách quan sát/kiểm tra giá trị trung gian.
5. Đề xuất đúng **một bước tiếp theo**. Không tự chuyển bước nếu người dùng chưa yêu cầu.

## 3. Quy tắc viết code dễ hiểu

- Dùng **Python đơn giản, trực tiếp**, kiểu lập trình hướng thủ tục hoặc lớp nhỏ khi thực sự có ích.
- Tên biến và hàm rõ nghĩa: `queue_length`, `current_phase`, `choose_action`, `update_q_table`; tránh viết tắt khó hiểu.
- Một hàm nên đảm nhận **một nhiệm vụ**. Ưu tiên hàm ngắn, nhưng không chia nhỏ máy móc chỉ để đạt số dòng.
- Viết luồng điều khiển tuần tự, ít tầng gọi hàm/điều kiện lồng nhau. Đừng giấu công thức chính sau lớp trừu tượng.
- Có thể dùng `numpy`, `traci`, `sumolib`, `matplotlib`; chỉ thêm `torch` khi bắt đầu DQN.
- Không tự ý thêm framework hoặc phụ thuộc nặng: Gymnasium/Ray/RLlib/Stable-Baselines3, Hydra, Lightning, database, web app, Docker hoặc CI/CD.
- Không tạo interface, factory, plugin, registry, base class phức tạp, dependency injection, thiết kế nhiều lớp hay hệ cấu hình lớn cho một nhiệm vụ nhỏ.
- Không dùng mẹo Python khó đọc (metaclass, decorator phức tạp, monkey patching, chuỗi lambda, lập trình bất đồng bộ) khi không bắt buộc.
- Ưu tiên lệnh chạy trực tiếp qua CLI, tham số ít và được giải thích trong README.
- Tránh sao chép cùng một đoạn mô phỏng hoặc tính metric ở bốn thuật toán. Dùng chung một hàm rõ ràng.
- Không âm thầm thay đổi kết quả, sửa số liệu bằng tay hoặc che lỗi bằng `try/except: pass`.

### Chú thích và tài liệu

- Dùng chú thích **tiếng Việt dễ hiểu** tại những phần có lý thuyết, phép tính, quyết định pha và điều kiện đặc biệt.
- Mỗi hàm trọng tâm có docstring ngắn: làm gì, tham số, giá trị trả về và đơn vị (giây, xe, v.v.).
- Giải thích **vì sao**, không chú thích lặp lại từng dòng lệnh dễ hiểu.
- Giữ nguyên tên thuật toán, biến toán học và API tiếng Anh để tra cứu tài liệu.
- Không viết hàng trăm dòng giải thích chung chung bên trong code; đặt lý thuyết dài trong `docs/` hoặc README khi cần.

## 4. Phạm vi và bố cục mã đề xuất

Chỉ tạo tệp khi bước tương ứng bắt đầu; **không dựng trọn bộ khung rỗng**.

```text
TrafficSignalControl-ML/
├── AGENTS.md
├── README.md
├── requirements.txt
├── data/
│   ├── SOURCES.md
│   ├── cologne1/
│   ├── ingolstadt1/
│   └── cologne3/
├── src/
│   ├── simulation.py          # Mở/đóng SUMO, chạy từng bước
│   ├── traffic_signal.py      # Pha hợp lệ, chuyển pha an toàn
│   ├── observations.py        # Trạng thái và reward
│   ├── metrics.py             # Thu thập và tổng hợp metrics
│   ├── fixed_time.py          # FT
│   ├── max_pressure.py        # MP
│   ├── q_learning.py          # Q-Learning
│   └── dqn.py                 # Mạng Q và cập nhật DQN
├── scripts/
│   ├── inspect_scenario.py
│   ├── run_baseline.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
├── tests/                    # Một số kiểm tra nhỏ và smoke test
├── models/                   # Checkpoints sinh ra, không commit mặc định
└── results/                  # CSV, đồ thị, log sinh ra
```

Bố cục này là **định hướng**, không phải lý do để tạo thêm abstraction. Điều chỉnh nếu một cách đơn giản hơn đáp ứng được yêu cầu và đã giải thích lý do.

## 5. Trình tự triển khai: không bỏ qua cổng kiểm tra

### Giai đoạn 1 — Khảo sát dữ liệu và môi trường

- Kiểm tra phiên bản Python, SUMO/SUMO-GUI, `sumolib`, `traci` và khả năng khởi chạy SUMO.
- Đọc `*.sumocfg`, `*.net.xml`, `*.rou.xml` của `cologne1` và `ingolstadt1`.
- Ghi nhận: simulation begin/end, TLS IDs, số pha, trạng thái đèn mỗi pha, độ dài xanh/vàng/đỏ, controlled lanes/links, nguồn nhu cầu xe và số lượng xe.
- Chạy mô phỏng nguyên trạng, không điều khiển, để xác minh tệp đầu vào hợp lệ.
- **Lưu ý đã quan sát trong repository:** `cologne1` có chương trình 8 pha, mỗi pha 5 giây, không có pha vàng riêng; `ingolstadt1` có 6 pha gồm 3 pha xanh và 3 pha vàng. Phải xác minh lại từ mã/dữ liệu trước khi lập bản đồ hành động.
- Chưa triển khai thuật toán ở bước này.

**Hoàn thành khi:** có bản tóm tắt kịch bản, cách chạy và log SUMO hoạt động được.

### Giai đoạn 2 — Môi trường SUMO dùng chung

- Viết các thao tác đơn giản: `reset(seed)`, `step(action)`, `get_state()`, `get_reward()`, `close()`; có thể là class nhỏ khi phù hợp.
- Mọi thuật toán sử dụng **cùng mạng, demand, danh sách pha được phép, điều kiện chuyển tiếp và bộ đo metrics**.
- Phân biệt simulation step (ví dụ 1 giây), thời gian giữ pha xanh và thời điểm ra quyết định.
- Hành động chỉ được chọn giữa **pha xanh hợp lệ**, không chọn pha vàng như hành động độc lập.
- Xây dựng chuyển tiếp vàng/đỏ an toàn khi đổi pha; nếu kịch bản gốc không có pha chuyển tiếp, phải tạo/định nghĩa và kiểm chứng cách chuyển, **không tự chuyển xanh → xanh khi chưa xác thực an toàn**.
- Không tự suy diễn pha đối nghịch từ vị trí index trong XML. Xác minh `getControlledLinks`, signal states và xung đột giao thông.
- Ghi nhận cả thời gian chuyển pha trong reward và metrics; không bỏ qua thời gian xe chờ khi đèn vàng/đỏ.
- Nếu thời lượng giữa hai quyết định thay đổi, ghi lại `elapsed_seconds`, giải thích cách tổng hợp reward và discount theo thời gian. Không gọi nhầm bài toán SMDP là MDP bước thời gian cố định.

**Hoàn thành khi:** có thể chọn pha hợp lệ thủ công, quan sát qua SUMO-GUI và kiểm chứng không có chuyển tiếp nguy hiểm.

### Giai đoạn 3 — Fixed-Time (FT)

- Chọn chu kỳ theo thứ tự pha xanh hợp lệ, giữ mỗi pha xanh **30 giây** theo cấu hình thực nghiệm mặc định; thời gian vàng/đỏ tính riêng.
- Không nhầm với chương trình mặc định có sẵn của SUMO; ghi rõ đây là `FT-30` do dự án triển khai.
- Xác nhận thời gian xanh thực tế bằng log thời điểm vào/ra pha; không tính vàng/đỏ vào 30 giây xanh.

**Hoàn thành khi:** chạy FT trên hai kịch bản, có log pha và metrics.

### Giai đoạn 4 — Max-Pressure (MP)

- Định nghĩa rõ áp lực cho mỗi chuyển động từ làn vào `i` đến làn ra `j`: `pressure_ij = q_in_i - q_out_j` (hoặc biến thể có trọng số được công bố).
- Với mỗi pha hợp lệ, tính tổng áp lực các chuyển động được pha phục vụ; chọn pha có áp lực lớn nhất, vẫn tôn trọng giới hạn an toàn và thời gian xanh tối thiểu.
- Dùng đúng kết nối `incoming → outgoing` của SUMO; tránh cộng trùng các làn hoặc các movement.
- Không được dùng riêng tổng số xe chờ trên các làn vào rồi đặt tên là Max-Pressure. Nếu làm bản đơn giản hóa, ghi rõ đó là heuristic, không phải MP tiêu chuẩn.
- Kiểm tra một ví dụ tính áp lực bằng tay và đối chiếu giá trị trong code.

**Hoàn thành khi:** có log từng pressure, lựa chọn pha và metrics trên hai kịch bản.

### Giai đoạn 5 — Tabular Q-Learning

- Mô tả **state** (hàng đợi đã rời rạc hóa, pha hiện tại, thông tin cần thiết), **action** (pha xanh hợp lệ), **reward** (ví dụ âm tổng hàng đợi tích lũy), **episode** và **terminal state**.
- State phải có kích thước Q-table hợp lý; cân nhắc nhóm làn và 2–3 mức hàng đợi, ghi rõ các ngưỡng.
- Triển khai epsilon-greedy rõ ràng và cập nhật:
  `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]`.
- Nếu `done=True`, mục tiêu cập nhật không cộng giá trị Q ở trạng thái tiếp theo.
- Giải thích các giá trị `alpha`, `gamma`, `epsilon`, chính sách giảm epsilon và cách xử lý reward trong suốt khoảng chuyển pha.
- Hỗ trợ lưu/tải Q-table; tách `train` khỏi `evaluate` và **không cập nhật Q-table khi đánh giá**.
- Có ví dụ số học một bước cập nhật Q để người học tự tính lại.

**Hoàn thành khi:** chạy nhiều episode, lưu mô hình, tải lại, đánh giá với epsilon = 0 và vẽ được đường học.

### Giai đoạn 6 — Deep Q-Network (DQN)

- Chỉ thêm `torch` và phụ thuộc cần thiết tại giai đoạn này; ưu tiên chạy CPU được.
- Mô hình: MLP nhỏ, đơn giản; đầu vào là state dạng số, đầu ra là giá trị Q cho từng hành động hợp lệ.
- Phải có và giải thích: policy network, target network, replay buffer, minibatch sampling, epsilon-greedy, loss, optimizer và đồng bộ target network.
- Target TD chuẩn: `y = r + gamma * (1-done) * max_a' Q_target(s', a')`; nếu dùng khoảng quyết định biến thiên, điều chỉnh discount theo quy ước thời gian đã xác định ở giai đoạn 2.
- Không để gradient chạy xuyên qua target; không dùng target từ các hành động bất hợp lệ.
- Ghi rõ chuẩn hóa state, seed, kích thước buffer/batch và cách chọn checkpoint.
- Tách huấn luyện/đánh giá; đánh giá không có exploration, không cập nhật trọng số.
- **Không thêm Double DQN, Dueling DQN, PER, CNN, LSTM** trong phiên bản đầu nếu chưa được yêu cầu.

**Hoàn thành khi:** loss được tính đúng, checkpoint có thể tải lại và mô hình chạy đánh giá độc lập.

### Giai đoạn 7 — Đánh giá công bằng

- Chạy FT, MP, Q-Learning đã huấn luyện và DQN đã huấn luyện với cùng scenario, demand, thời gian mô phỏng, quy tắc chuyển pha và cách tính metrics.
- Metrics chính: **Average Waiting Time (s), Average Travel Time (s), Average Queue Length (vehicles), Average Time Loss (s), Throughput (vehicles)**.
- Ghi rõ phương pháp tính, mẫu số và đơn vị. Dùng `tripinfo` của SUMO cho các chỉ số theo chuyến; dùng dữ liệu làn theo từng simulation step cho queue.
- `Throughput` phải được định nghĩa rõ: số xe hoàn thành chuyến trong khoảng đo (hoặc xe/giờ, khi có chuẩn hóa).
- Báo cáo số xe khởi hành/thực sự vào mạng, hoàn thành, chưa hoàn thành và `departDelay` để tránh thiên lệch của chỉ số chỉ tính trên xe đến đích.
- Không cộng trùng xe trên danh sách controlled lanes. Queue trung bình phải là trung bình theo **thời gian mô phỏng**.
- Huấn luyện và đánh giá dùng tập seed tách biệt; đề xuất thử ít nhất 5 seed đánh giá khi có nguồn ngẫu nhiên thực sự. Không tuyên bố 5 seed là 5 nhu cầu giao thông khác nhau nếu route files bất biến.
- Với QL/DQN: ghi train seed, số episode, budget tương đối, thời điểm chọn checkpoint (validation) và eval seed. Không chọn checkpoint dựa trên tập test.
- Chỉ kết luận thuật toán tốt hơn nếu dữ liệu hỗ trợ; so sánh giá trị trung bình, độ phân tán và các tình huống thất bại. RL thua baseline vẫn là kết quả nghiên cứu hợp lệ.
- Lưu kết quả CSV, config, seed, checkpoint ID và cách tái lập. Không sửa số liệu hoặc loại lượt chạy xấu mà không giải thích.

**Hoàn thành khi:** có CSV/bảng/biểu đồ so sánh được tái lập từ cùng quy trình.

### Giai đoạn 8 — Thuyết trình từ code

- Viết hướng dẫn theo thứ tự: cấu hình SUMO → lấy state → quyết định action → chuyển pha → reward → học → metrics.
- Với mỗi thuật toán, nêu: ý tưởng, công thức, đường dẫn tệp, tên hàm, biến trong code, ví dụ đầu vào/đầu ra và cách kiểm tra.
- Nêu thẳng các giới hạn và điều chưa kiểm chứng; không dựng kết quả giả.

## 6. Quy tắc riêng về khoa học và thuật toán

- **FT:** thời gian xanh cố định, vàng/đỏ riêng; lịch pha phải hợp lệ.
- **MP:** trọng tâm là **chênh lệch áp lực đầu vào - đầu ra** trên chuyển động được pha phục vụ; không đánh tráo thành chọn làn đông nhất.
- **Q-Learning:** cập nhật Q-table thực sự, không dùng reward có sẵn để giả lập kết quả tốt hơn; phân biệt explore/exploit.
- **DQN:** phải học Q bằng mạng neural, có replay và target network; không được dùng mạng ngẫu nhiên rồi gọi là DQN đã huấn luyện.
- State/action/reward và quy tắc chuyển pha phải được mô tả trước khi code; không thay đổi ngầm giữa các lần thử.
- Không ép Q-Learning/DQN thắng baseline, không tinh chỉnh riêng tập test để có biểu đồ đẹp.
- Nếu tài liệu khoa học và giả định triển khai khác nhau, ghi rõ biến thể được chọn và lý do.

## 7. Kiểm thử tối thiểu, ưu tiên tính đúng

Không cần bộ kiểm thử enterprise phức tạp. Tuy nhiên phải có kiểm tra nhỏ cho những phần cốt lõi:

1. Khởi động/đóng SUMO và reset được sau mỗi episode.
2. Tất cả action được sinh đều nằm trong danh sách pha hợp lệ.
3. Chuyển pha an toàn và log thời gian xanh/vàng/đỏ đúng.
4. FT giữ đúng 30 giây xanh mỗi lần phục vụ pha (theo cấu hình mặc định).
5. MP có một trường hợp tính tay và chọn đúng pha.
6. Q-Learning có trường hợp số học tính tay, kể cả `done=True`.
7. DQN có kiểm tra shape đầu vào/đầu ra, gradient, terminal target, save/load.
8. Chạy thử ngắn (`smoke test`) từng thuật toán trên ít nhất một kịch bản.

Có thể viết kiểm tra thủ công đơn giản hoặc `pytest`; không mở rộng thành hệ kiểm thử lớn khi không cần.

## 8. Git, dữ liệu, kết quả và an toàn

- **Không xóa, sửa hoặc ghi đè** `data/**/*.net.xml`, `data/**/*.rou.xml`, `data/**/*.sumocfg`, `data/**/LICENSE` nếu chưa được người dùng chấp thuận rõ ràng.
- Tôn trọng nguồn và giấy phép được ghi trong `data/SOURCES.md`; không tự gỡ thông tin bản quyền.
- Không tự tạo lưu lượng xe mới, chỉnh route, seed hay thời gian mô phỏng với lý do làm kết quả đẹp hơn.
- Không tự commit, push, reset, force push, rebase, xóa nhánh hoặc xóa file quan trọng khi chưa được giao.
- Thêm `models/`, `results/`, log, cache phù hợp vào `.gitignore` khi các thư mục được tạo; không commit checkpoint lớn và file trung gian mặc định.
- Nếu phát hiện bug cũ hoặc dữ liệu bất thường ngoài phạm vi công việc, báo cáo trước, không sửa lan sang các phần khác.
- Mọi kết quả đo phải truy được về scenario, algorithm, config, seed, version và checkpoint.

## 9. Mẫu báo cáo bắt buộc sau mỗi nhiệm vụ

Trả lời người dùng bằng tiếng Việt, ngắn gọn nhưng có thể dùng để thuyết trình:

```text
1. Mục tiêu bước vừa làm
2. Các tệp/hàm đã sửa và vai trò của chúng
3. Luồng hoạt động: đầu vào → xử lý → đầu ra
4. Công thức hoặc nguyên lý liên quan; biến nào trong code tương ứng
5. Cách chạy và cách kiểm chứng; kết quả chạy thực tế
6. Điều chưa hoàn thành / giả định / hạn chế
7. Bước tiếp theo được đề xuất (chưa tự triển khai)
```

**Tiêu chuẩn cuối:** người học mở được một tệp code, chỉ vào từng hàm trọng tâm, giải thích đúng tại sao nó tồn tại, công thức được triển khai ở đâu và dữ liệu chạy qua nó như thế nào.
