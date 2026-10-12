# Đánh giá mã nguồn và phương pháp nghiên cứu — 12/10/2026

## Kết luận

Dự án có nền tảng mô phỏng và các thành phần RL hợp lý, nhưng chưa đủ bằng chứng để tuyên bố triển khai đúng nguyên bản RESCO hoặc đạt chuẩn đánh giá khoa học toàn diện. Không có một chuẩn duy nhất áp dụng cho tất cả bài báo: cần xác định bài báo mục tiêu, liệt kê giả định và công khai mọi khác biệt. Đây là đánh giá mã nguồn hiện tại, dữ liệu benchmark đã lưu và tài liệu nghiên cứu; không phải tái huấn luyện và tái chạy toàn bộ benchmark.

## Phát hiện chính

1. **P1 — Đánh giá thiếu xe chưa được phục vụ.** `src/metrics.py:300–309` tính mean_waiting_time trên xe hoàn thành. Tripinfo mặc định cũng chỉ ghi xe đến đích; runner chưa bật write-unfinished. Trong benchmark_runs.json, DQN Ingolstadt1 hoàn thành 1118/1716 xe (seed 42 và 2024), 1138/1716 (101), còn 535–553 xe pending insertion. Waiting 0.65–0.80 giây không đại diện toàn bộ nhu cầu. Reward chỉ phạt hàng đợi trên làn vào nên không trực tiếp phạt xe bị chặn ngoài mạng. Đây là một cơ chế có thể khuyến khích chính sách kém phục vụ; chưa chứng minh là nguyên nhân duy nhất. Cần báo cáo departDelay, pending demand, backlog và delay toàn hệ thống, cùng waiting của xe hoàn thành. Nếu bật write-unfinished, parser phải tách arrival âm khỏi số xe hoàn thành.

2. **P1 — Nhãn SUCCESS không phản ánh chất lượng hoặc an toàn.** `validate_run_metrics` chỉ kiểm tra số âm, NaN/Inf, zero completion và completion vượt 105%; không phân loại backlog cao, collision hoặc teleport. Kiểm tra trực tiếp cả ba bản ghi DQN Ingolstadt1 đều trả `(True, [])`. Bản ghi Cologne1 Max-Pressure seed 42 có 4 collision và 4 teleport; DQN seed 101 có 1 collision và 1 teleport, đều SUCCESS. Nên tách chạy thành công, dữ liệu hợp lệ, an toàn và chất lượng chính sách. Không đồng nhất teleport với collision hoặc suy nguyên nhân va chạm từ tổng đếm.

3. **P1 — Resume có thể dùng lại kết quả cũ sau khi thay đổi mô hình/mã.** `src/evaluation/experiment_runner.py:456–460` nhận diện lượt đã chạy chỉ bằng scenario, algorithm, simulation seed. Thay checkpoint hoặc sửa logic mà giữ ba giá trị này sẽ bỏ qua chạy lại. Cần định danh bằng cấu hình, hash checkpoint, mã nguồn, phiên bản SUMO và dữ liệu. `sumo_version` hiện bị ghi cứng thành 1.28.0, chưa phải bằng chứng về phiên bản thực tế.

4. **P2 — Max-Pressure là biến thể giản lược.** `compute_phase_pressure` phân bổ queue đầu vào theo tỷ lệ số làn đầu ra được phục vụ, rồi trừ queue của mỗi làn đầu ra duy nhất. Đây không phải công thức movement pressure tổng quát có turning ratios và saturation flows trong các nghiên cứu Max-Pressure. Ví dụ làn vào có hai hướng nhưng 90% xe đi một hướng: tỷ lệ 1/2 theo kết nối không thể hiện nhu cầu 90%. Cần đặt tên biến thể, nêu giả định và đối chiếu công thức với bài được chọn; không áp dụng trực tiếp bảo đảm ổn định của lý thuyết gốc.

5. **P2 — Tần suất quyết định giữa các thuật toán khác nhau.** Baseline được runner hỏi mỗi bước 1 giây, RL chọn mỗi 5 giây. Đây là khác biệt giao thức cần công khai hoặc chuẩn hóa khi so sánh thuật toán. RESCO 2021 mặc định quyết định mỗi 10 giây, gồm 3 giây vàng khi cần. Dùng cùng dữ liệu RESCO chưa đồng nghĩa tái lập giao thức RESCO.

6. **P2 — Chọn checkpoint chưa đánh giá đúng chính sách triển khai.** Cả hai training pipeline lưu best khi reward của episode huấn luyện cao nhất, với epsilon khác 0 và seed thay đổi theo episode. Không có validation riêng với epsilon=0 để chọn checkpoint. DQN Ingolstadt1 cho thấy chính sách greedy có thể khác mạnh kết quả lúc còn exploration. Cần chọn trên validation riêng, đóng băng trước test và đánh giá thêm nhiều training seeds. Ba simulation seeds hiện tại chỉ đo biến thiên mô phỏng của một mô hình, không đo biến thiên quá trình học.

7. **P2 — Xử lý kết thúc RL cần làm rõ.** Hai pipeline đặt done=terminated or truncated; một cắt ngắn vì giới hạn tính toán sẽ bị bỏ bootstrap như terminal thật. Nếu nghiên cứu bài toán tiếp diễn, cần bootstrap khi chỉ truncated. Nếu nghiên cứu finite horizon, cần định nghĩa terminal và cân nhắc đưa thời gian còn lại vào observation. Hiện environment đặt end_time là terminated nhưng max_steps là truncated, rồi learner gộp cả hai.

8. **P2 — State Q-Learning làm mất danh tính hướng chờ.** `StateDiscretizer` chỉ giữ tổng queue đang phục vụ và tổng queue đang chờ. Hai tình huống ùn tắc ở các hướng khác nhau có thể có cùng state nhưng cần chọn pha khác nhau. Không thể mặc nhiên coi biểu diễn này có tính Markov hoặc đảm bảo hội tụ tối ưu chỉ vì state nhỏ. Nên so sánh với queue/pressure theo pha hoặc hướng bằng thí nghiệm loại bỏ/thay thế thành phần.

9. **P2 — Baseline và tuyên bố thống kê cần tiết chế.** Fixed-Time hiện mặc định 30 giây cho mọi pha, chưa được tối ưu theo nhu cầu. Có thêm Original SUMO là hữu ích, nhưng phần trăm cải thiện so với lịch đều không chứng minh vượt baseline tối ưu. Chỉ 15 episodes và một training seed chưa chứng minh hội tụ. Q-Learning Ingolstadt1 waiting biến thiên 6.07, 37.02, 7.01 giây; Cologne1 25.29, 26.74, 53.94. Báo cáo phải thể hiện độ bất định, không chỉ chọn seed đẹp. np.std đang dùng ddof=0; cần ghi rõ định nghĩa, và bổ sung khoảng tin cậy nếu muốn suy luận thống kê.

## Những thành phần có cơ sở hợp lý

- Q-Learning có cập nhật Bellman, epsilon-greedy và masking hành động kế tiếp.
- DQN có online/target networks, replay, Huber loss, target sync và gradient clipping. Đây là DQN cơ bản điều chỉnh cho bài toán giao thông; không phải tự động là PressLight, MPLight hoặc bản IDQN của RESCO.
- Phase manager quản lý min-green và clearance; runner áp dụng trạng thái trước bước mô phỏng. Chưa đủ để chứng nhận an toàn giao thông cho mọi chuyển pha; collision đã lưu cần điều tra riêng theo thời gian, vị trí và phương tiện.
- Có dữ liệu gốc, checkpoint, log và benchmark đa simulation seed. Nguồn dữ liệu ghi rõ commit trong data/SOURCES.md; đánh giá này chưa xác minh hash từng tệp với upstream.

## Thứ tự xử lý đề xuất

1. Sửa phép đo/diễn giải delay và nhãn chất lượng, xác minh collision, đồng bộ báo cáo với bản ghi benchmark.
2. Bổ sung định danh kết quả để tránh resume dữ liệu cũ.
3. Chốt bài báo mục tiêu và đặc tả state/action/reward, pressure, chu kỳ quyết định, baseline và terminal.
4. Validation greedy riêng, nhiều training seeds, test demand khác và khoảng tin cậy; huấn luyện lại khi thay state/reward/giao thức.
5. Chỉ cập nhật kết luận sau khi có benchmark mới. Không lấy số bài kiểm thử đạt làm bằng chứng phương pháp nghiên cứu đúng.

## Nguồn đối chiếu

- Ault & Sharon (2021), Reinforcement Learning Benchmarks for Traffic Signal Control: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/f0935e4cd5920aa6c7c996a5ee53a70f-Paper-round1.pdf
- Mã và mô tả RESCO: https://github.com/Pi-Star-Lab/RESCO
- Variants of Max Pressure Control for Signalized Intersections: https://connected-corridors.berkeley.edu/sites/default/files/Variants%20of%20Max%20Pressure%20Control%20for%20Signalized%20Intersections.pdf
- SUMO TripInfo, đặc biệt output cho xe chưa đến đích: https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html
