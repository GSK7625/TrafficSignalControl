# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG

**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  
**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  
**Điều kiện mô phỏng:** Chu kỳ 3600s, pha vàng chuyển tiếp = 3.0s, đèn xanh tối thiểu (min_green) = 5.0s, dữ liệu hành trình TripInfo chuẩn.

---

## Kịch bản mạng lưới: `COLOGNE1`

| Thuật toán | Waiting Time (s) | Queue Length (xe) | Throughput (xe) | Completion Rate (%) | Time Loss (s) |
|---|---|---|---|---|---|
| **Fixed-Time** | 68.45 | 36.20 | 1953 | 96.9% | 163431.0 |
| **Max-Pressure** | 13.32 | 6.43 | 1998 | 99.2% | 66920.2 |
| **Q-Learning** | 26.40 | 13.61 | 1991 | 98.8% | 93767.5 |
| **DQN** | 11.41 | 5.72 | 1997 | 99.1% | 56928.9 |
