# BÁO CÁO SO SÁNH 4 THUẬT TOÁN ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG

**Dự án:** TrafficSignalControl  
**Giai đoạn 5:** Q-Learning và Deep Q-Network (DQN) so sánh với Fixed-Time và Max-Pressure.  
**Cấu hình đánh giá:** 3600 giây mô phỏng SUMO, Seed 42, Yellow clearance = 3.0s, Min green = 5.0s, Tripinfo ground-truth.

---

## Kịch bản: `COLOGNE1`

| Thuật toán | Waiting Time (s) | Queue Length (xe) | Throughput (xe) | Travel Time (s) | Time Loss (s) | Teleports | Reward |
|---|---|---|---|---|---|---|---|
| **Fixed-Time** | 68.45 | 36.20 | 1953 | 106.62 | 166733.9 | 0 | - |
| **Max-Pressure** | 13.32 | 6.43 | 1998 | 56.27 | 67048.9 | 1 | - |
| **Q-Learning** | 25.29 | 12.55 | 1989 | 70.40 | 95210.8 | 0 | -9033.4 |
| **DQN** | 11.21 | 5.64 | 1997 | 50.98 | 56446.6 | 0 | -4059.0 |
| **Original SUMO** | 29.21 | 15.60 | 1981 | 77.13 | 108848.2 | 0 | - |

## Kịch bản: `INGOLSTADT1`

| Thuật toán | Waiting Time (s) | Queue Length (xe) | Throughput (xe) | Travel Time (s) | Time Loss (s) | Teleports | Reward |
|---|---|---|---|---|---|---|---|
| **Fixed-Time** | 18.97 | 5.04 | 1703 | 50.62 | 50683.5 | 0 | - |
| **Max-Pressure** | 4.11 | 0.78 | 1704 | 33.75 | 21900.0 | 0 | - |
| **Q-Learning** | 6.07 | 1.74 | 1704 | 35.71 | 25208.7 | 0 | -1253.4 |
| **DQN** | 0.67 | 1.93 | 1118 | 22.98 | 138921.7 | 0 | -1386.8 |
| **Original SUMO** | 17.17 | 5.86 | 1694 | 48.50 | 47178.4 | 0 | - |
