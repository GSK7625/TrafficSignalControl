# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG

**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  
**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  

---

## Kịch bản mạng lưới: `COLOGNE1`


| Thuật toán | Completed Wait Time (s) | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport |
|---|---|---|---|---|---|---|
| **Fixed-Time** | 41.78 ± 0.94 | 48.47 ± 1.50 | 21.65 ± 0.42 | 1983 | 98.4% | 0 / 0 |
| **Max-Pressure** | 21.61 ± 0.46 | 29.53 ± 0.89 | 10.77 ± 0.30 | 1994 | 99.0% | ⚠️ 1 / 1 (max 1.0) |
| **Q-Learning** | 12.10 ± 0.38 | 14.29 ± 0.51 | 6.09 ± 0.13 | 1999 | 99.2% | ⚠️ 1 / 1 (max 1.0) |

## Kịch bản mạng lưới: `INGOLSTADT1`


| Thuật toán | Completed Wait Time (s) | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport |
|---|---|---|---|---|---|---|
| **Fixed-Time** | 18.82 ± 1.15 | 20.27 ± 1.22 | 4.92 ± 0.04 | 1698 | 99.0% | 0 / 0 |
| **Max-Pressure** | 5.25 ± 0.18 | 6.66 ± 0.32 | 1.17 ± 0.15 | 1701 | 99.2% | 0 / 0 |
| **Q-Learning** | 5.72 ± 0.81 | 7.17 ± 0.90 | 1.25 ± 0.15 | 1701 | 99.1% | 0 / 0 |
