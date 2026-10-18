# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG

**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  
**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  
**Điều kiện mô phỏng & an toàn chuẩn:** Chu kỳ 3600s, vàng chuyển tiếp = 3.0s, đỏ toàn bộ = 1.0s, xanh tối thiểu = 5.0s, xanh tối đa = 60.0s, dữ liệu hành trình TripInfo chuẩn.

---

## Kịch bản mạng lưới: `COLOGNE1`

| Thuật toán | Completed Wait Time (s)* | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport | Trạng thái |
|---|---|---|---|---|---|---|---|
| **Fixed-Time** | 77.95 ± 0.45 | 98.12 ± 2.49 | 40.48 ± 0.22 | 1969 | 97.8% | 0 / 0 | ⚠️ LOW_COMPLETION |
| **Max-Pressure** | 21.40 ± 0.46 | 29.03 ± 0.37 | 10.64 ± 0.23 | 1997 | 99.1% | 0 / 0 | ✅ SAFE |
| **Q-Learning** | 16.59 ± 0.42 | 20.05 ± 0.50 | 8.22 ± 0.14 | 1995 | 99.0% | 0 / 0 | ✅ SAFE |

> [!NOTE]
> [*] **Lưu ý về Survivorship Bias:** Completed Wait Time chỉ tính trên các xe đã về đích. Khi Completion (%) < 98%, chỉ số này có thể thấp phiến diện do bỏ qua các xe đang kẹt trong mạng; System Delay và Throughput phản ánh khách quan toàn bộ nhu cầu giao thông.

> [!WARNING]
> **Cảnh báo quan sát:**
> - **Fixed-Time**: Tỷ lệ hoàn thành thấp (97.8%). Có hiện tượng nghẽn mạng ngoài cổng (System Delay 98.12s).

## Kịch bản mạng lưới: `INGOLSTADT1`

| Thuật toán | Completed Wait Time (s)* | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport | Trạng thái |
|---|---|---|---|---|---|---|---|
| **Fixed-Time** | 18.76 ± 1.12 | 20.68 ± 1.54 | 4.88 ± 0.10 | 1699 | 99.0% | 0 / 0 | ✅ SAFE |
| **Max-Pressure** | 5.15 ± 0.44 | 6.43 ± 0.73 | 1.17 ± 0.05 | 1701 | 99.2% | 0 / 0 | ✅ SAFE |
| **Q-Learning** | 7.81 ± 0.86 | 9.49 ± 0.82 | 1.89 ± 0.15 | 1702 | 99.2% | 0 / 0 | ✅ SAFE |

> [!NOTE]
> [*] **Lưu ý về Survivorship Bias:** Completed Wait Time chỉ tính trên các xe đã về đích. Khi Completion (%) < 98%, chỉ số này có thể thấp phiến diện do bỏ qua các xe đang kẹt trong mạng; System Delay và Throughput phản ánh khách quan toàn bộ nhu cầu giao thông.
