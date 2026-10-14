# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG

**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  
**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  
**Điều kiện mô phỏng & an toàn chuẩn:** Chu kỳ 3600s, vàng chuyển tiếp = 3.0s, đỏ toàn bộ = 1.0s, xanh tối thiểu = 5.0s, xanh tối đa = 60.0s, dữ liệu hành trình TripInfo chuẩn.

---

## Kịch bản mạng lưới: `COLOGNE1`

| Thuật toán | Waiting Time (s) | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport | Trạng thái |
|---|---|---|---|---|---|---|---|
| **Fixed-Time** | 77.75 ± 0.32 | 96.49 ± 1.98 | 40.39 ± 0.17 | 1968 | 97.7% | 0 / 0 | ⚠️ LOW_COMPLETION |
| **Max-Pressure** | 21.16 ± 0.88 | 28.33 ± 1.28 | 10.49 ± 0.49 | 1996 | 99.1% | 0 / 0 | ✅ SAFE |
| **Q-Learning** | 31.00 ± 1.62 | 38.39 ± 3.51 | 15.33 ± 0.47 | 1975 | 98.0% | ⚠️ 1 / 1 (max 1) | 🚨 INCIDENTS |

> [!WARNING]
> **Cảnh báo quan sát:**
> - **Fixed-Time**: Tỷ lệ hoàn thành thấp (97.7%). Có hiện tượng nghẽn mạng ngoài cổng (System Delay 96.49s).
> - **Q-Learning**: Ghi nhận tổng cộng 1 va chạm và 1 teleport qua các seed (lớn nhất 1 va chạm/lượt). Cần kiểm tra dọn nút giao và xung đột luồng.

## Kịch bản mạng lưới: `INGOLSTADT1`

| Thuật toán | Waiting Time (s) | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport | Trạng thái |
|---|---|---|---|---|---|---|---|
| **Fixed-Time** | 18.76 ± 1.12 | 20.68 ± 1.54 | 4.88 ± 0.10 | 1699 | 99.0% | 0 / 0 | ✅ SAFE |
| **Max-Pressure** | 5.15 ± 0.44 | 6.43 ± 0.73 | 1.17 ± 0.05 | 1701 | 99.2% | 0 / 0 | ✅ SAFE |
| **Q-Learning** | 10.14 ± 2.08 | 12.32 ± 3.08 | 2.28 ± 0.36 | 1696 | 98.9% | 0 / 0 | ✅ SAFE |
