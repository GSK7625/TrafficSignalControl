# BÁO CÁO SO SÁNH HIỆU NĂNG ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG

- **Kịch bản:** `cologne1`
- **Số lượng hạt giống (Seeds):** `5` hạt giống ([42, 101, 2024, 777, 999])
- **Thời gian xanh tối thiểu (min_green):** `20.0s`
- **Mô hình Q-Learning:** `models/q_learning/q_learning_cologne1_best.json`
- **Thời gian mô phỏng:** `3600s` mỗi lượt

## Bảng tổng hợp các chỉ số hiệu năng (Mean ± Std)

| Thuật toán | Thời gian chờ TB (s) | Thời gian di chuyển TB (s) | Độ dài hàng đợi TB (xe) | Tổn thất thời gian TB (s) | Thông lượng Throughput (xe) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | 76.81 ± 0.67 | 116.17 ± 0.94 | 39.85 ± 0.35 | 93.23 ± 0.86 | 1969.4 ± 0.9 |
| **Max-Pressure** | 55.87 ± 1.86 | 96.67 ± 2.66 | 29.91 ± 1.03 | 73.77 ± 2.74 | 1978.6 ± 0.5 |
| **Q-Learning** | 53.02 ± 1.51 | 93.03 ± 2.08 | 28.43 ± 0.80 | 70.16 ± 2.10 | 1987.8 ± 1.9 |
