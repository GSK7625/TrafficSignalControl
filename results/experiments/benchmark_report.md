# BÁO CÁO KẾT QUẢ THỰC NGHIỆM SO SÁNH THUẬT TOÁN ĐIỀU KHIỂN TÍN HIỆU ĐÈN

- **Thời gian thực hiện:** 2026-10-12 19:53:58
- **Môi trường:** Python 3.11, SUMO 1.28.0, TraCI
- **Thuật toán so sánh:**
  1. **Original SUMO Program**: Chương trình tín hiệu có sẵn trong RESCO benchmark (baseline chuẩn)
  2. **Fixed-Time Controller**: Bộ điều khiển chu kỳ cố định tự triển khai (chu kỳ 30s)
  3. **Max-Pressure Controller**: Bộ điều khiển áp lực dòng xe (tính theo hàng đợi đầu vào - đầu ra)

---

## 1. Bảng tổng hợp các chỉ số đánh giá

### Kịch bản: `COLOGNE1`

| Chỉ số đánh giá | Original SUMO | Fixed-Time | Max-Pressure |
| :--- | :---: | :---: | :---: |
| Average Waiting Time (s) | 29.21 | 68.45 | 13.32 |
| Average Queue Length (vehs) | 15.60 | 36.20 | 6.43 |
| Throughput (Completed) | 1981 | 1953 | 1998 |
| Average Trip Duration (s) | 77.13 | 106.62 | 56.27 |
| Total Time Loss (s) | 108848.2 | 166733.9 | 67048.9 |
| Unfinished Vehicles (at end) | 33 | 55 | 15 |
| Undeparted Vehicles (pending) | 1 | 7 | 2 |
| Teleports count | 0 | 0 | 1 |
| Collisions count | 0 | 0 | 1 |
| TripInfo Completed | 1981 | 1953 | 1998 |
| TripInfo Mean Travel Time (s) | 77.13 | 106.62 | 56.27 |
| TripInfo Mean Waiting Time (s) | 29.21 | 68.45 | 13.32 |
| Wall-clock Runtime (s) | 15.18 | 24.16 | 17.78 |

### Kịch bản: `INGOLSTADT1`

| Chỉ số đánh giá | Original SUMO | Fixed-Time | Max-Pressure |
| :--- | :---: | :---: | :---: |
| Average Waiting Time (s) | 17.17 | 18.97 | 4.11 |
| Average Queue Length (vehs) | 5.86 | 5.04 | 0.78 |
| Throughput (Completed) | 1694 | 1703 | 1704 |
| Average Trip Duration (s) | 48.50 | 50.62 | 33.75 |
| Total Time Loss (s) | 47178.4 | 50683.5 | 21900.0 |
| Unfinished Vehicles (at end) | 21 | 12 | 11 |
| Undeparted Vehicles (pending) | 1 | 1 | 1 |
| Teleports count | 0 | 0 | 0 |
| Collisions count | 0 | 0 | 0 |
| TripInfo Completed | 1694 | 1703 | 1704 |
| TripInfo Mean Travel Time (s) | 48.50 | 50.62 | 33.75 |
| TripInfo Mean Waiting Time (s) | 17.17 | 18.97 | 4.11 |
| Wall-clock Runtime (s) | 11.01 | 15.36 | 14.45 |

---

## 2. Phân tích kết quả thực nghiệm

### 2.1. Đánh giá trên Cologne1
- **Lưu lượng:** Kịch bản 1 nút giao lớn với 8 pha đèn khả dĩ, lưu lượng tập trung theo các trục chính.
- **Hiệu quả Fixed-Time:** Hoạt động ổn định theo chu kỳ định trước, tuy nhiên do phân bổ thời gian xanh cố định đều cho các pha nên các pha vắng xe vẫn được cấp thời gian xanh.
- **Hiệu quả Max-Pressure:** Tự động phát hiện áp lực từ các nhánh có hàng đợi lớn và ưu tiên cấp quyền ưu tiên cho các pha có áp lực cao nhất; đồng thời cơ chế tie-breaking giữ nguyên pha hiện tại giúp triệt tiêu hiện tượng chuyển pha lãng phí.

### 2.2. Đánh giá trên Ingolstadt1
- **Lưu lượng:** Nút giao 3 pha xanh với các luồng rẽ phức tạp và xe phân bố không đồng đều theo các hướng tiếp cận.
- **So sánh thuật toán:** Max-Pressure thích ứng tức thì với các đợt xe dồn dập, giảm thiểu thời gian chờ tích lũy và độ dài hàng đợi trung bình.

---

## 3. Biểu đồ trực quan hóa

- [Waiting Time Comparison](charts/waiting_time_comparison.png)
- [Queue Length Comparison](charts/queue_length_comparison.png)
- [Throughput Comparison](charts/throughput_comparison.png)
- [Time Loss Comparison](charts/time_loss_comparison.png)
- [Combined Metrics Dashboard](charts/combined_metrics_dashboard.png)
