# `tests/` — Bộ Kiểm thử Tự động (Automated Test Suite)

Thư mục [`tests/`](file:///tests) chứa bộ kiểm thử tự động toàn diện bao quát 100% các thành phần logic của nền tảng, đảm bảo tính toàn vẹn, độ tin cậy và an toàn chuyển pha khi mở rộng hệ thống.

---

## 1. Tổng quan & Kết quả kiểm thử

Bộ kiểm thử hiện bao gồm **80 unit tests**, thực thi trong thời gian chưa đầy **1 giây** nhờ cơ chế mocking tinh gọn:

```powershell
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\dotru\Documents\TrafficSignalControl
collected 80 items

tests\test_compare_algorithms.py ..                                      [  2%]
tests\test_config_and_inspector.py ............                          [ 17%]
tests\test_controllers.py .................                              [ 38%]
tests\test_evaluation_runner.py ..                                       [ 41%]
tests\test_metrics.py .......                                            [ 50%]
tests\test_phase_manager.py ...................                          [ 73%]
tests\test_rl_components.py ..........                                   [ 86%]
tests\test_runner.py ...........                                         [100%]

============================= 80 passed in 0.54s ==============================
```

---

## 2. Chi tiết phân loại các nhóm kiểm thử (80 Tests)

| Tệp kiểm thử | Số tests | Mục tiêu & Phạm vi kiểm thử |
| :--- | :---: | :--- |
| [`test_phase_manager.py`](file:///tests/test_phase_manager.py) | **19** | Quy tắc sinh pha vàng dọn nút an toàn (`build_yellow_state`), khóa thời gian xanh tối thiểu (`min_green`), cưỡng bức chuyển pha khi chạm ngưỡng `max_green`, chu kỳ vàng chuyển tiếp (`yellow_duration`) và toàn đỏ (`all_red_duration`), đồng bộ pha ngoại vi. |
| [`test_controllers.py`](file:///tests/test_controllers.py) | **17** | Giải thuật Fixed-Time (chu kỳ tuần hoàn, tương thích min/max green, đồng bộ khi cưỡng bức đổi pha), giải thuật Max-Pressure (tránh đếm trùng làn, tie-breaking giữ pha hiện tại, tín hiệu cho phép rẽ `s`), và Tabular Q-Learning (chọn hành động $\varepsilon$-greedy, cập nhật Bellman, phân rã $\varepsilon$, nạp/xuất JSON Q-table). |
| [`test_config_and_inspector.py`](file:///tests/test_config_and_inspector.py) | **12** | Xác thực schema cấu hình mô phỏng `SimulationConfig`, nạp và bắt lỗi tệp YAML cấu hình, kiểm tra bộ phân tích cú pháp dữ liệu mạng `data_inspector` trên các kịch bản chuẩn. |
| [`test_runner.py`](file:///tests/test_runner.py) | **11** | Hàm lọc pha xanh hợp lệ `extract_green_phases`, khởi tạo và điều phối `SimulationRunner`, tự động chọn backend `libsumo` khi chạy headless, bắt buộc cổng `traci` khi chạy `--gui` hoặc truyền cờ `--no-libsumo`, kiểm tra cờ `save_run_details`. |
| [`test_rl_components.py`](file:///tests/test_rl_components.py) | **10** | Rời rạc hóa không gian trạng thái `StateDiscretizer`, hàm phần thưởng `NegativePressureReward` đo áp lực vào/ra và phạt dồn ứ hạ lưu, quy trình đánh giá RL, cơ chế lọc checkpoint đạt cổng khả thi (Viability Gate), tuân thủ ràng buộc `max_green`. |
| [`test_metrics.py`](file:///tests/test_metrics.py) | **7** | Trích xuất tệp XML `--tripinfo-output`, xử lý an toàn tệp rỗng hoặc thiếu, thu thập chỉ số từng bước với mock TraCI, hàm trích xuất `get_core_metrics`, kiểm tra an toàn thư mục xuất dữ liệu. |
| [`test_compare_algorithms.py`](file:///tests/test_compare_algorithms.py) | **2** | Hàm tổng hợp báo cáo đối chuẩn đa thuật toán, xử lý an toàn khi gặp các lượt chạy bị hủy hoặc thất bại (`FAILED`) mà không làm ngắt quãng luồng xử lý. |
| [`test_evaluation_runner.py`](file:///tests/test_evaluation_runner.py) | **2** | Hàm phân tích phiên bản SUMO đang cài đặt và hàm tính toán tỷ lệ cải thiện hiệu năng (`compute_improvement`) so với baseline. |

---

## 3. Hướng dẫn thực thi kiểm thử

```powershell
# 1. Chạy nhanh toàn bộ bộ kiểm thử
pytest -q

# 2. Chạy hiển thị chi tiết tên từng test
pytest -v

# 3. Chạy kiểm thử cho một tệp cụ thể
pytest tests/test_phase_manager.py -v
pytest tests/test_runner.py -v

# 4. Chạy kiểm thử với báo cáo dừng ngay khi gặp lỗi đầu tiên (-x)
pytest -x
```

---

## 4. Chiến lược Giả lập (Mocking Strategy)

Bộ kiểm thử tận dụng thư viện `unittest.mock` để giả lập các lời gọi TraCI và libsumo. Nhờ vậy:
- Không yêu cầu tiến trình SUMO hay socket mạng thật khi chạy unit test.
- Kiểm thử được toàn bộ các trường hợp biên nguy hiểm (ví dụ: mất kết nối, lỗi tệp, hành vi cưỡng bức khi vượt `max_green`).
- Tốc độ thực thi siêu tốc (< 1s), phù hợp tích hợp vào quy trình CI/CD.
