# Hướng Dẫn Dành Cho Agent (AGENT.md)

## Mục Tiêu
Viết mã nguồn đơn giản nhất có thể nhưng vẫn đúng yêu cầu, đúng lý thuyết và chạy được. Người mới học phải có khả năng đọc, lần theo luồng xử lý và giải thích tại sao code hoạt động.

---

## 1. Trước Khi Viết Code
1. **Đọc yêu cầu và phần mã nguồn liên quan**: Không tự ý bổ sung tính năng nằm ngoài phạm vi.
2. **Tóm tắt ngắn gọn**: Xác định rõ đầu vào, đầu ra, các bước xử lý và các điều kiện quan trọng.
3. **Đối chiếu lý thuyết nếu là thuật toán**: Xác định công thức / định nghĩa gốc và đối chiếu từng bước triển khai với lý thuyết. Không đánh đổi tính đúng đắn để rút ngắn code.
4. **Tôn trọng cấu trúc hiện tại**: Nếu sửa một dự án có sẵn, giữ nguyên cấu trúc và các quy ước bắt buộc; chỉ thay đổi phần thực sự cần thiết.

---

## 2. Quy Tắc Viết Code
- **Trực tiếp & dễ hiểu**: Chọn giải pháp trực tiếp, dễ hiểu trước khi nghĩ đến giải pháp tổng quát hoặc tối ưu hóa.
- **Hàm ngắn & biến rõ nghĩa**: Ưu tiên hàm ngắn, tên biến rõ nghĩa, luồng xử lý tuần tự và ít tầng gọi hàm lồng nhau.
- **Hạn chế trừu tượng hóa không cần thiết**: Chỉ tạo class, interface, abstraction hoặc design pattern khi thực sự cần, hoặc khi kiến trúc hiện tại bắt buộc.
- **Tối giản phụ thuộc**: Tránh thêm dependency, framework, configuration, helper, generic, decorator hoặc cơ chế bất đồng bộ không cần thiết.
- **Không tối ưu hóa sớm**: Không refactor diện rộng, tối ưu hóa sớm hoặc viết mã để phòng ngừa các tình huống giả định không thuộc yêu cầu.
- **Bình luận đúng chỗ**: Bình luận ngắn tại công thức, giả định hoặc quyết định khó hiểu; không bình luận lại những dòng code đã hiển nhiên.
- **Xử lý an toàn thiết yếu**: Giữ các xử lý lỗi, kiểm tra đầu vào và yêu cầu an toàn thực sự cần thiết; không lược bỏ chỉ vì muốn code ngắn.

---

## 3. Độ Chính Xác Về Lý Thuyết
- **Đặc tả thuật toán rõ ràng**: Với thuật toán, phải nêu rõ trạng thái (state) / đầu vào, hành động (action), cách cập nhật (update rule), điều kiện dừng (stopping condition) và các tham số liên quan.
- **Ánh xạ công thức sang code**: Giải thích công thức bằng ký hiệu toán học rồi chỉ ra chính xác vị trí code thực hiện công thức đó.
- **Trung thực về bản chất**: Không gọi một heuristic là thuật toán học máy, không tuyên bố hội tụ / tối ưu nếu chưa có cơ sở hoặc bằng chứng toán học / thực nghiệm.
- **Ưu tiên tính sáng tỏ**: Khi có nhiều cách triển khai hợp lệ, ưu tiên cách dễ chứng minh và dễ trình bày.

---

## 4. Kiểm Tra Vừa Đủ
- **Kiểm tra đường đi chính**: Luôn chạy chương trình hoặc kiểm tra luồng chính (happy path) có thể thực thi được.
- **Bao phủ ca thường và ca biên**: Thực hiện ít nhất một trường hợp thông thường và một trường hợp biên quan trọng khi phù hợp.
- **Đối chiếu tính tay**: Với công thức / thuật toán, đối chiếu ít nhất một ví dụ nhỏ tính tay với kết quả chương trình khi khả thi.
- **Tối giản hạ tầng test**: Tận dụng kiểm thử liên quan đã có. Không tự tạo bộ kiểm thử đồ sộ, mock phức tạp hoặc hạ tầng kiểm thử mới nếu không cần thiết.
- **Trung thực về kết quả test**: Nếu chưa chạy được hoặc chưa test xong, nêu đúng điều chưa kiểm tra; không tuyên bố code đã đúng hoặc test đã qua khi chưa xác minh.

---

## 5. Cách Trình Bày Kết Quả
Mỗi khi hoàn thành công việc, trình bày phản hồi theo 5 phần:
1. **Đã làm**: Những file / hàm thay đổi, ngắn gọn.
2. **Luồng chạy**: Giải thích theo thứ tự: Đầu vào $\rightarrow$ Xử lý $\rightarrow$ Đầu ra.
3. **Đối chiếu lý thuyết**: Công thức / nguyên lý tương ứng với đoạn code đã viết.
4. **Kiểm tra**: Đã chạy những gì, kết quả ra sao, những giới hạn còn lại.
5. **Giữ đơn giản**: Nếu có lựa chọn phức tạp hơn, giải thích tại sao không triển khai (khi người dùng chưa yêu cầu).

---

## 6. Nguyên Tắc Ưu Tiên
```
Đúng yêu cầu > Đúng bản chất và lý thuyết > Dễ đọc, dễ giải thích > Độ ngắn > Tối ưu hiệu năng không bắt buộc
```

- Khi các nguyên tắc xung đột, hỏi lại người dùng nếu cần quyết định quan trọng.
- Với các quyết định còn lại, luôn chọn phương án ít phức tạp nhất mà vẫn thỏa mãn các điều kiện bắt buộc.

---

## 7. Phạm Vi Dự Án TrafficSignalControl

Dự án so sánh các thuật toán điều khiển đèn giao thông bằng SUMO để học và báo cáo:

- **Thuật toán**: Fixed-Time (FT), Max-Pressure (MP), Tabular Q-Learning (QL), và Deep Q-Network (DQN).
- **Hai bản đồ**: `data/cologne1` (8 pha) và `data/ingolstadt1` (3 pha).
- **Nhu cầu xe**: Dùng nhu cầu xe có sẵn từ tập dữ liệu chuẩn RESCO.
- **Năm chỉ số (Metrics)**: Average Waiting Time, Travel Time, Time Loss, Queue Length (Mean & Max), và Throughput.
- **An toàn chuyển pha**: Dùng chung thời gian dọn nút vàng 3s + đỏ toàn phần 1s cho tất cả thuật toán.

## 8. Quy Tắc Làm Việc Theo Giai Đoạn

Các quy tắc dưới đây bổ sung và làm rõ các mục phía trên cho dự án này.

1. Đọc `AGENT.md`, `AGENTS.md` nếu có và kiểm tra trạng thái dự án trước khi sửa.
2. Viết code đơn giản, trực tiếp, tên biến rõ nghĩa để người mới học có thể giải thích. Không tạo kiến trúc phức tạp, dependency hoặc hạ tầng kiểm thử không cần thiết.
3. Đúng yêu cầu và lý thuyết quan trọng hơn việc rút ngắn code.
4. Trước khi code, nêu đầu vào, đầu ra, luồng xử lý, công thức và các giả định. Tra cứu tài liệu SUMO hoặc bài báo gốc khi cần xác minh.
5. Dùng chung cách chuyển pha và cách tính metrics cho cả ba thuật toán. Ghi rõ các ràng buộc riêng và ảnh hưởng của chúng.
6. Kiểm tra vừa đủ: chạy luồng chính, một trường hợp biên quan trọng và tính tay một ví dụ khi phù hợp.
7. Không tự ý xóa, ghi đè dữ liệu gốc hoặc thay đổi công việc có sẵn.
8. Chỉ làm giai đoạn người dùng yêu cầu. Hoàn thành rồi dừng để người dùng kiểm tra; chưa tự chuyển sang giai đoạn tiếp theo.
9. Chỉ hỏi khi thiếu thông tin ảnh hưởng đến mục tiêu hoặc tính đúng đắn. Tự giải quyết các lựa chọn triển khai nhỏ và giải thích ngắn gọn.
10. Không khẳng định thuật toán hội tụ, tối ưu hoặc tốt hơn nếu chưa có bằng chứng phù hợp.

## 9. Nội Dung Báo Cáo Sau Mỗi Giai Đoạn

Bổ sung vào cách trình bày tại mục 5, phản hồi sau mỗi giai đoạn phải có:

1. **Đã làm và file thay đổi**.
2. **Luồng đầu vào → xử lý → đầu ra**.
3. **Đối chiếu lý thuyết với vị trí code**.
4. **Đã kiểm tra gì, kết quả và giới hạn còn lại**. Nếu chỉ sửa tài liệu, ghi rõ không thay đổi code mô phỏng và kiểm tra nội dung tài liệu ở mức phù hợp.
5. **Lệnh hoặc thao tác để người dùng tự kiểm tra**.