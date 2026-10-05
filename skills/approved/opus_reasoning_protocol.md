# Opus Metacognitive Reasoning Protocol (OMRP) for TuMinh
## Giao Thức Tư Duy Phản Biện Chuẩn Claude Opus

Mỗi khi Agent TuMinh nhận một nhiệm vụ mới, trước khi gọi bất kỳ công cụ nào hoặc viết bất kỳ dòng mã nào, BẮT BUỘC phải thực hiện chuỗi tư duy tự phản biện theo 4 giai đoạn bên dưới bên trong thẻ `<thought>...</thought>`:

---

### Giai đoạn 1: Khảo sát & Phân rã mục tiêu (Deconstruction)
1. **Đích đến thực sự là gì?** Yêu cầu này cần đầu ra là gì? (Tạo file mới, sửa hàm cũ hay viết test?).
2. **Các ràng buộc ẩn (Hidden Constraints):**
   - Môi trường chạy là Windows hay Linux? (chú ý dấu phân cách đường dẫn `\`, subprocess encoding `utf-8`).
   - Có thư viện bên ngoài nào chưa được cài đặt không?
   - Có nguy cơ rò rỉ secrets / API keys không?

---

### Giai đoạn 2: Tự phản biện & Tìm rủi ro (Adversarial Critique)
*Tự đóng vai một Senior Reviewer khó tính để bắt bẻ giải pháp của chính mình:*
- *"Nếu mình sửa file này, có làm vỡ logic của file khác đang import nó không?"*
- *"Nếu đầu vào bị rỗng (`None`, `""`, mảng trống), code có bị crash không?"*
- *"Có trường hợp ngoại lệ nào (edge case, network timeout, file not found) mà mình chưa bọc trong `try...except` không?"*

---

### Giai đoạn 3: Lập kế hoạch hành động tối giản (Minimal Viable Action)
- Không viết code dư thừa, không suy đoán ngoài phạm vi yêu cầu.
- Liệt kê chính xác thứ tự các bước cần làm:
  - Bước 1: Đọc file để hiểu context.
  - Bước 2: Viết/Sửa code theo chuẩn PEP 8.
  - Bước 3: Viết unit test bao phủ các ca lỗi.
  - Bước 4: Chạy `pytest` để kiểm chứng thực tế.

---

### Giai đoạn 4: Đánh giá sau thực thi (Post-Execution Reflection)
- Sau khi chạy lệnh terminal:
  - Nếu kết quả lỗi $\rightarrow$ Không hoảng loạn thử bừa, mà phải đọc chính xác dòng lỗi (Traceback) để truy tìm nguyên nhân gốc rễ (Root Cause).
  - Nếu test PASS $\rightarrow$ Tự kiểm tra lại diff code một lần cuối trước khi tạo Git commit.
