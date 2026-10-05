# TỔNG HỢP TOÀN BỘ Ý TƯỞNG & KIẾN TRÚC CHIẾN LƯỢC: DỰ ÁN TỰ MÌNH (TUMINH)
> **Tài liệu đúc kết toàn bộ tư duy, chiến lược và giải pháp kỹ thuật từ sáng tới giờ.**  
> **Vị trí lưu trữ:** `TuMinh/docs/Y_TUONG_VA_KIEN_TRUC_TUMINH.md`  
> **Ngày lập:** 05/10/2026

---

## I. BẢN CHẤT CỦA CÁC CÔNG CỤ & CHIẾN THUẬT "HACK" 1 LƯỢT CHAT

### 1. Hiểu đúng về hạn mức GitHub Copilot Free:
* **2,000 Code Completions/tháng**: Tự động gợi ý code khi nhấn phím `Tab` trong IDE (VS Code). Dùng thoải mái hàng ngày, **không bao giờ đụng vào 50 lượt chat**.
* **50 Chat Requests/tháng**: Dùng cho khung chat hoặc Copilot CLI trong terminal.
* **Chiêu "Single-Session Project Hack" (Gom 1 lượt làm cả dự án)**:
  * Đưa ra 1 prompt khởi tạo toàn diện ban đầu (có cấu trúc đặc tả rõ ràng).
  * Copilot tự động kích hoạt chế độ **Brainstorming / Interactive Planning**.
  * Tất cả các câu hỏi trắc nghiệm duyệt lệnh (`↑/↓ select • enter accept`) tiếp theo **hoàn toàn KHÔNG bị tính thêm lượt chat nào**!
  * Giải quyết được 18 task lớn nhỏ chỉ với **đúng 1 lượt chat duy nhất**.

---

## II. LINH HỒN CỦA DỰ ÁN "TỰ MÌNH" (TUMINH)

### 1. Ý nghĩa:
Không phải là một chatbot "hỏi-đáp nói suông", mà là một **Autonomous AI Coding Agent (Kỹ sư AI tự hành)** độc lập trên máy tính cá nhân.

### 2. Vòng lặp hành động cốt lõi (ReAct Loop):
1. **Nhận việc (Ingestion)**: Đọc từ `tasks/tasks.json` (hoặc GitHub Issues).
2. **Khảo sát (Explore)**: Dùng tool đọc code thật, không bao giờ đoán mò.
3. **Lập kế hoạch (Plan)**: Chia nhỏ bài toán thành các bước tối giản.
4. **Viết code & Tự kiểm thử (Pytest)**: Bắt buộc chạy test thật trên máy.
5. **Tự sửa sai (Self-Correction)**: Nhìn log lỗi terminal và tự sửa lại cho đến khi test PASS xanh lá.
6. **Git Guard**: Chỉ commit khi test PASS 100%, không bao giờ commit code lỗi.

### 3. Giao thức tư duy Claude Opus (`skills/approved/opus_reasoning_protocol.md`):
* Bắt buộc suy nghĩ trong thẻ `<thought>...</thought>` trước khi gõ code.
* Tự phản biện đối kháng (Adversarial Critique): Đóng vai Senior khó tính tìm mọi rủi ro, lỗi biên (edge cases) và bẫy hệ điều hành Windows trước khi lưu file.

---

## III. CHIẾN LƯỢC CHƯNG CẤT TRI THỨC (KNOWLEDGE DISTILLATION)

### 1. Ý tưởng đột phá:
* Thay vì trả tiền mua API đắt đỏ của OpenAI/Claude, ta dùng **chính Copilot (chạy model triệu đô) và Claude/Antigravity** làm "Thầy giáo".
* Thầy giải các bài toán hóc búa $\rightarrow$ TuMinh thu hoạch toàn bộ chuỗi tư duy và code chuẩn vào file `distill/data/`.
* Sau đó dùng chính chiếc card đồ họa của bạn để "nướng" lại vào model nhỏ chạy cục bộ.

### 2. Tối ưu "Đo ni đóng giày" cho phần cứng máy bạn:
* **Cấu hình máy thực tế**:
  * **GPU**: NVIDIA GeForce RTX 3060 Ti (8 GB VRAM)
  * **RAM hệ thống**: 64 GB
  * **CPU**: Intel Core i7-12700K
* **Bộ não nền (Base Model)**: `Qwen2.5-Coder-7B-Instruct` (Model lập trình 7B số 1 thế giới hiện nay).
* **Công nghệ huấn luyện**: **Unsloth (QLoRA 4-bit)**
  * Tiêu thụ VRAM khi train: **~ 6.2 GB / 8 GB** (an toàn 100%, không bao giờ tràn VRAM).
  * Tiêu thụ RAM máy khi train: **~ 8 – 12 GB / 64 GB** (máy vẫn dư tận 50GB RAM làm việc khác).
* **Mức tiêu thụ khi chạy hàng ngày (Inference qua Ollama)**:
  * VRAM: **~ 5.5 GB**
  * RAM máy: **~ 2 GB** (siêu nhẹ, bằng vài tab Chrome).
  * Tốc độ phản hồi: **40 – 60 từ/giây** (nhanh hơn tốc độ đọc của mắt người).
  * Ngữ cảnh tối ưu: **8,192 – 16,384 tokens** (đọc hiểu trọn vẹn 2,500 – 5,000 dòng code mỗi lượt).

---

## IV. TỰ ĐỘNG HÓA CHẠY NGẦM & TÀNG HÌNH

### 1. Cơ chế chạy ngầm khi bật máy:
* Tự động khởi động cùng Windows qua thư mục `shell:startup`.
* Chạy bằng **`start_background.vbs`** gọi `pythonw.exe daemon.py`:
  * **Hoàn toàn tàng hình**: Không bao giờ chớp màn hình đen CMD làm vướng mắt.
  * Tự động đánh thức Copilot giải ngầm 1 bài toán khó mỗi ngày.
  * Bắn thông báo **Windows Toast** sang trọng ở góc màn hình khi hoàn thành bài học.

### 2. Mẹo "Dồn task dùng 1 lượt cho cả tháng":
* Khi mất mạng hoặc suốt cả tháng: Cứ ghi dồn 30 - 50 nhiệm vụ vào file `tasks/tasks.json`.
* Cuối tháng chỉ mở Copilot lên **ĐÚNG 1 LẦN DUY NHẤT** yêu cầu giải quyết hết danh sách $\rightarrow$ Tiết kiệm 100% quota!

---

## V. TẠI SAO TUMINH KHÔNG BỊ ẢO GIÁC?
1. **Trọng tài Pytest**: Code sai cú pháp hay bịa hàm là terminal ném lỗi đỏ ngay, AI buộc phải sửa lại theo thực tế.
2. **Grounding công cụ thật**: Phải mở file đọc code thật mới được sửa, không được đoán mò.
3. **Dataset chưng cất siêu sạch**: Chỉ những bài tập đạt điểm 10 (Pass 100% test) mới được đưa vào giáo trình dạy cho TuMinh.

---

## VI. SƠ ĐỒ TIẾN HÓA CỦA DỰ ÁN TUMINH

```
   [HÔM NAY]                                    [TƯƠNG LAI GẦN]
Copilot + Antigravity                      TuMinh Model 7B Local
(Lập trình viên hướng dẫn)             (Trợ lý trung thành chạy trên máy)
          │                                              │
          ▼                                              ▼
• Cần Internet                             • Chạy 100% OFFLINE
• Bị giới hạn 50 lượt/tháng                • HOÀN TOÀN MIỄN PHÍ VĨNH VIỄN
• Model ở Cloud nước ngoài                 • Model thuộc sở hữu riêng trên PC bạn
```
