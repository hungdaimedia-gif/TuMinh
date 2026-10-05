# BIÊN BẢN XÁC MINH KỸ THUẬT — DỰ ÁN TỰ MINH
**Thời điểm xác minh:** 08:02 PM, 05/10/2026
**Người thực hiện:** GitHub Copilot CLI (phối hợp cùng chủ dự án)
**Môi trường:** `.venv` (Python 3.11.14), Windows, GPU NVIDIA GeForce RTX 3060 Ti (8192 MiB VRAM), `torch 2.5.1+cu121`

---

## 1. Xác minh toàn bộ bộ kiểm thử tự động (pytest)

Lệnh chạy:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Kết quả:** `317 passed, 3 skipped` — **100% xanh**, không bài test nào fail.

3 bài skip đều do thiếu quyền tạo symlink trên Windows (`WinError 1314`), không phải lỗi logic:
- `tests/test_distill_curator.py:230`
- `tests/test_workspace.py:48`
- `tests/test_workspace.py:68`

## 2. Xác minh cơ chế GPU VRAM Guard (hồ sơ `rtx_3060_ti_8gb`)

Môi trường xác nhận thật (không giả lập): GPU `NVIDIA GeForce RTX 3060 Ti`, 8192 MiB VRAM — khớp chính xác hồ sơ phần cứng cấu hình trong `distill/trainer.py`. `torch.cuda.is_available() = True`.

**2.1. Unit test:** `pytest tests/test_distill_trainer.py -v` → **20/20 PASS**, bao gồm:
- `test_rtx_3060_ti_profile_rejects_aggressive_settings`
- `test_backend_out_of_memory_is_reported_as_typed_error_and_cleans_staging`

**2.2. Kiểm thử trực tiếp qua CLI thật** (`python -m distill.cli train`), dùng dataset hợp lệ tối thiểu và cấu hình tạm thời (không phải model thật, không tải mạng):

| # | Kịch bản | Cấu hình | Kết quả | Tầng chặn |
|---|----------|----------|---------|-----------|
| 1 | Aggressive | `sequence_length=4096` (> giới hạn 2048) | Exit code `2`, chặn tức thì | **VRAM Guard** — chặn trước khi chạm CUDA/backend: `"the rtx_3060_ti_8gb profile is conservative, not a VRAM guarantee..."` |
| 2 | Conservative (offline) | `sequence_length=512`, `batch_size=1`, `HF_HUB_OFFLINE=1` | Exit code `2` | VRAM Guard cho qua → CUDA check cho qua (GPU thật = `True`) → chặn ở tầng phụ thuộc QLoRA còn thiếu (`peft`/`datasets`/`bitsandbytes`), đúng theo README: cần cài riêng `distill/requirements-training.txt` |

Không có lượt tải mô hình, không có OOM thật, không đụng tới model thật trong suốt quá trình kiểm thử — an toàn tuyệt đối. Dữ liệu tạm đã được dọn dẹp sau khi xác minh xong.

## 3. Kết luận

- Bộ kiểm thử tự động của dự án đạt **100% xanh** trên môi trường `.venv`.
- Cơ chế **GPU VRAM Guard** cho hồ sơ `rtx_3060_ti_8gb` hoạt động đúng thiết kế theo nguyên tắc *"conservative, not a guarantee"*: chặn cấu hình vượt ngưỡng bằng kiểm tra tĩnh **trước khi** chạm phần cứng GPU thật, và chỉ cho phép tiếp tục khi cấu hình nằm trong giới hạn an toàn (`sequence_length ≤ 2048`, `batch_size = 1`).
- Cột mốc kỹ thuật ngày 05/10/2026 được xác nhận **đạt yêu cầu**, sẵn sàng cho bước tiếp theo (cài đặt `distill/requirements-training.txt` khi thực sự cần chạy huấn luyện QLoRA thật với model đã được cấp phép).
