# Kịch bản tiếp nhận Trajectory từ Orca Desktop (chuẩn TADP/1)

**Mục đích:** Hướng dẫn xuất trajectory từ các task Orca Desktop đã hoàn thành
(Claude / OpenCode-DeepSeek / Codex) và nạp an toàn vào kho tri thức chắt lọc
(distillation) của Tự Minh, tuân thủ giao thức TADP/1 hiện có trong
`distill/importers.py` (`ExternalImporter`).

---

## 1. Định dạng export mà Tự Minh chấp nhận (TADP/1)

File export là **JSONL UTF-8**, dòng đầu tiên là `header`, các dòng sau là
`event`. Ví dụ tối thiểu hợp lệ:

```jsonl
{"record_type":"header","schema_version":1,"trajectory_id":"orca-task-0001","task_id":"orca-task-0001","task_outcome":"succeeded","verification_passed":true,"eval_passed":true}
{"schema_version":1,"task_id":"orca-task-0001","sequence_number":1,"event_type":"user_message","timestamp":"2026-10-05T12:00:01Z","data":{"message":{"role":"user","content":"..."}}}
{"schema_version":1,"task_id":"orca-task-0001","sequence_number":2,"event_type":"assistant_message","timestamp":"2026-10-05T12:00:02Z","data":{"message":{"role":"assistant","content":"..."}}}
{"schema_version":1,"task_id":"orca-task-0001","sequence_number":3,"event_type":"tool_message","timestamp":"2026-10-05T12:00:03Z","data":{"message":{"role":"tool","tool_name":"pytest","content":"passed"}}}
{"schema_version":1,"task_id":"orca-task-0001","sequence_number":4,"event_type":"verification","timestamp":"2026-10-05T12:00:04Z","data":{"passed":true}}
```

**event_type cho phép:** `task`, `user_message`, `assistant_message`,
`tool_message`, `verification`.

**Giới hạn an toàn bắt buộc (đã code cứng trong `ExternalImporter`):**
- Toàn bộ file nguồn ≤ **64 MB**.
- Tối đa **1000 events** mỗi trajectory.
- Mỗi event ≤ **64 KB**.
- Nội dung phải là UTF-8 hợp lệ; các chuỗi trông giống secret/token/API key/
  mật khẩu sẽ bị từ chối tự động (regex phát hiện `api_key=`, `Bearer ...`,
  `ghp_...`, `AKIA...`, URL có credential nhúng).

---

## 2. Bước 1 — Xuất trajectory từ Orca Desktop

Trong Orca Desktop / Antigravity IDE, sau khi một task của worker agent
(Claude, OpenCode/DeepSeek, Codex) hoàn tất và **đã qua kiểm chứng
(verification_passed / eval_passed = true)**:

1. Xuất log hội thoại + kết quả tool-call của task sang JSON/JSONL thô.
2. Dùng script chuyển đổi mẫu bên dưới để quy đổi sang đúng khuôn JSONL TADP/1.
3. Đặt file vào thư mục tạm, ví dụ `imports/orca/<task_id>.jsonl`.

### Script chuyển đổi mẫu: `tools/orca_export_to_tadp1.py`

Xem file script đã tạo kèm tài liệu này. Cách dùng:

```powershell
.venv\Scripts\python tools\orca_export_to_tadp1.py `
  --orca-export imports/orca/raw/<task_id>.json `
  --output imports/orca/<task_id>.jsonl
```

---

## 3. Bước 2 — Nạp vào Tự Minh qua `distill.cli import-external`

```powershell
.venv\Scripts\python -m distill.cli import-external `
  --source-name "orca-desktop" `
  --license-or-authorization "Nội bộ - đã được chủ dự án phê duyệt dùng cho chắt lọc" `
  --authorized-by "hungdaimedia-gif" `
  --source-model-id "claude-sonnet-5" `
  --source-model-revision "2026-09" `
  --output data\distill\raw\orca-<task_id>.export.jsonl `
  imports\orca\<task_id>.jsonl
```

Cờ bắt buộc: `--source-name`, `--license-or-authorization`, `--authorized-by`.
Cờ đa-agent TADP/1 (tùy chọn nhưng khuyến nghị để truy vết nguồn gốc model):
`--source-model-id`, `--source-model-revision`.

> Lưu ý: `--output` không được trùng với file đầu vào (`--path` / positional
> input), importer sẽ từ chối ghi đè nguồn.

## 4. Bước 3 — Chắt lọc (curate) và chuẩn bị huấn luyện

```powershell
.venv\Scripts\python -m distill.cli curate `
  --input data\distill\raw\orca-<task_id>.export.jsonl `
  --output data\distill\curated\dataset-orca.jsonl

.venv\Scripts\python -m distill.cli train `
  --config distill\config.example.toml `
  --dataset data\distill\curated\dataset-orca.jsonl
```

`train` chỉ chạy đầy đủ khi môi trường QLoRA tùy chọn (Linux/WSL2,
`distill/requirements-training.txt`) đã được cài; trên Windows gốc, lệnh sẽ
fail-closed với `TrainingEnvironmentError` — đây là hành vi **đúng theo
thiết kế**, không phải lỗi.

---

## 5. Kiểm tra sau khi nạp

- `pytest tests/test_distill_importers.py tests/test_distill_cli.py -q`
  để đảm bảo không phá vỡ hành vi hiện có.
- Kiểm tra `candidate-manifest.json` (sau khi train trong môi trường WSL2 đầy
  đủ) có `training_config.model_id` / `dataset_hash` đúng như kỳ vọng trước
  khi `benchmark` và `promote`.
