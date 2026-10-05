# TuMinh Self-Distillation & Continuous Evolution Engine
## Architecture Addendum — Optimal Claude / Anthropic Design Specification

**Status:** Approved Master Blueprint  
**Target Hardware:** NVIDIA GeForce RTX 3060 Ti (8 GB VRAM), 64 GB System RAM, Intel Core i7-12700K  
**Author:** AI System Architect (Claude / Antigravity Design Philosophy)  
**Date:** 2026-10-05  

---

## 1. Executive Summary & Core Objective

The primary TuMinh specification defines the **Autonomous Runtime** (Agent Loop, Tools, Task Queue, Reflection).  
This addendum specifies the **Self-Distillation & Evolution Engine** — the mechanism that harvests high-quality reasoning and coding trajectories from teacher models (GitHub Copilot / Claude Opus), converts them into a synthetic training curriculum, and fine-tunes a local coding model on consumer GPU hardware (**RTX 3060 Ti 8GB**) to run 100% offline and free.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. HARVESTING STAGE (Khâu Thu Hoạch Tri Thức Từ Copilot / Claude)           │
│                                                                             │
│   [Copilot / Claude Teacher Session]                                        │
│         │                                                                   │
│         ▼ (Bắt trọn chuỗi tư duy: Prompt -> CoT -> Code -> Test Verified)   │
│   `distill/harvester.py`                                                    │
│         │                                                                   │
│         ▼                                                                   │
│   `distill/data/raw_trajectories.jsonl`                                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CURATION & EVAL GATING (Khâu Sàng Lọc & Kiểm Định Chất Lượng)            │
│                                                                             │
│   `distill/curator.py`                                                      │
│   - Loại bỏ các bước thừa, log rác, API keys / secrets.                     │
│   - CHỈ giữ lại các phiên mà Test/Eval đạt kết quả 100% PASS.               │
│   - Chuẩn hóa định dạng ShareGPT / ChatML với trường `thinking` sâu.        │
│         │                                                                   │
│         ▼                                                                   │
│   `distill/data/train_dataset.jsonl`                                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. LOCAL DISTILLATION PIPELINE (Khâu Huấn Luyện Trên RTX 3060 Ti 8GB)       │
│                                                                             │
│   `distill/trainer.py` (Unsloth + PyTorch CUDA)                             │
│   - Base Model: `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`               │
│   - Hyperparameters tối ưu cho 8GB VRAM:                                   │
│     * Batch Size = 1 (Gradient Accumulation Steps = 4)                      │
│     * Max Sequence Length = 4,096 tokens                                    │
│     * LoRA rank (r) = 16, alpha = 16, target_modules = all linear           │
│     * Memory footprint: ~6.2 GB VRAM (An toàn tuyệt đối dưới ngưỡng 8GB)    │
│         │                                                                   │
│         ▼                                                                   │
│   Xuất file trọng số: `distill/models/tuminh-v1-q4_k_m.gguf`                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. LOCAL DEPLOYMENT & FLYWHEEL (Khâu Triển Khai & Khép Kín Vòng Lặp)        │
│                                                                             │
│   `distill/deploy.py`                                                       │
│   - Tự động nạp model vào Ollama: `ollama create tuminh -f Modelfile`       │
│   - Cập nhật `.env`:                                                        │
│     `MODEL_BASE_URL=http://localhost:11434/v1`                              │
│     `MODEL_NAME=tuminh`                                                     │
│   - TuMinh từ nay chạy 100% Offline, không tốn Token, tốc độ 40-50 tokens/s! │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout Addendum

Bổ sung phân hệ `/distill` vào kiến trúc gốc của dự án `TuMinh`:

```text
TuMinh/
├── agent/                  # Runtime chính của Agent (nhận việc, loop, tool call)
├── tools/                  # Công cụ file, git, command allowlist
├── memory/                 # Reflection store (bài học kinh nghiệm)
├── tasks/                  # Hàng đợi công việc (tasks.json)
├── skills/                 # Thư viện skill đã duyệt & đề xuất mới
│
├── distill/                # [PHÂN HỆ CHƯNG CẤT TRI THỨC]
│   ├── harvester.py        # Thu thập log từ các phiên giải quyết task thành công
│   ├── curator.py          # Lọc sạch dữ liệu, format sang JSONL chuẩn SFT
│   ├── trainer.py          # Script Unsloth QLoRA 4-bit tối ưu cho RTX 3060 Ti
│   ├── export_gguf.py      # Chuyển đổi LoRA weights sang GGUF (Q4_K_M)
│   ├── deploy.py           # Đăng ký model vào Ollama và cập nhật .env
│   ├── Modelfile           # Cấu hình system prompt và temperature cho Ollama
│   └── data/
│       ├── raw/            # Trajectory logs thô
│       └── train.jsonl     # Tập dữ liệu chưng cất hoàn chỉnh
│
├── tests/                  # Unit tests với mock/fake LLM
└── docs/                   # Tài liệu thiết kế
```

---

## 3. Data Format Specification (Cấu trúc dữ liệu chưng cất)

Mỗi mẫu dữ liệu trong `train.jsonl` phải ghi lại đầy đủ **quy trình tư duy (Chain-of-Thought)** của model thầy:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Viết hàm Python validate_cpf(cpf: str) -> bool kiểm tra mã định danh và kèm pytest test cases."
    },
    {
      "from": "thought",
      "value": "1. Khảo sát: Cần loại bỏ ký tự không phải số. Kiểm tra độ dài 11 chữ số và không phải các chuỗi lặp như '11111111111'.\n2. Tính toán 2 số kiểm tra theo trọng số.\n3. Viết unit test bao phủ: valid, invalid digits, repeated digits, invalid length."
    },
    {
      "from": "gpt",
      "value": "```python\n# Mã nguồn hoàn chỉnh pass test...\n```"
    }
  ]
}
```

---

## 4. Hardware-Tuned Training Configuration (RTX 3060 Ti 8GB)

Để huấn luyện thành công model 7B trên card 8GB VRAM mà không bị lỗi `CUDA Out of Memory`:

| Tham số | Giá trị | Giải thích lý do kỹ thuật |
| :--- | :--- | :--- |
| **Base Model** | `Qwen2.5-Coder-7B-Instruct-bnb-4bit` | Model 7B lập trình tốt nhất hiện nay, bản 4-bit chiếm ~4.5 GB VRAM |
| **Framework** | **Unsloth** | Giảm 60% VRAM sử dụng và tăng tốc 2x so với HuggingFace PEFT gốc |
| **Max Seq Length** | `4096` | Đủ dài cho một hàm hoặc module code hoàn chỉnh |
| **Per Device Batch** | `1` | Tránh tràn bộ nhớ đệm activation trên VRAM 8GB |
| **Gradient Accumulation** | `4` hoặc `8` | Giả lập Effective Batch Size = 4 hoặc 8 giúp loss hội tụ ổn định |
| **LoRA Target Modules** | `q, k, v, o, gate, up, down` | Tinh chỉnh toàn bộ attention và MLP layers |
| **Optimizers** | `adamw_8bit` | Tiết kiệm thêm 1 GB VRAM bộ nhớ optimizer |
| **Gradient Checkpointing** | `True` (Unsloth auto) | Đánh đổi 10% tốc độ để tiết kiệm 30% VRAM |

---

## 5. Continuous Evolution Gate (Giao thức tự tiến hóa khép kín)

Để đảm bảo model sau khi chưng cất **chỉ thông minh hơn chứ không bị thoái hóa**:
1. Trước khi nạp model mới vào Ollama, chạy bộ kiểm thử đánh giá:
   ```bash
   python -m evals.run_benchmark --model tuminh-candidate
   ```
2. Nếu tỷ lệ giải đúng test case (Pass Rate) của model mới $\ge$ model cũ:
   * Chấp thuận nạp model mới làm model mặc định.
   * Ghi nhận phiên bản (vd: `tuminh:v1.1`).
3. Nếu model mới bị thoái hóa:
   * Giữ nguyên model cũ, ghi lại nhật ký lỗi và đề xuất bổ sung thêm dữ liệu cho bài học bị hỏng.

---

**Kết luận:**  
Bản thiết kế này cung cấp **công thức toàn diện** để biến dự án `TuMinh` từ một agent phụ thuộc vào cloud thành một **AI Agent tự hành độc lập, tự học và tự tiến hóa vĩnh viễn trên chính chiếc PC của bạn**.
