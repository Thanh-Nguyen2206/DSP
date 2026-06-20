# 🚀 PROJECT CONTEXT & AI HANDOFF DOCUMENT: Vietnamese Personalized Voice Studio

> **LƯU Ý DÀNH CHO AI TIẾP QUẢN:** Đây là tài liệu Hand-off toàn diện của dự án Capstone (Môn DSP391m - FPTU). Hãy đọc kỹ toàn bộ bối cảnh, cấu trúc thư mục, các ràng buộc phần cứng và những lỗi đã được giải quyết để tránh lặp lại sai lầm. Bạn phải luôn tuân thủ mô hình "Pair-Programming", hỏi ý kiến user trước khi thực hiện thay đổi lớn.

---

## 1. 🎯 TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)
- **Tên đề tài đề xuất (không dính kỹ thuật):** **Nền tảng tạo giọng nói tiếng Việt cá nhân hóa cho nội dung số**.
- **Cách diễn đạt khi báo cáo:** Tập trung vào bài toán tạo giọng nói cá nhân hóa, hỗ trợ sản xuất nội dung, học tập và thuyết minh tiếng Việt. Không đặt tên đề tài theo tên model/kiến trúc.
- **Mục tiêu kỹ thuật nội bộ:** Xây dựng hệ thống Voice Cloning (Sao chép giọng nói) cho tiếng Việt, ứng dụng kiến trúc **F5-TTS** (Flow Matching + Diffusion Transformer).
- **Mô hình gốc (Base Model):** `hynt/F5-TTS-Vietnamese-ViVoice` (Hugging Face).
- **Ràng buộc phần cứng (CRITICAL):** Chạy trên **Google Colab - NVIDIA T4 GPU (16GB VRAM)**.
  - BẮT BUỘC sử dụng **Mixed Precision (FP16)**. (T4 không hỗ trợ BF16).
  - BẮT BUỘC cấu hình Batch Size cực nhỏ (VD: `batch_size = 1` hoặc `2`) kết hợp với **Gradient Accumulation** (VD: `grad_accum = 8`) để tránh lỗi Out Of Memory (OOM).

---

## 2. 📂 CẤU TRÚC THƯ MỤC & FILE ĐÃ HOÀN THÀNH (FILE STRUCTURE)

Dự án nằm trong thư mục: `f5tts_vietnamese/`

```text
f5tts_vietnamese/
├── app.py                     # Giao diện Gradio (Phase 4 - Đã hoàn thiện)
├── verify_env.py              # Script kiểm tra môi trường & GPU (Phase 1)
├── setup.sh                   # Script bash cài đặt môi trường (Phase 1)
├── requirements.txt           # Danh sách thư viện Python (Phase 1)
├── configs/
│   ├── train_config.yaml      # Cấu hình baseline (T4 Optimized)
│   └── experiment_matrix.yaml # Ma trận 6 biến thể model/technique
├── scripts/
│   ├── data_prep.py           # Pipeline xử lý dữ liệu âm thanh (Phase 2)
│   ├── train.py               # Script Fine-Tuning chính (Phase 3)
│   ├── run_experiments.py     # Chạy 3-6 experiment theo ma trận
│   ├── select_best_model.py   # Chọn model tối ưu từ bảng metric
│   └── preflight_check.py     # Kiểm tra nhanh cấu trúc/config trước khi train
├── checkpoints/               # (Tự động sinh ra) Nơi lưu model.pt sau khi train
└── data/
    ├── raw/                   # (Tự tạo) Thư mục chứa audio gốc của người dùng
    ├── processed/             # (Tự động sinh ra) Chứa audio .wav đã cắt
    └── metadata/              # (Tự động sinh ra) Chứa metadata.csv
```

---

## 3. ⏳ TIẾN ĐỘ THỰC HIỆN (PROGRESS & PHASES)

### ✅ Phase 1: Environment Setup (Hoàn thành)
- Đã tạo `requirements.txt`, `setup.sh` để cài đặt môi trường (PyTorch CUDA 11.8/12.1, torchaudio, f5-tts, gradio, librosa).
- Đã tạo `verify_env.py` để test GPU.
  - *Lỗi đã fix:* Sửa thuộc tính `total_mem` thành `total_memory` của PyTorch CUDA.

### ✅ Phase 2: Data Preprocessing Pipeline (Hoàn thành)
- Script: `scripts/data_prep.py`.
- **Chức năng:** Đọc file audio từ `data/raw/`, xoá khoảng lặng (silence removal), cắt thành các đoạn 3-10 giây, lọc nhiễu SNR, chuyển về chuẩn `24kHz, Mono, .wav` và tự động sinh file `data/metadata/metadata.csv`.
- **Đã test:** User đã chạy thành công và sinh ra file `metadata.csv` (với dummy text).

### ✅ Phase 3: Fine-Tuning Script (Hoàn thành & Train xong 100 Epochs)
- Script: `scripts/train.py` & `configs/train_config.yaml`.
- **Chức năng:** Khởi tạo kiến trúc CFM+DiT, tải Pre-trained weights, đọc Dataset, và Fine-tune bằng Accelerate.
- **Những LỖI LỚN ĐÃ GIẢI QUYẾT (QUAN TRỌNG):**
  1. **Lỗi tải Checkpoint 404:** Hardcode tên file `model_last.safetensors` bị lỗi. Đã viết lại hàm tự động quét repo Hugging Face (`list_repo_files`) để fallback sang file checkpoint thực tế có sẵn.
  2. **Lỗi Dataset format:** Đã viết lại Class `VoiceDataset` để tự động parse (bóc tách) metadata.csv thông minh, hỗ trợ cả đường dẫn tương đối và tuyệt đối (`/absolute/path.wav|Text`).
  3. **Lỗi Off-by-One Tensor Shape (319 vs 320):** Lỗi kinh điển của F5-TTS khi tính `mel_lengths`. 
     - *Cách fix:* Sửa thuật toán tính toán `mel_lens = alen // hop_length + 1` trong `collate_fn`.
     - Thêm cơ chế **Auto-Trim/Pad (Safety Check)** trong vòng lặp training: Bắt buộc `lens.amax()` phải bằng tuyệt đối với chiều dài thực tế do mô hình sinh ra để tránh văng `RuntimeError: broadcast mismatch`.

### ✅ Phase 4: Gradio Inference UI (Hoàn thành)
- Script: `app.py`.
- **Chức năng:** Giao diện Web. Nhận Reference Audio, Reference Text và Input Text -> Trả ra âm thanh giả lập.
- **Những LỖI ĐÃ GIẢI QUYẾT:**
  1. **Lỗi Vocoder 401/404:** Thư viện `Vocos.from_pretrained('charactrix/vocos...')` bị sập do HuggingFace. Đã sửa thành việc gọi thẳng hàm nội bộ `load_vocoder()` của thư viện `f5_tts`.
  2. **Lỗi Tham số API:** Sửa sai sót truyền tham số `text_gen` thành tên chuẩn xác `gen_text` khi gọi hàm `infer_process()`.
  3. **Logic Load Model:** Đã import lại module `build_model` từ `train.py` để kiến trúc inference 100% khớp với cấu trúc lúc training. Hàm tự động tìm trong `checkpoints/` thư mục `step_...` mới nhất để chèn weights.

### ✅ Phase 5: Multi-Model Experimentation (Đã bổ sung theo yêu cầu giảng viên)
- **Mục tiêu:** Train thêm 3-6 biến thể để không phụ thuộc vào một cấu hình duy nhất.
- **Ma trận experiment:** `configs/experiment_matrix.yaml` gồm 6 model:
  1. `m01_baseline_stable`: baseline ổn định.
  2. `m02_low_lr_preserve_voice`: learning rate thấp để giữ đặc trưng giọng.
  3. `m03_fast_adaptation`: thích nghi nhanh, ít epoch hơn.
  4. `m04_noise_robust`: augmentation nhẹ với noise/gain/time-shift.
  5. `m05_regularized_general`: regularization mạnh hơn để giảm overfit.
  6. `m06_large_effective_batch`: batch vật lý nhỏ nhưng effective batch lớn hơn qua gradient accumulation.
- **Cách chạy:** `python scripts/preflight_check.py`, sau đó `python scripts/run_experiments.py --dry-run` để kiểm tra, rồi `python scripts/run_experiments.py`.
- **Cách chọn model tối ưu:** Điền metric vào `evaluation/model_scores.csv`, chạy `python scripts/select_best_model.py`. Script sẽ sinh `evaluation/best_model_report.md` và `evaluation/best_model.json`.
- **Inference:** `app.py` ưu tiên `evaluation/best_model.json` hoặc biến môi trường `F5TTS_EXPERIMENT` để load đúng model tốt nhất.

---

## 4. 🚧 NHỮNG VIỆC CHƯA LÀM / BƯỚC TIẾP THEO (TODOs / MISSING PIECES)

Vì đây là đồ án Capstone, các bước tiếp theo AI mới cần chú ý hỗ trợ người dùng:

1. **Dữ liệu thật (Real Data):** Hiện tại ở Phase 2, User báo cáo đang dùng "dummy text" (Văn bản giả) trong `metadata.csv`. Cần có bước gán nhãn (transcription) chính xác 100% văn bản cho các file âm thanh đã bị cắt nhỏ (có thể dùng Whisper để auto-transcribe text thực tế).
2. **Đánh giá chất lượng mô hình (Evaluation):** Đã có script chọn model từ bảng metric (`scripts/select_best_model.py`), nhưng vẫn cần chạy thực nghiệm thật để điền WER, CER, MOS, speaker similarity và latency.
3. **Triển khai (Deployment):** Script `app.py` hiện tại chạy tốt trên Colab (sử dụng `share=True`), nhưng để làm sản phẩm đồ án, có thể cần đóng gói Docker hoặc đẩy lên Hugging Face Spaces / AWS / GCP.
4. **Tối ưu Tốc độ Sinh âm thanh:** Inference hiện tại chạy ổn, nhưng có thể cần thiết lập bước nhảy `nfe_step` cho luồng ODE solver trong F5-TTS để sinh giọng nói nhanh hơn.

---

## 5. 💡 QUY TẮC LÀM VIỆC DÀNH CHO AI MỚI

1. **KHÔNG** phá vỡ cấu trúc của `train.py` và `collate_fn` vì nó đã được "đo ni đóng giày" rất kỹ cho lỗi Tensor Shape mismatch. 
2. Luôn ghi nhớ đây là dự án chạy trên **NVIDIA T4**. Cấm đề xuất các giải pháp tốn VRAM như tăng `batch_size` lên quá cao hoặc dùng `BF16`.
3. Luôn giữ phong thái giao tiếp **Pair-Programming**: Chỉ đưa ra giải pháp/code cho một module/vấn đề cụ thể, sau đó **dừng lại** để đợi User test trên Colab và xác nhận thành công (Test Pass) mới được đi tiếp. Đừng bao giờ viết nhồi nhét tất cả các bước vào một prompt.
4. Giao diện (Gradio) luôn phải duy trì các khối `try/except` để bắt lỗi, không làm sập ứng dụng nếu đầu vào của người dùng bị sai format.
