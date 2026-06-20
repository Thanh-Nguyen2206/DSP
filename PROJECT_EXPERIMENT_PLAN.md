# Nền tảng tạo giọng nói tiếng Việt cá nhân hóa cho nội dung số

## 1. Tên đề tài

**Tên đề xuất:** Nền tảng tạo giọng nói tiếng Việt cá nhân hóa cho nội dung số

Tên này tập trung vào giá trị sản phẩm và bài toán ứng dụng, không đặt theo kỹ thuật như F5-TTS, diffusion, transformer hay voice cloning. Khi cần giải thích chuyên sâu, phần kỹ thuật được trình bày trong chương phương pháp.

## 2. Mục tiêu thực nghiệm

Thay vì chỉ huấn luyện một mô hình, dự án huấn luyện 6 biến thể để chọn cấu hình tối ưu theo dữ liệu thực tế. Các biến thể giữ an toàn cho Google Colab T4 bằng cách dùng FP16, batch size nhỏ và gradient accumulation.

## 3. Sáu biến thể model

| ID | Tên | Kỹ thuật chính | Mục đích |
|---|---|---|---|
| M01 | Baseline ổn định | Fine-tune chuẩn | Làm mốc so sánh |
| M02 | LR thấp giữ giọng | Learning rate thấp hơn | Giảm méo đặc trưng giọng |
| M03 | Thích nghi nhanh | LR cao hơn, ít epoch hơn | Kiểm tra tốc độ hội tụ |
| M04 | Bền vững với nhiễu | Noise, gain, time-shift nhẹ | Tăng khả năng chịu audio tham chiếu chưa sạch |
| M05 | Regularization mạnh | Weight decay và gradient clipping chặt hơn | Giảm overfit khi dữ liệu ít |
| M06 | Effective batch lớn | Tăng gradient accumulation | Làm gradient ổn định hơn nhưng không tăng VRAM |

Ma trận cấu hình nằm ở `configs/experiment_matrix.yaml`.

## 4. Quy trình chạy

Kiểm tra cấu trúc dự án và cấu hình trước khi train:

```bash
python scripts/preflight_check.py
```

Kiểm tra danh sách experiment:

```bash
python scripts/run_experiments.py --dry-run
```

Chạy toàn bộ 6 experiment:

```bash
python scripts/run_experiments.py
```

Chạy một vài experiment được chọn:

```bash
python scripts/run_experiments.py --only m01_baseline_stable,m04_noise_robust
```

Mỗi model sẽ lưu riêng:

```text
checkpoints/experiments/<experiment_name>/
logs/experiments/<experiment_name>/
configs/generated/<experiment_name>.yaml
```

## 5. Tiêu chí chọn model tối ưu

Sau khi sinh audio test cho từng model, điền bảng:

```text
evaluation/model_scores.csv
```

Các metric cần điền:

- `speaker_similarity`: độ giống giọng người nói, càng cao càng tốt.
- `cer`: Character Error Rate, càng thấp càng tốt.
- `wer`: Word Error Rate, càng thấp càng tốt.
- `mos`: điểm nghe chủ quan 1-5, càng cao càng tốt.
- `latency_sec`: thời gian sinh audio, càng thấp càng tốt.

Chọn model tốt nhất:

```bash
python scripts/select_best_model.py
```

Script sẽ tạo:

```text
evaluation/best_model_report.md
evaluation/best_model.json
```

`app.py` sẽ ưu tiên `evaluation/best_model.json` để load model tốt nhất khi demo.

## 6. Công thức chấm điểm

Điểm tổng hợp:

```text
0.35 * speaker_similarity
+ 0.25 * (1 - CER)
+ 0.15 * (1 - WER)
+ 0.15 * (MOS / 5)
+ 0.10 * latency_score
```

Công thức này ưu tiên giống giọng và độ rõ nội dung, nhưng vẫn tính trải nghiệm nghe và tốc độ.
