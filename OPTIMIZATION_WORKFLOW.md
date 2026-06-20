# Quy trinh toi uu chat luong giong noi

## 1. Tai sao hien tai co `not trained yet`

`M01` den `M06` hien tai moi la cac cau hinh/chien luoc train. Chung chi tro thanh model that khi da sinh ra checkpoint:

```text
checkpoints/experiments/<ten_variant>/step_.../model.pt
```

Neu chua co file nay, web khong the dung variant do de tao giong. Khi do he thong chi co the dung `Base zero-shot`, day chi la che do test pipeline va co the cho audio hon loan, khong ro chu, khong giong reference.

## 2. Viec can lam de toi uu that su

1. Dat audio that cua nguoi can clone vao:

```text
data/raw/
```

2. Chay tien xu ly audio:

```bash
python scripts/data_prep.py --input_dir data/raw --output_dir data/processed
```

3. Mo file metadata va dien transcript chinh xac 100%:

```text
data/metadata/metadata.csv
```

Transcript sai la nguyen nhan rat lon lam model doc khong ro chu.

4. Kiem tra truoc khi train:

```bash
python scripts/preflight_check.py
python scripts/run_experiments.py --dry-run
```

5. Train 6 variant theo yeu cau cua thay:

```bash
python scripts/run_experiments.py
```

Co the train rieng tung variant:

```bash
python scripts/run_experiments.py --only m01_baseline_stable
python scripts/run_experiments.py --only m02_low_lr_preserve_voice
python scripts/run_experiments.py --only m03_fast_adaptation
python scripts/run_experiments.py --only m04_noise_robust
python scripts/run_experiments.py --only m05_regularized_general
python scripts/run_experiments.py --only m06_large_effective_batch
```

6. Nghe thu output cua tung variant, dien metric vao:

```text
evaluation/model_scores.csv
```

7. Chon model tot nhat:

```bash
python scripts/select_best_model.py
```

8. Restart web:

```bash
python app.py
```

Luc nay `Auto best available` moi load checkpoint that va cho ket qua on dinh hon.

## 3. Quy tac input khi demo web

- Reference audio nen sach, it nhieu, 3-10 giay.
- Reference transcript phai khop dung tung tu trong audio.
- Text can tao nen la cau day du 8-20 tu khi demo.
- Khong dung `Base zero-shot` de danh gia chat luong clone giong.
- Neu muon giong that su giong reference, bat buoc phai co checkpoint fine-tuned.
