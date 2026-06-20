# AI Handoff - Vietnamese Personalized Voice Generation Project

## 1. Project Identity

Project folder:

```text
/Users/vudjeuvuj84gmail.com/Downloads/STUDY/FPTU/2026/Summer 2026/DSP391m/f5tts_vietnamese
```

Public topic name:

```text
Personalized Vietnamese Voice Generation Platform
```

Vietnamese non-technical topic framing:

```text
Nền tảng tạo giọng nói tiếng Việt cá nhân hóa cho nội dung số
```

Internal technical goal:

- Build a Vietnamese personalized voice generation / voice cloning demo.
- Use F5-TTS Vietnamese as the base model.
- Train 3-6 variants as requested by the instructor, then select the best checkpoint for demo.

Base model:

```text
hynt/F5-TTS-Vietnamese-ViVoice
```

Hardware target for real training:

```text
Google Colab NVIDIA T4, FP16, small batch size, gradient accumulation
```

## 2. Current Truth: Why Output Is Still Not Good

The current local project has no trained checkpoints yet:

```text
checkpoints/experiments/<variant>/step_.../model.pt
```

Because these files do not exist yet, `M01` to `M06` are currently training recipes, not usable trained models.

The web app can run `Base zero-shot`, but that mode is only a pipeline test. It can produce unclear or chaotic output, especially when:

- reference transcript does not exactly match the reference audio;
- generated text is too short;
- reference audio has noise, clipping, or low volume;
- no fine-tuned speaker checkpoint has been loaded.

To get a reliable demo, the missing work is training/evaluating the variants and selecting the best model.

## 3. Important Files

```text
app.py
```

Gradio web app. It loads F5-TTS, exposes the UI, validates inputs, preprocesses reference audio, runs inference, and reports generation status.

```text
configs/train_config.yaml
```

Base training config optimized for NVIDIA T4.

```text
configs/experiment_matrix.yaml
```

Six training variants:

- `m01_baseline_stable`
- `m02_low_lr_preserve_voice`
- `m03_fast_adaptation`
- `m04_noise_robust`
- `m05_regularized_general`
- `m06_large_effective_batch`

```text
configs/generated/
```

Generated per-variant config files.

```text
scripts/data_prep.py
```

Audio preprocessing pipeline. Takes audio from `data/raw`, outputs processed clips to `data/processed`, and creates `data/metadata/metadata.csv`.

```text
scripts/train.py
```

Fine-tuning script. Contains the dataset class, F5-TTS model construction, base model loading, checkpoint saving/loading, and training loop.

```text
scripts/run_experiments.py
```

Runs the six configured model variants.

```text
scripts/select_best_model.py
```

Ranks variants from `evaluation/model_scores.csv` and creates `evaluation/best_model.json`.

```text
scripts/preflight_check.py
```

Checks required files, Python syntax, YAML config, T4 constraints, and experiment count.

```text
evaluation/model_scores.csv
```

Metric template for selecting the best model. Needs speaker similarity, CER, WER, MOS, latency.

```text
OPTIMIZATION_WORKFLOW.md
```

Step-by-step Vietnamese workflow for making output quality reliable.

## 4. What Has Been Done

### Web App

- Changed UI content to English.
- Kept the public project title non-technical.
- Added compact runtime status.
- Added `Usage Guide, Model Functions, and Data Resources`.
- Added `Optimization Readiness`.
- Added `Prepare Project Folders` button.
- Added `Model Variant` dropdown with all choices visible:
  - `Auto best available`
  - `Base zero-shot (quick test)`
  - `M01` to `M06`
- Added clear status:
  - `ready` means checkpoint exists and can be used.
  - `needs training` means the variant exists as a training recipe but cannot generate final cloned speech yet.
- Changed generation behavior:
  - If a selected variant needs training, the app does not generate misleading noisy audio.
  - It returns status text with the correct training command.
  - `Base zero-shot` still works for pipeline testing only.
- Added input quality warnings:
  - short reference audio;
  - long reference audio;
  - quiet reference audio;
  - clipped reference audio;
  - short reference transcript;
  - short generated text;
  - likely repetitive reference transcript;
  - no fine-tuned checkpoint loaded.
- Added F5-TTS reference preprocessing via `preprocess_ref_audio_text`.
- Added output post-processing to prevent clipping and invalid samples.
- Added quality presets:
  - `Balanced clarity`
  - `Higher clarity (slower)`
  - `Fast preview`

### Training / Experiment Setup

- Six experiment variants are configured in `configs/experiment_matrix.yaml`.
- Generated configs exist under `configs/generated/`.
- Preflight check passes.
- The project folders exist:
  - `data/raw`
  - `data/processed`
  - `data/metadata`
  - `checkpoints/experiments`
  - `logs/experiments`
  - `evaluation`

### Git Hygiene

- Added `.gitignore`.
- Checkpoints, raw audio, processed audio, logs, local Gradio files, cache files, and generated audio are ignored.
- `.gitkeep` files preserve required empty folders.

## 5. What Is Still Missing

### Missing Data

No real training dataset is currently present.

Needed:

```text
data/raw/
data/processed/
data/metadata/metadata.csv
```

`metadata.csv` must contain exact transcript text. If transcript is wrong, the model will learn wrong text/audio alignment and output quality will be poor.

### Missing Checkpoints

No trained checkpoint exists yet:

```text
checkpoints/experiments/<variant>/step_.../model.pt
```

Until these exist, M01-M06 cannot be used for final generation.

### Missing Evaluation

`evaluation/model_scores.csv` still needs real metrics:

- speaker similarity;
- CER;
- WER;
- MOS;
- latency.

After metrics are filled:

```bash
python scripts/select_best_model.py
```

This creates:

```text
evaluation/best_model.json
evaluation/best_model_report.md
```

Then restart the web app and use `Auto best available`.

## 6. Correct End-to-End Workflow

### Step 1 - Add Real Audio

Put clean speaker audio into:

```text
data/raw/
```

Recommended:

- clean speech;
- low background noise;
- consistent microphone;
- enough speech for fine-tuning;
- exact transcripts available.

### Step 2 - Preprocess

```bash
python scripts/data_prep.py --input_dir data/raw --output_dir data/processed
```

### Step 3 - Fix Metadata

Open:

```text
data/metadata/metadata.csv
```

Fill exact transcript for every audio clip.

### Step 4 - Check Project

```bash
python scripts/preflight_check.py
python scripts/run_experiments.py --dry-run
```

### Step 5 - Train Variants

Train all:

```bash
python scripts/run_experiments.py
```

Train one:

```bash
python scripts/run_experiments.py --only m01_baseline_stable
```

### Step 6 - Evaluate

Generate test samples for each model and fill:

```text
evaluation/model_scores.csv
```

### Step 7 - Select Best Model

```bash
python scripts/select_best_model.py
```

### Step 8 - Restart Web

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

Use:

```text
Auto best available
```

## 7. Verification Already Run

These checks have passed locally:

```bash
python -m py_compile app.py
python scripts/preflight_check.py
```

The Gradio server starts at:

```text
http://127.0.0.1:7860
```

API check confirmed that model dropdown includes:

- `Auto best available`
- `Base zero-shot (quick test)`
- all six M01-M06 variants marked `needs training`.

Selecting an untrained variant returns status text and does not generate misleading noisy audio.

## 8. Notes For The Next AI

- Do not claim output quality is fixed until a fine-tuned checkpoint exists.
- Do not hide M01-M06 from the UI; keep them visible with clear status.
- Do not let untrained variants silently fallback to base output without warning.
- If the user wants actual voice match, prioritize dataset preparation, transcript correction, and training on T4.
- If output is chaotic in `Base zero-shot`, that is expected behavior for this project state.
- Avoid committing raw speaker audio, checkpoints, and logs to GitHub.

## 9. Git / GitHub Status

This project folder has been initialized as its own Git repository so it does not accidentally include unrelated course folders from the parent workspace.

Remote:

```text
https://github.com/Thanh-Nguyen2206/DSP.git
```

The local commit is ready, but pushing from this machine was blocked by GitHub authentication:

```text
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/Thanh-Nguyen2206/DSP.git/'
```

To upload, authenticate Git on this machine with a valid GitHub token or GitHub CLI, then run:

```bash
git push -u origin main
```
