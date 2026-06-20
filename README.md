# Personalized Vietnamese Voice Generation Platform

Project for Vietnamese personalized speech generation using F5-TTS. The public-facing topic name avoids technical model names; the technical implementation uses a Vietnamese F5-TTS base model plus six fine-tuning variants.

## Current Status

- Web app: `app.py`
- Base model: `hynt/F5-TTS-Vietnamese-ViVoice`
- Six experiment configs: `configs/experiment_matrix.yaml`
- Training entrypoint: `scripts/run_experiments.py`
- Best-model selector: `scripts/select_best_model.py`
- Current limitation: no trained checkpoints are committed. `Base zero-shot` is only for pipeline testing and may not produce stable cloned speech.

## Run Web

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Train Workflow

```bash
python scripts/data_prep.py --input_dir data/raw --output_dir data/processed
python scripts/preflight_check.py
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py
python scripts/select_best_model.py
python app.py
```

See `AI_HANDOFF_FULL_PROJECT_STATUS.md` and `OPTIMIZATION_WORKFLOW.md` for detailed handoff context.
