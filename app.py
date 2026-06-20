import os
import sys
import json
import yaml
import torch
import traceback
import shutil
import subprocess
import tempfile
import uuid
import gradio as gr
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to sys.path for local script imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# =============================================================================
# GLOBAL VARIABLES & CONFIG
# =============================================================================
CONFIG_PATH = os.path.join(current_dir, "configs", "train_config.yaml")
CHECKPOINTS_DIR = os.path.join(current_dir, "checkpoints")
BEST_MODEL_PATH = os.path.join(current_dir, "evaluation", "best_model.json")
EXPERIMENT_MATRIX_PATH = os.path.join(current_dir, "configs", "experiment_matrix.yaml")
MOCK_MODE = os.getenv("F5TTS_MOCK", "0") == "1"
PROJECT_TITLE = "Personalized Vietnamese Voice Generation Platform"
LOCAL_PREVIEW_VOICE = os.getenv("F5TTS_PREVIEW_VOICE", "Linh")
SHARE_GRADIO = os.getenv("F5TTS_SHARE", "0") == "1"
AUTO_MODEL_CHOICE = "Auto best available"
BASE_MODEL_CHOICE = "Base zero-shot (quick test)"
NO_TRAINED_MODEL_CHOICE = "No trained variants found"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
vocoder = None
vocab_char_map = None
runtime_state: Dict[str, Any] = {
    "mode": "mock" if MOCK_MODE else "real",
    "device": device,
    "checkpoint_source": "not initialized",
    "checkpoint_dir": "",
    "checkpoint_path": "",
    "experiment": "",
    "active_model_key": "base",
    "requested_model_key": "auto",
    "using_finetuned": False,
    "message": "Runtime has not been initialized yet.",
}

GENERATION_PRESETS = {
    "Balanced clarity": {
        "nfe_step": 40,
        "cfg_strength": 2.0,
        "speed": 0.96,
        "description": "Recommended default for clearer Vietnamese speech.",
    },
    "Higher clarity (slower)": {
        "nfe_step": 48,
        "cfg_strength": 2.1,
        "speed": 0.94,
        "description": "Uses more solver steps and a slightly slower speaking rate.",
    },
    "Fast preview": {
        "nfe_step": 24,
        "cfg_strength": 1.8,
        "speed": 1.0,
        "description": "Faster local checks, lower quality.",
    },
}

OPEN_DATA_RESOURCES = [
    {
        "name": "F5-TTS Vietnamese ViVoice",
        "kind": "Base model / research reference",
        "url": "https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice",
        "use": "Current base model. Good for zero-shot testing and as the starting point for fine-tuning.",
        "note": "Model card says it was trained on Vi-Voice and VLSP 2021-2023, about 1000 hours total.",
    },
    {
        "name": "F5-TTS Vietnamese training pipeline",
        "kind": "GitHub training recipe",
        "url": "https://github.com/nguyenthienhy/F5-TTS-Vietnamese",
        "use": "Reference for training and inference rules, especially clean reference audio and exact transcript.",
        "note": "Useful for reducing hallucination and poor pronunciation during inference.",
    },
    {
        "name": "Bud500",
        "kind": "Open Vietnamese ASR dataset",
        "url": "https://github.com/apluka34/Bud500",
        "use": "Best used for ASR/transcript evaluation and robustness checks, not direct single-speaker cloning.",
        "note": "Around 500 hours, broad topics and regional accents.",
    },
    {
        "name": "Vietnamese speech dataset collection",
        "kind": "Hugging Face collection",
        "url": "https://huggingface.co/collections/doof-ferb/vietnamese-speech-dataset",
        "use": "Catalog for VIVOS, FPT FOSD, VietSpeech, viVoice, and other Vietnamese speech resources.",
        "note": "Check each dataset license/access before training or public demo use.",
    },
    {
        "name": "CodeLinkIO Vietnamese Voice Dataset",
        "kind": "TTS dataset listing",
        "url": "https://github.com/CodeLinkIO/vietnamese-voice-dataset",
        "use": "Southern female single-speaker TTS-style data if access is granted.",
        "note": "Research-use listing; download requires contacting the maintainers.",
    },
    {
        "name": "vietTTS / InfoRe notes",
        "kind": "Vietnamese TTS toolkit",
        "url": "https://github.com/NTT123/vietTTS",
        "use": "Reference for denoising, alignment, and Vietnamese TTS data preparation ideas.",
        "note": "The project is older, but the data preparation notes are still useful.",
    },
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_runtime_modules():
    """Lazy import F5-TTS so preview mode can still open without full ML deps."""
    if MOCK_MODE:
        return None, None, None, None, None, None

    from f5_tts.model import CFM
    from f5_tts.infer.utils_infer import infer_process, load_vocoder
    from scripts.train import build_model, download_vocab, download_and_load_weights

    return CFM, infer_process, load_vocoder, build_model, download_vocab, download_and_load_weights


def load_experiment_matrix() -> Dict[str, Any]:
    matrix_path = Path(EXPERIMENT_MATRIX_PATH)
    if not matrix_path.exists():
        return {"experiments": []}

    with matrix_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"experiments": []}


def load_experiments() -> list:
    try:
        return load_experiment_matrix().get("experiments", [])
    except Exception as e:
        print(f"⚠️ Could not read the experiment matrix: {e}")
        return []


def get_experiment(name: str) -> Optional[Dict[str, Any]]:
    for exp in load_experiments():
        if exp.get("name") == name:
            return exp
    return None


def _relative_or_name(path: str) -> str:
    if not path:
        return "none"
    try:
        return str(Path(path).relative_to(Path(current_dir)))
    except ValueError:
        return Path(path).name


def _checkpoint_status_for_experiment(name: str) -> Tuple[str, Optional[Path]]:
    checkpoint_dir = Path(CHECKPOINTS_DIR) / "experiments" / name
    checkpoint_path, _ = find_latest_checkpoint_file(str(checkpoint_dir))
    if checkpoint_path:
        return "ready", checkpoint_path
    return "needs training", None


def build_model_choices() -> list:
    choices = [AUTO_MODEL_CHOICE, BASE_MODEL_CHOICE]
    for exp in load_experiments():
        name = exp["name"]
        status, _ = _checkpoint_status_for_experiment(name)
        choices.append(f"{exp.get('display_name', name)} | {name} | {status}")
    return choices


def build_training_plan_choices() -> list:
    choices = []
    for exp in load_experiments():
        name = exp["name"]
        status, _ = _checkpoint_status_for_experiment(name)
        choices.append(f"{exp.get('display_name', name)} | {name} | {status}")
    return choices


def parse_model_choice(model_choice: str) -> str:
    if not model_choice or model_choice == AUTO_MODEL_CHOICE:
        return "auto"
    if model_choice == BASE_MODEL_CHOICE:
        return "base"
    if model_choice == NO_TRAINED_MODEL_CHOICE:
        return "none"
    parts = [part.strip() for part in model_choice.split("|")]
    if len(parts) >= 2:
        return parts[1]
    return "auto"


def selection_for_model_choice(model_choice: str) -> Dict[str, str]:
    key = parse_model_choice(model_choice)
    if key == "none":
        return {
            "checkpoint_source": "no trained model",
            "experiment": "none",
            "checkpoint_dir": "",
            "requested_model_key": "none",
        }
    if key == "base":
        return {
            "checkpoint_source": "manual model selector",
            "experiment": "base_zero_shot",
            "checkpoint_dir": "",
            "requested_model_key": "base",
        }
    if key == "auto":
        selection = resolve_checkpoint_selection()
        selection["requested_model_key"] = "auto"
        return selection

    return {
        "checkpoint_source": "manual model selector",
        "experiment": key,
        "checkpoint_dir": str(Path(CHECKPOINTS_DIR) / "experiments" / key),
        "requested_model_key": key,
    }


def build_model_choice_details(model_choice: str) -> str:
    key = parse_model_choice(model_choice)
    if key == "none":
        return (
            "**No trained variants are available yet.**\n\n"
            "Run the training workflow first. Until a checkpoint exists, the only usable generation engine is base zero-shot."
        )
    if key == "base":
        return (
            "**Selected model:** Base zero-shot.\n\n"
            "Use this for quick testing only. It does not use a fine-tuned speaker checkpoint, "
            "so the voice may not match the reference speaker exactly."
        )

    if key == "auto":
        selection = resolve_checkpoint_selection()
        checkpoint_path, _ = find_latest_checkpoint_file(selection["checkpoint_dir"])
        if checkpoint_path:
            return (
                "**Selected model:** Auto best available.\n\n"
                f"The app will load `{selection.get('experiment')}` from "
                f"`{_relative_or_name(str(checkpoint_path))}`."
            )
        return (
            "**Selected model:** Auto best available.\n\n"
            "No selected fine-tuned checkpoint is available yet. Train/evaluate the variants first, "
            "or choose `Base zero-shot` only for quick pipeline testing."
        )

    exp = get_experiment(key)
    status, checkpoint_path = _checkpoint_status_for_experiment(key)
    display_name = exp.get("display_name", key) if exp else key
    description = exp.get("description", "No description available.") if exp else "No description available."
    checkpoint_text = _relative_or_name(str(checkpoint_path)) if checkpoint_path else "needs training"
    return (
        f"**Selected model:** {display_name}\n\n"
        f"{description}\n\n"
        f"**Checkpoint:** `{checkpoint_text}`. "
        f"{'This variant can be loaded during generation.' if status == 'ready' else 'This variant is visible for planning, but it must be trained before it can generate reliable cloned speech.'}"
    )


def build_training_choice_details(training_choice: str) -> str:
    key = parse_model_choice(training_choice)
    exp = get_experiment(key)
    if not exp:
        return "Select a training variant to see what it does."

    status, checkpoint_path = _checkpoint_status_for_experiment(key)
    checkpoint_text = _relative_or_name(str(checkpoint_path)) if checkpoint_path else "needs training"
    command = f"python scripts/run_experiments.py --only {key}"
    return (
        f"**{exp.get('display_name', key)}**\n\n"
        f"{exp.get('description', '')}\n\n"
        f"- Status: `{status}`\n"
        f"- Checkpoint: `{checkpoint_text}`\n"
        f"- Train this variant: `{command}`\n\n"
        "After training, evaluate the generated samples, fill `evaluation/model_scores.csv`, "
        "run `python scripts/select_best_model.py`, then restart this web app."
    )


def build_project_readiness_markdown() -> str:
    data_dir = Path(current_dir) / "data"
    processed_dir = data_dir / "processed"
    metadata_path = data_dir / "metadata" / "metadata.csv"
    checkpoints_dir = Path(CHECKPOINTS_DIR)
    trained = [
        exp["name"]
        for exp in load_experiments()
        if _checkpoint_status_for_experiment(exp["name"])[0] == "ready"
    ]

    data_status = "ready" if metadata_path.exists() and processed_dir.exists() else "missing"
    checkpoint_status = f"{len(trained)} trained variant(s)" if trained else "no trained checkpoints"
    best_status = "selected" if Path(BEST_MODEL_PATH).exists() else "not selected"

    next_step = (
        "Collect/prepare real voice data first."
        if data_status == "missing"
        else "Train the six variants."
        if not trained
        else "Evaluate variants and select the best model."
        if best_status == "not selected"
        else "Use Auto best available for generation."
    )

    return (
        "### Optimization Readiness\n"
        f"- Dataset: `{data_status}` (`data/processed` + `data/metadata/metadata.csv`).\n"
        f"- Trained checkpoints: `{checkpoint_status}`.\n"
        f"- Best model selection: `{best_status}`.\n"
        f"- Next action: **{next_step}**\n\n"
        "**Why the current audio can sound chaotic:** the app is currently using base zero-shot because no trained checkpoint exists. "
        "Base zero-shot can test the pipeline, but it is not reliable enough for a stable cloned voice demo."
    )


def build_compact_status_markdown() -> str:
    if MOCK_MODE:
        return (
            f"**Runtime:** Local Preview Mode using macOS `{LOCAL_PREVIEW_VOICE}` system voice. "
            "**Voice cloning is not active.**"
        )

    engine = "fine-tuned checkpoint" if runtime_state.get("using_finetuned") else "base zero-shot"
    active = runtime_state.get("active_model_key") or "base"
    checkpoint = _relative_or_name(runtime_state.get("checkpoint_path", ""))
    return (
        f"**Runtime:** `{engine}` on `{runtime_state.get('device', device)}`. "
        f"**Active model:** `{active}`. **Checkpoint:** `{checkpoint}`."
    )


def build_usage_guide_markdown() -> str:
    variant_lines = []
    for exp in load_experiments():
        status, _ = _checkpoint_status_for_experiment(exp["name"])
        variant_lines.append(
            f"- **{exp.get('display_name', exp['name'])}**: {exp.get('description', '')} Status: `{status}`."
        )

    resource_lines = []
    for resource in OPEN_DATA_RESOURCES:
        resource_lines.append(
            f"- **[{resource['name']}]({resource['url']})** ({resource['kind']}): "
            f"{resource['use']} {resource['note']}"
        )

    return (
        "### Quick Use\n"
        "1. Upload a clean 3-10 second reference voice sample.\n"
        "2. Enter the exact transcript of that reference audio. Do not put the output text there.\n"
        "3. Choose a trained model. If no trained model exists, use `Base zero-shot` only for pipeline testing.\n"
        "4. Enter the Vietnamese text to generate and click `Generate Voice`.\n\n"
        "### What `needs training` means\n"
        "`M01`-`M06` are experiment recipes until training creates a checkpoint file. "
        "A usable trained model must exist at `checkpoints/experiments/<variant>/step_.../model.pt`. "
        "Without that file, the app cannot actually use that variant for generation.\n\n"
        "### Model Functions\n"
        "- **Auto best available**: loads `evaluation/best_model.json` when it exists; otherwise uses base zero-shot.\n"
        "- **Base zero-shot**: quick testing without a trained personal checkpoint.\n"
        + "\n".join(variant_lines)
        + "\n\n"
        "### Data & Training Resources\n"
        "These resources are not downloaded automatically. Use them for training/evaluation only after checking license and consent.\n"
        + "\n".join(resource_lines)
    )


def data_resource_details(resource_name: str) -> str:
    for resource in OPEN_DATA_RESOURCES:
        if resource["name"] == resource_name:
            return (
                f"**{resource['name']}**\n\n"
                f"- Type: {resource['kind']}\n"
                f"- Use in this project: {resource['use']}\n"
                f"- Note: {resource['note']}\n"
                f"- Link: {resource['url']}"
            )
    return "Select a resource to see how it fits this project."


def create_project_dirs() -> str:
    """Create the required local folders for data and checkpoints."""
    required_dirs = [
        Path(current_dir) / "data" / "raw",
        Path(current_dir) / "data" / "processed",
        Path(current_dir) / "data" / "metadata",
        Path(current_dir) / "checkpoints" / "experiments",
        Path(current_dir) / "logs" / "experiments",
        Path(current_dir) / "evaluation",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    return (
        "Created/verified project folders: `data/raw`, `data/processed`, "
        "`data/metadata`, `checkpoints/experiments`, `logs/experiments`, and `evaluation`."
    )


def create_project_dirs_and_status() -> Tuple[str, str]:
    message = create_project_dirs()
    return build_project_readiness_markdown(), message


def generate_local_preview_audio(text: str) -> str:
    """
    Local preview uses macOS system TTS to read the input text.

    This is not voice cloning. It is only for UI debugging when the local
    machine does not have f5_tts/checkpoints/GPU. Real inference uses F5-TTS.
    """
    if not shutil.which("say"):
        raise RuntimeError(
            "The local macOS 'say' command is not available. Use Real Mode "
            "with the full F5-TTS environment instead."
        )
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to convert preview audio to WAV.")

    output_dir = Path(current_dir) / "outputs" / "local_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="f5tts_preview_"))
    aiff_path = tmp_dir / "preview.aiff"
    wav_path = output_dir / f"preview_{uuid.uuid4().hex}.wav"

    subprocess.run(
        ["say", "-v", LOCAL_PREVIEW_VOICE, "-o", str(aiff_path), text],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(aiff_path),
            "-ar",
            "24000",
            "-ac",
            "1",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return str(wav_path)

def resolve_checkpoint_dir() -> str:
    """
    Prefer the best selected experiment checkpoint if model selection has run.

    Priority:
      1. F5TTS_EXPERIMENT=m04_noise_robust → checkpoints/experiments/...
      2. evaluation/best_model.json from scripts/select_best_model.py
      3. checkpoints/ legacy
    """
    experiment_name = os.getenv("F5TTS_EXPERIMENT", "").strip()
    if experiment_name:
        selected = os.path.join(CHECKPOINTS_DIR, "experiments", experiment_name)
        print(f"🎯 Using experiment from F5TTS_EXPERIMENT={experiment_name}")
        return selected

    best_model_file = Path(BEST_MODEL_PATH)
    if best_model_file.exists():
        try:
            payload = json.loads(best_model_file.read_text(encoding="utf-8"))
            checkpoint_dir = payload.get("checkpoint_dir")
            if checkpoint_dir:
                path = Path(checkpoint_dir)
                if not path.is_absolute():
                    path = Path(current_dir) / path
                print(f"🎯 Using selected best model: {payload.get('experiment', path.name)}")
                return str(path)
        except Exception as e:
            print(f"⚠️ Could not read best_model.json ({e}). Falling back to default checkpoints.")

    return CHECKPOINTS_DIR


def resolve_checkpoint_selection() -> Dict[str, str]:
    """Resolve the checkpoint source and directory used for this web session."""
    experiment_name = os.getenv("F5TTS_EXPERIMENT", "").strip()
    if experiment_name:
        return {
            "checkpoint_source": "F5TTS_EXPERIMENT",
            "experiment": experiment_name,
            "checkpoint_dir": str(Path(CHECKPOINTS_DIR) / "experiments" / experiment_name),
        }

    best_model_file = Path(BEST_MODEL_PATH)
    if best_model_file.exists():
        try:
            payload = json.loads(best_model_file.read_text(encoding="utf-8"))
            checkpoint_dir = payload.get("checkpoint_dir", "")
            if checkpoint_dir:
                path = Path(checkpoint_dir)
                if not path.is_absolute():
                    path = Path(current_dir) / path
                return {
                    "checkpoint_source": "evaluation/best_model.json",
                    "experiment": payload.get("experiment", path.name),
                    "checkpoint_dir": str(path),
                }
        except Exception as e:
            print(f"⚠️ Could not read best_model.json ({e}). Falling back to default checkpoints.")

    return {
        "checkpoint_source": "default checkpoints folder",
        "experiment": "legacy/default",
        "checkpoint_dir": CHECKPOINTS_DIR,
    }


def _step_number(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def find_latest_checkpoint_file(checkpoints_dir: str) -> Tuple[Optional[Path], str]:
    """Find the newest model checkpoint under a training output directory."""
    ckpt_path = Path(checkpoints_dir)
    if not ckpt_path.exists():
        return None, f"Checkpoint directory not found: {checkpoints_dir}"

    direct_candidates = [ckpt_path / "model.pt", ckpt_path / "model_last.pt"]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate, "Found a direct model checkpoint."

    all_steps = sorted(
        [
            path for path in ckpt_path.glob("step_*")
            if path.is_dir() and _step_number(path) >= 0
        ],
        key=_step_number,
    )
    if not all_steps:
        return None, f"No step_* checkpoints found in {checkpoints_dir}"

    for step_dir in reversed(all_steps):
        for filename in ("model.pt", "model_last.pt"):
            candidate = step_dir / filename
            if candidate.exists():
                return candidate, f"Found latest checkpoint in {step_dir.name}."

    return None, f"No model.pt/model_last.pt file found under {checkpoints_dir}"


def _unwrap_model_state_dict(payload: Any) -> Any:
    """Support both raw state_dict files and wrapped checkpoint payloads."""
    if not isinstance(payload, dict):
        return payload

    for key in ("model", "state_dict", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    ema_value = payload.get("ema_model_state_dict")
    if isinstance(ema_value, dict):
        return {
            key.replace("ema_model.", "", 1) if key.startswith("ema_model.") else key: value
            for key, value in ema_value.items()
        }

    if payload and all(isinstance(key, str) for key in payload.keys()):
        if all(key.startswith("module.") for key in payload.keys()):
            return {key.replace("module.", "", 1): value for key, value in payload.items()}
    return payload

def load_latest_checkpoint(model, selection: Dict[str, str]):
    """
    Find and load the latest fine-tuned model.pt checkpoint.
    """
    info = {
        **selection,
        "checkpoint_path": "",
        "using_finetuned": False,
        "message": "",
    }

    checkpoint_path, message = find_latest_checkpoint_file(selection["checkpoint_dir"])
    if checkpoint_path is None:
        info["message"] = f"{message}. Using the base zero-shot model."
        print(f"⚠️ {info['message']}")
        return model, info

    print(f"🔄 Loading fine-tuned weights from: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _unwrap_model_state_dict(payload)
    
    # Load state dict into the model.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    info.update({
        "checkpoint_path": str(checkpoint_path),
        "using_finetuned": True,
        "message": f"Fine-tuned weights loaded from {checkpoint_path.name}.",
    })
    print(f"✅ {info['message']}")
    return model, info


def load_base_weights_into_model() -> str:
    """Reset the current model object back to the base Vietnamese checkpoint."""
    global model

    if MOCK_MODE:
        return "Mock mode does not load ML weights."
    if model is None:
        raise RuntimeError("The model is not initialized yet.")

    _, _, _, _, _, download_and_load_weights = load_runtime_modules()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("🔄 Resetting model to base zero-shot weights...")
    model = download_and_load_weights(model, config)
    model = model.to(device)
    model.eval()
    return "Base zero-shot weights loaded."


def set_runtime_to_base(message: str, requested_model_key: str = "base") -> None:
    runtime_state.update({
        "mode": "real",
        "device": device,
        "checkpoint_source": "manual model selector",
        "checkpoint_dir": "",
        "checkpoint_path": "",
        "experiment": "base_zero_shot",
        "active_model_key": "base",
        "requested_model_key": requested_model_key,
        "using_finetuned": False,
        "message": message,
    })


def ensure_selected_model_loaded(model_choice: str) -> Tuple[list, bool]:
    """Load the selected model variant when its checkpoint exists."""
    global model

    warnings = []
    if MOCK_MODE:
        return warnings, True

    selection = selection_for_model_choice(model_choice)
    requested_key = selection.get("requested_model_key", "auto")

    if requested_key == "base":
        if runtime_state.get("active_model_key") != "base":
            message = load_base_weights_into_model()
        else:
            message = "Base zero-shot model is already active."
        set_runtime_to_base(message, requested_model_key="base")
        return warnings, True

    checkpoint_path, message = find_latest_checkpoint_file(selection.get("checkpoint_dir", ""))
    if checkpoint_path is None:
        if requested_key == "auto":
            warnings.append(
                f"{message}. No best fine-tuned checkpoint is available yet. "
                "Train/evaluate the variants first, or choose 'Base zero-shot (quick test)' to test the pipeline only."
            )
        else:
            warnings.append(
                f"{message}. This model variant is configured but needs training before it can generate reliable cloned speech."
            )
        return warnings, False

    active_key = runtime_state.get("active_model_key")
    active_checkpoint = runtime_state.get("checkpoint_path")
    if active_key == selection.get("experiment") and active_checkpoint == str(checkpoint_path):
        runtime_state["requested_model_key"] = requested_key
        return warnings, True

    print(f"🔄 Switching model to selected checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _unwrap_model_state_dict(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.append(f"Selected checkpoint loaded with {len(missing)} missing keys.")
        print(f"⚠️ Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        warnings.append(f"Selected checkpoint loaded with {len(unexpected)} unexpected keys.")
        print(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model = model.to(device)
    model.eval()

    runtime_state.update({
        "mode": "real",
        "device": device,
        "checkpoint_source": selection.get("checkpoint_source", "manual model selector"),
        "checkpoint_dir": selection.get("checkpoint_dir", ""),
        "checkpoint_path": str(checkpoint_path),
        "experiment": selection.get("experiment", requested_key),
        "active_model_key": selection.get("experiment", requested_key),
        "requested_model_key": requested_key,
        "using_finetuned": True,
        "message": f"Loaded selected checkpoint: {_relative_or_name(str(checkpoint_path))}.",
    })
    return warnings, True


def init_system():
    """Initialize F5-TTS runtime once when the web app starts."""
    global model, vocoder, vocab_char_map, runtime_state
    
    if MOCK_MODE:
        print("🧪 F5TTS_MOCK=1: local preview mode is enabled. Real cloning is disabled.")
        runtime_state.update({
            "mode": "mock",
            "device": "system",
            "checkpoint_source": "mock preview",
            "checkpoint_dir": "",
            "checkpoint_path": "",
            "experiment": "",
            "active_model_key": "mock",
            "requested_model_key": "mock",
            "using_finetuned": False,
            "message": "Local Preview Mode uses macOS system TTS and does not clone a voice.",
        })
        return

    print("🚀 Initializing F5-TTS real voice cloning runtime...")
    _, _, load_vocoder, build_model, download_vocab, download_and_load_weights = load_runtime_modules()
    
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    vocab_char_map = download_vocab(config)
    
    model = build_model(config, vocab_char_map)
    
    print("\n⬇️  Loading base model...")
    model = download_and_load_weights(model, config)
    
    print("\n⬇️  Checking fine-tuned checkpoints...")
    selection = resolve_checkpoint_selection()
    model, checkpoint_info = load_latest_checkpoint(model, selection)
    runtime_state.update({
        "mode": "real",
        "device": device,
        "active_model_key": (
            checkpoint_info.get("experiment", "base")
            if checkpoint_info.get("using_finetuned")
            else "base"
        ),
        "requested_model_key": "auto",
        **checkpoint_info,
    })
    
    model = model.to(device)
    model.eval()
    
    print("\n🔊 Initializing vocoder...")
    try:
        vocoder = load_vocoder(device=device)
    except Exception as e:
        raise RuntimeError(f"Could not initialize the F5-TTS vocoder: {e}")
    
    print("\n✅ Real voice cloning runtime is ready.")


def _normalize_text(text: str) -> str:
    normalized = " ".join((text or "").strip().split())
    if normalized and normalized[-1] not in ".!?;:":
        normalized += "."
    return normalized


def _audio_duration_seconds(audio_path: str) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def _basic_audio_stats(audio_path: str) -> Dict[str, Optional[float]]:
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        duration = float(len(data)) / float(sr) if sr else None
        rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        return {"duration": duration, "rms": rms, "peak": peak}
    except Exception:
        return {"duration": None, "rms": None, "peak": None}


def _has_repeated_tokens(text: str) -> bool:
    tokens = [token.strip(".,!?;:").lower() for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 4:
        return False

    for idx in range(len(tokens) - 2):
        if tokens[idx] == tokens[idx + 1] or tokens[idx] == tokens[idx + 2]:
            return True
    return False


def _quality_settings(quality_preset: str) -> Dict[str, Any]:
    return GENERATION_PRESETS.get(quality_preset, GENERATION_PRESETS["Balanced clarity"])


def _validate_generation_inputs(ref_audio_path: str, ref_text: str, input_text: str) -> Tuple[str, str, list]:
    warnings = []
    if not ref_audio_path:
        raise ValueError("Please upload or record a reference voice audio file.")
    if not ref_text or not ref_text.strip():
        raise ValueError(
            "Reference Audio Transcript is required. It must exactly match the words spoken in the reference audio."
        )
    if not input_text or len(input_text.strip()) == 0:
        raise ValueError("Please enter the text to synthesize.")

    stats = _basic_audio_stats(ref_audio_path)
    duration = stats["duration"]
    if duration is not None:
        if duration < 3.0:
            warnings.append(
                f"Reference audio is {duration:.1f}s. Use 3-10 seconds of clean speech for better speaker matching."
            )
        elif duration > 12.0:
            warnings.append(
                f"Reference audio is {duration:.1f}s. F5-TTS will clip long references, so 3-10 seconds is safer."
            )
    if stats["rms"] is not None and stats["rms"] < 0.015:
        warnings.append("Reference audio is very quiet. Record closer to the microphone or normalize the clip before training/testing.")
    if stats["peak"] is not None and stats["peak"] > 0.98:
        warnings.append("Reference audio may be clipped. Use a cleaner recording with more headroom.")

    if len(ref_text.split()) < 4:
        warnings.append("Reference transcript is very short. A longer exact sentence usually preserves voice identity better.")
    if len(input_text.split()) < 6:
        warnings.append("Generated text is very short. Use a complete 8-20 word sentence for a more stable demo.")
    if _has_repeated_tokens(ref_text):
        warnings.append("Reference transcript looks repetitive. Make sure it exactly matches the spoken words in the reference audio.")
    if not runtime_state.get("using_finetuned") and not MOCK_MODE:
        warnings.append(
            "No fine-tuned checkpoint is loaded. Base zero-shot is only a pipeline test and may sound unstable or unlike the reference speaker."
        )

    return _normalize_text(ref_text), _normalize_text(input_text), warnings


def _postprocess_generated_audio(audio):
    """Prevent clipping and invalid samples before Gradio writes the WAV file."""
    import numpy as np

    processed = np.asarray(audio, dtype=np.float32)
    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
    if processed.size == 0:
        return processed, 0.0, False

    peak = float(np.max(np.abs(processed)))
    normalized = False
    if peak > 0.95:
        processed = processed / peak * 0.95
        normalized = True
    return processed, peak, normalized

# =============================================================================
# INFERENCE LOGIC
# =============================================================================

def synthesize_audio(model_choice, ref_audio_path, ref_text, input_text, quality_preset):
    """
    Gradio callback.
    - model_choice: selected model variant from the UI
    - ref_audio_path: reference voice audio
    - ref_text: transcript of the reference audio
    - input_text: text to synthesize
    """
    try:
        ref_text, input_text, warnings = _validate_generation_inputs(ref_audio_path, ref_text, input_text)
        model_warnings, model_ready = ensure_selected_model_loaded(model_choice)
        warnings = model_warnings + warnings
        settings = _quality_settings(quality_preset)

        if not model_ready:
            key = parse_model_choice(model_choice)
            train_hint = (
                f"`python scripts/run_experiments.py --only {key}`"
                if key not in ("auto", "base", "none")
                else "`python scripts/run_experiments.py`"
            )
            warning_lines = "\n".join([f"- {warning}" for warning in warnings]) or "- No input warnings."
            status = (
                "### Generation Status\n"
                "- Generation was not started because the selected model is not available yet.\n"
                f"- Selected model: `{model_choice}`.\n"
                "- Output audio is intentionally empty to avoid returning misleading noisy speech.\n"
                f"- Next training command: {train_hint}\n\n"
                "**What to do next**\n"
                "1. Prepare real audio data in `data/raw`.\n"
                "2. Run preprocessing and fill exact transcripts in `data/metadata/metadata.csv`.\n"
                "3. Train the selected variant or all six variants.\n"
                "4. Select the best model, restart the web app, then use `Auto best available`.\n\n"
                "**Input/model warnings**\n"
                f"{warning_lines}"
            )
            return None, status

        if MOCK_MODE:
            preview_path = generate_local_preview_audio(input_text.strip())
            print(f"🧪 Local preview with macOS voice {LOCAL_PREVIEW_VOICE}: {preview_path}")
            status = (
                "### Generation Status\n"
                "- Completed in Local Preview Mode.\n"
                "- This audio is generated by macOS system TTS and does not clone the reference speaker."
            )
            return preview_path, status
            
        print(f"\n🎙️ Starting real voice cloning inference:")
        print(f"   - Ref audio: {Path(ref_audio_path).name}")
        print(f"   - Input text: {input_text[:50]}...")
        
        _, infer_process, _, _, _, _ = load_runtime_modules()
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        messages = []

        def collect_info(message):
            messages.append(str(message))
            print(message)

        processed_ref_audio, processed_ref_text = preprocess_ref_audio_text(
            ref_audio_path,
            ref_text,
            show_info=collect_info,
        )
        
        audio, sr, spect = infer_process(
            ref_audio=processed_ref_audio,
            ref_text=processed_ref_text,
            gen_text=input_text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos",
            nfe_step=settings["nfe_step"],
            cfg_strength=settings["cfg_strength"],
            speed=settings["speed"],
            device=device,
            show_info=collect_info,
        )

        if audio is None:
            raise RuntimeError("F5-TTS returned empty audio. Please check the input text and reference audio.")
        audio, peak_before_postprocess, normalized_audio = _postprocess_generated_audio(audio)
        
        print("✅ Voice cloning completed successfully.")
        warning_lines = "\n".join([f"- {warning}" for warning in warnings]) or "- No input warnings."
        postprocess_message = (
            f"normalized to 0.95 peak headroom (previous peak {peak_before_postprocess:.3f})"
            if normalized_audio
            else f"no normalization needed (peak {peak_before_postprocess:.3f})"
        )
        status = (
            "### Generation Status\n"
            "- Completed with F5-TTS real inference.\n"
            f"- Selected model: `{model_choice}`.\n"
            f"- Active engine: {'fine-tuned checkpoint' if runtime_state.get('using_finetuned') else 'base zero-shot model'}.\n"
            f"- Active model key: `{runtime_state.get('active_model_key', 'base')}`.\n"
            f"- Quality preset: `{quality_preset}` ({settings['description']})\n"
            f"- Settings: `nfe_step={settings['nfe_step']}`, `cfg_strength={settings['cfg_strength']}`, `speed={settings['speed']}`.\n"
            f"- Reference preprocessing: `{Path(processed_ref_audio).name}`.\n"
            f"- Audio post-processing: {postprocess_message}.\n"
            "\n"
            "**Input warnings**\n"
            f"{warning_lines}"
        )
        return (sr, audio), status
        
    except ValueError as e:
        error_msg = str(e)
        print(f"\n⚠️ Generation blocked: {error_msg}")
        raise gr.Error(error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Synthesis error: {error_msg}")
        traceback.print_exc()
        raise gr.Error(f"System error: {error_msg}")

# =============================================================================
# GRADIO UI
# =============================================================================

def create_ui():
    """Create the Gradio web UI."""
    with gr.Blocks(title=PROJECT_TITLE) as app:
        mode_label = (
            f"🧪 Local Preview Mode - macOS `{LOCAL_PREVIEW_VOICE}` system TTS, not voice cloning"
            if MOCK_MODE
            else (
                "🚀 Fine-Tuned Voice Cloning Mode"
                if runtime_state.get("using_finetuned")
                else "🚀 Base Zero-Shot Mode - no fine-tuned checkpoint is loaded"
            )
        )
        gr.Markdown(
            f"""
            # 🎙️ {PROJECT_TITLE}
            ### Generate personalized Vietnamese speech for narration, learning, and digital content.
            
            **Current mode:** {mode_label}
            """
        )
        gr.Markdown(build_compact_status_markdown())

        with gr.Accordion("Usage Guide, Model Functions, and Data Resources", open=False):
            readiness_status = gr.Markdown(build_project_readiness_markdown())
            create_dirs_btn = gr.Button("Prepare Project Folders", variant="secondary")
            create_dirs_status = gr.Markdown("")
            create_dirs_btn.click(
                fn=create_project_dirs_and_status,
                inputs=[],
                outputs=[readiness_status, create_dirs_status],
            )
            gr.Markdown(build_usage_guide_markdown())
            training_choices = build_training_plan_choices()
            if training_choices:
                training_plan = gr.Dropdown(
                    choices=training_choices,
                    value=training_choices[0],
                    label="Training Variant Plan",
                    info="These variants are training plans. They become selectable for generation only after a checkpoint exists.",
                )
                training_plan_info = gr.Markdown(build_training_choice_details(training_choices[0]))
                training_plan.change(
                    fn=build_training_choice_details,
                    inputs=[training_plan],
                    outputs=[training_plan_info],
                )
            data_resource = gr.Dropdown(
                choices=[resource["name"] for resource in OPEN_DATA_RESOURCES],
                value=OPEN_DATA_RESOURCES[0]["name"],
                label="Training/Evaluation Resource",
                info="Choose a resource to see how it can help this project.",
            )
            data_resource_info = gr.Markdown(data_resource_details(OPEN_DATA_RESOURCES[0]["name"]))
            data_resource.change(
                fn=data_resource_details,
                inputs=[data_resource],
                outputs=[data_resource_info],
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Model")
                generation_model_choices = build_model_choices()
                default_model_choice = AUTO_MODEL_CHOICE
                model_choice = gr.Dropdown(
                    choices=generation_model_choices,
                    value=default_model_choice,
                    label="Model Variant",
                    info="All variants are visible. Variants marked 'needs training' must be trained before they can generate reliable cloned speech.",
                )
                selected_model_info = gr.Markdown(build_model_choice_details(AUTO_MODEL_CHOICE))
                model_choice.change(
                    fn=build_model_choice_details,
                    inputs=[model_choice],
                    outputs=[selected_model_info],
                )

                gr.Markdown("### 2. Reference Voice")
                ref_audio_input = gr.Audio(
                    label="Reference Audio (.wav, 3-10 seconds)", 
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                ref_text_input = gr.Textbox(
                    label="Reference Audio Transcript",
                    placeholder="Enter the exact sentence spoken in the reference audio. This must match the audio.",
                    lines=2
                )
                
                gr.Markdown("### 3. Text To Generate")
                input_text = gr.Textbox(
                    label="Vietnamese Text",
                    placeholder="The generated audio will speak exactly this text.",
                    lines=5
                )
                quality_preset = gr.Radio(
                    choices=list(GENERATION_PRESETS.keys()),
                    value="Higher clarity (slower)",
                    label="Generation Quality",
                    info="Use Higher clarity when the output is unclear. It is slower, especially on CPU.",
                )
                submit_btn = gr.Button("🚀 Generate Voice", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 4. Output")
                output_audio = gr.Audio(
                    label="Generated Audio", 
                    interactive=False,
                    autoplay=True
                )
                generation_status = gr.Markdown("Generation status will appear here after you click Generate Voice.")
                
        submit_btn.click(
            fn=synthesize_audio,
            inputs=[model_choice, ref_audio_input, ref_text_input, input_text, quality_preset],
            outputs=[output_audio, generation_status],
            api_name="synthesize"
        )
        
    return app

if __name__ == "__main__":
    init_system()
    
    app = create_ui()
    
    print("\n🌐 Starting Gradio web app...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=SHARE_GRADIO)
