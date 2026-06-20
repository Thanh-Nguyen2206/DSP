"""
=============================================================================
F5-TTS Vietnamese Fine-tuning - Environment Verification Script

Mục đích: Kiểm tra toàn bộ thư viện đã cài đúng + GPU khả dụng
Chạy  : python verify_env.py
Output : Bảng trạng thái ✅/❌ cho từng dependency

Chạy script này SAU KHI hoàn tất setup.sh
=============================================================================
"""

import sys
import importlib
import importlib.metadata as importlib_metadata
from typing import Optional, Tuple

try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # packaging thường có sẵn, nhưng verify không nên sập nếu thiếu.
    InvalidVersion = None
    Version = None


def check_package(
    package_name: str,
    import_name: Optional[str] = None,
    min_version: Optional[str] = None,
    dist_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Kiểm tra một package đã được cài đặt chưa.
    
    Args:
        package_name: Tên hiển thị của package
        import_name: Tên khi import (nếu khác package_name)
        min_version: Version tối thiểu cần có
        dist_name: Tên distribution trên pip nếu khác import_name
    
    Returns:
        (success, version_or_error)
    """
    name = import_name or package_name
    try:
        module = importlib.import_module(name)
        # Lấy version nếu có. Ưu tiên metadata vì một số module không có __version__.
        try:
            version = importlib_metadata.version(dist_name or name)
        except importlib_metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "installed (no version info)")

        if min_version and Version is not None:
            try:
                if Version(str(version)) < Version(min_version):
                    return False, f"{version} < required {min_version}"
            except (InvalidVersion, TypeError):
                return False, f"cannot verify version: {version}"

        return True, version
    except ImportError as e:
        return False, str(e)


def check_gpu() -> Tuple[bool, str]:
    """
    Kiểm tra GPU NVIDIA có khả dụng qua PyTorch hay không.
    In ra: tên GPU, VRAM, CUDA version
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_ver = torch.version.cuda
            return True, f"{gpu_name} | {vram_gb:.1f}GB VRAM | CUDA {cuda_ver}"
        else:
            return False, "CUDA không khả dụng (torch.cuda.is_available() = False)"
    except ImportError:
        return False, "PyTorch chưa được cài đặt"


def check_ffmpeg() -> Tuple[bool, str]:
    """
    Kiểm tra ffmpeg đã cài trong hệ thống chưa.
    Cần thiết cho pydub để xử lý audio formats (mp3, m4a, etc.)
    """
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return True, f"Found at: {path}"
    else:
        return False, "ffmpeg chưa cài. Chạy: sudo apt install ffmpeg"


def main():
    """
    Chạy toàn bộ kiểm tra và in kết quả dạng bảng.
    Exit code 0 nếu tất cả pass, 1 nếu có lỗi.
    """
    print("=" * 65)
    print(" F5-TTS Vietnamese - Environment Verification")
    print(f" Python: {sys.version}")
    print("=" * 65)
    print()
    
    # =========================================================================
    # Danh sách các package cần kiểm tra
    # Format: (tên hiển thị, tên import, mức độ quan trọng, min_version, dist_name)
    #   - CRITICAL: Bắt buộc phải có, thiếu = không chạy được
    #   - REQUIRED: Cần thiết, thiếu = thiếu tính năng quan trọng
    #   - OPTIONAL: Có thì tốt, không có vẫn chạy cơ bản được
    # =========================================================================
    checks = [
        # === Core ML Framework ===
        ("PyTorch",        "torch",         "CRITICAL", "2.3.0",  "torch"),
        ("torchaudio",     "torchaudio",    "CRITICAL", "2.3.0",  "torchaudio"),
        
        # === F5-TTS ===
        ("F5-TTS",         "f5_tts",        "CRITICAL", "0.3.4",  "f5-tts"),
        
        # === HuggingFace ===
        ("transformers",   "transformers",  "CRITICAL", "4.40.0", "transformers"),
        ("accelerate",     "accelerate",    "CRITICAL", "0.30.0", "accelerate"),
        ("datasets",       "datasets",      "REQUIRED", "2.19.0", "datasets"),
        ("safetensors",    "safetensors",   "REQUIRED", "0.4.3",  "safetensors"),
        
        # === Audio Processing ===
        ("librosa",        "librosa",       "CRITICAL", "0.10.2", "librosa"),
        ("soundfile",      "soundfile",     "CRITICAL", "0.12.1", "soundfile"),
        ("pydub",          "pydub",         "REQUIRED", "0.25.1", "pydub"),
        
        # === Vietnamese NLP ===
        ("underthesea",    "underthesea",   "REQUIRED", "6.8.0",  "underthesea"),
        
        # === Web UI ===
        ("Gradio",         "gradio",        "REQUIRED", "4.31.0", "gradio"),
        
        # === Evaluation ===
        ("resemblyzer",    "resemblyzer",   "REQUIRED", "0.1.3",  "resemblyzer"),
        ("jiwer",          "jiwer",         "OPTIONAL", "3.0.3",  "jiwer"),
        
        # === Utilities ===
        ("matplotlib",     "matplotlib",    "REQUIRED", "3.8.0",  "matplotlib"),
        ("numpy",          "numpy",         "CRITICAL", "1.24.0", "numpy"),
        ("scipy",          "scipy",         "REQUIRED", "1.12.0", "scipy"),
        ("tqdm",           "tqdm",          "REQUIRED", "4.66.0", "tqdm"),
        ("PyYAML",         "yaml",          "REQUIRED", "6.0.1",  "PyYAML"),
        ("tensorboard",    "tensorboard",   "OPTIONAL", "2.16.0", "tensorboard"),
    ]
    
    # =========================================================================
    # Chạy kiểm tra
    # =========================================================================
    all_passed = True
    critical_failed = False
    results = []
    
    for display_name, import_name, level, min_version, dist_name in checks:
        ok, info = check_package(display_name, import_name, min_version, dist_name)
        results.append((display_name, ok, info, level))
        if not ok:
            all_passed = False
            if level == "CRITICAL":
                critical_failed = True
    
    # In kết quả dạng bảng
    print(f"{'Package':<18} {'Status':<4} {'Level':<10} {'Info'}")
    print("-" * 65)
    
    for name, ok, info, level in results:
        icon = "✅" if ok else "❌"
        # Truncate info nếu quá dài
        info_short = info[:35] + "..." if len(info) > 38 else info
        print(f"{name:<18} {icon:<4} {level:<10} {info_short}")
    
    # =========================================================================
    # Kiểm tra GPU
    # =========================================================================
    print()
    print("-" * 65)
    print("GPU Check:")
    gpu_ok, gpu_info = check_gpu()
    gpu_icon = "✅" if gpu_ok else "⚠️ "
    print(f"  {gpu_icon} {gpu_info}")
    
    # Cảnh báo nếu không phải T4 hoặc VRAM < 15GB
    if gpu_ok:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 14.0:
            print(f"  ⚠️  VRAM ({vram_gb:.1f}GB) thấp hơn khuyến nghị (15GB+).")
            print(f"      Cần giảm batch_size và tăng gradient_accumulation_steps.")
    
    # =========================================================================
    # Kiểm tra ffmpeg
    # =========================================================================
    print()
    print("System Dependencies:")
    ff_ok, ff_info = check_ffmpeg()
    ff_icon = "✅" if ff_ok else "⚠️ "
    print(f"  {ff_icon} ffmpeg: {ff_info}")
    
    # =========================================================================
    # Kiểm tra Mixed Precision (FP16) support
    # Bắt buộc cho T4 để tiết kiệm VRAM
    # =========================================================================
    print()
    print("Training Capabilities:")
    try:
        import torch
        if torch.cuda.is_available():
            # T4 hỗ trợ FP16 (compute capability 7.5)
            capability = torch.cuda.get_device_capability(0)
            cap_str = f"{capability[0]}.{capability[1]}"
            
            # FP16: cần compute capability >= 7.0
            fp16_ok = capability[0] >= 7
            fp16_icon = "✅" if fp16_ok else "❌"
            print(f"  {fp16_icon} FP16 Mixed Precision: "
                  f"Compute Capability {cap_str} "
                  f"{'(supported)' if fp16_ok else '(NOT supported)'}")
            
            # BF16: cần compute capability >= 8.0 (Ampere+)
            # T4 là Turing (7.5) nên KHÔNG hỗ trợ BF16
            bf16_ok = capability[0] >= 8
            bf16_icon = "✅" if bf16_ok else "ℹ️ "
            print(f"  {bf16_icon} BF16 Mixed Precision: "
                  f"{'(supported)' if bf16_ok else '(not supported - dùng FP16 thay thế)'}")
        else:
            print("  ⚠️  Không kiểm tra được (GPU không khả dụng)")
    except ImportError:
        print("  ❌ PyTorch chưa cài, không kiểm tra được")
    
    # =========================================================================
    # Tổng kết
    # =========================================================================
    print()
    print("=" * 65)
    if critical_failed:
        print(" ❌ CÓ PACKAGE CRITICAL BỊ THIẾU!")
        print("    Chạy lại: bash setup.sh")
        print("=" * 65)
        sys.exit(1)
    elif not all_passed:
        print(" ⚠️  Một số package optional/required bị thiếu.")
        print("    Hệ thống vẫn chạy được cơ bản, nhưng nên cài đầy đủ.")
        print("    Chạy: pip install -r requirements.txt")
        print("=" * 65)
        sys.exit(0)
    else:
        print(" ✅ TẤT CẢ KIỂM TRA ĐỀU PASS!")
        print("    Sẵn sàng cho Phase 2: Data Collection & Preprocessing")
        print("=" * 65)
        sys.exit(0)


if __name__ == "__main__":
    main()
