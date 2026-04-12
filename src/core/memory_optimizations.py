"""
Memory optimization utilities for LoRA-Harvester.

Provides xformers and Flash Attention 2 integration for reduced VRAM
usage during model inference (Florence-2, SAM 2, WD14).
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _check_xformers() -> bool:
    """Return True if xformers is installed and functional."""
    try:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401
        return True
    except ImportError:
        return False


def _check_flash_attn() -> bool:
    """Return True if flash-attn v2 is installed and CUDA is available."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_backends() -> dict:
    """
    Probe which attention backends are available.

    Returns:
        Dict with keys 'xformers', 'flash_attn', 'sdpa' and bool values.
    """
    return {
        'xformers': _check_xformers(),
        'flash_attn': _check_flash_attn(),
        'sdpa': hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
    }


def enable_xformers(model: torch.nn.Module) -> bool:
    """
    Enable xformers memory-efficient attention on *model* if possible.

    Works with HuggingFace diffusers/transformers models that expose
    ``enable_xformers_memory_efficient_attention()``.

    Returns:
        True if xformers was enabled, False otherwise.
    """
    if not _check_xformers():
        logger.debug("xformers not available — skipping")
        return False

    if hasattr(model, 'enable_xformers_memory_efficient_attention'):
        try:
            model.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
            return True
        except Exception as e:
            logger.warning("xformers enable failed: %s", e)
            return False

    logger.debug("Model does not support xformers API")
    return False


def apply_torch_optimizations(device: str = 'cuda') -> None:
    """
    Apply global PyTorch performance flags.

    Called once at startup to set backend math defaults.
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return

    # Enable TF32 for faster matmuls on Ampere+ GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN auto-tuner — picks fastest convolution algorithm.
    torch.backends.cudnn.benchmark = True

    logger.info(
        "PyTorch optimizations applied (TF32=%s, cuDNN benchmark=%s)",
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.benchmark,
    )


def get_optimal_dtype(device: str = 'cuda') -> torch.dtype:
    """
    Return the best inference dtype for the current hardware.

    - BF16 on Ampere+ GPUs (compute capability >= 8.0)
    - FP16 on older CUDA GPUs
    - FP32 on CPU
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return torch.float32

    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8:
        return torch.bfloat16
    return torch.float16


def estimate_vram_mb() -> Optional[float]:
    """Return free VRAM in MB, or None if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 * 1024)
