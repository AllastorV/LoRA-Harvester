"""
FrameUpscaler — Real-ESRGAN wrapper for LoRA-Harvester.

Lazy-loads model on first use so startup is never slowed by missing deps.
Fully graceful: if realesrgan / basicsr is not installed, is_available()
returns False and upscale() transparently returns the original frame.

Requires (optional — feature disabled if missing):
    pip install realesrgan>=0.3.0 basicsr>=1.4.2
    pip install gfpgan>=1.3.8   # only if face_enhance=True

Usage:
    from src.core.upscaler import FrameUpscaler
    u = FrameUpscaler(model_name="RealESRGAN_x4plus_anime_6B")
    enhanced = u.upscale(bgr_frame)   # returns ndarray same dtype
"""

from __future__ import annotations
import logging
import sys
import types
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_torchvision_compat() -> None:
    """Shim ``torchvision.transforms.functional_tensor`` for basicsr.

    basicsr<=1.4.2 (pulled in by realesrgan) does
    ``from torchvision.transforms.functional_tensor import rgb_to_grayscale``.
    That submodule was REMOVED in torchvision>=0.17, so the import explodes
    with ModuleNotFoundError and the whole upscaler appears "deps missing".

    The function still exists at ``torchvision.transforms.functional`` — we
    register a tiny alias module so the old import path resolves. Lives in
    repo code so it survives realesrgan/basicsr/torchvision reinstalls
    (patching site-packages would not).
    """
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return  # genuinely present (older torchvision) — nothing to do
    except Exception:
        pass
    try:
        import torchvision.transforms.functional as _F
    except Exception:
        return  # torchvision itself missing — let the real import error surface
    shim = types.ModuleType(name)
    shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules[name] = shim
    logger.debug("Installed torchvision.transforms.functional_tensor compat shim")


class FrameUpscaler:
    """
    Wraps Real-ESRGAN (and optionally GFPGAN face-restore) with lazy loading.
    Thread-safety: each worker should create its own instance (model not shared).
    """

    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus_anime_6B",
        tile: int = 0,
        tile_pad: int = 10,
        use_gpu: bool = True,
        face_enhance: bool = False,
        denoise_strength: float = 0.5,
    ):
        self.model_name = model_name
        self.tile = tile
        self.tile_pad = tile_pad
        self.use_gpu = use_gpu
        self.face_enhance = face_enhance
        self.denoise_strength = denoise_strength

        self._upsampler = None
        self._face_enhancer = None
        self._available: Optional[bool] = None  # None = not yet checked
        self._load_error: str = ""
        self._scale: int = 4  # updated after load

    # ──────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the model can be used (deps present, weights exist)."""
        if self._available is None:
            self._ensure_loaded()
        return bool(self._available)

    def get_scale(self) -> int:
        """Return the upscale factor (2 or 4). Call after is_available()."""
        return self._scale

    def upscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Upscale a BGR (3ch) or BGRA (4ch, transparent PNG) frame.
        Alpha is split off, the colour is upscaled, then the alpha is resized
        back on so transparent PNGs stay transparent (no green/black matte).
        Returns the enhanced frame on success, or the original frame on any error.
        """
        if not self.is_available():
            return frame_bgr

        import cv2

        # Split alpha so the model only ever sees 3-channel colour.
        alpha = None
        color = frame_bgr
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            color = frame_bgr[:, :, :3]
            alpha = frame_bgr[:, :, 3]
        elif frame_bgr.ndim == 2:
            color = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        try:
            if self.face_enhance and self._face_enhancer is not None:
                _, _, output = self._face_enhancer.enhance(
                    color, has_aligned=False, only_center_face=False,
                    paste_back=True
                )
            else:
                output, _ = self._upsampler.enhance(color, outscale=self._scale)

            # Re-attach alpha, scaled to the upscaled colour size.
            if alpha is not None and output is not None:
                out_h, out_w = output.shape[:2]
                if (alpha.shape[0], alpha.shape[1]) != (out_h, out_w):
                    alpha = cv2.resize(alpha, (out_w, out_h),
                                       interpolation=cv2.INTER_LINEAR)
                output = cv2.merge([output[:, :, 0], output[:, :, 1],
                                    output[:, :, 2], alpha])
            return output
        except RuntimeError as e:
            err = str(e)
            if 'CUDA out of memory' in err or 'out of memory' in err.lower():
                # Retry with tiling (alpha already stripped into `color`)
                retry = self._retry_with_tile(color)
                if retry is not None:
                    if alpha is not None:
                        out_h, out_w = retry.shape[:2]
                        if (alpha.shape[0], alpha.shape[1]) != (out_h, out_w):
                            alpha = cv2.resize(alpha, (out_w, out_h),
                                               interpolation=cv2.INTER_LINEAR)
                        retry = cv2.merge([retry[:, :, 0], retry[:, :, 1],
                                           retry[:, :, 2], alpha])
                    return retry
                logger.warning("Upscaler OOM even with tile=256 — returning original frame")
            else:
                logger.warning("Upscaler error: %s — returning original frame", e)
            return frame_bgr
        except Exception as e:
            logger.warning("Upscaler unexpected error: %s — returning original frame", e)
            return frame_bgr

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _retry_with_tile(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Attempt upscale with tile=256 after an OOM."""
        try:
            # RealESRGANer stores the tile param as `tile_size`
            self._upsampler.tile_size = 256
            output, _ = self._upsampler.enhance(frame_bgr, outscale=self._scale)
            logger.info("Upscaler: OOM retry with tile=256 succeeded")
            return output
        except Exception:
            return None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model. Sets self._available True/False exactly once."""
        try:
            self._load_model()
            self._available = True
        except ImportError as e:
            self._available = False
            self._load_error = str(e)
            logger.warning(
                "Real-ESRGAN not available (deps missing). "
                "Install: pip install realesrgan basicsr\n"
                "Details: %s", e
            )
        except Exception as e:
            self._available = False
            self._load_error = str(e)
            logger.warning("FrameUpscaler failed to load: %s", e)

    def _load_model(self) -> None:
        """
        Actually import and initialise RealESRGANer.
        Raises ImportError if realesrgan/basicsr not installed.
        Raises RuntimeError if model weights missing and download fails.
        """
        import torch

        # basicsr (via realesrgan) needs the removed functional_tensor module.
        _ensure_torchvision_compat()

        from src.core.upscale_models import resolve
        cfg = resolve(self.model_name)
        self._scale = cfg.get('scale', 4)
        arch = cfg.get('arch', 'RRDBNet')
        num_feat = cfg.get('num_feat', 64)
        num_block = cfg.get('num_block', 23)
        num_grow_ch = cfg.get('num_grow_ch', 32)
        model_path = cfg['path']

        # Choose device
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            # FP16 only on Turing+ (capability >= 7.0)
            half = torch.cuda.get_device_capability()[0] >= 7

            # Auto-select tile size based on free VRAM if user left tile=0.
            # Real-ESRGAN x4 on a 512-tile uses ~2.5 GB; x4 tileless needs ~6+ GB.
            # This prevents silent OOM crashes on mid-range GPUs.
            if self.tile == 0:
                try:
                    free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
                    if free_mb < 3000:        # < 3 GB  → small tile, safe for 4 GB cards
                        self.tile = 256
                        logger.info("Auto tile=256 (free VRAM %.0f MB < 3 GB)", free_mb)
                    elif free_mb < 5000:      # 3–5 GB → medium tile
                        self.tile = 512
                        logger.info("Auto tile=512 (free VRAM %.0f MB < 5 GB)", free_mb)
                    # else ≥ 5 GB free → keep tile=0 (no tiling, fastest)
                except Exception:
                    pass  # Non-fatal; proceed with user's tile value
        else:
            device = torch.device('cpu')
            half = False
            if self.use_gpu:
                logger.warning("Upscaler: CUDA not available, running on CPU (slow)")
            # On CPU, always tile to avoid exhausting system RAM
            if self.tile == 0:
                self.tile = 512
                logger.info("Auto tile=512 for CPU inference (RAM safety)")

        # Build architecture
        if arch == 'RRDBNet':
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=num_feat, num_block=num_block,
                num_grow_ch=num_grow_ch, scale=self._scale
            )
        elif arch == 'SRVGGNetCompact':
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3,
                num_feat=num_feat, num_conv=num_block,
                upscale=self._scale, act_type='prelu'
            )
        else:
            raise ValueError(f"Unknown upscale arch: {arch}")

        from realesrgan import RealESRGANer
        self._upsampler = RealESRGANer(
            scale=self._scale,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=half,
            device=device,
        )

        # Optional GFPGAN face restoration
        if self.face_enhance:
            try:
                from gfpgan import GFPGANer
                from src.core.model_paths import MODELS_DIR
                gfpgan_model = MODELS_DIR / "gfpgan" / "GFPGANv1.3.pth"
                if not gfpgan_model.exists():
                    logger.warning(
                        "GFPGAN weights not found at %s — face enhancement disabled. "
                        "Download from https://github.com/TencentARC/GFPGAN/releases",
                        gfpgan_model
                    )
                    self.face_enhance = False
                else:
                    self._face_enhancer = GFPGANer(
                        model_path=str(gfpgan_model),
                        upscale=self._scale,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self._upsampler,
                    )
                    logger.info("GFPGAN face enhancer loaded")
            except ImportError:
                logger.warning(
                    "gfpgan not installed — face enhancement disabled. "
                    "Install: pip install gfpgan"
                )
                self.face_enhance = False

        logger.info(
            "FrameUpscaler ready: model=%s scale=%d arch=%s device=%s half=%s",
            self.model_name, self._scale, arch, device, half
        )
