"""
Advanced Captioning Module for LoRA-Harvester
WD14/Danbooru tagging with comprehensive tag management and filtering.
For natural-language captions, see Florence2Captioner.
"""

# Ensure CUDA DLLs are discoverable for onnxruntime-gpu on Windows
import sys
import os
if sys.platform == 'win32':
    _cuda_bin_dirs = []
    # Check CUDA_PATH env variable first
    _cuda_env = os.environ.get('CUDA_PATH', '')
    if _cuda_env:
        _cuda_bin_dirs.append(os.path.join(_cuda_env, 'bin'))
    # Check common install locations
    for _v in ('v12.1', 'v12.2', 'v12.3', 'v12.4', 'v12.5', 'v12.6'):
        _cuda_bin_dirs.append(rf'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{_v}\bin')
    for _d in _cuda_bin_dirs:
        if os.path.isdir(_d):
            # add_dll_directory is required on Windows for DLL search (Python 3.8+)
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass
            # Also update PATH as fallback
            if _d not in os.environ.get('PATH', ''):
                os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')

# Import onnxruntime FIRST before any torch import to avoid DLL conflicts
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

import json
import logging
import threading
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
import cv2

try:
    from PIL import Image as _PIL_Image
except ImportError:
    _PIL_Image = None

logger = logging.getLogger(__name__)


@dataclass
class TagSettings:
    """
    Comprehensive tag configuration settings
    """
    # Trigger word (added at the beginning)
    trigger_word: str = ""
    
    # Tag limits
    max_tags: int = 30
    min_confidence: float = 0.35
    
    # Negative tags (to exclude)
    negative_tags: List[str] = field(default_factory=list)
    
    # Priority tags (always include if detected)
    priority_tags: List[str] = field(default_factory=list)
    
    # Tag replacements (old -> new)
    tag_replacements: Dict[str, str] = field(default_factory=dict)
    
    # Tag categories to include/exclude
    include_categories: List[str] = field(default_factory=lambda: [
        'general', 'character', 'copyright', 'artist', 'meta'
    ])
    
    # Formatting
    tag_separator: str = ", "
    use_underscores: bool = False  # tag_name vs tag name
    use_escape_parentheses: bool = True  # \( \) for SD
    lowercase: bool = True
    
    # Prefix/Suffix for final caption
    caption_prefix: str = ""
    caption_suffix: str = ""
    
    # Keep character/series names
    keep_character_tags: bool = True
    keep_series_tags: bool = True
    
    # Quality/Rating tags
    include_quality_tags: bool = False
    include_rating_tags: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TagSettings':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CaptionResult:
    """Result from captioning"""
    wd14_tags: List[Tuple[str, float]] = field(default_factory=list)
    final_caption: str = ""
    tag_count: int = 0
    filtered_tags: List[str] = field(default_factory=list)
    

class WD14Tagger:
    """
    WD14/Danbooru-style tagger
    Generates anime/illustration tags with confidence scores
    Uses SmilingWolf's WD14 models
    """
    
    # Tag categories from WD14
    CATEGORIES = {
        0: 'general',
        1: 'artist', 
        2: 'copyright',
        3: 'character',
        4: 'meta'
    }
    
    # Rating tags
    RATING_TAGS = ['general', 'sensitive', 'questionable', 'explicit']
    
    def __init__(self,
                 model_name: str = "wd-v1-4-vit-tagger-v2",
                 device: str = None):
        """
        Args:
            model_name: WD14 model variant:
                - "wd-v1-4-vit-tagger-v2" (recommended)
                - "wd-v1-4-convnext-tagger-v2"
                - "wd-v1-4-swinv2-tagger-v2"
            device: cuda/cpu
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tags = None
        self.tag_categories = None
        self._input_name = None
        self._output_name = None
        self._loaded = False
        self._load_error = None  # Track loading errors
        self._load_lock = threading.Lock()

        logger.info("WD14 Tagger initialized (lazy loading) - model=%s device=%s", model_name, self.device)

    def _load_model(self):
        """Lazy load WD14 model (thread-safe)"""
        if self._loaded:
            return
        if self._load_error:
            raise self._load_error
        with self._load_lock:
            if self._loaded:  # Double-checked locking
                return
            if self._load_error:
                raise self._load_error

            logger.info("Loading WD14 tagger...")

            try:
                # Try using timm + huggingface_hub
                from huggingface_hub import hf_hub_download
                import pandas as pd

                # Use globally imported ort
                if not ONNX_AVAILABLE:
                    raise ImportError("onnxruntime not available")

                # Handle both full repo ID and short model name
                if "/" in self.model_name:
                    repo_id = self.model_name
                else:
                    repo_id = f"SmilingWolf/{self.model_name}"

                # Download model files into project-local models/wd14/
                from src.core.model_paths import WD14_DIR, ensure_dirs
                ensure_dirs()
                _cache_dir = str(WD14_DIR)

                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="model.onnx",
                    cache_dir=_cache_dir,
                )

                tags_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="selected_tags.csv",
                    cache_dir=_cache_dir,
                )

                # Load ONNX model with optimization
                available = ort.get_available_providers()
                if self.device == 'cuda' and 'CUDAExecutionProvider' in available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                elif self.device == 'cuda' and 'CUDAExecutionProvider' not in available:
                    logger.warning("CUDAExecutionProvider not available — install onnxruntime-gpu for GPU acceleration")
                    providers = ['CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 0  # auto
                self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
                active_provider = self.model.get_providers()[0]
                logger.info("WD14 ONNX provider: %s (requested: %s)", active_provider, providers[0])

                # Cache input/output names
                self._input_name = self.model.get_inputs()[0].name
                self._output_name = self.model.get_outputs()[0].name

                # Load tags
                tags_df = pd.read_csv(tags_path)
                self.tags = tags_df['name'].tolist()

                # Get category for each tag
                if 'category' in tags_df.columns:
                    self.tag_categories = tags_df['category'].tolist()
                else:
                    self.tag_categories = [0] * len(self.tags)

                # Validate model output vs tag count
                model_output_shape = self.model.get_outputs()[0].shape
                logger.info("WD14 output shape: %s, tag count: %d", model_output_shape, len(self.tags))
                if len(model_output_shape) >= 2:
                    output_dim = model_output_shape[-1]
                    if isinstance(output_dim, int) and output_dim != len(self.tags):
                        logger.warning("WD14 model output dim (%d) != tag count (%d) — predictions may be misaligned",
                                       output_dim, len(self.tags))

                self._loaded = True
                logger.info("WD14 model loaded successfully (%d tags)", len(self.tags))

            except ImportError as e:
                self._load_error = ImportError(f"Missing package: {e}. Run: pip install onnxruntime pandas huggingface_hub")
                raise self._load_error
            except Exception as e:
                self._load_error = RuntimeError(f"Failed to load WD14: {e}")
                logger.error("WD14 load error: %s", e)
                raise self._load_error

    def cleanup(self):
        """Release ONNX model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        self.tags = None
        self.tag_categories = None
        self._input_name = None
        self._output_name = None
        self._loaded = False
        self._load_error = None

    def _preprocess(self, image: Union[np.ndarray, object]) -> np.ndarray:
        """Preprocess image for WD14 (matches SmilingWolf reference implementation)"""
        target_size = 448

        # Convert to BGR numpy if PIL
        if not isinstance(image, np.ndarray):
            image = np.array(image.convert('RGB'))[:, :, ::-1]  # RGB PIL -> BGR numpy

        # Handle alpha channel and grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # Composite alpha onto white background
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            image = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

        # Pad to square (white background) then resize
        h, w = image.shape[:2]
        max_dim = max(h, w, target_size)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        padded = cv2.copyMakeBorder(
            image, pad_h, max_dim - h - pad_h, pad_w, max_dim - w - pad_w,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Convert to float32, add batch dimension (BGR, 0-255 range)
        img_array = resized.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    
    @staticmethod
    def _ensure_sigmoid(predictions: np.ndarray) -> np.ndarray:
        """Apply sigmoid if model outputs raw logits instead of probabilities."""
        max_val = float(np.max(predictions))
        min_val = float(np.min(predictions))
        # If values are outside [0, 1] range, they are logits — apply sigmoid
        if max_val > 1.0 or min_val < 0.0:
            logger.info("WD14 output range [%.2f, %.2f] → applying sigmoid", min_val, max_val)
            predictions = 1.0 / (1.0 + np.exp(-predictions.astype(np.float64))).astype(np.float32)
        return predictions

    def predict(self,
                image: Union[np.ndarray, object],
                threshold: float = 0.35) -> List[Tuple[str, float, int]]:
        """
        Predict tags for image

        Args:
            image: Input image
            threshold: Minimum confidence threshold

        Returns:
            List of (tag_name, confidence, category_id) tuples
        """
        if not self._loaded:
            self._load_model()

        # Preprocess
        img_input = self._preprocess(image)

        # Run inference (use cached names)
        predictions = self.model.run(
            [self._output_name], {self._input_name: img_input}
        )[0][0]
        del img_input

        # Apply sigmoid if model outputs logits
        predictions = self._ensure_sigmoid(predictions)

        # Filter by threshold and sort
        results = []
        for idx, conf in enumerate(predictions):
            if conf >= threshold and idx < len(self.tags):
                tag = self.tags[idx]
                category = self.tag_categories[idx] if idx < len(self.tag_categories) else 0
                results.append((tag, float(conf), category))

        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            logger.warning("WD14 predict: 0 tags above threshold %.2f (max confidence: %.4f)",
                           threshold, float(np.max(predictions)) if len(predictions) > 0 else 0.0)

        return results

    def predict_batch(self,
                      images: List[Union[np.ndarray, object]],
                      threshold: float = 0.35) -> List[List[Tuple[str, float, int]]]:
        """
        Predict tags for multiple images (processes one at a time for ONNX compatibility)

        Args:
            images: List of input images
            threshold: Minimum confidence threshold

        Returns:
            List of tag results for each image
        """
        if not self._loaded:
            self._load_model()

        if not images:
            return []

        # Process images one at a time (ONNX model expects batch_size=1)
        all_results = []
        for img in images:
            preprocessed = self._preprocess(img)
            predictions = self.model.run(
                [self._output_name], {self._input_name: preprocessed}
            )[0][0]
            del preprocessed

            # Apply sigmoid if model outputs logits
            predictions = self._ensure_sigmoid(predictions)

            results = []
            for idx, conf in enumerate(predictions):
                if conf >= threshold and idx < len(self.tags):
                    tag = self.tags[idx]
                    category = self.tag_categories[idx] if idx < len(self.tag_categories) else 0
                    results.append((tag, float(conf), category))
            results.sort(key=lambda x: x[1], reverse=True)
            all_results.append(results)

        return all_results


class AdvancedCaptioner:
    """
    WD14 captioner with advanced tag management.
    For natural-language captions, use Florence2Captioner.
    """

    def __init__(self,
                 wd14_model: str = "wd-v1-4-vit-tagger-v2",
                 tag_settings: TagSettings = None,
                 enable_wd14: bool = True):
        self.tag_settings = tag_settings or TagSettings()
        self.enable_wd14 = enable_wd14

        self.wd14 = WD14Tagger(wd14_model) if enable_wd14 else None

        self.stats = {
            'images_processed': 0,
            'tags_generated': 0,
            'tags_filtered': 0,
        }

        logger.info(
            "Advanced Captioner initialized - WD14=%s trigger='%s' max_tags=%d neg_tags=%d",
            enable_wd14, self.tag_settings.trigger_word,
            self.tag_settings.max_tags, len(self.tag_settings.negative_tags),
        )

    def cleanup(self):
        """Release all models from memory"""
        if self.wd14:
            self.wd14.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("AdvancedCaptioner cleaned up")

    # ── Constant sets (computed once, shared by all instances) ──────────────
    _QUALITY_TAGS: Set[str] = {
        'masterpiece', 'best quality', 'high quality', 'low quality',
        'worst quality', 'normal quality', 'absurdres', 'highres',
        'very aesthetic', 'aesthetic', 'displeasing', 'very displeasing',
    }
    _RATING_TAGS: Set[str] = {
        'general', 'sensitive', 'questionable', 'explicit', 'safe',
        'rating:general', 'rating:sensitive', 'rating:questionable', 'rating:explicit',
    }
    _CATEGORY_NAMES: Dict[int, str] = {
        0: 'general', 1: 'artist', 2: 'copyright', 3: 'character', 4: 'meta',
    }

    @staticmethod
    def _normalize(tag: str) -> str:
        """Single normalization function used everywhere for consistency."""
        return tag.lower().replace('_', ' ').strip()

    def _format_tag(self, tag: str) -> str:
        """Apply formatting rules (lowercase, underscores, escaping) to a single tag."""
        s = self.tag_settings
        if s.lowercase:
            tag = tag.lower()
        # Apply replacements on both original and normalized form
        norm = self._normalize(tag)
        if tag in s.tag_replacements:
            tag = s.tag_replacements[tag]
        elif norm in s.tag_replacements:
            tag = s.tag_replacements[norm]
        # Space / underscore
        if s.use_underscores:
            tag = tag.replace(' ', '_')
        else:
            tag = tag.replace('_', ' ')
        # Escape parentheses for Stable Diffusion
        if s.use_escape_parentheses:
            tag = tag.replace('(', '\\(').replace(')', '\\)')
        return tag.strip()

    def process_tags(self, raw_tags: List[Tuple[str, float, int]]) -> List[str]:
        """
        Process and filter raw WD14 tags according to TagSettings.

        Pipeline:
          1. Confidence threshold
          2. Rating / quality tag removal (if disabled)
          3. Category gate (with priority-tag override)
          4. Character / copyright gate
          5. Negative tag matching (exact + prefix* + *suffix wildcards)
          6. Format & deduplicate
          7. Inject missing priority tags at the front
          8. Enforce max_tags limit

        Args:
            raw_tags: List of (tag_name, confidence, category_id) from WD14.

        Returns:
            Ordered list of formatted, deduplicated tag strings.
        """
        settings = self.tag_settings

        # ── Pre-compute lookup structures (once per call) ────────────────
        neg_exact: Set[str] = set()
        neg_prefix: List[str] = []
        neg_suffix: List[str] = []
        for neg in settings.negative_tags:
            n = self._normalize(neg)
            if not n:
                continue
            if n.endswith('*') and n.startswith('*'):
                # *pattern* → contains check (stored as substring)
                core = n[1:-1]
                if core:
                    neg_prefix.append(core)   # reuse prefix list for contains
                    neg_suffix.append(core)
            elif n.endswith('*'):
                neg_prefix.append(n[:-1])
            elif n.startswith('*'):
                neg_suffix.append(n[1:])
            else:
                neg_exact.add(n)

        priority_normalized: Set[str] = {
            self._normalize(p) for p in settings.priority_tags if p.strip()
        }

        include_cats = set(settings.include_categories)

        # ── Main filter loop ─────────────────────────────────────────────
        seen: Set[str] = set()          # dedup by normalized form
        processed: List[str] = []
        filtered_count = 0

        for tag_raw, conf, category in raw_tags:
            # 1. Confidence gate
            if conf < settings.min_confidence:
                continue

            norm = self._normalize(tag_raw)

            # 2. Rating tags
            if not settings.include_rating_tags and norm in self._RATING_TAGS:
                filtered_count += 1
                continue

            # 3. Quality tags
            if not settings.include_quality_tags and norm in self._QUALITY_TAGS:
                filtered_count += 1
                continue

            cat_name = self._CATEGORY_NAMES.get(category, 'general')
            is_priority = norm in priority_normalized

            # 4. Category gate (priority tags bypass)
            if cat_name not in include_cats and not is_priority:
                filtered_count += 1
                continue

            # 5. Character / copyright gates
            if cat_name == 'character' and not settings.keep_character_tags and not is_priority:
                filtered_count += 1
                continue
            if cat_name == 'copyright' and not settings.keep_series_tags and not is_priority:
                filtered_count += 1
                continue

            # 6. Negative tag matching
            if norm in neg_exact:
                filtered_count += 1
                continue
            if any(norm.startswith(p) for p in neg_prefix):
                filtered_count += 1
                continue
            if any(norm.endswith(s) for s in neg_suffix):
                filtered_count += 1
                continue

            # 7. Dedup check (before formatting so we compare apples-to-apples)
            if norm in seen:
                continue
            seen.add(norm)

            # 8. Format
            processed.append(self._format_tag(tag_raw))

        # ── Inject missing priority tags at the front ────────────────────
        for ptag in reversed(settings.priority_tags):
            pnorm = self._normalize(ptag)
            if pnorm and pnorm not in seen:
                seen.add(pnorm)
                processed.insert(0, self._format_tag(ptag))

        # ── Enforce max_tags ─────────────────────────────────────────────
        if len(processed) > settings.max_tags:
            processed = processed[:settings.max_tags]

        # ── Stats ────────────────────────────────────────────────────────
        self.stats['tags_generated'] += len(processed)
        self.stats['tags_filtered'] += filtered_count

        return processed
    
    def format_caption(self,
                       tags: List[str] = None,
                       mode: str = "tags_only") -> str:
        """
        Assemble the final caption string from WD14 tags.

        Order of assembly:
          [caption_prefix] [trigger_word] [body] [caption_suffix]

        Returns:
            Formatted caption string (never None).
        """
        settings = self.tag_settings
        tags = tags or []
        sep = settings.tag_separator

        body = sep.join(tags)

        # Prepend trigger word (trigger is a tag → use tag separator)
        if settings.trigger_word:
            tw = settings.trigger_word.strip()
            if body:
                body = f"{tw}{sep}{body}"
            else:
                body = tw

        # Wrap with prefix / suffix using tag separator so commas stay consistent
        caption = body
        if settings.caption_prefix:
            pfx = settings.caption_prefix.strip()
            caption = f"{pfx}{sep}{caption}" if caption else pfx
        if settings.caption_suffix:
            sfx = settings.caption_suffix.strip()
            caption = f"{caption}{sep}{sfx}" if caption else sfx

        return caption.strip()
    
    def caption_image(self,
                      image: Union[np.ndarray, object],
                      mode: str = "tags_only") -> CaptionResult:
        """
        Generate caption for a single image using WD14.

        Args:
            image: OpenCV BGR ndarray or PIL Image.
            mode:  Caption mode (tags_only).

        Returns:
            Populated CaptionResult.
        """
        result = CaptionResult()

        if self.wd14 and self.enable_wd14:
            try:
                raw_tags = self.wd14.predict(image, self.tag_settings.min_confidence)
                result.wd14_tags = [(t, c) for t, c, _ in raw_tags]
                result.filtered_tags = self.process_tags(raw_tags)
                result.tag_count = len(result.filtered_tags)
                logger.debug("WD14: %d raw → %d processed tags", len(raw_tags), result.tag_count)
            except Exception as e:
                logger.error("WD14 tagging failed: %s", e, exc_info=True)
        elif not self.enable_wd14:
            logger.warning("WD14 tagging is disabled — no auto-tags will be generated")

        result.final_caption = self.format_caption(result.filtered_tags, mode)

        self.stats['images_processed'] += 1
        return result
    
    def caption_directory(self,
                          directory: str,
                          mode: str = "tags_only",
                          overwrite: bool = False,
                          save_json: bool = False,
                          progress_callback=None,
                          batch_size: int = 8,
                          recursive: bool = False) -> Dict:
        """
        Caption all images in a directory with batch processing for speed
        
        Args:
            directory: Path to image directory
            mode: Caption mode
            overwrite: Overwrite existing captions
            save_json: Also save detailed JSON
            progress_callback: callback(current, total, filename)
            batch_size: Number of images to process in each batch
            
        Returns:
            Statistics dictionary
        """
        dir_path = Path(directory)
        
        # Find images (recursive or top-level only)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        all_images = []
        glob_fn = dir_path.rglob if recursive else dir_path.glob
        for ext in extensions:
            all_images.extend(glob_fn(ext))
            all_images.extend(glob_fn(ext.upper()))
        
        stats = {
            'total': len(all_images),
            'captioned': 0,
            'skipped': 0,
            'errors': 0,
            'zero_tags': 0
        }
        
        logger.info("Captioning %d images (batch_size=%d mode=%s dir=%s)",
                    len(all_images), batch_size, mode, directory)
        
        # Filter images that need processing
        images_to_process = []
        for img_path in all_images:
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists() and not overwrite:
                stats['skipped'] += 1
            else:
                images_to_process.append(img_path)
        
        # Process in batches
        total_to_process = len(images_to_process)
        stopped = False
        
        for batch_start in range(0, total_to_process, batch_size):
            if stopped:
                break
                
            batch_paths = images_to_process[batch_start:batch_start + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load batch images
            for img_path in batch_paths:
                try:
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        batch_images.append(image)
                        valid_paths.append(img_path)
                    else:
                        logger.warning("Failed to read image: %s", img_path)
                        stats['errors'] += 1
                except Exception as e:
                    logger.warning("Error loading image %s: %s", img_path, e)
                    stats['errors'] += 1
            
            if not batch_images:
                continue

            # Batch predict WD14 tags
            try:
                if self.wd14 and self.enable_wd14:
                    batch_tags = self.wd14.predict_batch(batch_images, self.tag_settings.min_confidence)
                else:
                    batch_tags = [[] for _ in batch_images]
                    if batch_start == 0:
                        logger.warning("WD14 disabled — captions will have no auto-tags")

                # Process each image result
                for img_idx, (img_path, raw_tags) in enumerate(zip(valid_paths, batch_tags)):
                    # Check callback for stop
                    if progress_callback:
                        current_progress = batch_start + img_idx
                        should_continue = progress_callback(current_progress, total_to_process, img_path.name)
                        if should_continue is False:
                            logger.info("Captioning stopped by user")
                            stopped = True
                            break

                    try:
                        # Process tags
                        processed_tags = self.process_tags(raw_tags)

                        if not processed_tags and self.enable_wd14:
                            stats['zero_tags'] += 1
                            logger.debug("0 tags for %s (raw=%d)", img_path.name, len(raw_tags))

                        # Format caption
                        final_caption = self.format_caption(processed_tags, mode)

                        # Save text caption
                        caption_path = img_path.with_suffix('.txt')
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(final_caption)

                        # Save JSON if requested
                        if save_json:
                            json_path = img_path.with_suffix('.json')
                            json_data = {
                                'tags': processed_tags,
                                'tag_count': len(processed_tags),
                                'final_caption': final_caption,
                            }
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)

                        stats['captioned'] += 1

                    except Exception as e:
                        logger.warning("Error captioning %s: %s", img_path.name, e)
                        stats['errors'] += 1

            except Exception as e:
                logger.warning("Batch captioning error, falling back to single: %s", e)
                # Fallback to single processing for this batch
                for img_path, image in zip(valid_paths, batch_images):
                    try:
                        result = self.caption_image(image, mode)
                        caption_path = img_path.with_suffix('.txt')
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(result.final_caption)
                        stats['captioned'] += 1
                    except Exception as fallback_err:
                        logger.warning("Fallback caption error for %s: %s", img_path.name, fallback_err)
                        stats['errors'] += 1
            finally:
                # Free batch memory to prevent accumulation
                del batch_images
                del valid_paths

        if stats['zero_tags'] > 0:
            logger.warning("⚠️ %d/%d images had 0 auto-tags (only trigger word written). "
                           "Check WD14 model loading and min_confidence (%.2f).",
                           stats['zero_tags'], stats['captioned'], self.tag_settings.min_confidence)
        logger.info("Captioning complete - captioned=%d skipped=%d errors=%d zero_tags=%d",
                    stats['captioned'], stats['skipped'], stats['errors'], stats['zero_tags'])
        
        return stats
    
    def update_settings(self, **kwargs):
        """Update tag settings"""
        for key, value in kwargs.items():
            if hasattr(self.tag_settings, key):
                setattr(self.tag_settings, key, value)
    
    def add_negative_tags(self, tags: List[str]):
        """Add tags to negative list"""
        self.tag_settings.negative_tags.extend(tags)
        self.tag_settings.negative_tags = list(set(self.tag_settings.negative_tags))
    
    def add_priority_tags(self, tags: List[str]):
        """Add tags to priority list"""
        self.tag_settings.priority_tags.extend(tags)
        self.tag_settings.priority_tags = list(set(self.tag_settings.priority_tags))
    
    def set_tag_replacement(self, old_tag: str, new_tag: str):
        """Set tag replacement"""
        self.tag_settings.tag_replacements[old_tag] = new_tag
    
    def get_stats(self) -> Dict:
        """Get captioning statistics"""
        return self.stats.copy()
    
    def save_settings(self, filepath: str):
        """Save current settings to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.tag_settings.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Settings saved to %s", filepath)

    def load_settings(self, filepath: str):
        """Load settings from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tag_settings = TagSettings.from_dict(data)
        logger.info("Settings loaded from %s", filepath)


# Preset configurations for common use cases
CAPTIONER_PRESETS = {
    'anime_character': TagSettings(
        trigger_word="",
        max_tags=25,
        min_confidence=0.35,
        negative_tags=[
            'simple_background', 'white_background', 'grey_background',
            'watermark', 'signature', 'artist_name', 'twitter_username',
            'patreon_username', 'web_address', 'dated', 'copyright_name'
        ],
        include_categories=['general', 'character'],
        keep_character_tags=True,
        keep_series_tags=False,
        include_quality_tags=False,
        include_rating_tags=False
    ),
    
    'style_lora': TagSettings(
        trigger_word="",
        max_tags=20,
        min_confidence=0.4,
        negative_tags=[
            '1girl', '1boy', 'solo', 'multiple_girls', 'multiple_boys',
            'character_name', 'copyright_name'
        ],
        include_categories=['general', 'meta'],
        keep_character_tags=False,
        keep_series_tags=False,
        include_quality_tags=True,
        include_rating_tags=False
    ),
    
    'realistic_photo': TagSettings(
        trigger_word="",
        max_tags=15,
        min_confidence=0.5,
        negative_tags=[
            'anime', 'manga', 'illustration', 'drawing', 'sketch',
            'watermark', 'signature'
        ],
        include_categories=['general'],
        keep_character_tags=False,
        keep_series_tags=False,
        include_quality_tags=False,
        include_rating_tags=False
    ),
    
    'concept_art': TagSettings(
        trigger_word="",
        max_tags=30,
        min_confidence=0.3,
        negative_tags=[
            'watermark', 'signature', 'text', 'username'
        ],
        include_categories=['general', 'meta'],
        keep_character_tags=False,
        keep_series_tags=False,
        include_quality_tags=True,
        include_rating_tags=False
    )
}


def create_captioner_from_preset(preset_name: str, 
                                  trigger_word: str = "",
                                  **kwargs) -> AdvancedCaptioner:
    """
    Create captioner from preset
    
    Args:
        preset_name: One of CAPTIONER_PRESETS keys
        trigger_word: Custom trigger word
        **kwargs: Override preset settings
        
    Returns:
        Configured AdvancedCaptioner
    """
    if preset_name not in CAPTIONER_PRESETS:
        available = ', '.join(CAPTIONER_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    # Copy the preset to avoid mutating the shared original
    import copy
    settings = copy.deepcopy(CAPTIONER_PRESETS[preset_name])
    
    # Override with custom values
    if trigger_word:
        settings.trigger_word = trigger_word
    
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    return AdvancedCaptioner(tag_settings=settings)
