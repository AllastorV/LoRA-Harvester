"""
Advanced Captioning Module for LoRA-Harvester
Supports BLIP (natural language) and WD14/Danbooru (anime tags)
With comprehensive tag management and filtering
"""

# Import onnxruntime FIRST before any torch import to avoid DLL conflicts
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from PIL import Image
import cv2


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
    blip_caption: str = ""
    wd14_tags: List[Tuple[str, float]] = field(default_factory=list)
    final_caption: str = ""
    tag_count: int = 0
    filtered_tags: List[str] = field(default_factory=list)
    

class BLIPCaptioner:
    """
    BLIP-based natural language captioning
    Generates descriptive captions like "a woman with long brown hair"
    """
    
    def __init__(self, 
                 model_type: str = "blip-base",
                 device: str = None):
        """
        Args:
            model_type: "blip-base" or "blip-large"
            device: cuda/cpu (auto-detect if None)
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.processor = None
        self.model = None
        self._loaded = False
        
        print(f"📝 BLIP Captioner initialized (lazy loading)")
        print(f"   Model: {model_type}")
        print(f"   Device: {self.device}")
    
    def _load_model(self):
        """Lazy load BLIP model"""
        if self._loaded:
            return
        
        print("📦 Loading BLIP model...")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            model_name = (
                "Salesforce/blip-image-captioning-large" 
                if self.model_type == "blip-large" 
                else "Salesforce/blip-image-captioning-base"
            )
            
            self.processor = BlipProcessor.from_pretrained(
                model_name,
                cache_dir=".cache/blip"
            )
            
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=".cache/blip",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            print("✅ BLIP model loaded")
            
        except ImportError:
            raise ImportError("transformers library required: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP: {e}")
    
    def generate(self, 
                 image: Union[np.ndarray, Image.Image],
                 max_length: int = 75,
                 min_length: int = 5,
                 num_beams: int = 4) -> str:
        """
        Generate caption for image
        
        Args:
            image: OpenCV (BGR) or PIL image
            max_length: Maximum caption length
            min_length: Minimum caption length
            num_beams: Beam search width
            
        Returns:
            Generated caption string
        """
        if not self._loaded:
            self._load_model()
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()


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
        self._loaded = False
        self._load_error = None  # Track loading errors
        
        print(f"🏷️ WD14 Tagger initialized (lazy loading)")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
    
    def _load_model(self):
        """Lazy load WD14 model"""
        if self._loaded:
            return
        
        if self._load_error:
            # Don't retry if already failed
            raise self._load_error
        
        print("📦 Loading WD14 tagger...")
        
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
            
            # Download model files
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.onnx",
                cache_dir=".cache/wd14"
            )
            
            tags_path = hf_hub_download(
                repo_id=repo_id,
                filename="selected_tags.csv",
                cache_dir=".cache/wd14"
            )
            
            # Load ONNX model
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)
            
            # Load tags
            tags_df = pd.read_csv(tags_path)
            self.tags = tags_df['name'].tolist()
            
            # Get category for each tag
            if 'category' in tags_df.columns:
                self.tag_categories = tags_df['category'].tolist()
            else:
                self.tag_categories = [0] * len(self.tags)
            
            self._loaded = True
            print(f"✅ WD14 model loaded ({len(self.tags)} tags)")
            
        except ImportError as e:
            self._load_error = ImportError(f"Missing package: {e}. Run: pip install onnxruntime pandas huggingface_hub")
            raise self._load_error
        except Exception as e:
            self._load_error = RuntimeError(f"Failed to load WD14: {e}")
            print(f"❌ WD14 load error: {e}")
            raise self._load_error
    
    def _preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image for WD14"""
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize to 448x448 (WD14 input size)
        image = image.convert('RGB')
        image = image.resize((448, 448), Image.LANCZOS)
        
        # Convert to numpy and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB -> BGR for WD14
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, 
                image: Union[np.ndarray, Image.Image],
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
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        predictions = self.model.run([output_name], {input_name: img_input})[0][0]
        
        # Filter by threshold and sort
        results = []
        for idx, conf in enumerate(predictions):
            if conf >= threshold and idx < len(self.tags):
                tag = self.tags[idx]
                category = self.tag_categories[idx] if idx < len(self.tag_categories) else 0
                results.append((tag, float(conf), category))
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def predict_batch(self, 
                      images: List[Union[np.ndarray, Image.Image]],
                      threshold: float = 0.35) -> List[List[Tuple[str, float, int]]]:
        """
        Predict tags for multiple images at once (batch processing for speed)
        
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
        
        # Preprocess all images
        batch_input = np.concatenate([self._preprocess(img) for img in images], axis=0)
        
        # Run batch inference
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        batch_predictions = self.model.run([output_name], {input_name: batch_input})[0]
        
        # Process results for each image
        all_results = []
        for predictions in batch_predictions:
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
    Combined BLIP + WD14 captioner with advanced tag management
    """
    
    def __init__(self,
                 blip_model: str = "blip-base",
                 wd14_model: str = "wd-v1-4-vit-tagger-v2",
                 tag_settings: TagSettings = None,
                 enable_blip: bool = True,
                 enable_wd14: bool = True):
        """
        Initialize advanced captioner
        
        Args:
            blip_model: BLIP model variant
            wd14_model: WD14 model variant
            tag_settings: Tag configuration settings
            enable_blip: Enable BLIP captioning
            enable_wd14: Enable WD14 tagging
        """
        self.tag_settings = tag_settings or TagSettings()
        self.enable_blip = enable_blip
        self.enable_wd14 = enable_wd14
        
        # Initialize captioners (lazy loaded)
        self.blip = BLIPCaptioner(blip_model) if enable_blip else None
        self.wd14 = WD14Tagger(wd14_model) if enable_wd14 else None
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'tags_generated': 0,
            'tags_filtered': 0
        }
        
        print("="*50)
        print("🎨 Advanced Captioner Initialized")
        print("="*50)
        print(f"   BLIP: {'✓' if enable_blip else '✗'} ({blip_model})")
        print(f"   WD14: {'✓' if enable_wd14 else '✗'} ({wd14_model})")
        print(f"   Trigger: '{self.tag_settings.trigger_word}'")
        print(f"   Max tags: {self.tag_settings.max_tags}")
        print(f"   Negative tags: {len(self.tag_settings.negative_tags)}")
        print("="*50)
    
    def process_tags(self, raw_tags: List[Tuple[str, float, int]]) -> List[str]:
        """
        Process and filter tags according to settings
        
        Args:
            raw_tags: List of (tag, confidence, category) tuples
            
        Returns:
            Filtered and formatted tag list
        """
        settings = self.tag_settings
        processed = []
        filtered_out = []
        
        # Category mapping
        category_names = {0: 'general', 1: 'artist', 2: 'copyright', 3: 'character', 4: 'meta'}
        
        for tag, conf, category in raw_tags:
            # Skip if below confidence threshold
            if conf < settings.min_confidence:
                continue
            
            # Get category name
            cat_name = category_names.get(category, 'general')
            
            # Check category inclusion
            if cat_name not in settings.include_categories:
                # But check priority tags first
                if tag not in settings.priority_tags:
                    filtered_out.append(tag)
                    continue
            
            # Handle character tags
            if cat_name == 'character' and not settings.keep_character_tags:
                filtered_out.append(tag)
                continue
            
            # Handle series/copyright tags
            if cat_name == 'copyright' and not settings.keep_series_tags:
                filtered_out.append(tag)
                continue
            
            # Skip quality tags if disabled
            if not settings.include_quality_tags:
                quality_words = ['masterpiece', 'best quality', 'high quality', 'low quality', 
                                'worst quality', 'normal quality', 'absurdres', 'highres']
                if any(q in tag.lower() for q in quality_words):
                    filtered_out.append(tag)
                    continue
            
            # Skip rating tags if disabled
            if not settings.include_rating_tags:
                if tag.lower() in ['general', 'sensitive', 'questionable', 'explicit', 'safe']:
                    filtered_out.append(tag)
                    continue
            
            # Check negative tags (normalize both for comparison)
            is_negative = False
            tag_normalized = tag.lower().replace('_', ' ').strip()
            for neg in settings.negative_tags:
                neg_normalized = neg.lower().replace('_', ' ').strip()
                # Support wildcards
                if neg_normalized.endswith('*'):
                    if tag_normalized.startswith(neg_normalized[:-1]):
                        is_negative = True
                        break
                elif neg_normalized.startswith('*'):
                    if tag_normalized.endswith(neg_normalized[1:]):
                        is_negative = True
                        break
                elif neg_normalized in tag_normalized or tag_normalized in neg_normalized:
                    # Partial match for flexibility
                    is_negative = True
                    break
            
            if is_negative:
                filtered_out.append(tag)
                continue
            
            # Apply replacements
            if tag in settings.tag_replacements:
                tag = settings.tag_replacements[tag]
            
            # Format tag
            if settings.lowercase:
                tag = tag.lower()
            
            if settings.use_underscores:
                tag = tag.replace(' ', '_')
            else:
                tag = tag.replace('_', ' ')
            
            # Escape parentheses for Stable Diffusion
            if settings.use_escape_parentheses:
                tag = tag.replace('(', '\\(').replace(')', '\\)')
            
            processed.append(tag)
        
        # Add priority tags that weren't detected
        for priority_tag in settings.priority_tags:
            formatted = priority_tag.lower() if settings.lowercase else priority_tag
            if formatted not in processed:
                processed.insert(0, formatted)
        
        # Limit tag count
        if len(processed) > settings.max_tags:
            processed = processed[:settings.max_tags]
        
        # Update stats
        self.stats['tags_generated'] += len(processed)
        self.stats['tags_filtered'] += len(filtered_out)
        
        return processed
    
    def format_caption(self, 
                       blip_caption: str = "",
                       tags: List[str] = None,
                       mode: str = "tags_only") -> str:
        """
        Format final caption
        
        Args:
            blip_caption: BLIP-generated caption
            tags: Processed tags list
            mode: "tags_only", "blip_only", "blip_first", "tags_first", "combined"
            
        Returns:
            Formatted caption string
        """
        settings = self.tag_settings
        tags = tags or []
        
        # Build caption based on mode
        if mode == "tags_only":
            caption = settings.tag_separator.join(tags)
        elif mode == "blip_only":
            caption = blip_caption
        elif mode == "blip_first":
            caption = f"{blip_caption}{settings.tag_separator}{settings.tag_separator.join(tags)}"
        elif mode == "tags_first":
            caption = f"{settings.tag_separator.join(tags)}{settings.tag_separator}{blip_caption}"
        else:  # combined
            caption = f"{blip_caption}{settings.tag_separator}{settings.tag_separator.join(tags)}"
        
        # Add trigger word at beginning
        if settings.trigger_word:
            caption = f"{settings.trigger_word}{settings.tag_separator}{caption}"
        
        # Add prefix/suffix
        if settings.caption_prefix:
            caption = f"{settings.caption_prefix} {caption}"
        if settings.caption_suffix:
            caption = f"{caption} {settings.caption_suffix}"
        
        return caption.strip()
    
    def caption_image(self,
                      image: Union[np.ndarray, Image.Image],
                      mode: str = "tags_only") -> CaptionResult:
        """
        Generate caption for a single image
        
        Args:
            image: Input image (OpenCV BGR or PIL)
            mode: Caption mode
            
        Returns:
            CaptionResult with all caption data
        """
        result = CaptionResult()
        
        # Generate BLIP caption
        if self.blip and self.enable_blip:
            try:
                result.blip_caption = self.blip.generate(image)
            except Exception as e:
                print(f"⚠️ BLIP error: {e}")
        
        # Generate WD14 tags
        if self.wd14 and self.enable_wd14:
            try:
                raw_tags = self.wd14.predict(image, self.tag_settings.min_confidence)
                result.wd14_tags = [(t, c) for t, c, _ in raw_tags]
                
                # Process tags
                processed_tags = self.process_tags(raw_tags)
                result.filtered_tags = processed_tags
                result.tag_count = len(processed_tags)
            except Exception as e:
                import traceback
                print(f"⚠️ WD14 error: {e}")
                traceback.print_exc()
        
        # Format final caption
        result.final_caption = self.format_caption(
            result.blip_caption,
            result.filtered_tags,
            mode
        )
        
        self.stats['images_processed'] += 1
        
        return result
    
    def caption_directory(self,
                          directory: str,
                          mode: str = "tags_only",
                          overwrite: bool = False,
                          save_json: bool = False,
                          progress_callback=None,
                          batch_size: int = 8) -> Dict:
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
        
        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        all_images = []
        for ext in extensions:
            all_images.extend(dir_path.glob(ext))
            all_images.extend(dir_path.glob(ext.upper()))
        
        stats = {
            'total': len(all_images),
            'captioned': 0,
            'skipped': 0,
            'errors': 0
        }
        
        print(f"🎨 Captioning {len(all_images)} images (batch size: {batch_size})...")
        print(f"   Mode: {mode}")
        print(f"   Directory: {directory}")
        
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
        processed_count = 0
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
                        stats['errors'] += 1
                except Exception:
                    stats['errors'] += 1
            
            if not batch_images:
                continue
            
            # Batch predict WD14 tags
            try:
                if self.wd14 and self.enable_wd14:
                    batch_tags = self.wd14.predict_batch(batch_images, self.tag_settings.min_confidence)
                else:
                    batch_tags = [[] for _ in batch_images]
                
                # Process each image result
                for img_idx, (img_path, raw_tags) in enumerate(zip(valid_paths, batch_tags)):
                    # Check callback for stop
                    if progress_callback:
                        current_progress = batch_start + img_idx
                        should_continue = progress_callback(current_progress, total_to_process, img_path.name)
                        if should_continue is False:
                            print("\\n⏹️ Captioning stopped by user")
                            stopped = True
                            break
                    
                    try:
                        # Process tags
                        processed_tags = self.process_tags(raw_tags)
                        
                        # Format caption
                        final_caption = self.format_caption("", processed_tags, mode)
                        
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
                                'final_caption': final_caption
                            }
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)
                        
                        stats['captioned'] += 1
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error with {img_path.name}: {e}")
                        stats['errors'] += 1
                        
            except Exception as e:
                print(f"⚠️ Batch error: {e}")
                # Fallback to single processing for this batch
                for img_path, image in zip(valid_paths, batch_images):
                    try:
                        result = self.caption_image(image, mode)
                        caption_path = img_path.with_suffix('.txt')
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(result.final_caption)
                        stats['captioned'] += 1
                    except Exception:
                        stats['errors'] += 1
        
        print(f"\\n✅ Captioning complete!")
        print(f"   Captioned: {stats['captioned']}")
        print(f"   Skipped: {stats['skipped']}")
        print(f"   Errors: {stats['errors']}")
        
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
        print(f"💾 Settings saved to {filepath}")
    
    def load_settings(self, filepath: str):
        """Load settings from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tag_settings = TagSettings.from_dict(data)
        print(f"📂 Settings loaded from {filepath}")


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
    
    settings = CAPTIONER_PRESETS[preset_name]
    
    # Override with custom values
    if trigger_word:
        settings.trigger_word = trigger_word
    
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    return AdvancedCaptioner(tag_settings=settings)
