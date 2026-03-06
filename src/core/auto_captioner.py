"""
Auto Captioning Module for LoRA-Harvester
Generates captions for images using BLIP model
Useful for training with captions (SD-scripts, kohya-ss, etc.)
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
import cv2
import numpy as np


class AutoCaptioner:
    """
    Automatic image captioning using BLIP
    Generates .txt caption files alongside images
    """
    
    def __init__(self, 
                 model_type: str = "blip-base",
                 prefix: str = "",
                 suffix: str = "",
                 max_length: int = 75):
        """
        Initialize captioner
        
        Args:
            model_type: "blip-base" or "blip-large"
            prefix: Prefix to add to all captions (e.g., "photo of")
            suffix: Suffix to add to all captions
            max_length: Maximum caption length
        """
        self.prefix = prefix
        self.suffix = suffix
        self.max_length = max_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.processor = None
        self.model = None
        self.model_type = model_type
        self._loaded = False
        
        print(f"📝 AutoCaptioner initialized (lazy loading)")
        print(f"   Model: {model_type}")
        print(f"   Device: {self.device}")
    
    def _load_model(self):
        """Lazy load BLIP model"""
        if self._loaded:
            return
        
        print("📦 Loading BLIP model...")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            if self.model_type == "blip-large":
                model_name = "Salesforce/blip-image-captioning-large"
            else:
                model_name = "Salesforce/blip-image-captioning-base"
            
            self.processor = BlipProcessor.from_pretrained(
                model_name,
                cache_dir=".cache"
            )
            
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=".cache",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            print("✅ BLIP model loaded")
            
        except ImportError:
            print("❌ transformers library required for captioning")
            print("   pip install transformers")
            raise
        except Exception as e:
            print(f"❌ Failed to load BLIP: {e}")
            raise
    
    def generate_caption(self, image: np.ndarray) -> str:
        """
        Generate caption for a single image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Generated caption string
        """
        if not self._loaded:
            self._load_model()
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Process image
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Add prefix/suffix
        if self.prefix:
            caption = f"{self.prefix} {caption}"
        if self.suffix:
            caption = f"{caption} {self.suffix}"
        
        return caption.strip()
    
    def generate_captions_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Generate captions for multiple images (batch processing)
        
        Args:
            images: List of OpenCV images
            
        Returns:
            List of captions
        """
        if not self._loaded:
            self._load_model()
        
        # Convert all images
        pil_images = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
        
        # Process batch
        inputs = self.processor(pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate captions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode all
        captions = []
        for out in outputs:
            caption = self.processor.decode(out, skip_special_tokens=True)
            
            if self.prefix:
                caption = f"{self.prefix} {caption}"
            if self.suffix:
                caption = f"{caption} {self.suffix}"
            
            captions.append(caption.strip())
        
        return captions
    
    def caption_directory(self, 
                         directory: str, 
                         overwrite: bool = False,
                         progress_callback=None) -> Dict:
        """
        Generate captions for all images in a directory
        
        Args:
            directory: Path to image directory
            overwrite: Overwrite existing captions
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Statistics dictionary
        """
        dir_path = Path(directory)
        
        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        images = []
        for ext in extensions:
            images.extend(dir_path.glob(ext))
            images.extend(dir_path.glob(ext.upper()))
        
        stats = {
            'total': len(images),
            'captioned': 0,
            'skipped': 0,
            'errors': 0
        }
        
        print(f"📝 Captioning {len(images)} images in {directory}")
        
        for idx, img_path in enumerate(images):
            # Check for existing caption
            caption_path = img_path.with_suffix('.txt')
            
            if caption_path.exists() and not overwrite:
                stats['skipped'] += 1
                continue
            
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    stats['errors'] += 1
                    continue
                
                # Generate caption
                caption = self.generate_caption(image)
                
                # Save caption
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                
                stats['captioned'] += 1
                
                if progress_callback:
                    progress_callback(idx + 1, len(images), img_path.name)
                    
            except Exception as e:
                print(f"⚠️ Error captioning {img_path.name}: {e}")
                stats['errors'] += 1
        
        print(f"✅ Captioning complete!")
        print(f"   Captioned: {stats['captioned']}")
        print(f"   Skipped: {stats['skipped']}")
        print(f"   Errors: {stats['errors']}")
        
        return stats


class TagGenerator:
    """
    Generate tags/keywords for images using CLIP or WD14 tagger
    Complements captions with structured tags
    """
    
    # Common LoRA training tags
    TRIGGER_TEMPLATES = {
        'person': ['1girl', '1boy', '1person', 'solo'],
        'character': ['{name}', 'character_{name}'],
        'style': ['{style}_style', 'in_the_style_of_{style}'],
        'concept': ['{concept}', 'concept_{concept}']
    }
    
    def __init__(self, trigger_word: str = ""):
        """
        Args:
            trigger_word: Custom trigger word to include
        """
        self.trigger_word = trigger_word
    
    def generate_tags_from_detections(self, 
                                      detections: Dict,
                                      category: str) -> List[str]:
        """
        Generate tags based on object detections
        
        Args:
            detections: Detection results from detector
            category: Primary category (person/animal/object)
            
        Returns:
            List of tags
        """
        tags = []
        
        # Add trigger word first
        if self.trigger_word:
            tags.append(self.trigger_word)
        
        # Count detections
        person_count = len(detections.get('person', []))
        animal_count = len(detections.get('animal', []))
        
        # Person tags
        if person_count == 1:
            tags.append('solo')
            tags.append('1person')
        elif person_count > 1:
            tags.append(f'{person_count}people')
            tags.append('multiple_people')
        
        # Animal tags
        for animal in detections.get('animal', []):
            class_name = animal.get('class_name', 'animal')
            tags.append(class_name)
        
        # Object tags (top 3 most confident)
        objects = sorted(
            detections.get('object', []),
            key=lambda x: x['confidence'],
            reverse=True
        )[:3]
        
        for obj in objects:
            tags.append(obj.get('class_name', 'object'))
        
        return tags
    
    def format_tags(self, tags: List[str], separator: str = ", ") -> str:
        """
        Format tags for caption file
        
        Args:
            tags: List of tags
            separator: Tag separator
            
        Returns:
            Formatted tag string
        """
        # Clean and deduplicate
        cleaned = []
        seen = set()
        
        for tag in tags:
            tag = tag.lower().strip().replace(' ', '_')
            if tag and tag not in seen:
                cleaned.append(tag)
                seen.add(tag)
        
        return separator.join(cleaned)
