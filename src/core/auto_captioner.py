"""
Tag Generator Module for LoRA-Harvester

Generates structured tags from YOLO/ensemble detection results.
Produces category and count tags (solo, 1person, class names) from
detection boxes as a lightweight complement to WD14 tags.

(AutoCaptioner / BLIP class has been removed — use AdvancedCaptioner
with WD14, or Florence2Captioner for natural-language captions.)
"""

from typing import List, Dict


class TagGenerator:
    """
    Generate structured tags from YOLO/ensemble detection results.
    Used as a lightweight complement to WD14 tags — produces category
    and count tags (solo, 1person, class names) from detection boxes.
    """

    def __init__(self, trigger_word: str = "", separator: str = ", "):
        self.trigger_word = trigger_word
        self.separator = separator

    def generate_tags_from_detections(self,
                                      detections: Dict,
                                      category: str) -> List[str]:
        """
        Build a tag list from detector output.

        Args:
            detections: Dict with keys 'person', 'animal', 'object' → lists of dicts.
            category:   Primary subject category (informational, not used to filter).

        Returns:
            Deduplicated, ordered list of tags.
        """
        tags: List[str] = []
        seen: set = set()

        def _add(tag: str):
            t = tag.lower().strip().replace(' ', '_')
            if t and t not in seen:
                tags.append(t)
                seen.add(t)

        # Trigger word always first
        if self.trigger_word:
            _add(self.trigger_word)

        # Person count tags
        persons = detections.get('person', [])
        if len(persons) == 1:
            _add('solo')
            _add('1person')
        elif len(persons) > 1:
            _add(f'{len(persons)}people')
            _add('multiple_people')

        # Animal class names
        for animal in detections.get('animal', []):
            _add(animal.get('class_name', 'animal'))

        # Top-3 most confident object class names
        objects = sorted(
            detections.get('object', []),
            key=lambda x: x.get('confidence', 0),
            reverse=True,
        )[:3]
        for obj in objects:
            _add(obj.get('class_name', 'object'))

        return tags

    def format_tags(self, tags: List[str], separator: str = None) -> str:
        """Deduplicate and join tags into a single string."""
        sep = separator or self.separator
        cleaned: List[str] = []
        seen: set = set()
        for tag in tags:
            t = tag.lower().strip().replace(' ', '_')
            if t and t not in seen:
                cleaned.append(t)
                seen.add(t)
        return sep.join(cleaned)
