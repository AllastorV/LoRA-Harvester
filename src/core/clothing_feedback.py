"""Explicit, reversible human corrections, not self-training on model guesses.

Exact replay is image+selection+library bound. At most two human-labeled crops
can assist a later model comparison; they never bypass the visibility rules.
"""
from __future__ import annotations
import base64
from dataclasses import asdict
import io
import json
from pathlib import Path
import time
import uuid
from .clothing_io import (FileLock, atomic_json, atomic_bytes, crop_rgb, load_rgb,
                          digest_json, file_digest, digest_bytes, validate_box)
from .clothing_profiles import tag_key


def selection_key(image_digest, bbox):
    validate_box(bbox)
    return digest_json({'image': image_digest, 'bbox': [round(float(v), 8) for v in bbox] if bbox else None})


class ClothingFeedbackStore:
    def __init__(self, store):
        self.store = store
        self.root = store.root / 'feedback'
        self.path = self.root / 'corrections.json'
        if self.root.is_symlink() or self.root.resolve() != self.root.absolute():
            raise ValueError('Feedback storage must not redirect outside the library.')

    def load(self):
        if not self.path.exists():
            return []
        if self.path.is_symlink():
            raise ValueError('Feedback metadata must not be a symbolic link.')
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or not isinstance(data.get('records'), list):
            raise ValueError('Unsupported correction memory; nothing was overwritten.')
        for record in data['records']:
            if (not isinstance(record, dict) or not isinstance(record.get('visible_tags'), list)
                    or type(record.get('active')) is not bool
                    or record.get('status') not in ('matched', 'no_match')):
                raise ValueError('Invalid correction record; review the memory file.')
        return data['records']

    def fingerprint(self):
        return digest_json(self.load())

    def remember(self, image, bbox, *, profile_id, visible_tags, expected_image_digest,
                 rejected_profile_id='', note='', use_as_example=True):
        """Human UI action only. Returns a ClothingResult ready for preview, not writing."""
        from .clothing_matcher import ClothingResult
        image = Path(image)
        if image.is_symlink():
            raise ValueError('Symbolic-link images are not supported.')
        image = image.resolve()
        validate_box(bbox)
        if not bbox:
            raise ValueError('Select the main person before recording a correction.')
        digest = file_digest(image)
        if not expected_image_digest or digest != expected_image_digest:
            raise ValueError('Image changed after preview. Load and select it again.')
        profiles = {p.id: p for p in self.store.load() if p.enabled}
        profile = profiles.get(profile_id) if profile_id else None
        if profile_id and profile is None:
            raise ValueError('Choose an enabled outfit profile.')
        if not isinstance(visible_tags, list) or any(not isinstance(t, str) for t in visible_tags):
            raise ValueError('Visible tags must be a list of strings.')
        requested = {tag_key(t) for t in visible_tags}
        if len(requested) != len(visible_tags):
            raise ValueError('Visible tags must be unique strings.')
        if profile:
            legal = {tag_key(p.tag) for p in profile.parts}
            if not requested or not requested.issubset(legal):
                raise ValueError('Select at least one genuinely visible part of this outfit.')
            visible_tags = [p.tag for p in profile.parts if tag_key(p.tag) in requested]
        elif requested:
            raise ValueError('A no-match correction cannot contain outfit tags.')
        if len(note) > 1500:
            raise ValueError('Correction note is too long.')
        cropped = crop_rgb(load_rgb(image), bbox)
        from PIL import Image
        cropped.thumbnail((768, 768), Image.Resampling.LANCZOS)
        content = io.BytesIO(); cropped.save(content, 'PNG')
        crop_bytes = content.getvalue()
        name = digest_bytes(crop_bytes) + '.png'
        path = self.root / 'images' / name
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Unsafe feedback image location.')
        if not path.exists():
            atomic_bytes(path, crop_bytes)
        if file_digest(image) != digest:
            raise ValueError('Image changed while recording the correction.')
        library_digest = self.store.library_fingerprint(list(profiles.values()))
        rejected = profiles.get(rejected_profile_id)
        record = {'id': uuid.uuid4().hex, 'key': selection_key(digest, bbox), 'active': True,
                  'created': time.time(), 'image_digest': digest, 'source_name': image.name,
                  'bbox': list(bbox), 'status': 'matched' if profile else 'no_match',
                  'profile_id': profile.id if profile else '',
                  'profile_digest': self.store.fingerprint(profile) if profile else '',
                  'library_digest': library_digest, 'visible_tags': visible_tags,
                  'rejected_profile_id': rejected.id if rejected and rejected != profile else '',
                  'rejected_profile_digest': self.store.fingerprint(rejected) if rejected and rejected != profile else '',
                  'crop': 'images/' + name, 'crop_digest': digest_bytes(crop_bytes),
                  'use_as_example': bool(use_as_example), 'note': note.strip()}
        with FileLock(self.root / 'feedback.lock'):
            records = self.load()
            for old in records:
                if old['key'] == record['key']:
                    old['active'] = False
            records.append(record)
            atomic_json(self.path, {'schema_version': 1, 'records': records})
        return self.result(record, image, profiles, self.fingerprint())

    def forget(self, image, bbox):
        key = selection_key(file_digest(Path(image)), bbox)
        count = 0
        with FileLock(self.root / 'feedback.lock'):
            records = self.load()
            for record in records:
                if record['key'] == key and record['active']:
                    record['active'] = False
                    count += 1
            atomic_json(self.path, {'schema_version': 1, 'records': records})
        return count

    def exact(self, image, bbox, profiles):
        if not bbox:
            return None
        image = Path(image).resolve()
        digest = file_digest(image)
        key = selection_key(digest, bbox)
        active = {p.id: p for p in profiles if p.enabled}
        library_digest = self.store.library_fingerprint(list(active.values()))
        records = self.load()
        for record in reversed(records):
            if not record['active'] or record['key'] != key or record['library_digest'] != library_digest:
                continue
            profile = active.get(record['profile_id'])
            if record['profile_id'] and (profile is None or self.store.fingerprint(profile) != record['profile_digest']):
                continue
            return self.result(record, image, active, digest_json(records))
        return None

    def result(self, record, image, profiles, fingerprint):
        from .clothing_matcher import ClothingResult
        p = profiles.get(record['profile_id'])
        parts = []
        if p:
            visible = {tag_key(t) for t in record['visible_tags']}
            parts = [{'tag': part.tag, 'id': f'p{i}', 'visibility': 'visible' if tag_key(part.tag) in visible else 'not_visible',
                      'score': 1.0, 'evidence': 'Explicit user annotation, not a model confidence.'}
                     for i, part in enumerate(p.parts)]
        return ClothingResult(str(Path(image).resolve()), record['status'],
                              'Human correction memory; not an independently verified model prediction.',
                              image_digest=record['image_digest'], profile_id=record['profile_id'],
                              profile_name=p.name if p else '', profile_digest=record['profile_digest'],
                              tags=[p.master_tag] + record['visible_tags'] if p else [],
                              bbox=record['bbox'], score=0.0, parts=parts,
                              library_digest=record['library_digest'], feedback_digest=fingerprint,
                              decision_source='human')

    def _crop_path(self, record):
        path = self.root / record['crop']
        folder = self.root / 'images'
        if folder.is_symlink() or folder.resolve() != folder.absolute():
            return None
        if (path.is_symlink() or not path.resolve().is_relative_to((self.root / 'images').resolve())
                or not path.is_file() or file_digest(path) != record['crop_digest']):
            return None
        return path

    def examples(self, profile, target_encoded, limit=2):
        """Select relevant human examples using a small perceptual/color descriptor.

        Ranking is only an efficiency heuristic, never an outfit identity score.
        A negative example is retained when available to illustrate known mistakes.
        """
        if not 0 <= limit <= 2:
            raise ValueError('At most two feedback examples may be placed in a comparison.')
        fingerprint = self.store.fingerprint(profile)
        eligible = []
        for record in self.load():
            if not record['active'] or not record['use_as_example']:
                continue
            positive = record['profile_id'] == profile.id and record['profile_digest'] == fingerprint
            negative = record['rejected_profile_id'] == profile.id and record['rejected_profile_digest'] == fingerprint
            if (positive or negative) and (path := self._crop_path(record)):
                eligible.append((record, path, positive))
        if not eligible or not limit:
            return []
        from PIL import Image
        def feature(im):
            im = im.convert('RGB').resize((8, 8))
            # No embedding network or extra model download.
            return list(im.tobytes())
        with Image.open(io.BytesIO(base64.b64decode(target_encoded, validate=True))) as im:
            target = feature(im)
        ranked = []
        for record, path, positive in eligible:
            with Image.open(path) as im:
                vector = feature(im)
            distance = sum(abs(a - b) for a, b in zip(target, vector))
            ranked.append((distance, -record['created'], record, path, positive))
        ranked.sort(key=lambda item: item[:2])
        selected = []
        for polarity in (True, False):
            choice = next((r for r in ranked if r[4] is polarity), None)
            if choice and len(selected) < limit:
                selected.append(choice)
        for choice in ranked:
            if len(selected) >= limit:
                break
            if choice not in selected:
                selected.append(choice)
        return [{'image': base64.b64encode(path.read_bytes()).decode('ascii'),
                 'label': {'matches_this_profile': positive,
                           'visible_tags': record['visible_tags'] if positive else [],
                           'note': record['note']}}
                for _, _, record, path, positive in selected]
