"""Versioned outfit library. Literal user tags are never silently corrected."""
from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional
from .clothing_io import (atomic_bytes, atomic_json, digest_json, file_digest,
                          load_rgb, validate_box, FileLock)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'clothing'


def tag_key(tag: str) -> str:
    return ' '.join(tag.replace('_', ' ').casefold().split())


def parse_tags(prompt: str):
    result, seen = [], set()
    for token in re.split(r'[,\r\n]+', prompt):
        token = token.strip()
        if token and tag_key(token) not in seen:
            if len(token) > 240 or any(ord(c) < 32 for c in token):
                raise ValueError('A tag is too long or contains control characters.')
            result.append(token)
            seen.add(tag_key(token))
    return result


@dataclass
class ClothingPart:
    tag: str
    description: str = ''


@dataclass
class ClothingReference:
    path: str
    bbox: Optional[list] = None


@dataclass
class ClothingProfile:
    name: str
    master_tag: str
    parts: list[ClothingPart]
    references: list[ClothingReference] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enabled: bool = True
    family: str = ''  # optional label for human organization, not shared identity
    notes: str = ''
    style: str = 'anime_2.5d'

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError('Profile enabled must be a boolean.')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', self.id):
            raise ValueError('Invalid profile ID.')
        if not self.name.strip() or len(self.name) > 200:
            raise ValueError('Enter a profile name (maximum 200 characters).')
        master = parse_tags(self.master_tag)
        if len(master) != 1 or master[0] != self.master_tag.strip():
            raise ValueError('Master tag must be exactly one comma-free tag.')
        if not self.parts or len(self.parts) > 64:
            raise ValueError('An outfit needs 1–64 visible-part tags.')
        keys = [tag_key(self.master_tag)]
        for part in self.parts:
            if parse_tags(part.tag) != [part.tag.strip()]:
                raise ValueError('Each part must have exactly one tag.')
            keys.append(tag_key(part.tag))
            if len(part.description) > 2000:
                raise ValueError('Part description is too long.')
        if len(set(keys)) != len(keys):
            raise ValueError('Master and part tags must be unique within a profile.')
        if len(self.notes) > 6000:
            raise ValueError('Profile notes are too long.')
        if self.style != 'anime_2.5d':
            raise ValueError('This release supports anime / 2.5D outfits.')
        for ref in self.references:
            validate_box(ref.bbox)
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['parts'] = [ClothingPart(**p) for p in value.get('parts', [])]
        value['references'] = [ClothingReference(**r) for r in value.get('references', [])]
        return cls(**value).validate()


@dataclass
class ClothingSettings:
    endpoint: str = 'http://127.0.0.1:11434'
    model: str = 'qwen3-vl:4b'
    identity_threshold: float = 0.85
    part_threshold: float = 0.80
    person_mode: str = 'largest'
    max_side: int = 768
    num_ctx: int = 8192
    timeout_seconds: int = 180
    use_cache: bool = True
    unload_after_job: bool = True
    auto_after_video: bool = False
    auto_after_caption: bool = False

    def validate(self):
        for name in ('use_cache', 'unload_after_job', 'auto_after_video', 'auto_after_caption'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean, not a string.')
        for name in ('max_side', 'num_ctx', 'timeout_seconds'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer.')
        for name in ('identity_threshold', 'part_threshold'):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0 <= val <= 1:
                raise ValueError('Thresholds must be numbers in [0, 1].')
        if self.person_mode not in ('largest', 'center', 'manual'):
            raise ValueError('Unknown person-selection mode.')
        if not 384 <= self.max_side <= 1536 or not 4096 <= self.num_ctx <= 32768:
            raise ValueError('Invalid image size or context window.')
        if not 15 <= self.timeout_seconds <= 1800:
            raise ValueError('Request timeout must be 15–1800 seconds.')
        from .clothing_backend import validate_endpoint, validate_model_name
        validate_endpoint(self.endpoint)
        validate_model_name(self.model)
        return self

    @classmethod
    def from_dict(cls, value):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in known}).validate()


class ClothingProfileStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.path = self.root / 'profiles.json'

    def load(self) -> list[ClothingProfile]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1:
            raise ValueError('Unsupported outfit-library schema. No data was overwritten.')
        profiles = [ClothingProfile.from_dict(p) for p in data['profiles']]
        self._validate_profiles(profiles)
        return profiles

    @staticmethod
    def _validate_profiles(profiles):
        ids, masters = set(), set()
        for p in profiles:
            p.validate()
            if p.id in ids or tag_key(p.master_tag) in masters:
                raise ValueError('Each outfit/color variant needs its own unique master tag.')
            ids.add(p.id)
            masters.add(tag_key(p.master_tag))
        if any(tag_key(part.tag) in masters for p in profiles for part in p.parts):
            raise ValueError('An outfit master tag cannot also be a part tag in another profile.')

    def upsert(self, profile: ClothingProfile):
        profile.validate()
        for ref in profile.references:
            self.reference_path(ref)  # Reject paths outside the managed library.
        with FileLock(self.root / 'library.lock'):
            profiles = self.load()
            profiles = [p for p in profiles if p.id != profile.id] + [profile]
            self._validate_profiles(profiles)
            atomic_json(self.path, {'schema_version': 1,
                                   'profiles': [p.to_dict() for p in profiles]})

    def delete(self, profile_id: str):
        # Keep imported references: accidental profile deletion must not erase files.
        with FileLock(self.root / 'library.lock'):
            profiles = [p for p in self.load() if p.id != profile_id]
            atomic_json(self.path, {'schema_version': 1,
                                   'profiles': [p.to_dict() for p in profiles]})

    def reference_path(self, ref: ClothingReference) -> Path:
        p = (self.root / ref.path).resolve()
        if not p.is_relative_to((self.root / 'references').resolve()):
            raise ValueError('Reference path must be inside data/clothing/references.')
        if not p.is_file():
            raise FileNotFoundError(f'Reference image is missing: {ref.path}')
        return p

    def import_reference(self, path: Path) -> ClothingReference:
        image = load_rgb(Path(path))
        data = io.BytesIO()
        image.save(data, 'PNG')
        from .clothing_io import digest_bytes
        name = digest_bytes(data.getvalue()) + '.png'
        dest = self.root / 'references' / name
        if not dest.exists():
            atomic_bytes(dest, data.getvalue())
        return ClothingReference(path='references/' + name)

    def fingerprint(self, profile: ClothingProfile) -> str:
        return digest_json({'profile': profile.to_dict(),
                            'images': [file_digest(self.reference_path(r))
                                       for r in profile.references]})

    def library_fingerprint(self, profiles=None) -> str:
        """Includes every enabled rival, not only the selected outfit."""
        profiles = self.load() if profiles is None else profiles
        return digest_json(sorted(
            [(p.id, self.fingerprint(p)) for p in profiles if p.enabled], key=lambda item: item[0]))

    def load_settings(self) -> ClothingSettings:
        path = self.root / 'settings.json'
        if not path.exists():
            return ClothingSettings()
        return ClothingSettings.from_dict(json.loads(path.read_text(encoding='utf-8')))

    def save_settings(self, settings: ClothingSettings):
        settings.validate()
        with FileLock(self.root / 'settings.lock'):
            atomic_json(self.root / 'settings.json', asdict(settings))
