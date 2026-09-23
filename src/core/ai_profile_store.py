"""Compare-and-swap profile commits for an explicitly reviewed AI update."""
from .ai_control import ControlError, digest
from .clothing_io import FileLock, atomic_json


def commit_profile(store, profile, expected_revision=None):
    profile.validate()
    for ref in profile.references:
        store.reference_path(ref)
    with FileLock(store.root / 'library.lock'):
        profiles = store.load()
        existing = next((p for p in profiles if p.id == profile.id), None)
        if existing is None:
            if expected_revision is not None:
                raise ControlError('stale_profile', 'The profile was removed before commit.')
        elif expected_revision is None or digest(existing.to_dict()) != expected_revision:
            raise ControlError('stale_profile', 'Profile changed before commit; no overwrite.')
        profiles = [p for p in profiles if p.id != profile.id] + [profile]
        store._validate_profiles(profiles)
        atomic_json(store.path, {'schema_version': 1, 'profiles': [p.to_dict() for p in profiles]})
