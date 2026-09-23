"""Caption-driven stratified dataset selection and non-destructive export.

No missing label is guessed from pixels. Manual labels are bound to the source
image AND caption hashes. Sampling never deletes sources or duplicates rare items.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import shutil
import threading
import uuid
from .clothing_io import FileLock, atomic_json, file_digest, digest_json
from .clothing_profiles import ClothingProfileStore, parse_tags, tag_key
from .dataset_scanner import scan_dataset
from .dataset_files import transfer_image_pair

UNKNOWN = 'unknown'
AMBIGUOUS = 'ambiguous'
POSE_ALIASES = {
    'standing': {'standing'}, 'sitting': {'sitting', 'seated'},
    'kneeling': {'kneeling', 'on knees'}, 'squatting': {'squatting', 'crouching'},
    'lying': {'lying', 'lying down', 'on back', 'on side', 'on stomach'},
    'walking': {'walking'}, 'running': {'running'}, 'jumping': {'jumping'},
    'bending': {'bent over', 'bending forward'},
}
VIEW_ALIASES = {'front': {'front view', 'facing viewer'},
                'back': {'from behind', 'back view'},
                'side': {'from side', 'side view', 'profile'},
                'three_quarter': {'three-quarter view', 'three quarter view'}}
ELEVATION_ALIASES = {'high': {'from above', "bird's-eye view", 'overhead view'},
                     'low': {'from below', "worm's-eye view"},
                     'eye_level': {'eye level', 'eye-level'}}
ANGLE_CHOICES = [UNKNOWN, 'front', 'back', 'side', 'three_quarter', 'high', 'low', 'eye_level'] + [
    v + '+' + e for v in VIEW_ALIASES for e in ELEVATION_ALIASES]
DIMENSIONS = {'outfit': ('outfit',), 'pose': ('pose',), 'angle': ('angle',),
              'joint': ('outfit', 'pose', 'angle')}

class BalanceCancelled(RuntimeError):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise BalanceCancelled('Dataset işlemi durduruldu. Kaynak dosyalar değiştirilmedi.')


def _optional_hash(path):
    if path.is_symlink():
        raise ValueError('Symbolic-link sidecars are not supported.')
    return file_digest(path) if path.is_file() else ''


def _labels(tags, mapping):
    return [name for name, aliases in mapping.items() if tags.intersection(aliases)]


def caption_labels(text, masters):
    tags = {tag_key(t) for t in parse_tags(text)}
    outfits = [master for master in masters if tag_key(master) in tags]
    poses = _labels(tags, POSE_ALIASES)
    # Specific motion overrides standing; lying variants are already one category.
    if len(poses) > 1 and 'standing' in poses and any(p in poses for p in ('walking', 'running', 'jumping')):
        poses.remove('standing')
    view, elevation = _labels(tags, VIEW_ALIASES), _labels(tags, ELEVATION_ALIASES)
    angle = AMBIGUOUS if len(view) > 1 or len(elevation) > 1 else '+'.join(view + elevation) or UNKNOWN
    return {'outfit': outfits[0] if len(outfits) == 1 else (AMBIGUOUS if outfits else UNKNOWN),
            'pose': poses[0] if len(poses) == 1 else (AMBIGUOUS if poses else UNKNOWN),
            'angle': angle}


@dataclass
class BalanceItem:
    path: str
    relative_path: str
    image_digest: str
    caption_digest: str
    json_digest: str
    outfit: str
    pose: str
    angle: str
    caption: str = ''
    issues: list[str] = field(default_factory=list)
    duplicate_of: str = ''
    overridden: bool = False

    def bucket(self, dimensions):
        return tuple(getattr(self, field) for field in dimensions)


def _override_path(root):
    root = Path(root).resolve()
    meta = root / '.lh-dataset'
    if meta.is_symlink() or meta.resolve() != meta.absolute():
        raise ValueError('Dataset metadata must stay inside the dataset.')
    path = meta / 'balance_overrides.json'
    if path.is_symlink():
        raise ValueError('Dataset overrides must not be a symbolic link.')
    return path


def load_overrides(root):
    path = _override_path(root)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not isinstance(data.get('items'), dict):
        raise ValueError('Unsupported balance override format; nothing was overwritten.')
    return data['items']


def save_override(root, item: BalanceItem, labels, *, clear=False):
    root = Path(root).resolve()
    image = Path(item.path)
    if image.is_symlink() or not image.resolve().is_relative_to(root):
        raise ValueError('Image is outside this dataset.')
    if (file_digest(image) != item.image_digest
            or _optional_hash(image.with_suffix('.txt')) != item.caption_digest):
        raise ValueError('Image/caption changed after the scan. Scan again.')
    if set(labels) != {'outfit', 'pose', 'angle'} or not all(isinstance(v, str) and v.strip() for v in labels.values()):
        raise ValueError('Provide outfit, pose and angle labels.')
    if labels['pose'] not in set(POSE_ALIASES) | {UNKNOWN, AMBIGUOUS} or labels['angle'] not in ANGLE_CHOICES + [AMBIGUOUS]:
        raise ValueError('Unknown pose or angle label.')
    if any(len(v) > 240 or any(ord(c) < 32 for c in v) for v in labels.values()):
        raise ValueError('Invalid balance label.')
    path = _override_path(root)
    with FileLock(path.parent / 'balance.lock'):
        data = load_overrides(root)
        if clear:
            data.pop(item.relative_path, None)
        else:
            data[item.relative_path] = {'image_digest': item.image_digest,
                                        'caption_digest': item.caption_digest, 'labels': labels}
        atomic_json(path, {'schema_version': 1, 'items': data})


def scan_balance(root, *, recursive=True, store=None, cancel=None, progress=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Choose an existing dataset folder.')
    progress = progress or (lambda *args: None)
    store = store or ClothingProfileStore()
    # Disabled profiles still label existing data; enabled controls model inference,
    # not whether an already recorded master tag can be counted.
    masters = [p.master_tag for p in store.load()]
    overrides = load_overrides(root)
    pairs = sorted(scan_dataset(root, recursive), key=lambda p: p.image.relative_to(root).as_posix().casefold())
    items, seen = [], {}
    stems = Counter((str(pair.image.parent).casefold(), pair.image.stem.casefold()) for pair in pairs)
    for index, pair in enumerate(pairs):
        check_cancel(cancel)
        image = pair.image
        if not image.resolve().is_relative_to(root):
            continue
        issues, text = [], ''
        digest = file_digest(image)
        try:
            caption_hash = _optional_hash(image.with_suffix('.txt'))
            metadata_hash = _optional_hash(image.with_suffix('.json'))
            if caption_hash:
                text = image.with_suffix('.txt').read_text(encoding='utf-8-sig')
                if not text.strip():
                    issues.append('empty_caption')
            else:
                issues.append('missing_caption')
            labels = caption_labels(text, masters)
        except (ValueError, UnicodeError, OSError) as exc:
            caption_hash, metadata_hash = '', ''
            labels = dict.fromkeys(('outfit', 'pose', 'angle'), UNKNOWN)
            issues.append('unreadable_caption_or_metadata: ' + str(exc))
        relative = image.relative_to(root).as_posix()
        record = overrides.get(relative)
        overridden = False
        if record:
            if record.get('image_digest') == digest and record.get('caption_digest') == caption_hash:
                values = record.get('labels', {})
                if set(values) == set(labels) and all(isinstance(v, str) for v in values.values()):
                    labels.update(values)
                    overridden = True
                else:
                    issues.append('invalid_override')
            else:
                issues.append('stale_override')
        if stems[(str(image.parent).casefold(), image.stem.casefold())] > 1:
            issues.append('same_stem_caption_conflict')
        duplicate = seen.get(digest, '')
        if duplicate:
            issues.append('exact_duplicate')
        else:
            seen[digest] = relative
        items.append(BalanceItem(str(image), relative, digest, caption_hash, metadata_hash,
                                 **labels, caption=text, issues=issues, duplicate_of=duplicate, overridden=overridden))
        progress(index + 1, len(pairs), image.name)
    # Byte-identical images with divergent annotations are not arbitrarily kept
    # from whichever filename sorts first. Require a human to resolve the conflict.
    by_digest = defaultdict(list)
    for item in items:
        by_digest[item.image_digest].append(item)
    for group in by_digest.values():
        if len({(x.caption_digest, x.outfit, x.pose, x.angle) for x in group}) > 1:
            for item in group:
                item.issues.append('duplicate_label_conflict')
    return {'schema_version': 1, 'root': str(root), 'masters': masters,
            'items': items, 'summary': summarize(items), 'overrides_digest': digest_json(overrides)}


def summarize(items):
    return {'total': len(items), 'exact_duplicates': sum(bool(i.duplicate_of) for i in items),
            'issues': dict(Counter(issue.split(':', 1)[0] for i in items for issue in i.issues)),
            **{name: dict(sorted(Counter(getattr(i, name) for i in items).items()))
               for name in ('outfit', 'pose', 'angle')}}


def make_plan(report, *, dimension='joint', per_group=0, seed=42, include_unknown=False, deduplicate=True):
    if dimension not in DIMENSIONS or type(per_group) is not int or per_group < 0 or type(seed) is not int:
        raise ValueError('Invalid sampling settings.')
    dimensions = DIMENSIONS[dimension]
    groups, skipped = defaultdict(list), Counter()
    for item in report['items']:
        if any(issue.startswith(('missing_caption', 'empty_caption', 'unreadable_',
                                 'same_stem_', 'duplicate_label_conflict', 'invalid_override', 'stale_override'))
               for issue in item.issues):
            skipped['data_issue'] += 1
            continue
        if item.duplicate_of and deduplicate:
            skipped['exact_duplicate'] += 1
            continue
        bucket = item.bucket(dimensions)
        if AMBIGUOUS in bucket or (not include_unknown and UNKNOWN in bucket):
            skipped['unlabeled_or_ambiguous'] += 1
            continue
        groups[bucket].append(item)
    # 0 means equal per observed bucket; absent combinations cannot be generated.
    target = per_group or min((len(g) for g in groups.values()), default=0)
    selected, buckets = [], []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda i: i.relative_path)
        # Stable per-bucket RNG; adding a different group cannot reshuffle this one.
        rng = random.Random(digest_json({'seed': seed, 'bucket': key}))
        picked = rng.sample(group, min(target, len(group)))
        selected.extend(picked)
        buckets.append({'labels': dict(zip(dimensions, key)), 'available': len(group),
                        'selected': len(picked), 'shortfall': max(0, target - len(group)),
                        'suggested_repeat': min(5, max(1, round(target / len(group))))})
    selected.sort(key=lambda i: i.relative_path)
    plan = {'schema_version': 1, 'root': report['root'], 'dimension': dimension, 'seed': seed,
            'target_per_group': target, 'include_unknown': include_unknown,
            'deduplicate': deduplicate, 'buckets': buckets,
            'skipped': dict(skipped), 'selected': [asdict(i) for i in selected],
            'source_count': len(report['items']), 'selected_count': len(selected),
            'overrides_digest': report.get('overrides_digest', digest_json({}))}
    plan['fingerprint'] = digest_json(plan)
    return plan


def validate_plan(plan):
    payload = {k: v for k, v in plan.items() if k != 'fingerprint'}
    if plan.get('schema_version') != 1 or plan.get('fingerprint') != digest_json(payload):
        raise ValueError('Sampling plan changed or is invalid; build a new plan.')
    root = Path(plan['root']).resolve()
    if digest_json(load_overrides(root)) != plan['overrides_digest']:
        raise ValueError('Manual labels changed after the plan. Scan and build a new plan.')
    paths = set()
    for item in plan['selected']:
        image = Path(item['path'])
        if image.is_symlink() or not image.resolve().is_relative_to(root) or str(image.resolve()) in paths:
            raise ValueError('Plan contains an unsafe or duplicate source path.')
        paths.add(str(image.resolve()))
        if (file_digest(image) != item['image_digest']
                or _optional_hash(image.with_suffix('.txt')) != item['caption_digest']
                or _optional_hash(image.with_suffix('.json')) != item['json_digest']):
            raise ValueError('Dataset changed after preview: ' + image.name)
    return root


def export_plan(plan, destination, *, cancel=None, progress=None):
    """Publish a new, self-contained dataset. No overwrite/rename of user files."""
    check_cancel(cancel)
    root = validate_plan(plan)
    if not plan['selected']:
        raise ValueError('No eligible images selected. Resolve unknown labels or choose a different dimension.')
    destination = Path(destination).absolute()
    if destination.is_symlink() or destination.exists():
        raise ValueError('Choose a NEW output folder; existing folders are never overwritten.')
    if destination.resolve().is_relative_to(root) or root.is_relative_to(destination.resolve()):
        raise ValueError('Export must be separate from the source dataset, not inside it or its parent.')
    progress = progress or (lambda *args: None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / ('.lh-balance-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        published = []
        for index, item in enumerate(plan['selected']):
            check_cancel(cancel)
            image = Path(item['path'])
            # Hash checks are repeated at copy time, not just at initial preview.
            if (_optional_hash(image.with_suffix('.txt')) != item['caption_digest']
                    or _optional_hash(image.with_suffix('.json')) != item['json_digest']):
                raise ValueError('Caption or metadata changed during export.')
            name = f'{index + 1:06d}_{item["image_digest"][:10]}{image.suffix.lower()}'
            out = transfer_image_pair(image, stage / 'images', copy=True, target_name=name,
                                      expected_image_digest=item['image_digest'])
            if (_optional_hash(out.with_suffix('.txt')) != item['caption_digest']
                    or _optional_hash(out.with_suffix('.json')) != item['json_digest']):
                raise ValueError('Copied sidecar differs from the preview; export was cancelled.')
            published.append({'source': str(image), 'output': 'images/' + out.name,
                              'image_digest': item['image_digest']})
            progress(index + 1, len(plan['selected']), image.name)
        check_cancel(cancel)
        # Manual labels affect the plan, not captions. No silent tag injection.
        atomic_json(stage / 'balance_plan.json', plan)
        atomic_json(stage / 'manifest.json', {'schema_version': 1, 'files': published,
                                             'note': 'Copies only. Manual balance labels did not edit captions.'})
        # No existing directory should ever be replaced even if another process
        # claimed the target while we were copying.
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
        return {'folder': str(destination), 'images': len(published), 'plan': str(destination / 'balance_plan.json')}
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # Only our unique staging folder, never a dataset.
