"""Native Ostris AI Toolkit recipes. No torch/toolkit import in the host process.

Reviewed against upstream config_modules.py/data_loader.py on 2026-09-22.
Only the explicitly exposed image-LoRA families are supported, not every model
in upstream. JSON indices freeze captions without copying/re-encoding images.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import uuid
from typing import Callable

SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
OTHER_IMAGES = {'.bmp', '.tif', '.tiff', '.gif', '.avif'}
SKIP_DIRS = {'_approved', '_rejected', '__pycache__', '_controls', 'venv', '.venv'}
MAX_CAPTION_BYTES = 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2,
                      allow_nan=False).encode('utf-8')


def file_hash(path: Path, cancel: threading.Event | None = None) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            _cancelled(cancel)
            hasher.update(block)
    return hasher.hexdigest()


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError('İşlem iptal edildi / Operation cancelled.')


def _path(text: str, label: str, *, exists: bool = False) -> Path:
    if not isinstance(text, str) or not text.strip() or any(c in text for c in '\x00\r\n'):
        raise ValueError(f'{label}: geçerli bir yol gerekli / valid path required.')
    original = Path(text).expanduser().absolute()
    # Refuse symbolic links / junctions in any component (including output).
    for part in (original, *original.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError(f'{label}: bağlantı yolu desteklenmiyor / linked path: {part}')
    result = original.resolve()
    if exists and not result.exists():
        raise ValueError(f'{label}: bulunamadı / not found: {result}')
    return result


@dataclass(frozen=True)
class ToolkitOptions:
    dataset_dir: str
    output_dir: str
    base_model: str
    name: str = 'my_lora'
    architecture: str = 'sdxl'
    steps: int = 3000
    batch_size: int = 1
    gradient_accumulation: int = 1
    rank: int = 32
    alpha: int = 16
    learning_rate: float = 0.0001
    optimizer: str = 'adamw8bit'
    dtype: str = 'bf16'
    resolution: int = 1024
    save_every: int = 250
    max_saves: int = 10
    trigger_word: str = ''
    shuffle_tokens: bool = False
    keep_tokens: int = 1
    caption_dropout: float = 0.0
    flip_x: bool = False
    cache_latents: bool = True
    gradient_checkpointing: bool = True
    recursive: bool = True
    honor_folder_repeats: bool = True
    require_captions: bool = True
    train_text_encoder: bool = False
    min_snr_gamma: float = 5.0
    noise_offset: float = 0.0
    network_dropout: float = 0.0
    quantize: bool = False
    sample_prompts: tuple[str, ...] = ()
    sample_every: int = 250
    sample_steps: int = 20
    sample_guidance: float = 5.0
    seed: int = 42
    allow_downloads: bool = False
    device_index: int = 0
    num_workers: int = 0

    def validate(self) -> None:
        if (not isinstance(self.name, str) or
                not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', self.name) or
                self.name.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))}):
            raise ValueError('LoRA adı: 1–80 harf/rakam/_/-, yol veya uzantı değil / invalid name.')
        if self.architecture not in ('sd1', 'sdxl', 'flux'):
            raise ValueError('Desteklenen aileler / supported families: SD1.5, SDXL, FLUX.1-dev.')
        integer_bounds = {
            'steps': (1, 1000000), 'batch_size': (1, 64), 'gradient_accumulation': (1, 128),
            'rank': (1, 256), 'alpha': (1, 256), 'resolution': (256, 2048),
            'save_every': (1, 1000000), 'max_saves': (1, 1000), 'keep_tokens': (0, 128),
            'sample_every': (1, 1000000), 'sample_steps': (1, 150), 'seed': (0, 2147483647),
            'device_index': (0, 31), 'num_workers': (0, 16),
        }
        for key, (lo, hi) in integer_bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{key}: {lo}..{hi} aralığında tam sayı gerekli / integer required.')
        if self.resolution % 64:
            raise ValueError('Çözünürlük 64 katı olmalı / resolution must be a multiple of 64.')
        for key, lo, hi in [('learning_rate', 1e-10, 1.0),
                             ('min_snr_gamma', 0, 100), ('noise_offset', 0, 1),
                             ('network_dropout', 0, 0.9), ('caption_dropout', 0, 0.99),
                             ('sample_guidance', 0, 30)]:
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key}: geçersiz değer / invalid value.')
        for key in ('shuffle_tokens', 'flip_x', 'cache_latents', 'gradient_checkpointing',
                    'recursive', 'honor_folder_repeats', 'require_captions',
                    'train_text_encoder', 'quantize', 'allow_downloads'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key}: boolean gerekli / boolean required.')
        if self.optimizer not in ('adamw8bit', 'adamw', 'adafactor'):
            raise ValueError('Desteklenmeyen optimizer / unsupported optimizer.')
        if self.dtype not in ('fp16', 'bf16', 'fp32'):
            raise ValueError('dtype: fp16 / bf16 / fp32.')
        if not isinstance(self.trigger_word, str) or len(self.trigger_word) > 256 or any(c in self.trigger_word for c in '\r\n\x00'):
            raise ValueError('Trigger tek satır olmalı / trigger must be a single line.')
        if not isinstance(self.sample_prompts, (tuple, list)) or len(self.sample_prompts) > 16:
            raise ValueError('En fazla 16 örnek prompt / up to 16 sample prompts.')
        if any(not isinstance(p, str) or not p.strip() or len(p) > 4000 or '\x00' in p for p in self.sample_prompts):
            raise ValueError('Geçersiz örnek prompt / invalid sample prompt.')
        if self.architecture != 'flux' and self.quantize:
            raise ValueError('Bu profilde quantize yalnız FLUX içindir / quantize is FLUX-only.')
        if self.architecture == 'flux' and (self.train_text_encoder or self.min_snr_gamma != 0 or self.noise_offset != 0):
            raise ValueError('FLUX profili: text encoder kapalı, min-SNR ve noise offset 0 olmalı.')
        if self.architecture == 'flux' and self.dtype != 'bf16':
            raise ValueError('Bu FLUX profili bf16 gerektirir / this FLUX profile requires bf16.')
        model = self.base_model
        if not isinstance(model, str) or not model.strip() or any(c in model for c in '\x00\r\n'):
            raise ValueError('Ana model gerekli / base model required.')
        # Remote identifiers must be repository IDs, never arbitrary URLs.
        looks_local = Path(model).expanduser().exists() or model.startswith(('.', '/', '~')) or '\\' in model or ':' in model or model.lower().endswith(('.safetensors', '.ckpt', '.pt'))
        if looks_local:
            path = _path(model, 'Model', exists=True)
            if path.is_file() and path.suffix.lower() != '.safetensors':
                raise ValueError('Tek dosya model için .safetensors kullan / use .safetensors checkpoints.')
            if self.architecture == 'flux' and path.is_file():
                raise ValueError('FLUX için Diffusers klasörü veya HF model kimliği gerekli.')
        elif not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model):
            raise ValueError('Model yerel yol veya owner/repo biçiminde olmalı / invalid model ID.')
        ds = _path(self.dataset_dir, 'Dataset', exists=True)
        out = _path(self.output_dir, 'Çıktı / output')
        if not ds.is_dir() or (out.exists() and not out.is_dir()):
            raise ValueError('Dataset ve çıktı klasör olmalı / directories required.')
        if out == ds or ds in out.parents or out in ds.parents:
            raise ValueError('Dataset ile eğitim çıktısı iç içe olamaz / dataset and output must be separate.')


@dataclass
class DatasetSnapshot:
    root: Path
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def image_count(self):
        return len(self.rows)

    @property
    def weighted_count(self):
        return sum(row['repeats'] for row in self.rows)

    @property
    def fingerprint(self):
        return digest(json_bytes(self.rows))


def scan_snapshot(options: ToolkitOptions, cancel=None,
                  progress: Callable[[int, int, str], None] | None = None) -> DatasetSnapshot:
    options.validate()
    root = _path(options.dataset_dir, 'Dataset', exists=True)
    snapshot = DatasetSnapshot(root)
    files = []
    def walk_error(error):
        raise error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        _cancelled(cancel)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS
                         and not Path(folder, d).is_symlink()
                         and not (hasattr(Path(folder,d), 'is_junction') and Path(folder,d).is_junction())) if options.recursive else []
        stems = set()
        for name in sorted(names):
            image = Path(folder, name)
            if name.startswith('.'):
                continue
            if image.suffix.lower() in OTHER_IMAGES:
                raise ValueError(f'Desteklenmeyen görsel / unsupported image: {image.name}. PNG/JPG/WebP olarak ayrı dışa aktar.')
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image.is_symlink():
                raise ValueError(f'Bağlantı görsel desteklenmiyor / linked image: {image}')
            if image.stem.casefold() in stems:
                raise ValueError(f'Caption adı çakışıyor / caption stem conflict: {image}')
            stems.add(image.stem.casefold())
            files.append(image)
    for index, image in enumerate(files):
        _cancelled(cancel)
        before = image.stat()
        ihash = file_hash(image, cancel)
        after = image.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Görsel taranırken değişti / image changed: {image}')
        # Validate without decoding full pixel arrays into the GUI process.
        from PIL import Image
        with Image.open(image) as im:
            if getattr(im, 'n_frames', 1) != 1:
                raise ValueError(f'Hareketli görsel desteklenmiyor / animated image: {image}')
            im.verify()
        caption = image.with_suffix('.txt')
        if caption.is_symlink():
            raise ValueError(f'Caption bağlantı olamaz / linked caption: {caption}')
        raw = None
        if caption.exists():
            with caption.open('rb') as stream:
                raw = stream.read(MAX_CAPTION_BYTES + 1)
            if len(raw) > MAX_CAPTION_BYTES:
                raise ValueError(f'Caption 1 MiB sınırını aşıyor / oversized caption: {caption}')
            text = raw.decode('utf-8-sig')
        else:
            text = ''
        if '\x00' in text:
            raise ValueError(f'Caption NUL içeriyor / invalid caption: {caption}')
        if options.require_captions and not text.strip():
            raise ValueError(f'Eksik/boş caption / missing or empty caption: {image.name}')
        repeats = 1
        if options.honor_folder_repeats:
            for parent in (image.parent, *image.parent.parents):
                match = re.match(r'^(\d+)_', parent.name)
                if match:
                    repeats = int(match.group(1))
                    if not 1 <= repeats <= 1000:
                        raise ValueError(f'Klasör tekrar sayısı 1..1000 olmalı / invalid repeats: {parent}')
                    break
                if parent == root:
                    break
        snapshot.rows.append({'path': str(image), 'caption': text,
                              'image_sha256': ihash, 'caption_sha256': digest(raw) if raw is not None else None,
                              'repeats': repeats})
        if progress and (index % 25 == 0 or index + 1 == len(files)):
            progress(index + 1, len(files), f"Dataset {index + 1} / {len(files)}")
    if not snapshot.rows:
        raise ValueError('Eğitim görseli bulunamadı / no training images found.')
    blank = sum(not r['caption'].strip() for r in snapshot.rows)
    if blank:
        snapshot.warnings.append(f'{blank} görsel boş caption ile işlenecek / empty captions.')
    if options.cache_latents:
        snapshot.warnings.append('AI Toolkit latent önbelleğini kaynak görsellerin yanında oluşturabilir; özgün görsel/caption içerikleri değiştirilmez.')
    return snapshot


@dataclass
class ToolkitBundle:
    directory: Path
    config_path: Path
    manifest_path: Path
    output_path: Path
    yaml_text: str
    image_count: int
    weighted_count: int
    warnings: list[str]


def make_config(options: ToolkitOptions, index_paths: dict[int, Path]) -> dict:
    """No Kohya-only keys or automatic epoch-to-step conversion."""
    options.validate()
    is_flux = options.architecture == 'flux'
    model_path = options.base_model
    if Path(model_path).expanduser().exists():
        model_path = str(Path(model_path).expanduser().resolve())
    train = {
        'batch_size': options.batch_size, 'steps': options.steps,
        'gradient_accumulation': options.gradient_accumulation,
        'gradient_accumulation_steps': 1,
        'train_unet': True, 'train_text_encoder': options.train_text_encoder,
        'gradient_checkpointing': options.gradient_checkpointing,
        'noise_scheduler': 'flowmatch' if is_flux else 'ddpm',
        'optimizer': options.optimizer, 'lr': float(options.learning_rate),
        'lr_scheduler': 'constant', 'dtype': options.dtype,
        'disable_sampling': not bool(options.sample_prompts), 'skip_first_sample': True,
        'cache_text_embeddings': False,
    }
    if not is_flux:
        train['noise_offset'] = float(options.noise_offset)
        if options.min_snr_gamma:
            train['min_snr_gamma'] = float(options.min_snr_gamma)
    model = {'name_or_path': model_path, 'arch': options.architecture,
             'is_xl': options.architecture == 'sdxl', 'is_flux': is_flux}
    if is_flux:
        model['quantize'] = options.quantize
    datasets = []
    for repeats, path in sorted(index_paths.items()):
        datasets.append({
            'dataset_path': str(path.resolve()), 'type': 'image', 'caption_ext': 'txt',
            'resolution': [options.resolution], 'buckets': True, 'num_repeats': repeats,
            'shuffle_tokens': options.shuffle_tokens, 'keep_tokens': options.keep_tokens,
            'caption_dropout_rate': float(options.caption_dropout), 'flip_x': options.flip_x,
            'cache_latents_to_disk': options.cache_latents, 'cache_latents': False,
            'num_workers': options.num_workers,
        })
    proc = {
        'type': 'sd_trainer', 'training_folder': str(Path(options.output_dir).expanduser().resolve()),
        'device': f'cuda:{options.device_index}',
        'network': {'type': 'lora', 'linear': options.rank, 'linear_alpha': options.alpha,
                    'dropout': float(options.network_dropout)},
        'save': {'dtype': 'float32' if options.dtype == 'fp32' else 'float16',
                 'save_every': options.save_every, 'max_step_saves_to_keep': options.max_saves,
                 'push_to_hub': False},
        'datasets': datasets, 'train': train, 'model': model,
        'logging': {'log_every': 10, 'use_wandb': False},
        'sample': {'sampler': 'flowmatch' if is_flux else 'ddpm', 'sample_every': options.sample_every,
                   'width': options.resolution, 'height': options.resolution,
                   'prompts': list(options.sample_prompts), 'neg': '', 'seed': options.seed,
                   'walk_seed': False, 'guidance_scale': float(options.sample_guidance),
                   'sample_steps': options.sample_steps},
    }
    if options.trigger_word.strip():
        proc['trigger_word'] = options.trigger_word.strip()
    return {'job': 'extension', 'config': {'name': options.name, 'process': [proc]},
            'meta': {'name': '[name]', 'version': '1.0', 'authoring_tool': 'LoRA-Harvester v5'}}


def model_identity(options: ToolkitOptions) -> dict:
    """Detect ordinary local model replacements without hashing multi-GB weights.

    Metadata signatures are not a cryptographic model-integrity guarantee. A
    remote repository ID is not a pinned revision; that boundary is documented.
    """
    path = Path(options.base_model).expanduser()
    if not path.exists():
        return {'kind': 'repository', 'id': options.base_model, 'revision': 'unverified'}
    path = path.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    records = []
    for file in files:
        stat = file.stat()
        item = {'name': file.name if path.is_file() else str(file.relative_to(path)),
                'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if file.suffix.lower() == '.json' and stat.st_size <= 1024 * 1024:
            item['sha256'] = file_hash(file)
        records.append(item)
    return {'kind': 'local', 'path': str(path), 'files': records}


def recipe_signature(options: ToolkitOptions, snapshot: DatasetSnapshot) -> str:
    values = asdict(options)
    # An explicit resume may extend the run and change checkpoint/sample cadence.
    for key in ('steps', 'save_every', 'max_saves', 'sample_prompts', 'sample_every',
                'sample_steps', 'sample_guidance', 'allow_downloads', 'output_dir', 'name'):
        values.pop(key, None)
    return digest(json_bytes({'options': values, 'dataset': snapshot.fingerprint,
                              'model_identity': model_identity(options)}))


def _write_new(path: Path, data: bytes):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def build_bundle(options: ToolkitOptions, *, cancel=None, progress=None) -> ToolkitBundle:
    import yaml
    snapshot = scan_snapshot(options, cancel, progress)
    out = _path(options.output_dir, 'Output')
    out.mkdir(parents=True, exist_ok=True)
    parent = out / '.lh-toolkit-jobs'
    _path(str(parent), 'Job folder')
    parent.mkdir(exist_ok=True)
    directory = parent / f'{options.name}-{uuid.uuid4().hex[:12]}'
    directory.mkdir()
    try:
        grouped = {}
        for row in snapshot.rows:
            grouped.setdefault(row['repeats'], {})[row['path']] = {'caption': row['caption']}
        paths = {}
        for repeats, rows in sorted(grouped.items()):
            _cancelled(cancel)
            path = directory / f'dataset-r{repeats}.json'
            _write_new(path, json_bytes(rows))
            paths[repeats] = path
        config = make_config(options, paths)
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        config_path = directory / 'training.yaml'
        _write_new(config_path, text.encode('utf-8'))
        manifest = {
            'schema_version': SCHEMA_VERSION, 'engine': 'ai-toolkit', 'options': asdict(options),
            'rows': snapshot.rows, 'recipe_signature': recipe_signature(options, snapshot),
            'model_identity': model_identity(options),
            'files': {p.name: file_hash(p) for p in [*paths.values(), config_path]},
            'output_path': str(out / options.name),
        }
        manifest_path = directory / 'harvester-job.json'
        _write_new(manifest_path, json_bytes(manifest))
        return ToolkitBundle(directory, config_path, manifest_path, out / options.name,
                             text, snapshot.image_count, snapshot.weighted_count, snapshot.warnings)
    except BaseException:
        shutil.rmtree(directory)
        raise


def validate_bundle(manifest_path: Path, *, cancel=None) -> dict:
    manifest_path = _path(str(manifest_path), 'Job manifest', exists=True)
    if manifest_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('İş manifesti çok büyük / oversized manifest.')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('engine') != 'ai-toolkit':
        raise ValueError('Uyumsuz iş manifesti / unsupported job manifest.')
    options = ToolkitOptions(**manifest['options'])
    options.validate()
    for name, expected in manifest['files'].items():
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Geçersiz manifest yolu / invalid manifest path.')
        file = _path(str(manifest_path.parent / name), 'Job file', exists=True)
        if file_hash(file, cancel) != expected:
            raise ValueError('Yapılandırma değişti; yeniden oluştur / configuration changed; rebuild.')
    fresh = scan_snapshot(options, cancel)
    if fresh.rows != manifest['rows'] or recipe_signature(options, fresh) != manifest['recipe_signature']:
        raise ValueError('Dataset, caption veya yerel model değişti; yeniden oluştur / dataset or model changed; rebuild.')
    expected_output = str(Path(options.output_dir).expanduser().resolve() / options.name)
    if manifest['output_path'] != expected_output:
        raise ValueError('Çıktı yolu uyuşmuyor / output mismatch.')
    # Rebuild the expected recipe: an edited YAML must not acquire extra commands,
    # remote logging destinations or a different dataset through a stale preview.
    import yaml
    paths = {r: manifest_path.parent / f'dataset-r{r}.json' for r in {row['repeats'] for row in fresh.rows}}
    expected_config = make_config(options, paths)
    if yaml.safe_load((manifest_path.parent / 'training.yaml').read_text('utf-8')) != expected_config:
        raise ValueError('YAML manifest ile uyuşmuyor / recipe mismatch.')
    for repeats, path in paths.items():
        expected_index = {r['path']: {'caption': r['caption']} for r in fresh.rows if r['repeats'] == repeats}
        if json.loads(path.read_text('utf-8')) != expected_index:
            raise ValueError('Dataset indeksi değişti / dataset index changed.')
    return manifest
