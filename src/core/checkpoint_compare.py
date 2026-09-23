"""Controlled LoRA/checkpoint comparisons through a local WebUI txt2img API.

No weights are downloaded, executed here, or uploaded. The user runs their own
Forge/A1111 service with --api. A fixed recipe improves control, not a promise of
bitwise identity across hardware, extensions or WebUI versions.
"""
from __future__ import annotations
import base64
from dataclasses import asdict, dataclass, field
import html
import http.client
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
import uuid
from .clothing_io import FileLock, atomic_bytes, atomic_json, digest_bytes, digest_json

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'data' / 'evaluations'

class ComparisonError(RuntimeError):
    pass

class ComparisonCancelled(ComparisonError):
    pass


def local_address(endpoint):
    try:
        url = urlsplit(endpoint)
        port = 7860 if url.port is None else url.port
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid local WebUI address.') from exc
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or not 1 <= port <= 65535 or url.username or url.password
            or url.path not in ('', '/') or url.query or url.fragment):
        raise ValueError('Only local http://127.0.0.1:7860 / localhost WebUI endpoints are allowed.')
    return ('127.0.0.1' if url.hostname == 'localhost' else url.hostname), port


@dataclass
class ComparisonConfig:
    endpoint: str = 'http://127.0.0.1:7860'
    mode: str = 'lora'
    candidates: list[str] = field(default_factory=list)
    base_checkpoint: str = ''
    prompts: list[str] = field(default_factory=list)
    negative_prompt: str = ''
    seeds: list[int] = field(default_factory=lambda: [42, 1234])
    sampler: str = 'Euler'
    scheduler: str = 'Automatic'
    steps: int = 28
    cfg_scale: float = 6.0
    width: int = 1024
    height: int = 1024
    clip_skip: int = 1
    vae: str = 'Automatic'
    lora_weight: float = 1.0
    include_baseline: bool = True
    timeout_seconds: int = 600

    def validate(self):
        local_address(self.endpoint)
        if self.mode not in ('lora', 'checkpoint'):
            raise ValueError('Choose lora or checkpoint mode.')
        if not isinstance(self.candidates, list) or any(not isinstance(c, str) for c in self.candidates):
            raise ValueError('Candidate identifiers must be a list of strings.')
        if not isinstance(self.prompts, list) or any(not isinstance(c, str) for c in self.prompts):
            raise ValueError('Prompts must be a list of strings.')
        if not isinstance(self.seeds, list):
            raise ValueError('Seeds must be a list of integers.')
        if not isinstance(self.base_checkpoint, str) or not isinstance(self.negative_prompt, str):
            raise ValueError('Checkpoint and negative prompt must be strings.')
        if not 1 <= len(self.candidates) <= 16 or len(set(self.candidates)) != len(self.candidates):
            raise ValueError('Select 1–16 different candidates.')
        if any(not isinstance(c, str) or not c.strip() or len(c) > 500 for c in self.candidates):
            raise ValueError('Invalid candidate identifier.')
        if self.mode == 'lora':
            if not self.base_checkpoint.strip():
                raise ValueError('A fixed base checkpoint is required for LoRA comparison.')
            if any(any(c in name for c in ':<>\r\n') for name in self.candidates):
                raise ValueError('LoRA identifiers cannot contain colon, brackets or line breaks.')
        if not 1 <= len(self.prompts) <= 32 or any(not p.strip() or len(p) > 8000 for p in self.prompts):
            raise ValueError('Provide 1–32 non-empty prompts (one per line, maximum 8000 characters each).')
        if any(re.search(r'<\s*(?:lora|lyco|hypernet)\s*:', p, re.I) for p in self.prompts + [self.negative_prompt]):
            raise ValueError('Remove LoRA/extra-network directives from shared prompts; candidates are added by the comparison.')
        if (not 1 <= len(self.seeds) <= 32 or any(type(s) is not int or not 0 <= s <= 2**32-1 for s in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('Seeds must be distinct integers in 0..4294967295; random seed -1 is not allowed.')
        for name, lo, hi in [('steps', 1, 150), ('clip_skip', 1, 12), ('timeout_seconds', 30, 3600),
                             ('width', 256, 2048), ('height', 256, 2048)]:
            v = getattr(self, name)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f'Invalid {name}.')
        if self.width % 64 or self.height % 64:
            raise ValueError('Image dimensions must be multiples of 64.')
        for name, lo, hi in [('cfg_scale', 0.1, 30), ('lora_weight', -2, 2)]:
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f'Invalid {name}.')
        for name in ('sampler', 'scheduler', 'vae'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 500 or any(ord(c) < 32 for c in value):
                raise ValueError(f'Invalid {name}.')
        if not isinstance(self.negative_prompt, str) or len(self.negative_prompt) > 8000:
            raise ValueError('Negative prompt is too long.')
        if type(self.include_baseline) is not bool:
            raise ValueError('include_baseline must be a boolean.')
        if len(self.jobs()) > 1024:
            raise ValueError('Comparison exceeds 1024 images; reduce candidates, prompts or seeds.')
        return self

    def jobs(self):
        candidates = ([None] if self.mode == 'lora' and self.include_baseline else []) + self.candidates
        return [{'candidate': c, 'prompt_index': pi, 'seed': seed}
                for c in candidates for pi in range(len(self.prompts)) for seed in self.seeds]

    def payload(self, job):
        candidate = job['candidate']
        prompt = self.prompts[job['prompt_index']]
        if self.mode == 'lora' and candidate is not None:
            prompt += f', <lora:{candidate}:{self.lora_weight:g}>'
        checkpoint = self.base_checkpoint if self.mode == 'lora' else candidate
        return {'prompt': prompt, 'negative_prompt': self.negative_prompt,
                'seed': job['seed'], 'subseed': 0, 'subseed_strength': 0,
                'seed_resize_from_h': 0, 'seed_resize_from_w': 0,
                'sampler_name': self.sampler, 'scheduler': self.scheduler,
                'steps': self.steps, 'cfg_scale': self.cfg_scale,
                'width': self.width, 'height': self.height, 'batch_size': 1, 'n_iter': 1,
                'restore_faces': False, 'tiling': False, 'enable_hr': False,
                'do_not_save_samples': True, 'do_not_save_grid': True,
                'send_images': True, 'save_images': False, 'alwayson_scripts': {},
                'override_settings': {'sd_model_checkpoint': checkpoint,
                                      'CLIP_stop_at_last_layers': self.clip_skip,
                                      'sd_vae': self.vae, 'randn_source': 'CPU',
                                      'eta_noise_seed_delta': 0, 'sd_lora': 'None',
                                      'lora_add_hashes_to_infotext': True, 'add_model_hash_to_info': True},
                'override_settings_restore_afterwards': True}


class WebUIClient:
    def __init__(self, endpoint='http://127.0.0.1:7860', timeout=600, cancel=None):
        self.host, self.port = local_address(endpoint)
        self.timeout = timeout
        self.cancel = cancel or threading.Event()

    def request(self, method, path, data=None, timeout=None):
        if self.cancel.is_set():
            raise ComparisonCancelled('Stopped before the next request.')
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            encoded = json.dumps(data, allow_nan=False).encode('utf-8') if data is not None else None
            connection.request(method, path, encoded, {'Content-Type': 'application/json'})
            response = connection.getresponse()
            # Redirects and remote proxies are not followed. No API credentials are stored.
            raw = response.read(48_000_001)
            if response.status != 200:
                raise ComparisonError(f'WebUI HTTP {response.status}: {raw[:1000].decode("utf-8", "replace")}')
            if len(raw) > 48_000_000:
                raise ComparisonError('WebUI response is too large.')
            return json.loads(raw)
        except ComparisonError:
            raise
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise ComparisonError('Local WebUI request failed. Enable --api and keep it running. '
                                  'The server may still finish a timed-out generation; this request was NOT retried. '
                                  + str(exc)) from exc
        finally:
            connection.close()

    def catalog(self):
        checkpoints = self.request('GET', '/sdapi/v1/sd-models', timeout=20)
        loras = self.request('GET', '/sdapi/v1/loras', timeout=20)
        samplers = self.request('GET', '/sdapi/v1/samplers', timeout=20)
        if (not isinstance(checkpoints, list) or not isinstance(loras, list) or not isinstance(samplers, list)
                or any(not isinstance(x, dict) for x in checkpoints + loras + samplers)):
            raise ComparisonError('Unexpected WebUI model catalog.')
        return {'checkpoints': checkpoints, 'loras': loras, 'samplers': samplers}

    def options(self):
        data = self.request('GET', '/sdapi/v1/options', timeout=20)
        if not isinstance(data, dict):
            raise ComparisonError('Unexpected WebUI settings response.')
        return data

    def ensure_idle(self):
        data = self.request('GET', '/sdapi/v1/progress?skip_current_image=true', timeout=10)
        if not isinstance(data, dict):
            raise ComparisonError('Unexpected WebUI status response.')
        state = data.get('state')
        if not isinstance(state, dict) or type(state.get('job_count')) is not int:
            raise ComparisonError('Unexpected WebUI job state.')
        if state['job_count'] != 0:
            raise ComparisonError('WebUI is running another job. Wait for it to finish before starting a comparison.')

    def generate(self, payload):
        return self.request('POST', '/sdapi/v1/txt2img', payload)


def validate_catalog(config, catalog):
    checkpoints = {m.get('title') for m in catalog['checkpoints']}
    loras = {m.get('name') for m in catalog['loras']}
    samplers = {m.get('name') for m in catalog['samplers']}
    if config.sampler not in samplers:
        raise ValueError('Selected sampler is not available on this WebUI.')
    pool = loras if config.mode == 'lora' else checkpoints
    missing = set(config.candidates) - pool
    if missing or (config.mode == 'lora' and config.base_checkpoint not in checkpoints):
        raise ValueError('Selected models are missing or renamed. Refresh the catalog. ' + ', '.join(sorted(missing)))


def decode_sample(response, expected_seed, expected_size):
    from PIL import Image, PngImagePlugin
    if not isinstance(response, dict) or not isinstance(response.get('images'), list) or len(response['images']) != 1:
        raise ComparisonError('Expected exactly one generated image.')
    info = response.get('info')
    try:
        info = json.loads(info) if isinstance(info, str) else info
        if not isinstance(info, dict):
            raise ValueError('Missing sample metadata')
        seeds = info.get('all_seeds', [info.get('seed')])
        if seeds != [expected_seed]:
            raise ValueError('Returned seed is missing or differs from the fixed seed')
        encoded = response['images'][0]
        if encoded.startswith('data:image/'):
            encoded = encoded.split(',', 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.size != expected_size:
                raise ValueError('Returned dimensions differ from the requested comparison size')
            im.load()
            png = PngImagePlugin.PngInfo()
            infotexts = info.get('infotexts', [])
            if infotexts and isinstance(infotexts[0], str):
                png.add_text('parameters', infotexts[0])
            output = io.BytesIO()
            im.convert('RGB').save(output, 'PNG', pnginfo=png)
        return output.getvalue(), info
    except (ValueError, TypeError, OSError, IndexError, AttributeError) as exc:
        raise ComparisonError('Invalid generated sample; no successful result was assumed: ' + str(exc)) from exc



def verify_actual(info, config, job, catalog):
    """Reject explicit mismatches; label fields the server did not substantiate.

    Metadata is server-reported, not an independent proof of correct model math.
    Incompatible networks sometimes leave an image but no hash; do not hide this.
    """
    warnings = []
    for key, expected in [('steps', config.steps), ('cfg_scale', config.cfg_scale),
                          ('sampler_name', config.sampler), ('clip_skip', config.clip_skip)]:
        value = info.get(key)
        if value is None:
            warnings.append('Server did not report ' + key)
        elif value != expected:
            raise ComparisonError(f'Actual {key} differs: requested {expected!r}, returned {value!r}.')
    checkpoint = config.base_checkpoint if config.mode == 'lora' else job['candidate']
    model = next(m for m in catalog['checkpoints'] if m.get('title') == checkpoint)
    expected_hash = model.get('sha256') or model.get('hash')
    actual_hash = info.get('sd_model_hash')
    if expected_hash and actual_hash:
        a, b = str(expected_hash).lower(), str(actual_hash).lower()
        if min(len(a), len(b)) < 8 or not (a.startswith(b) or b.startswith(a)):
            raise ComparisonError('The generated checkpoint hash differs from the selected model.')
    else:
        warnings.append('Checkpoint identity not hash-verified by server metadata')
    extra = info.get('extra_generation_params') or {}
    if not isinstance(extra, dict):
        raise ComparisonError('Malformed extra generation parameters.')
    value = extra.get('Lora hashes', '')
    hashes = dict((k.strip(), v.strip()) for k, v in
                  (item.rsplit(':', 1) for item in str(value).split(',') if ':' in item))
    candidate = job['candidate'] if config.mode == 'lora' else None
    if candidate is not None:
        key = candidate.replace(',', '')
        if key not in hashes:
            warnings.append('LoRA activation UNVERIFIED: server did not report the selected LoRA hash')
        if set(hashes) - {key}:
            raise ComparisonError('An additional, unrequested LoRA was activated by the server.')
    elif hashes:
        raise ComparisonError('A LoRA was active in a no-LoRA sample.')
    comments = str(info.get('comments', '')) + ' '.join(str(t) for t in info.get('infotexts', []))
    if 'networks with errors:' in comments.lower():
        raise ComparisonError('WebUI reported a network-loading/application error.')
    # Not all API builds return the actual scheduler or VAE hash. Preserve the
    # full metadata, and make absent validation explicit instead of fabricating it.
    actual_schedule = info.get('scheduler') or extra.get('Schedule type')
    if config.scheduler.lower() != 'automatic':
        if not actual_schedule:
            warnings.append('Actual scheduler was not reported separately')
        elif str(actual_schedule).casefold() != config.scheduler.casefold():
            raise ComparisonError('Actual scheduler differs from the requested schedule.')
    return warnings


def comparison_html(manifest):
    cfg = manifest['config']
    columns = ([None] if cfg['mode'] == 'lora' and cfg['include_baseline'] else []) + cfg['candidates']
    cells = {(r['candidate'], r['prompt_index'], r['seed']): r for r in manifest['results']}
    esc = lambda value: html.escape(str(value), quote=True)
    lines = ['<!doctype html><html lang="tr"><meta charset="utf-8"><title>LoRA comparison</title>',
             '<style>body{font:15px system-ui;background:#111827;color:#e5e7eb;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #374151;padding:10px;vertical-align:top}img{width:280px;max-height:440px;object-fit:contain}pre{white-space:pre-wrap;max-width:1050px}a{color:#93c5fd}</style>',
             '<h1>Kontrollü checkpoint karşılaştırması</h1>',
             '<p>Aynı ayarlar farklı sistem/eklenti sürümlerinde piksel eşitliği garantilemez. Otomatik kalite sıralaması yapılmadı.</p>',
             '<pre>' + esc(json.dumps({k:v for k,v in cfg.items() if k not in ('candidates', 'prompts')}, ensure_ascii=False, indent=2)) + '</pre>',
             '<table><thead><tr><th>Prompt / seed</th>' + ''.join('<th>' + esc(c or 'Base / LoRA yok') + '</th>' for c in columns) + '</tr></thead><tbody>']
    for pi, prompt in enumerate(cfg['prompts']):
        for seed in cfg['seeds']:
            row = '<tr><td><b>Seed ' + esc(seed) + '</b><pre>' + esc(prompt) + '</pre></td>'
            for candidate in columns:
                result = cells.get((candidate, pi, seed))
                if result:
                    # Filenames are generated by this module, not API/model input.
                    row += '<td><a href="' + esc(result['file']) + '"><img loading="lazy" src="' + esc(result['file']) + '"></a><p>' + esc('; '.join(result.get('warnings', []))) + '</p></td>'
                else:
                    row += '<td>Üretilmedi</td>'
            lines.append(row + '</tr>')
    lines += ['</tbody></table><p>Durum: ' + esc(manifest['status']) + '</p>',
              '<p>' + esc(manifest.get('error', '')) + '</p></html>']
    return '\n'.join(lines)


def run_comparison(config, output_parent=DEFAULT_OUTPUT, *, client=None, cancel=None, progress=None):
    config.validate()
    cancel = cancel or threading.Event()
    if cancel.is_set():
        raise ComparisonCancelled('Comparison cancelled before starting.')
    progress = progress or (lambda *a: None)
    client = client or WebUIClient(config.endpoint, config.timeout_seconds, cancel)
    output_parent = Path(output_parent).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    # One Harvester client per service. This cannot lock unrelated WebUI browsers;
    # users are asked not to change extensions/options during a comparison.
    locks = DEFAULT_OUTPUT / '.locks'
    lock_id = digest_json(local_address(config.endpoint))
    with FileLock(locks / (lock_id + '.lock')):
        catalog = client.catalog()
        validate_catalog(config, catalog)
        options = client.options()
        client.ensure_idle()
        run = output_parent / ('comparison-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
        run.mkdir()
        manifest = {'schema_version': 1, 'config': asdict(config), 'catalog': catalog,
                    'webui_options_before': options, 'recipe_hash': digest_json(asdict(config)),
                    'status': 'running', 'results': [], 'created': time.time(),
                    'reproducibility': 'Fixed requested settings; not bitwise reproducibility across environments.'}
        def save():
            atomic_json(run / 'comparison.json', manifest)
            atomic_bytes(run / 'index.html', comparison_html(manifest).encode('utf-8'))
        save()
        try:
            jobs = config.jobs()
            for index, job in enumerate(jobs):
                if cancel.is_set():
                    manifest['status'] = 'cancelled'; break
                payload = config.payload(job)
                response = client.generate(payload)
                data, info = decode_sample(response, job['seed'], (config.width, config.height))
                warnings = verify_actual(info, config, job, catalog)
                filename = f'sample_{index + 1:05d}.png'
                atomic_bytes(run / filename, data)
                manifest['results'].append({**job, 'file': filename, 'sha256': digest_bytes(data),
                                            'request': payload, 'actual': info, 'warnings': warnings})
                save()
                progress(index + 1, len(jobs), f'{job["candidate"] or "Base"} | seed {job["seed"]}')
            else:
                manifest['status'] = 'complete'
        except ComparisonCancelled:
            manifest['status'] = 'cancelled'
        except Exception as exc:
            manifest['status'] = 'error'
            manifest['error'] = str(exc)
        finally:
            manifest['finished'] = time.time()
            save()
        return {'folder': str(run), 'report': str(run / 'index.html'),
                'manifest': str(run / 'comparison.json'), 'status': manifest['status'],
                'generated': len(manifest['results']), 'total': len(config.jobs()),
                'warning_samples': sum(bool(r.get('warnings')) for r in manifest['results']),
                'error': manifest.get('error', '')}
