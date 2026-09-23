"""Run only inside the selected AI Toolkit venv (not Harvester's Python).

The upstream CLI stays in charge of training. A process-local end-of-step hook
reports progress and observes cooperative stop requests *after* save/sample.
No toolkit files are modified. See AI_TOOLKIT_NOTLARI.md for compatibility scope.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

PREFIX = 'LH_AITK_EVENT '


def event(kind, **kwargs):
    print('\n' + PREFIX + json.dumps({'event': kind, **kwargs}, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stop-file', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dtype', choices=('fp16', 'bf16', 'fp32'), default='bf16')
    args = parser.parse_args()
    root, config, stop = Path(args.root).resolve(), Path(args.config).resolve(), Path(args.stop_file).resolve()
    if stop.exists():
        event('stopped', step=0); return 0
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    torch.cuda.set_device(args.device)
    if args.dtype == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected CUDA device does not support bf16; choose fp16 for SD/SDXL.')
    spec = importlib.util.spec_from_file_location('_lh_external_toolkit_run', root / 'run.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    if stop.exists():
        event('stopped', step=0); return 0
    accelerator = getattr(upstream, 'accelerator', None)
    if accelerator is not None and str(accelerator.device) != f'cuda:{args.device}':
        raise RuntimeError('Toolkit selected a different CUDA device; refusing mismatched run.')
    original = upstream.get_job
    state = {'stopped': False, 'hooks': 0, 'last_emit': 0.0}

    def get_job(*a, **kw):
        job = original(*a, **kw)
        processes = getattr(job, 'process', [])
        if len(processes) != 1:
            raise RuntimeError('Harvester expects exactly one image LoRA training process.')
        process = processes[0]
        hook = getattr(process, 'end_step_hook', None)
        if not callable(hook):
            raise RuntimeError('Unsupported AI Toolkit version: end_step_hook missing.')

        def end_step(instance):
            hook()
            step = int(getattr(instance, 'step_num', 0))
            total = int(getattr(instance.train_config, 'steps', 0))
            state['hooks'] += 1
            now = time.monotonic()
            if now - state['last_emit'] >= .15 or step >= total or stop.exists():
                event('step', step=step, total=total)
                state['last_emit'] = now
            if stop.exists():
                state['stopped'] = True
                event('stopped', step=step)
                # Upstream catches KeyboardInterrupt and calls its on_error.
                # We never signal the OS during a save or claim unsaved steps saved.
                raise KeyboardInterrupt('Harvester requested stop at step boundary.')
        process.end_step_hook = types.MethodType(end_step, process)
        return job

    upstream.get_job = get_job
    sys.argv = [str(root / 'run.py'), str(config)]
    try:
        upstream.main()
    except SystemExit as exc:
        if state['stopped']:
            return 0
        # A CLI exiting without completion/hook evidence is not success.
        if exc.code not in (None, 0):
            raise
        raise RuntimeError('Toolkit exited without confirmed completion.') from exc
    if state['stopped']:
        return 0
    if state['hooks'] == 0:
        raise RuntimeError('No training step observed; check resume target and Toolkit compatibility.')
    event('done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
