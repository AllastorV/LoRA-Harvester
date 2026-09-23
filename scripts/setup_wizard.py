"""Unified bootstrap UI, independent of the application's optional ML packages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import webbrowser
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.setup_manager import SetupManager, diagnose, SetupError, ROOT


def cli(args):
    if args.check:
        print(json.dumps(diagnose(), ensure_ascii=False, indent=2))
        return 0
    components = set(filter(None, args.components.split(',')))
    if not args.yes:
        print('Project:', ROOT)
        print('Components:', ', '.join(sorted(components)))
        print('Only the project environment is changed. Packages may be downloaded. User data is preserved.')
        if input('Continue? [y/N]: ').strip().casefold() not in ('y', 'yes'):
            return 0
    try:
        SetupManager().install(components, torch_channel=args.channel, rebuild=args.rebuild)
        return 0
    except (SetupError, OSError) as exc:
        print('[ERROR]', exc)
        return 1


def gui(args):
    import tkinter as tk
    from tkinter import ttk, messagebox
    root = tk.Tk()
    root.title('LoRA-Harvester | Setup and Repair')
    root.geometry('920x760')
    root.minsize(760, 610)
    events = queue.Queue()
    cancel = threading.Event()
    active = [None]
    outer = ttk.Frame(root, padding=18)
    outer.pack(fill='both', expand=True)
    ttk.Label(outer, text='Setup and Repair', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
    ttk.Label(outer, text='Check the system → select components → install or repair → check again',
              font=('Segoe UI', 11)).pack(anchor='w', pady=(4, 10))
    ttk.Label(outer, text=str(ROOT), wraplength=850).pack(anchor='w')
    ttk.Label(outer, text='Global Python is unchanged. Datasets, outfit profiles, and model folders are preserved.',
              wraplength=850).pack(anchor='w', pady=8)
    choices = ttk.LabelFrame(outer, text='Components', padding=12)
    choices.pack(fill='x')
    defaults = {'core'} if args.focus == 'core' else {args.focus}
    labels = {'core': 'Core application / install missing packages',
              'gpu': 'PyTorch GPU support (replaces the installed version with the selected version)',
              'clothing': 'Outfit model: Ollama qwen3-vl:4b (several GB download)',
              'upscale': 'Optional: Real-ESRGAN / GFPGAN',
              'anime': 'Optional: Anime detection / dghs-imgutils',
              'faces': 'Optional: InsightFace (may require Windows build tools)'}
    variables = {}
    widgets = []
    for key, label in labels.items():
        variables[key] = tk.BooleanVar(value=key in defaults)
        button = ttk.Checkbutton(choices, text=label, variable=variables[key])
        button.pack(anchor='w', pady=2)
        widgets.append(button)
    options = ttk.Frame(choices)
    options.pack(fill='x', pady=6)
    ttk.Label(options, text='Torch 2.6.0 channel:').pack(side='left')
    channel = ttk.Combobox(options, values=['cu124', 'cu118', 'cpu'], state='readonly', width=10)
    channel.set(args.channel)
    channel.pack(side='left', padx=8)
    ttk.Label(options, text='ONNX/WD14:').pack(side='left', padx=(15, 0))
    onnx = ttk.Combobox(options, values=['Keep current', 'Repair CPU', 'Repair CUDA'], state='readonly', width=18)
    onnx.set('Mevcudu koru')
    onnx.pack(side='left', padx=8)
    rebuild = tk.BooleanVar(value=False)
    rebuild_widget = ttk.Checkbutton(choices,
        text='Back up and rebuild an incompatible environment (user data is preserved)', variable=rebuild)
    rebuild_widget.pack(anchor='w', pady=4)
    controls = ttk.Frame(outer)
    controls.pack(fill='x', pady=10)
    progress = ttk.Progressbar(outer, mode='indeterminate')
    progress.pack(fill='x')
    output = tk.Text(outer, wrap='word', height=12, state='disabled', font=('Consolas', 10))
    output.pack(fill='both', expand=True, pady=8)
    status = ttk.Label(outer, text='Ready. The first check does not download or change packages.')
    status.pack(anchor='w')

    def append(text):
        output.configure(state='normal')
        output.insert('end', text + '\n')
        output.see('end')
        output.configure(state='disabled')

    def work(fn):
        if active[0] is not None:
            return
        cancel.clear()
        for w in widgets + [check, install, rebuild_widget]:
            w.configure(state='disabled')
        channel.configure(state='disabled'); onnx.configure(state='disabled')
        stop.configure(state='normal'); progress.start(12)
        def runner():
            try:
                value = fn()
                events.put(('done', value))
            except Exception as exc:
                events.put(('error', str(exc)))
        active[0] = threading.Thread(target=runner, daemon=False)
        active[0].start()

    def check_now():
        work(lambda: diagnose(log=lambda msg: events.put(('log', msg))))

    def install_now():
        components = {k for k, v in variables.items() if v.get()}
        if onnx.current() == 1:
            components.add('onnx_cpu')
        elif onnx.current() == 2:
            components.add('onnx_gpu')
        if not components:
            messagebox.showinfo('Select a component', 'Select a component to install or repair.')
            return
        if not messagebox.askyesno('Confirm installation',
            'Close the application first. Selected packages will be installed or updated in the project environment.\n'
            'Packages or models may be downloaded. Continue?'):
            return
        picked_channel, do_rebuild = channel.get(), rebuild.get()
        manager = SetupManager(log=lambda msg: events.put(('log', msg)), cancel=cancel)
        work(lambda: manager.install(components, torch_channel=picked_channel, rebuild=do_rebuild))

    def stop_now():
        cancel.set()
        status.configure(text='Will stop after the current step. pip is not interrupted while writing files.')

    check = ttk.Button(controls, text='1. Check system', command=check_now)
    check.pack(side='left')
    install = ttk.Button(controls, text='2. Install / repair selected', command=install_now)
    install.pack(side='left', padx=8)
    stop = ttk.Button(controls, text='Stop after current step', command=stop_now, state='disabled')
    stop.pack(side='left')
    ttk.Button(controls, text='Download Ollama',
               command=lambda: webbrowser.open('https://ollama.com/download/windows')).pack(side='right')

    def poll():
        try:
            while True:
                kind, data = events.get_nowait()
                if kind == 'log':
                    append(data)
                elif kind in ('done', 'error'):
                    # Queue delivery happens just before the thread returns. Wait
                    # cooperatively, never let a second installer overlap it.
                    if active[0] is not None and active[0].is_alive():
                        events.put((kind, data)); break
                    active[0] = None
                    progress.stop(); stop.configure(state='disabled')
                    for w in widgets + [check, install, rebuild_widget]:
                        w.configure(state='normal')
                    channel.configure(state='readonly'); onnx.configure(state='readonly')
                    if kind == 'error':
                        append('[ERROR] ' + data); status.configure(text='Incomplete. Check the error in the log.')
                    else:
                        append(json.dumps(data, ensure_ascii=False, indent=2))
                        issues = data.get('issues', []) if isinstance(data, dict) else []
                        status.configure(text='Check complete: issues found.' if issues else 'Check complete. Review optional component warnings.')
        except queue.Empty:
            pass
        root.after(80, poll)

    def close():
        if active[0] is not None:
            messagebox.showinfo('Operation in progress', 'Use Stop after current step first. Installation will not be interrupted mid-step.')
            return
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    root.after(80, poll)
    root.mainloop()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--focus', choices=['core', 'gpu', 'clothing'], default='core')
    p.add_argument('--check', action='store_true', help='Read-only JSON diagnostics')
    p.add_argument('--cli', action='store_true')
    p.add_argument('--components', default='core')
    p.add_argument('--channel', choices=['cu124', 'cu118', 'cpu'], default='cu124')
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--yes', action='store_true')
    args = p.parse_args()
    if args.check or args.cli:
        return cli(args)
    try:
        return gui(args)
    except ImportError:
        print('Tkinter is unavailable. Read-only check: --check; terminal setup: --cli')
        return 1
    except Exception as exc:
        print('Could not open setup wizard:', exc, '\nTerminal check: python scripts/setup_wizard.py --check')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
