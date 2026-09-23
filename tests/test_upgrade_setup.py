"""No package is installed here: interpreter probes and pip are controlled fakes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import pytest
from src.core import setup_manager as sm


def fake_env(tmp_path, monkeypatch):
    env = tmp_path/'venv'; env.mkdir()
    exe = sm.python_in(env); exe.parent.mkdir(); exe.write_text('not executed')
    (env/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    info = {'version':[3,11,9], 'bits':64, 'prefix':str(env), 'base_prefix':str(tmp_path/'base')}
    monkeypatch.setattr(sm, 'probe_python', lambda *a, **k: dict(info))
    return env, exe, info


def test_missing_venv_error_is_actionable(tmp_path):
    with pytest.raises(sm.SetupError, match='venv'):
        sm.verified_venv(tmp_path)


def test_interpreter_paths(tmp_path):
    assert sm.python_in(tmp_path, True) == tmp_path/'Scripts/python.exe'
    assert sm.python_in(tmp_path, False) == tmp_path/'bin/python'


def test_verified_environment_is_project_local(tmp_path, monkeypatch):
    _, exe, _ = fake_env(tmp_path, monkeypatch)
    assert sm.verified_venv(tmp_path) == exe


@pytest.mark.parametrize('change', [{'version':[3,13,0]}, {'bits':32}, {'prefix':'/other'}, {'base_prefix':'SAME'}])
def test_reject_wrong_environment(tmp_path, monkeypatch, change):
    env, _, info = fake_env(tmp_path, monkeypatch)
    info.update(change)
    if info.get('base_prefix') == 'SAME': info['base_prefix']=str(env)
    with pytest.raises(sm.SetupError): sm.verified_venv(tmp_path)


@pytest.mark.parametrize('line', ['include-system-site-packages = true', 'include-system-site-packages=true', 'INCLUDE-SYSTEM-SITE-PACKAGES = True'])
def test_system_site_packages_are_never_shared(tmp_path, monkeypatch, line):
    env, _, _ = fake_env(tmp_path, monkeypatch)
    (env/'pyvenv.cfg').write_text(line)
    with pytest.raises(sm.SetupError): sm.verified_venv(tmp_path)


def test_venv_directory_symlink_is_rejected(tmp_path):
    target=tmp_path/'external'; target.mkdir(); (tmp_path/'venv').symlink_to(target, target_is_directory=True)
    with pytest.raises(sm.SetupError): sm.verified_venv(tmp_path)


def test_os_lock_releases_after_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with sm.setup_lock(tmp_path):
            with pytest.raises(sm.SetupError):
                with sm.setup_lock(tmp_path): pass
            raise RuntimeError('injected')
    with sm.setup_lock(tmp_path): pass


def test_active_app_blocks_installer_before_any_mutation(tmp_path, monkeypatch):
    manager=sm.SetupManager(tmp_path, log=lambda x:None)
    monkeypatch.setattr(manager, 'ensure_env', lambda *a: pytest.fail('venv was touched'))
    with sm.setup_lock(tmp_path, '.runtime.lock'):
        with pytest.raises(sm.SetupError): manager.install({'core'})
    assert not (tmp_path/'logs').exists()


def test_pip_rechecks_scope_and_uses_absolute_interpreter(tmp_path, monkeypatch):
    _, exe, _=fake_env(tmp_path, monkeypatch); manager=sm.SetupManager(tmp_path)
    commands=[]; monkeypatch.setattr(manager,'run',lambda args,**kw:commands.append(args))
    manager.pip('install','example==1')
    assert commands == [[exe,'-m','pip','--isolated','install','example==1']]
    exe.unlink()
    with pytest.raises(sm.SetupError): manager.pip('install','other')
    assert len(commands)==1


def test_host_package_redirects_are_removed(monkeypatch):
    for key in ('PYTHONHOME','PYTHONPATH','PIP_TARGET','PIP_PREFIX','PIP_USER'):
        monkeypatch.setenv(key,'untrusted')
    env=sm.clean_env()
    assert not any(k in env for k in ('PYTHONHOME','PYTHONPATH','PIP_TARGET','PIP_PREFIX','PIP_USER'))
    assert env['PIP_REQUIRE_VIRTUALENV']=='true'
    assert env['PIP_CONFIG_FILE']==os.devnull


def test_invalid_components_and_cuda_combination_fail_before_changes(tmp_path):
    m=sm.SetupManager(tmp_path)
    for components,channel in [({'oops'},'cu124'), (set(),'cu124'), ({'onnx_gpu','onnx_cpu'},'cu124'),
                               ({'gpu'},'cu999'), ({'gpu','onnx_gpu'},'cu118')]:
        with pytest.raises(sm.SetupError): m.install(components,torch_channel=channel)
    assert not list(tmp_path.iterdir())


def test_cancel_before_process_means_nothing_is_launched(tmp_path, monkeypatch):
    event=threading.Event();event.set()
    monkeypatch.setattr(sm.subprocess,'Popen',lambda *a,**k:pytest.fail('launched after cancellation'))
    with pytest.raises(sm.SetupCancelled): sm.SetupManager(tmp_path,cancel=event).run(['never'])


def test_invalid_venv_is_not_deleted_or_renamed_without_opt_in(tmp_path, monkeypatch):
    env,exe,info=fake_env(tmp_path,monkeypatch);info['bits']=32
    marker=env/'keep.txt';marker.write_text('keep')
    with pytest.raises(sm.SetupError): sm.SetupManager(tmp_path).ensure_env()
    assert marker.read_text()=='keep' and not list(tmp_path.glob('venv-backup-*'))


def test_failed_pip_stops_following_steps(tmp_path, monkeypatch):
    env,exe,_=fake_env(tmp_path,monkeypatch)
    m=sm.SetupManager(tmp_path,log=lambda x:None)
    monkeypatch.setattr(m,'snapshot',lambda *a:None)
    commands=[]
    def run(args,**kw):
        commands.append(args)
        if 'pip' in args: raise sm.SetupError('injected pip failure')
    monkeypatch.setattr(m,'run',run)
    with pytest.raises(sm.SetupError,match='injected'): m.install({'core'})
    assert len(commands)==2
    assert commands[0]==[exe,'-m','ensurepip','--upgrade']
    assert 'requirements-core.txt' not in str(commands)


def test_clothing_only_does_not_need_to_rebuild_python(tmp_path,monkeypatch):
    m=sm.SetupManager(tmp_path,log=lambda x:None); called=[]
    monkeypatch.setattr(m,'ensure_env',lambda *a:pytest.fail('unexpected Python setup'))
    monkeypatch.setattr(m,'install_clothing',lambda:called.append('ollama'))
    monkeypatch.setattr(sm,'diagnose',lambda *a,**kw:{'issues':['venv missing']})
    m.install({'clothing'})
    assert called==['ollama']


def test_single_bootstrap_has_no_global_pip():
    root=Path(__file__).resolve().parents[1]
    text=(root/'install.bat').read_text()
    assert 'pip install' not in text and 'pip uninstall' not in text
    assert 'nvidia-smi -L' in text and '--cli --yes' in text
    assert '--components' in text and 'setup_wizard.py' in text
    assert not any((root/name).exists() for name in
                   ['install_gpu.bat','install_clothing.bat','scripts/install_gpu.ps1'])
    assert '"%PYTHON%" main.py' in (root/'run.bat').read_text()
    assert r'%~dp0venv\Scripts\python.exe' in (root/'run.bat').read_text()


def test_diagnostics_do_not_attempt_repair(tmp_path,monkeypatch):
    monkeypatch.setattr(sm.shutil,'which',lambda *a:None)
    class Offline:
        def open(self,*a,**kw): raise OSError('offline')
    monkeypatch.setattr(sm.urllib.request,'build_opener',lambda *a:Offline())
    result=sm.diagnose(tmp_path)
    assert not result['venv_ok'] and result['issues']
    assert list(tmp_path.iterdir())==[]


def test_gpu_repair_replaces_cpu_wheel_even_at_same_public_version(tmp_path,monkeypatch):
    env,exe,_=fake_env(tmp_path,monkeypatch)
    (tmp_path/'requirements-compat.txt').write_text('numpy<2')
    m=sm.SetupManager(tmp_path,log=lambda x:None);commands=[]
    monkeypatch.setattr(m,'snapshot',lambda *a:None)
    monkeypatch.setattr(m,'run',lambda args,**kw:commands.append(args))
    monkeypatch.setattr(sm.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(a[0],0,stdout='',stderr=''))
    monkeypatch.setattr(sm,'diagnose',lambda *a,**kw:{'issues':[],'cuda_available':True})
    m.install({'gpu'})
    torch_cmd=next(c for c in commands if 'torch==2.6.0' in c)
    assert '--force-reinstall' in torch_cmd
    assert torch_cmd[0]==exe and 'https://download.pytorch.org/whl/cu124' in torch_cmd
    assert '-c' in torch_cmd and str(tmp_path/'requirements-compat.txt') in torch_cmd


def test_healthy_gpu_wheel_is_not_reinstalled(tmp_path,monkeypatch):
    fake_env(tmp_path,monkeypatch)
    m=sm.SetupManager(tmp_path,log=lambda x:None);commands=[]
    monkeypatch.setattr(m,'snapshot',lambda *a:None)
    monkeypatch.setattr(m,'run',lambda args,**kw:commands.append(args))
    monkeypatch.setattr(sm.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(
        a[0],0,stdout='2.6.0+cu124 0.21.0+cu124 2.6.0+cu124 True\n',stderr=''))
    monkeypatch.setattr(sm,'diagnose',lambda *a,**kw:{'issues':[],'cuda_available':True})
    m.install({'gpu'})
    assert not any('torch==2.6.0' in command for command in commands)


def test_auto_cpu_replaces_existing_cuda_wheel(tmp_path,monkeypatch):
    fake_env(tmp_path,monkeypatch)
    m=sm.SetupManager(tmp_path,log=lambda x:None);commands=[]
    monkeypatch.setattr(m,'snapshot',lambda *a:None)
    monkeypatch.setattr(m,'run',lambda args,**kw:commands.append(args))
    monkeypatch.setattr(sm.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(
        a[0],0,stdout='2.6.0+cu124 0.21.0+cu124 2.6.0+cu124 False\n',stderr=''))
    monkeypatch.setattr(sm,'diagnose',lambda *a,**kw:{'issues':[],'cuda_available':False})
    m.install({'core'},torch_channel='cpu')
    torch_cmd=next(command for command in commands if 'torch==2.6.0' in command)
    assert '--force-reinstall' in torch_cmd and 'https://download.pytorch.org/whl/cpu' in torch_cmd


def test_working_newer_environment_can_launch_without_using_install_profile(tmp_path,monkeypatch):
    env,exe,info=fake_env(tmp_path,monkeypatch);info['version']=[3,13,5]
    assert sm.verified_venv(tmp_path,require_supported=False)==exe
    with pytest.raises(sm.SetupError):sm.verified_venv(tmp_path)
