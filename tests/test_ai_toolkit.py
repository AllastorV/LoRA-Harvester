"""Real recipe/filesystem/subprocess tests; no model weights or CUDA needed.

The subprocess fixture is a deliberately tiny fake Toolkit + fake torch API,
inside a real temporary Python venv. It tests our integration, NOT model quality.
"""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import venv

from PIL import Image
import pytest
import yaml

from src.training.ai_toolkit_config import (ToolkitOptions, build_bundle, validate_bundle,
    scan_snapshot, make_config, recipe_signature, file_hash)
from src.training.ai_toolkit_runner import ToolkitRunner, check_runtime, validate_checkout, training_environment


@pytest.fixture
def options(tmp_path):
    ds=tmp_path/'dataset'; ds.mkdir()
    Image.new('RGB',(48,64),(10,20,30)).save(ds/'01.png')
    (ds/'01.txt').write_bytes(b'\xef\xbb\xbfmaster, serafaku, blue skirt\r\n')
    return ToolkitOptions(str(ds),str(tmp_path/'output'),'owner/base',cache_latents=False,steps=4)


@pytest.mark.parametrize('key,value',[
    ('steps',0),('steps',True),('batch_size',0),('gradient_accumulation',0),
    ('rank',0),('alpha',257),('resolution',1000),('save_every',0),('max_saves',0),
    ('learning_rate',0),('learning_rate',float('nan')),('learning_rate',float('inf')),
    ('caption_dropout',1),('noise_offset',-1),
    ('network_dropout',1),('min_snr_gamma',-1),('name','../escape'),('name','CON'),
    ('name','name.txt'),('name',''),('name','bad name'),('dtype','float16'),
    ('optimizer','fake'),('architecture','unsupported'),('quantize',True),
    ('shuffle_tokens','false'),('keep_tokens',-1),('seed',-1),('sample_steps',0),
    ('sample_guidance',float('nan')),('sample_prompts',(' ',)),('sample_prompts',('a',)*17),
    ('trigger_word','a\nb'),('base_model','https://bad/model'),('base_model','x'),
    ('base_model','./missing.safetensors'),('device_index',-1),('num_workers',17),
])
def test_invalid_options(options,key,value):
    with pytest.raises((ValueError,TypeError)):
        replace(options,**{key:value}).validate()


def test_sdxl_native_fields_and_master(options):
    bundle=build_bundle(replace(options,learning_rate=0.000005,shuffle_tokens=True))
    config=yaml.safe_load(bundle.yaml_text)['config']['process'][0]
    assert config['type']=='sd_trainer'
    assert config['model']['arch']=='sdxl' and config['model']['is_xl']
    assert config['train']['gradient_accumulation']==1
    assert config['train']['gradient_accumulation_steps']==1
    assert config['train']['lr']==.000005
    assert config['train']['disable_sampling'] and not config['train']['train_text_encoder']
    assert config['datasets'][0]['keep_tokens']==1
    ds=json.loads(Path(config['datasets'][0]['dataset_path']).read_text())
    assert list(ds.values())[0]['caption']=='master, serafaku, blue skirt\r\n'
    assert config['save']['push_to_hub'] is False
    assert not config['logging']['use_wandb']
    assert 'max_train_epochs' not in bundle.yaml_text
    assert 'network_dim' not in bundle.yaml_text
    assert 'text_encoder_lr' not in bundle.yaml_text
    validate_bundle(bundle.manifest_path)


@pytest.mark.parametrize('arch,changes',[
    ('sd1',{'resolution':512,'dtype':'fp16'}),
    ('sdxl',{'train_text_encoder':True}),
    ('flux',{'min_snr_gamma':0.,'quantize':True}),
])
def test_family_configs(options,arch,changes):
    b=build_bundle(replace(options,architecture=arch,**changes))
    p=yaml.safe_load(b.yaml_text)['config']['process'][0]
    assert p['model']['arch']==arch
    assert p['train']['noise_scheduler']==('flowmatch' if arch=='flux' else 'ddpm')
    assert ('quantize' in p['model'])==(arch=='flux')
    validate_bundle(b.manifest_path)


@pytest.mark.parametrize('changes',[{'dtype':'fp16'},{'train_text_encoder':True},{'noise_offset':.1},{'min_snr_gamma':5}])
def test_flux_rejects_incompatible(options,changes):
    o=replace(options,architecture='flux',min_snr_gamma=0)
    with pytest.raises(ValueError): replace(o,**changes).validate()


def test_samples_and_trigger(options):
    b=build_bundle(replace(options,sample_prompts=('master, portrait','master, side view'),trigger_word='master',gradient_accumulation=4))
    p=yaml.safe_load(b.yaml_text)['config']['process'][0]
    assert not p['train']['disable_sampling']
    assert p['sample']['walk_seed'] is False and p['sample']['seed']==42
    assert p['trigger_word']=='master' and p['train']['gradient_accumulation']==4


def test_numeric_folders_do_not_duplicate_parent_images(options):
    root=Path(options.dataset_dir)
    for directory in ('4_clothing','4_clothing/nested','2_other'):
        d=root/directory; d.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(40,50)).save(d/'p.png'); (d/'p.txt').write_text('master, shirt')
    snapshot=scan_snapshot(options)
    assert len(snapshot.rows)==4
    assert sorted(r['repeats'] for r in snapshot.rows)==[1,2,4,4]
    b=build_bundle(options)
    p=yaml.safe_load(b.yaml_text)['config']['process'][0]
    assert len(p['datasets'])==3
    assert sum(len(json.loads(Path(d['dataset_path']).read_text())) for d in p['datasets'])==4
    assert b.weighted_count==11


def test_disable_repeats_and_recursive(options):
    root=Path(options.dataset_dir); d=root/'12_concept'; d.mkdir()
    Image.new('RGB',(40,40)).save(d/'p.jpg'); (d/'p.txt').write_text('tags')
    assert scan_snapshot(replace(options,recursive=False)).image_count==1
    assert scan_snapshot(replace(options,honor_folder_repeats=False)).weighted_count==2


@pytest.mark.parametrize('directory',['.lh-clothing','.lh-dataset','_rejected','_approved','_controls','venv','.venv'])
def test_metadata_pruned(options,directory):
    d=Path(options.dataset_dir)/directory; d.mkdir(); (d/'bad.png').write_bytes(b'bad')
    assert scan_snapshot(options).image_count==1


def test_progress_and_source_unchanged(options):
    root=Path(options.dataset_dir)
    old={p.name:p.read_bytes() for p in root.iterdir()}
    calls=[]; b=build_bundle(options,progress=lambda *a:calls.append(a))
    assert calls==[(1,1,'Dataset 1 / 1')]
    validate_bundle(b.manifest_path)
    assert {p.name:p.read_bytes() for p in root.iterdir()}==old


@pytest.mark.parametrize('mode',['missing','empty','nonutf8','nul','large'])
def test_bad_captions_fail(options,mode):
    f=Path(options.dataset_dir)/'01.txt'
    if mode=='missing': f.unlink()
    else: f.write_bytes({'empty':b'', 'nonutf8':b'\xff', 'nul':b'a\x00b', 'large':b'x'*(1024*1024+1)}[mode])
    with pytest.raises((ValueError,UnicodeError)): build_bundle(options)
    assert not Path(options.output_dir).exists()


def test_allow_empty_caption_is_explicit(options):
    (Path(options.dataset_dir)/'01.txt').unlink()
    b=build_bundle(replace(options,require_captions=False))
    assert any('empty' in s for s in b.warnings)
    validate_bundle(b.manifest_path)


@pytest.mark.parametrize('extra',['01.jpg','01.PNG'])
def test_stem_collision(options,extra):
    Image.new('RGB',(32,32)).save(Path(options.dataset_dir)/extra)
    with pytest.raises(ValueError,match='conflict'): scan_snapshot(options)


def test_unsupported_image_is_not_silently_omitted(options):
    Image.new('RGB',(32,32)).save(Path(options.dataset_dir)/'02.bmp')
    with pytest.raises(ValueError,match='unsupported'): scan_snapshot(options)


def test_corrupt_image(options):
    (Path(options.dataset_dir)/'01.png').write_bytes(b'not an image')
    with pytest.raises(Exception): scan_snapshot(options)


def test_cancelled_build_does_not_write(options):
    event=threading.Event(); event.set()
    with pytest.raises(InterruptedError): build_bundle(options,cancel=event)
    assert not Path(options.output_dir).exists()


@pytest.mark.parametrize('where',['dataset','caption','yaml','index','new_file'])
def test_stale_bundle_refused(options,where):
    b=build_bundle(options); root=Path(options.dataset_dir)
    if where=='dataset': Image.new('RGB',(42,48)).save(root/'01.png')
    elif where=='caption': (root/'01.txt').write_text('edited')
    elif where=='yaml': b.config_path.write_text('job: hacked')
    elif where=='index': next(b.directory.glob('dataset-*.json')).write_text('{}')
    else:
        Image.new('RGB',(32,32)).save(root/'02.png'); (root/'02.txt').write_text('more')
    with pytest.raises(ValueError): validate_bundle(b.manifest_path)


def test_edit_yaml_and_manifest_hash_still_rejected(options):
    b=build_bundle(options)
    c=yaml.safe_load(b.yaml_text); c['config']['process'][0]['save']['push_to_hub']=True
    b.config_path.write_text(yaml.safe_dump(c))
    m=json.loads(b.manifest_path.read_text()); m['files']['training.yaml']=file_hash(b.config_path)
    b.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='recipe mismatch'): validate_bundle(b.manifest_path)


@pytest.mark.parametrize('suffix',['','/output','/../'])
def test_overlapping_output_refused(options,suffix):
    with pytest.raises(ValueError): replace(options,output_dir=options.dataset_dir+suffix).validate()


def test_symlink_refused(options,tmp_path):
    if not hasattr(os,'symlink'): pytest.skip('symlinks unavailable')
    (tmp_path/'link').symlink_to(Path(options.dataset_dir),target_is_directory=True)
    with pytest.raises(ValueError): replace(options,dataset_dir=str(tmp_path/'link')).validate()


def test_resume_signature_and_sampling(options):
    snapshot=scan_snapshot(options)
    assert recipe_signature(options,snapshot)==recipe_signature(replace(options,steps=5000,save_every=500),snapshot)
    assert recipe_signature(options,snapshot)!=recipe_signature(replace(options,rank=64),snapshot)


def test_environment_does_not_modify_host(options,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','host/path'); monkeypatch.setenv('LOCAL_RANK','3')
    env=training_environment(options)
    assert 'PYTHONPATH' not in env and 'LOCAL_RANK' not in env
    assert env['HF_HUB_OFFLINE']=='1' and env['WANDB_MODE']=='disabled'
    assert env['SEED']=='42' and env['ACCELERATE_TORCH_DEVICE']=='cuda:0'
    assert os.environ['PYTHONPATH']=='host/path'


def test_download_opt_in_respects_user_offline(options,monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    assert training_environment(replace(options,allow_downloads=True))['HF_HUB_OFFLINE']=='1'


FAKE_RUN = '''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
class Process:
    def __init__(self, config):
        self.config = config
        self.step_num = 0
        manifest = json.loads((Path(config).parent/'harvester-job.json').read_text())
        self.train_config = SimpleNamespace(steps=manifest['options']['steps'])
    def end_step_hook(self): pass
    def run(self):
        if os.environ.get('LH_FAKE_MODE') == 'error': raise RuntimeError('simulated GPU failure')
        if os.environ.get('LH_FAKE_MODE') == 'early_exit': sys.exit(0)
        for i in range(self.train_config.steps):
            self.step_num = i + 1
            print('loss=0.123', flush=True)
            if os.environ.get('LH_FAKE_MODE') == 'slow': time.sleep(.3)
            self.end_step_hook()
        manifest = json.loads((Path(self.config).parent/'harvester-job.json').read_text())
        if os.environ.get('LH_FAKE_MODE') != 'no_output':
            (Path(manifest['output_path']) / 'my_lora.safetensors').write_bytes(b'fake checkpoint for integration test')
class Job:
    def __init__(self,config): self.process=[Process(config)]
    def run(self): self.process[0].run()
    def cleanup(self): pass
def get_job(config,*args): return Job(config)
def main():
    job=get_job(sys.argv[1])
    try: job.run(); job.cleanup()
    except KeyboardInterrupt: sys.exit(0)
'''
FAKE_TORCH = '''
__version__='fake-test-only'
class cuda:
    @staticmethod
    def is_available(): return True
    @staticmethod
    def device_count(): return 1
    @staticmethod
    def get_device_name(i): return 'FAKE DEVICE - NO GPU EXECUTION'
    @staticmethod
    def is_bf16_supported(): return True
    @staticmethod
    def set_device(i): pass
'''


@pytest.fixture(scope='module')
def toolkit(tmp_path_factory):
    root=tmp_path_factory.mktemp('fake toolkit with spaces')
    for d in ('toolkit','jobs/process'): (root/d).mkdir(parents=True)
    (root/'run.py').write_text(FAKE_RUN)
    (root/'toolkit/config_modules.py').write_text('# gradient_accumulation dataset_path keep_tokens\n')
    (root/'toolkit/job.py').write_text('')
    (root/'toolkit/data_loader.py').write_text('')
    (root/'jobs/process/BaseSDTrainProcess.py').write_text('class X:\n def end_step_hook(self): self.end_step_hook()\n')
    (root/'torch.py').write_text(FAKE_TORCH)
    for mod in ('yaml','diffusers','transformers','accelerate','safetensors','bitsandbytes'):
        (root/f'{mod}.py').write_text('# fake module\n')
    env=root/'venv'; venv.EnvBuilder(with_pip=False).create(env)
    python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    return root,python


def test_real_separate_python_probe(toolkit,options):
    root,python=toolkit
    result=check_runtime(str(root),str(python),options)
    assert result['prefix']!=result['base_prefix']
    assert result['devices']==['FAKE DEVICE - NO GPU EXECUTION']


def test_host_python_is_rejected(toolkit):
    with pytest.raises(ValueError,match='separate|dedicated'):
        check_runtime(str(toolkit[0]),sys.executable)


def test_subprocess_run_and_log(toolkit,options):
    b=build_bundle(options); events=[]; lines=[]
    r=ToolkitRunner().run(*map(str,toolkit),b.manifest_path,progress=events.append,log=lines.append)
    assert r.status=='completed' and r.step==4
    assert any(e['step']==4 for e in events)
    assert Path(r.log_path).exists() and 'loss=0.123' in Path(r.log_path).read_text()
    assert Path(r.output_path,'my_lora.safetensors').exists()
    with pytest.raises(ValueError,match='output exists'):
        ToolkitRunner().run(*map(str,toolkit),b.manifest_path)
    extended=build_bundle(replace(options,steps=8))
    r2=ToolkitRunner().run(*map(str,toolkit),extended.manifest_path,resume=True)
    assert r2.status=='completed' and r2.log_path!=r.log_path and r2.step==8


@pytest.mark.parametrize('mode',['error','no_output','early_exit'])
def test_false_success_refused(toolkit,options,monkeypatch,mode):
    monkeypatch.setenv('LH_FAKE_MODE',mode)
    b=build_bundle(options)
    result=ToolkitRunner().run(*map(str,toolkit),b.manifest_path)
    assert result.status=='failed'


def test_safe_stop_is_not_success(toolkit,options,monkeypatch):
    monkeypatch.setenv('LH_FAKE_MODE','slow')
    b=build_bundle(options); runner=ToolkitRunner()
    def progress(e): runner.stop()
    result=runner.run(*map(str,toolkit),b.manifest_path,progress=progress)
    assert result.status=='stopped' and result.returncode==0 and result.step<4


def test_cancel_before_launch(toolkit,options):
    b=build_bundle(options); runner=ToolkitRunner(); runner.stop()
    with pytest.raises(InterruptedError): runner.run(*map(str,toolkit),b.manifest_path)
    assert not b.output_path.exists()


def test_untracked_existing_output_refused(toolkit,options):
    b=build_bundle(options); b.output_path.mkdir(); (b.output_path/'old.safetensors').write_bytes(b'important')
    with pytest.raises(ValueError,match='untracked'):
        ToolkitRunner().run(*map(str,toolkit),b.manifest_path,resume=True)
    assert (b.output_path/'old.safetensors').read_bytes()==b'important'


def test_resume_mismatching_recipe_refused(toolkit,options):
    b=build_bundle(options); ToolkitRunner().run(*map(str,toolkit),b.manifest_path)
    other=build_bundle(replace(options,rank=64))
    with pytest.raises(ValueError,match='mismatch'):
        ToolkitRunner().run(*map(str,toolkit),other.manifest_path,resume=True)


def test_incompatible_checkout_refused(toolkit,tmp_path):
    root=tmp_path/'bad'; root.mkdir()
    with pytest.raises(ValueError): validate_checkout(str(root),str(toolkit[1]))


def test_changed_local_model_rejected(options,tmp_path):
    model=tmp_path/'model.safetensors';model.write_bytes(b'fake local model')
    b=build_bundle(replace(options,base_model=str(model)))
    model.write_bytes(b'different longer local model')
    with pytest.raises(ValueError,match='model changed'):
        validate_bundle(b.manifest_path)


def test_changed_local_diffusers_config_rejected(options,tmp_path):
    model=tmp_path/'base';model.mkdir();(model/'model_index.json').write_text('{"v":1}')
    b=build_bundle(replace(options,base_model=str(model)))
    (model/'model_index.json').write_text('{"v":2}')
    with pytest.raises(ValueError):validate_bundle(b.manifest_path)


def test_revalidate_dataset_after_runtime_probe(options,toolkit,monkeypatch):
    import src.training.ai_toolkit_runner as module
    original=module.check_runtime
    def changing(*args,**kw):
        result=original(*args,**kw)
        (Path(options.dataset_dir)/'01.txt').write_text('changed while checking runtime')
        return result
    monkeypatch.setattr(module,'check_runtime',changing)
    b=build_bundle(options)
    with pytest.raises(ValueError,match='dataset or model changed'):
        ToolkitRunner().run(*map(str,toolkit),b.manifest_path)
    assert not b.output_path.exists()


def test_same_output_os_lock_prevents_second_trainer(options,toolkit):
    from src.core.clothing_io import FileLock
    b=build_bundle(options)
    with FileLock(b.directory.parent/f'{options.name}.lock'):
        with pytest.raises(RuntimeError):ToolkitRunner().run(*map(str,toolkit),b.manifest_path)
    assert not b.output_path.exists()


def test_checkout_missing_hook_refused(toolkit,tmp_path):
    import shutil
    root=tmp_path/'bad';shutil.copytree(toolkit[0],root,ignore=shutil.ignore_patterns('venv','__pycache__'))
    (root/'jobs/process/BaseSDTrainProcess.py').write_text('class X: pass')
    with pytest.raises(ValueError,match='step-boundary'):
        validate_checkout(str(root),str(toolkit[1]))


def test_resume_target_cannot_be_behind_last_step(options,toolkit):
    b=build_bundle(options);ToolkitRunner().run(*map(str,toolkit),b.manifest_path)
    with pytest.raises(ValueError,match='increase target'):
        ToolkitRunner().run(*map(str,toolkit),b.manifest_path,resume=True)


def test_no_half_bundle_on_write_failure(options,monkeypatch):
    import src.training.ai_toolkit_config as module
    original=module._write_new
    def fail(path,data):
        if path.name=='training.yaml':raise OSError('disk full')
        original(path,data)
    monkeypatch.setattr(module,'_write_new',fail)
    with pytest.raises(OSError):build_bundle(options)
    parent=Path(options.output_dir)/'.lh-toolkit-jobs'
    assert not list(parent.iterdir())


def test_two_bundles_are_separate(options):
    a=build_bundle(options);b=build_bundle(options)
    assert a.directory!=b.directory
    assert a.config_path.read_text()==a.yaml_text
    validate_bundle(a.manifest_path);validate_bundle(b.manifest_path)
