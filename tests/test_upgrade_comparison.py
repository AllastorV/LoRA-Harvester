"""Real localhost HTTP, synthetic images. Does NOT test diffusion quality or CUDA."""
import base64
import copy
import io
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from PIL import Image
from src.core import checkpoint_compare as cc

CATALOG={'checkpoints':[{'title':'base [aaaaaaaaaa]','sha256':'a'*64},
                        {'title':'other [bbbbbbbbbb]','sha256':'b'*64}],
         'loras':[{'name':'epoch_10'},{'name':'epoch_12'}], 'samplers':[{'name':'Euler'}]}


def config(**changes):
    values=dict(candidates=['epoch_10','epoch_12'],base_checkpoint='base [aaaaaaaaaa]',
                prompts=['full body, standing'],seeds=[42,77],width=256,height=256,steps=10)
    values.update(changes)
    return cc.ComparisonConfig(**values)


def sample(payload):
    bio=io.BytesIO();Image.new('RGB',(payload['width'],payload['height']),(20,30,40)).save(bio,'PNG')
    checkpoint=payload['override_settings']['sd_model_checkpoint']
    info={'all_seeds':[payload['seed']],'steps':payload['steps'],'cfg_scale':payload['cfg_scale'],
          'sampler_name':payload['sampler_name'],'clip_skip':payload['override_settings']['CLIP_stop_at_last_layers'],
          'sd_model_hash':'aaaaaaaaaa' if checkpoint.startswith('base') else 'bbbbbbbbbb',
          'extra_generation_params':{},'infotexts':['prompt\nSteps: 10, Seed: '+str(payload['seed'])]}
    if '<lora:' in payload['prompt']:
        name=payload['prompt'].split('<lora:',1)[1].split(':',1)[0]
        info['extra_generation_params']['Lora hashes']=name+': abcdef123456'
    return {'images':[base64.b64encode(bio.getvalue()).decode()], 'info':json.dumps(info)}


class FakeClient:
    def __init__(self, cancel=None, fail_at=None):
        self.payloads=[];self.cancel=cancel;self.fail_at=fail_at
    def catalog(self): return copy.deepcopy(CATALOG)
    def options(self): return {'sd_model_checkpoint':'previous','sd_lora':'auto-added'}
    def ensure_idle(self): pass
    def generate(self,payload):
        self.payloads.append(copy.deepcopy(payload))
        if self.fail_at==len(self.payloads): raise cc.ComparisonError('injected generation failure')
        if self.cancel: self.cancel.set()
        return sample(payload)


@pytest.fixture(autouse=True)
def isolated_locks(tmp_path,monkeypatch):
    monkeypatch.setattr(cc,'DEFAULT_OUTPUT',tmp_path/'locks-home')


@pytest.mark.parametrize('address',['https://localhost:7860','http://evil.test:7860',
    'http://127.0.0.1:0','http://localhost:99999','http://a:b@localhost:7860',
    'http://localhost:7860/api','http://localhost:7860?x=1','file:///tmp/test'])
def test_only_local_root_http_addresses(address):
    with pytest.raises(ValueError): cc.local_address(address)


def test_local_addresses():
    assert cc.local_address('http://localhost:7860')==('127.0.0.1',7860)
    assert cc.local_address('http://[::1]:7861/')==('::1',7861)


@pytest.mark.parametrize('changes',[{'seeds':[-1]}, {'seeds':[1,1]}, {'seeds':[True]},
    {'width':300}, {'lora_weight':float('nan')}, {'prompts':['<lora:other:1>']},
    {'candidates':[{}]}, {'prompts':[2]}, {'base_checkpoint':''}, {'sampler':''},
    {'candidates':['a:a']}, {'include_baseline':'true'}, {'seeds':'42'}])
def test_reject_uncontrolled_recipe(changes):
    with pytest.raises(ValueError): config(**changes).validate()


def test_fixed_pairing_and_no_lora_baseline():
    cfg=config().validate();jobs=cfg.jobs()
    assert len(jobs)==6
    by_candidate={c:[j['seed'] for j in jobs if j['candidate']==c] for c in [None,*cfg.candidates]}
    assert all(v==[42,77] for v in by_candidate.values())
    for job in jobs:
        payload=cfg.payload(job)
        assert payload['seed']==job['seed'] and payload['batch_size']==payload['n_iter']==1
        assert payload['override_settings_restore_afterwards'] is True
        assert payload['override_settings']['sd_lora']=='None'
        assert payload['override_settings']['sd_model_checkpoint']==cfg.base_checkpoint
        assert ('<lora:' in payload['prompt']) == (job['candidate'] is not None)


def test_checkpoint_mode_changes_only_checkpoint_identity():
    cfg=config(mode='checkpoint', candidates=[m['title'] for m in CATALOG['checkpoints']])
    for job in cfg.jobs():
        payload=cfg.payload(job)
        assert payload['override_settings']['sd_model_checkpoint']==job['candidate']
        assert '<lora:' not in payload['prompt']
    assert len(cfg.jobs())==4


def test_catalog_stale_names_are_rejected():
    with pytest.raises(ValueError): cc.validate_catalog(config(candidates=['missing']),CATALOG)
    with pytest.raises(ValueError): cc.validate_catalog(config(sampler='not-installed'),CATALOG)


def test_report_writes_all_fixed_samples_with_parameter_metadata(tmp_path):
    client=FakeClient();cfg=config()
    result=cc.run_comparison(cfg,tmp_path/'runs',client=client)
    assert result['status']=='complete' and result['generated']==6 and result['warning_samples']==0
    manifest=json.loads(Path(result['manifest']).read_text())
    assert len(manifest['results'])==6
    assert manifest['webui_options_before']['sd_model_checkpoint']=='previous'
    for row in manifest['results']:
        image=Path(result['folder'])/row['file']
        assert cc.digest_bytes(image.read_bytes())==row['sha256']
        with Image.open(image) as im: assert 'Seed:' in im.info['parameters']
    assert '<table>' in Path(result['report']).read_text()


def test_cancel_finishes_current_sample_and_preserves_it(tmp_path):
    event=threading.Event();client=FakeClient(cancel=event)
    result=cc.run_comparison(config(),tmp_path/'runs',client=client,cancel=event)
    assert result['status']=='cancelled' and result['generated']==1 and len(client.payloads)==1
    assert len(list(Path(result['folder']).glob('sample_*.png')))==1


def test_cancel_before_start_creates_nothing(tmp_path):
    event=threading.Event();event.set()
    with pytest.raises(cc.ComparisonCancelled):
        cc.run_comparison(config(),tmp_path/'runs',client=FakeClient(),cancel=event)
    assert not (tmp_path/'runs').exists()


def test_generation_failure_is_not_retried_or_marked_complete(tmp_path):
    client=FakeClient(fail_at=2)
    result=cc.run_comparison(config(),tmp_path/'runs',client=client)
    assert result['status']=='error' and result['generated']==1 and len(client.payloads)==2
    assert 'injected' in result['error']


@pytest.mark.parametrize('corruption',['seed','size','base64','multiple','metadata'])
def test_invalid_samples_are_rejected(corruption):
    cfg=config();response=sample(cfg.payload(cfg.jobs()[0]))
    if corruption=='seed': response['info']=json.dumps({'all_seeds':[99]})
    elif corruption=='size':
        raw=io.BytesIO();Image.new('RGB',(64,64)).save(raw,'PNG');response['images']=[base64.b64encode(raw.getvalue()).decode()]
    elif corruption=='base64':response['images']=['?invalid']
    elif corruption=='multiple':response['images']*=2
    elif corruption=='metadata':response['info']=[]
    with pytest.raises(cc.ComparisonError):cc.decode_sample(response,42,(256,256))


@pytest.mark.parametrize('key,value',[('steps',20),('cfg_scale',7),('sampler_name','DPM++ 2M'),
                                     ('clip_skip',2),('sd_model_hash','b'*10)])
def test_actual_mismatched_settings_are_rejected(key,value):
    cfg=config();job=cfg.jobs()[0];info=json.loads(sample(cfg.payload(job))['info']);info[key]=value
    with pytest.raises(cc.ComparisonError):cc.verify_actual(info,cfg,job,CATALOG)


def test_missing_hash_is_labeled_unverified_not_invented():
    cfg=config();job=next(j for j in cfg.jobs() if j['candidate']);info=json.loads(sample(cfg.payload(job))['info'])
    info['extra_generation_params']={}
    assert any('UNVERIFIED' in w for w in cc.verify_actual(info,cfg,job,CATALOG))


def test_unrequested_network_and_network_errors_rejected():
    cfg=config();job=cfg.jobs()[0];info=json.loads(sample(cfg.payload(job))['info'])
    info['extra_generation_params']={'Lora hashes':'unrequested: abcdef123456'}
    with pytest.raises(cc.ComparisonError):cc.verify_actual(info,cfg,job,CATALOG)
    info['extra_generation_params']={};info['comments']='Networks with errors: test (1)'
    with pytest.raises(cc.ComparisonError):cc.verify_actual(info,cfg,job,CATALOG)


def test_html_escapes_prompt_and_keeps_warnings(tmp_path):
    cfg=config(prompts=['<script>alert(1)</script>'])
    result=cc.run_comparison(cfg,tmp_path/'runs',client=FakeClient())
    text=Path(result['report']).read_text()
    assert '<script>' not in text and '&lt;script&gt;' in text


def test_real_local_http_contract_no_persistent_options_writes(tmp_path):
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def reply(self,data,status=200):
            raw=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            calls.append(('GET',self.path,None))
            mapping={'/sdapi/v1/sd-models':CATALOG['checkpoints'],'/sdapi/v1/loras':CATALOG['loras'],
                     '/sdapi/v1/samplers':CATALOG['samplers'],'/sdapi/v1/options':{'sd_model_checkpoint':'previous'},
                     '/sdapi/v1/progress?skip_current_image=true':{'state':{'job_count':0}}}
            self.reply(mapping.get(self.path,{}))
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(('POST',self.path,data));self.reply(sample(data))
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        cfg=config(endpoint=f'http://127.0.0.1:{server.server_port}',seeds=[42],candidates=['epoch_10'])
        result=cc.run_comparison(cfg,tmp_path/'runs')
        assert result['generated']==2 and result['status']=='complete'
        posts=[r for r in calls if r[0]=='POST']
        assert [r[1] for r in posts]==['/sdapi/v1/txt2img']*2
        assert not any(r[1] in ['/sdapi/v1/interrupt','/sdapi/v1/options'] for r in posts)
    finally: server.shutdown();server.server_close();thread.join(2)


def test_busy_server_refuses_generation(monkeypatch):
    client=cc.WebUIClient();monkeypatch.setattr(client,'request',lambda *a,**kw:{'state':{'job_count':-1}})
    with pytest.raises(cc.ComparisonError):client.ensure_idle()
