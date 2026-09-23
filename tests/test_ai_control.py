"""Real loopback/filesystem/protocol tests; no GPU, SDK or GUI simulation claims."""
from __future__ import annotations
import io
import json
import http.client
import subprocess
import sys
import threading
import time
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from src.core import ai_control as control
from src.core.ai_control import Broker, ControlError, ControlServer, PathScope, CATALOG, digest, validate, strict_json
from src.core.ai_client import LiveClient, connection_config
from src.core.ai_mcp import MCPServer, serve, tool_list
from src.core import ai_caption_journal as journal
from src.core.caption_sync import read_caption_snapshot
from src.core.clothing_profiles import ClothingProfile, ClothingProfileStore, ClothingPart
from src.core.ai_profile_store import commit_profile
from src.core.clothing_io import FileLock

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def broker(tmp_path):
    result = Broker(tmp_path / 'logs')
    result.configure(enabled=True, mode='review', roots=[tmp_path])
    return result

@pytest.fixture
def live(broker, tmp_path):
    server = ControlServer(broker, tmp_path / '.ai-control' / 'session.json')
    server.start()
    try:
        yield server, LiveClient(server.session_file)
    finally:
        server.stop()

@pytest.fixture
def image(tmp_path):
    p = tmp_path / 'dataset' / 'örnek.png'; p.parent.mkdir()
    Image.new('RGB', (32, 40), (20, 40, 60)).save(p)
    p.with_suffix('.txt').write_bytes(b'1girl, standing\r\n')
    return p


def test_transport_default_is_disabled(tmp_path):
    b = Broker()
    with pytest.raises(ControlError, match='Enable'):
        b.submit('app.state', {})
    assert b.requests == {}

@pytest.mark.parametrize('action,args', [
    ('app.navigate', {'route':'edit'}), ('dataset.scan',{}),
    ('caption.set',{'image':'/example.png','text':'new','expected_revision':'0'*64}),
])
def test_observe_refuses_actions(broker, tmp_path, action, args):
    broker.configure(enabled=True, mode='observe', roots=[tmp_path])
    with pytest.raises(ControlError) as exc:
        broker.submit(action, args)
    assert exc.value.code == 'read_only'


def test_exact_human_approval_and_rejection(broker):
    r = broker.submit('caption.set', {'image':'/example.png','text':'x','expected_revision':'0'*64})
    assert r['status'] == 'pending_approval' and broker.take() is None
    broker.approve(r['id'], False)
    assert broker.status(r['id'])['status'] == 'rejected'
    with pytest.raises(ControlError): broker.approve(r['id'], True)
    r2 = broker.submit('dataset.scan', {})
    broker.approve(r2['id'], True)
    assert broker.take()['id'] == r2['id']


def test_delegated_still_requires_profile_and_export_review(broker, tmp_path):
    broker.configure(enabled=True, mode='delegated', roots=[tmp_path])
    assert broker.submit('dataset.scan', {})['status'] == 'queued'
    assert broker.submit('balance.export', {'destination':'/new','expected_revision':'0'*64})['status'] == 'pending_approval'
    profile = {'name':'Blue','master_tag':'shhooldress','parts':[{'tag':'blue skirt'}],'references':[{'image':'/ref.png'}]}
    assert broker.submit('clothing.save_profile', profile)['status'] == 'pending_approval'


def test_duplicate_request_is_executed_at_most_once(broker):
    first = broker.submit('app.state', {}, 'repeat')
    broker.submit('app.state', {}, 'repeat')
    assert broker.take()['id'] == first['id'] and broker.take() is None
    broker.finish('repeat', {'ok':True})
    assert broker.submit('app.state', {}, 'repeat')['result'] == {'ok':True}
    assert broker.take() is None
    with pytest.raises(ControlError, match='different arguments'):
        broker.submit('dataset.list', {}, 'repeat')


def test_request_arguments_are_immutable_copies(broker):
    args = {'route':'edit'}
    broker.submit('app.navigate', args, 'nav')
    args['route'] = 'training'
    assert broker.take()['arguments']['route'] == 'edit'


def test_revocation_cancels_pending_and_removes_completed_data(broker, tmp_path):
    broker.submit('app.state', {}, 'done'); broker.take(); broker.finish('done', {'private_text':'secret'})
    broker.submit('dataset.scan', {}, 'pending')
    broker.submit('app.state', {}, 'running'); broker.take()
    broker.configure(enabled=True, mode='observe', roots=[tmp_path], share_images=False)
    assert broker.status('pending')['status'] == 'cancelled'
    assert broker.status('done')['result'] is None
    broker.finish('running', {'image':{'data':'secret bytes'}})
    assert broker.status('running')['result'] is None
    assert broker.status('running')['result_expired']
    assert 'secret' not in json.dumps(broker.history())


def test_result_expiry_keeps_idempotency_tombstone(broker, monkeypatch):
    monkeypatch.setattr(control, 'LEDGER_BUDGET', 100)
    for i in range(3):
        broker.submit('app.state', {}, f'id{i}'); broker.take(); broker.finish(f'id{i}', {'value':'a'*65})
    assert broker.status('id0')['result_expired']
    assert broker.submit('app.state', {}, 'id0')['status'] == 'succeeded'
    assert broker.take() is None
    assert all('result' not in r for r in broker.history())


def test_oversized_result_is_omitted_not_false_failure(broker, monkeypatch):
    monkeypatch.setattr(control, 'MAX_RESULT', 50)
    broker.submit('app.state', {}, 'big'); broker.take(); broker.finish('big', {'x':'z'*100})
    r = broker.status('big')
    assert r['status'] == 'succeeded' and r['result']['result_omitted']


def test_request_count_limit_does_not_evict_old_ids(tmp_path):
    b = Broker(limit=2); b.configure(enabled=True, mode='review', roots=[tmp_path])
    for i in range(2):
        b.submit('app.state', {}, str(i)); b.take(); b.finish(str(i), {})
    with pytest.raises(ControlError) as exc: b.submit('app.state', {}, 'third')
    assert exc.value.code == 'session_full' and len(b.requests) == 2


def test_pending_queue_is_bounded(broker):
    for i in range(32): broker.submit('app.state', {}, f'id{i}')
    with pytest.raises(ControlError) as exc: broker.submit('app.state', {}, 'overflow')
    assert exc.value.code == 'queue_full'


def test_progress_and_cancel_preserve_final_result(broker):
    r = broker.submit('app.state', {}, 'id'); broker.take()
    broker.progress('id', 1, 5, 'processing')
    assert broker.status('id')['progress']['current'] == 1
    broker.finish('id', {'written':1}, cancelled=True)
    broker.progress('id', 3, 5, 'late')
    assert broker.status('id')['status'] == 'cancelled'
    assert broker.status('id')['progress']['current'] == 1
    assert broker.status('id')['result']['written'] == 1


def test_event_cursor_and_log_do_not_include_user_content(broker, tmp_path):
    r = broker.submit('caption.set', {'image':'/secret/image.png','text':'secret caption','expected_revision':'0'*64})
    broker.approve(r['id'], True); broker.take(); broker.finish(r['id'], {'caption':'secret output'})
    events = broker.event_page(0)
    assert len(events['events']) == 4 and events['next'] == 4
    assert 'secret' not in json.dumps(events)
    assert 'secret' not in next((tmp_path/'logs').glob('*.jsonl')).read_text()

@pytest.mark.parametrize('action,args', [
    ('caption.set', {'image':'x','text':'x','expected_revision':'0'*64,'arbitrary':True}),
    ('app.navigate', {'route':'exec'}), ('app.state', []),
    ('dataset.list', {'limit':True}), ('dataset.list', {'limit':201}),
    ('caption.set', {'image':'x','text':'z'*65537,'expected_revision':'0'*64}),
    ('clothing.selection', {'image':'x','bbox':[0,0,float('nan'),1]}),
    ('clothing.selection', {'image':'x','bbox':[0,0,1]}),
    ('app.navigate', {'route':'edit\x00'}),
])
def test_strict_action_schema_rejects_unsafe_shapes(broker, action, args):
    with pytest.raises(ControlError): broker.submit(action, args)
    assert not broker.requests

@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json_rejects_ambiguous_content(text):
    with pytest.raises(ValueError): strict_json(text)


def test_local_scope_allows_only_selected_folder_and_new_sibling(tmp_path, image):
    scope = PathScope([image.parent])
    assert scope.check(str(image)) == image
    assert scope.check(str(image.parent/'new'), new=True) == image.parent/'new'
    outside = tmp_path/'other.txt'; outside.write_text('no')
    for p in (str(outside), '../other.txt', str(image.parent/'..'/'other.txt'), 'relative.png'):
        with pytest.raises(ControlError): scope.check(p)
    with pytest.raises(ControlError): scope.check(str(image), new=True)


def test_scope_rejects_symlink_files_and_directories(tmp_path, image):
    scope = PathScope([tmp_path])
    link = tmp_path/'link'; link.symlink_to(image.parent, target_is_directory=True)
    with pytest.raises(ControlError): scope.check(str(link/image.name))
    ref = image.parent/'ref.png'; ref.symlink_to(image)
    with pytest.raises(ControlError): scope.check(str(ref))


def test_metadata_is_not_public_input(tmp_path):
    p = tmp_path/'.lh-clothing'/'secret.txt'; p.parent.mkdir(); p.write_text('x')
    with pytest.raises(ControlError): PathScope([tmp_path]).check(str(p))


def test_windows_reparse_detection_works_without_is_junction(tmp_path, monkeypatch):
    fake = SimpleNamespace(st_mode=0o40755, st_file_attributes=0x400)
    monkeypatch.setattr(Path, 'lstat', lambda self: fake)
    assert control.redirected(tmp_path)


def test_http_local_auth_and_capability_schema(live):
    server, client = live
    result = client.request('GET', '/capabilities')
    assert len(result['actions']) == len(CATALOG) and result['mode'] == 'review'
    assert server.http.server_address[0] == '127.0.0.1'
    if os.name != 'nt':
        assert server.session_file.stat().st_mode & 0o077 == 0
    assert server.token not in json.dumps(result)

@pytest.mark.parametrize('bad', ['token','host','origin','unicode_auth'])
def test_http_rejects_unauthorized_browser_or_host(live, bad):
    server, _ = live
    headers = {'Host':f'127.0.0.1:{server.http.server_port}','Authorization':'Bearer '+server.token}
    if bad == 'token': headers['Authorization']='Bearer wrong'
    if bad == 'host': headers['Host']='attacker.example'
    if bad == 'origin': headers['Origin']='http://example.com'
    if bad == 'unicode_auth': headers['Authorization']='\xff'
    conn=http.client.HTTPConnection('127.0.0.1',server.http.server_port,timeout=3)
    try:
        conn.request('GET','/capabilities',headers=headers); response=conn.getresponse()
        assert response.status == 403
    finally: conn.close()


def test_http_enqueue_never_executes_gui_on_socket_thread(live, broker):
    _, client = live
    r = client.call('app.state', {}, 'http_read', wait=0)
    assert r['status']=='queued'
    pending=broker.take(); assert pending['id']=='http_read'
    broker.finish('http_read', {'route':'edit'})
    assert client.request('GET','/requests/http_read')['result']['route']=='edit'
    with pytest.raises(ControlError): client.request('POST','/approve', {'request_id':'http_read'})


def test_http_same_id_and_conflict(live, broker):
    _, client = live
    client.call('app.state', {}, 'id', wait=0)
    assert client.call('app.state', {}, 'id', wait=0)['status']=='queued'
    assert len(broker.requests)==1
    with pytest.raises(ControlError): client.call('dataset.list', {}, 'id', wait=0)


def test_http_oversized_or_ambiguous_bodies_rejected(live, broker):
    server,_=live
    for payload in (b'{"action":"app.state","action":"dataset.scan","arguments":{}}', None):
        conn=http.client.HTTPConnection('127.0.0.1',server.http.server_port,timeout=5)
        try:
            if payload is None:
                conn.putrequest('POST', '/command')
                conn.putheader('Authorization', 'Bearer '+server.token)
                conn.putheader('Content-Type', 'application/json')
                conn.putheader('Content-Length', str(control.MAX_BODY+1))
                conn.endheaders()  # Oversized length is rejected without reading a body.
            else:
                conn.request('POST','/command',payload,headers={'Authorization':'Bearer '+server.token,'Content-Type':'application/json'})
            assert conn.getresponse().status==400
        finally: conn.close()
    assert not broker.requests


def test_stop_removes_only_own_session(live):
    server,client=live
    data=json.loads(server.session_file.read_text()); data['instance']='other'
    server.session_file.write_text(json.dumps(data))
    server.stop()
    assert server.session_file.exists()


def test_stopped_bridge_client_fails_without_auto_restart(tmp_path):
    with pytest.raises(ControlError) as exc: LiveClient(tmp_path/'missing.json').call('app.state', {})
    assert exc.value.code=='not_connected'
    assert not (tmp_path/'missing.json').exists()


def test_session_path_cannot_be_redirected(tmp_path, broker):
    real=tmp_path/'real';real.mkdir();link=tmp_path/'link';link.symlink_to(real, target_is_directory=True)
    server=ControlServer(broker, link/'session.json')
    with pytest.raises(ControlError): server.start()
    assert server.http is None and not (real/'session.json').exists()


def test_mcp_initializes_without_connecting_or_importing_models(tmp_path):
    m=MCPServer(tmp_path/'none')
    result=m.handle({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25'}})['result']
    assert result['protocolVersion']=='2025-11-25' and result['capabilities']=={'tools':{}}
    tools=m.handle({'jsonrpc':'2.0','id':2,'method':'tools/list'})['result']['tools']
    assert len(tools)==len(CATALOG)+4
    assert not any(t['name'] in ('execute','shell','approve','eval') for t in tools)


def test_mcp_requires_initialization_and_ignores_notification_writes(tmp_path):
    m=MCPServer(tmp_path/'none')
    assert m.handle({'jsonrpc':'2.0','id':1,'method':'tools/list'})['error']['code']==-32002
    assert m.handle({'jsonrpc':'2.0','method':'tools/call','params':{'name':'harvester_caption_set'}}) is None


def test_mcp_tool_errors_are_iserror_not_fake_success(tmp_path):
    m=MCPServer(tmp_path/'none');m.initialized=True
    r=m.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'harvester_app_state'}})
    assert r['result']['isError'] and r['result']['structuredContent']['error']['code']=='not_connected'


def test_mcp_preserves_command_id_but_removes_it_from_action_args(tmp_path):
    m=MCPServer(tmp_path/'none');m.initialized=True
    m.client.call=Mock(return_value={'id':'stable','status':'queued'})
    m.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'harvester_app_state','arguments':{'command_id':'stable'}}})
    m.client.call.assert_called_once_with('app.state', {}, 'stable')


def test_mcp_image_separate_content_not_base64_text_copy():
    r=MCPServer._content({'status':'succeeded','result':{'image':{'type':'image','data':'privatebytes','mimeType':'image/jpeg'}}})
    assert r['content'][1]['type']=='image'
    assert 'privatebytes' not in r['content'][0]['text']
    assert r['structuredContent']['result']['image_in_content']


def test_mcp_stdio_framing_is_real_subprocess():
    messages=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25'}},
              {'jsonrpc':'2.0','method':'notifications/initialized'},
              {'jsonrpc':'2.0','id':2,'method':'tools/list'}]
    p=subprocess.run([sys.executable,str(ROOT/'mcp_server.py')],input=''.join(json.dumps(m)+'\n' for m in messages),text=True,capture_output=True,timeout=10,cwd=tempfile.gettempdir())
    assert p.returncode==0, p.stderr
    lines=p.stdout.splitlines(); assert len(lines)==2
    assert json.loads(lines[1])['id']==2 and 'tools' in json.loads(lines[1])['result']


def test_mcp_stdio_rejects_duplicate_keys_and_recovers(tmp_path):
    out=io.BytesIO()
    data=b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}\n'+b'{"jsonrpc":"2.0","id":3,"method":"ping"}\n'
    serve(tmp_path/'none',io.BytesIO(data),out)
    lines=[json.loads(p) for p in out.getvalue().splitlines()]
    assert lines[0]['error']['code']==-32700 and lines[1]['id']==3


def test_client_configuration_points_to_actual_paths_without_tokens(tmp_path):
    text=connection_config(tmp_path, tmp_path/'venv'/'pythonw.exe')
    assert 'python.exe' in text and 'pythonw.exe' not in text
    assert 'mcp_server.py' in text and 'token' not in text and 'mcp_servers.lora_harvester' in text
    blob=json.loads(text[text.index('{'):])
    assert blob['mcpServers']['lora_harvester']['args']==[str(tmp_path/'mcp_server.py')]


def test_ai_caption_roundtrip_preserves_bom_and_crlf(image):
    p=image.with_suffix('.txt');before=b'\xef\xbb\xbf1girl\r\n';p.write_bytes(before)
    snap,name=journal.save(read_caption_snapshot(image),'1girl, mavi etek\n','req')
    assert p.read_bytes()==b'\xef\xbb\xbf1girl, mavi etek\r\n'
    assert name.startswith('ai-caption-')
    assert journal.undo(image,name).raw==before and p.read_bytes()==before
    assert journal.undo(image,name).raw==before


def test_ai_caption_new_file_undo_restores_absence(image):
    p=image.with_suffix('.txt');p.unlink()
    _,name=journal.save(read_caption_snapshot(image),'new','req')
    assert p.read_text()=='new'
    journal.undo(image,name); assert not p.exists()


def test_ai_caption_save_and_undo_reject_later_edits(image):
    snapshot=read_caption_snapshot(image);image.with_suffix('.txt').write_text('human edit')
    with pytest.raises(ControlError): journal.save(snapshot,'bad','req')
    _,name=journal.save(read_caption_snapshot(image),'ai edit','req')
    image.with_suffix('.txt').write_text('later human edit')
    with pytest.raises(ControlError): journal.undo(image,name)
    assert image.with_suffix('.txt').read_text()=='later human edit'


def test_ai_caption_save_source_stamp_conflict(image):
    stamp=journal.image_stamp(image);snap=read_caption_snapshot(image)
    Image.new('RGB',(33,44)).save(image)
    with pytest.raises(ControlError,match='Source changed'):journal.save(snap,'new','r',expected_stamp=stamp)
    assert image.with_suffix('.txt').read_bytes()==snap.raw


def test_ai_caption_undo_source_changed_is_refused(image):
    _,name=journal.save(read_caption_snapshot(image),'ai edit','req')
    Image.new('RGB',(33,44)).save(image)
    with pytest.raises(ControlError,match='Source image changed'): journal.undo(image,name)


def test_ai_caption_writes_share_original_clothing_lock(image):
    snap=read_caption_snapshot(image)
    with FileLock(image.parent/'.lh-clothing'/'write.lock'):
        with pytest.raises(RuntimeError): journal.save(snap,'new','req')
    assert image.with_suffix('.txt').read_bytes()==snap.raw


def test_journal_write_failure_prevents_caption_change(image,monkeypatch):
    before=image.with_suffix('.txt').read_bytes()
    monkeypatch.setattr(journal,'atomic_json',Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError):journal.save(read_caption_snapshot(image),'new','req')
    assert image.with_suffix('.txt').read_bytes()==before


def test_after_write_journal_failure_is_recoverable(image,monkeypatch):
    before=image.with_suffix('.txt').read_bytes();real=journal.atomic_json
    def write(path,data):
        if data['phase']=='applied': raise OSError('interrupted marker')
        real(path,data)
    monkeypatch.setattr(journal,'atomic_json',write)
    _,name=journal.save(read_caption_snapshot(image),'new','req')
    assert json.loads((image.parent/'.lh-clothing'/'history'/name).read_text())['phase']=='prepared'
    journal.undo(image,name)
    assert image.with_suffix('.txt').read_bytes()==before


def test_undo_interruption_after_restore_is_resumable(image,monkeypatch):
    before=image.with_suffix('.txt').read_bytes();_,name=journal.save(read_caption_snapshot(image),'new','req')
    real=journal.atomic_json
    def write(path,data):
        if data['phase']=='undone': raise OSError('interrupted marker')
        real(path,data)
    monkeypatch.setattr(journal,'atomic_json',write)
    with pytest.raises(OSError):journal.undo(image,name)
    assert image.with_suffix('.txt').read_bytes()==before
    monkeypatch.setattr(journal,'atomic_json',real)
    assert journal.undo(image,name).raw==before


def test_ai_caption_rejects_wrong_journal_and_shared_basename(image):
    with pytest.raises(ControlError):journal.undo(image,'../anything.json')
    Image.new('RGB',(32,40)).save(image.with_suffix('.jpg'))
    with pytest.raises(RuntimeError):journal.save(read_caption_snapshot(image),'new','req')


def test_compare_and_swap_profile_keeps_changed_profile(image,tmp_path):
    store=ClothingProfileStore(tmp_path/'library')
    ref=store.import_reference(image)
    p=ClothingProfile('blue','shhooldress',[ClothingPart('blue skirt')],references=[ref],id='p')
    commit_profile(store,p)
    original=digest(p.to_dict())
    p.name='Human edited';store.upsert(p)
    p.name='Agent edited'
    with pytest.raises(ControlError):commit_profile(store,p,original)
    assert store.load()[0].name=='Human edited'


def test_compare_and_swap_profile_valid_revision_and_variant_rules(image,tmp_path):
    store=ClothingProfileStore(tmp_path/'library');ref=store.import_reference(image)
    p=ClothingProfile('blue','blueuniform',[ClothingPart('blue skirt')],references=[ref],id='blue')
    commit_profile(store,p);rev=digest(p.to_dict());p.name='Blue new name';commit_profile(store,p,rev)
    other=ClothingProfile('red','blueuniform',[ClothingPart('red skirt')],references=[ref],id='red')
    with pytest.raises(ValueError):commit_profile(store,other)
    assert len(store.load())==1
