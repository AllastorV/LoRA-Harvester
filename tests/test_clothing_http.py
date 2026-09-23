"""Real localhost HTTP integration with a fake Ollama server; no model downloads."""
from __future__ import annotations
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.clothing_backend import (OllamaClothingBackend, ClothingBackendError,
                                       ClothingCancelled, object_schema, BOOL)
from src.core.clothing_profiles import ClothingSettings


class FakeOllamaHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send_json(self, data, status=200):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.server.requests.append(('GET', self.path, None))
        self.send_json({'models': self.server.models})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(('POST', self.path, payload))
        if self.path == '/api/show':
            self.send_json(self.server.show)
        elif self.path == '/api/generate':
            self.server.entered.set()
            if self.server.block:
                self.server.release.wait(4)
            answer = self.server.answers.pop(0) if self.server.answers else {
                'done': True, 'response': '{"ok": true}'}
            self.send_json(answer)
        elif self.path == '/api/pull':
            raw = b'{"status":"pulling","total":100,"completed":50}\n{"status":"success"}\n'
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send_json({'error': 'not found'}, 404)


class OllamaTransportTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllamaHandler)
        self.server.daemon_threads = True
        self.server.requests = []
        self.server.models = [{'name': 'qwen3-vl:4b', 'digest': 'fake-sha', 'size': 1}]
        self.server.show = {'capabilities': ['completion', 'vision']}
        self.server.answers = []
        self.server.entered = threading.Event()
        self.server.release = threading.Event()
        self.server.block = False
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self.thread.start()
        self.settings = ClothingSettings(endpoint=f'http://127.0.0.1:{self.server.server_port}')
        self.backend = OllamaClothingBackend(self.settings)

    def tearDown(self):
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_model_check_requires_local_vision_capabilities(self):
        data = self.backend.check()
        self.assertEqual(data['digest'], 'fake-sha')
        self.assertIn('fake-sha', self.backend.fingerprint)

    def test_missing_model_has_actionable_error(self):
        self.server.models = []
        with self.assertRaisesRegex(ClothingBackendError, 'Download model'):
            self.backend.check()

    def test_text_only_model_is_rejected(self):
        self.server.show = {'capabilities': ['completion']}
        with self.assertRaises(ClothingBackendError):
            self.backend.check()

    def test_cloud_alias_is_rejected_even_on_local_server(self):
        self.server.show['remote_host'] = 'https://cloud.example'
        with self.assertRaises(ClothingBackendError):
            self.backend.check()

    def test_generation_sends_schema_images_and_no_sampling(self):
        self.backend.check()
        schema = object_schema({'ok': BOOL})
        result = self.backend.infer('visible clothes only', ['TARGET', 'REFERENCE'], schema)
        self.assertEqual(result, {'ok': True})
        sent = self.server.requests[-1][2]
        self.assertEqual(sent['images'], ['TARGET', 'REFERENCE'])
        self.assertEqual(sent['format'], schema)
        self.assertFalse(sent['stream'])
        self.assertFalse(sent['think'])
        self.assertEqual(sent['options']['temperature'], 0)

    def test_invalid_json_retried_once_then_rejected(self):
        self.server.answers = [{'done': True, 'response': 'bad json'}] * 2
        with self.assertRaises(ClothingBackendError):
            self.backend.infer('test', ['image'], object_schema({'ok': BOOL}))
        self.assertEqual(len(self.server.requests), 2)

    def test_wrong_schema_retried(self):
        self.server.answers = [{'done': True, 'response': '{"ok":"yes"}'},
                               {'done': True, 'response': '{"ok":true}'}]
        result = self.backend.infer('test', ['image'], object_schema({'ok': BOOL}))
        self.assertEqual(result, {'ok': True})
        self.assertEqual(len(self.server.requests), 2)

    def test_truncated_output_never_accepted(self):
        self.server.answers = [{'done': True, 'done_reason': 'length', 'response': '{"ok":true}'}] * 2
        with self.assertRaises(ClothingBackendError):
            self.backend.infer('test', ['image'], object_schema({'ok': BOOL}))

    def test_pull_progress_and_success(self):
        seen = []
        self.backend.pull(seen.append)
        self.assertEqual(seen[-1]['status'], 'success')
        self.assertEqual(seen[0]['completed'], 50)

    def test_unload_sends_keep_alive_zero(self):
        self.backend.unload()
        payload = self.server.requests[-1][2]
        self.assertEqual(payload['keep_alive'], 0)
        self.assertNotIn('images', payload)

    def test_cancel_interrupts_waiting_http_request(self):
        self.server.block = True
        errors = []
        def invoke():
            try:
                self.backend.infer('test', ['image'], object_schema({'ok': BOOL}))
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        self.assertTrue(self.server.entered.wait(1))
        self.backend.cancel()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], ClothingCancelled)

    def test_obviously_oversized_prompt_rejected_before_http(self):
        with self.assertRaises(ClothingBackendError):
            self.backend.infer('x' * 60000, ['image1', 'image2'], object_schema({'ok': BOOL}))
        self.assertEqual(self.server.requests, [])

    def test_non_string_response_is_retried_and_rejected_cleanly(self):
        self.server.answers = [{'done': True, 'response': None}] * 2
        with self.assertRaises(ClothingBackendError):
            self.backend.infer('test', ['image'], object_schema({'ok': BOOL}))
        self.assertEqual(len(self.server.requests), 2)

    def test_pipeline_stop_predicate_interrupts_request_without_cancel_method(self):
        stop = threading.Event()
        class StopEvent(threading.Event):
            def is_set(self):
                return super().is_set() or stop.is_set()
        self.backend.cancel_event = StopEvent()
        self.server.block = True
        errors = []
        def invoke():
            try:
                self.backend.infer('test', ['image'], object_schema({'ok': BOOL}))
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        self.assertTrue(self.server.entered.wait(1))
        stop.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], ClothingCancelled)


if __name__ == '__main__':
    unittest.main()
