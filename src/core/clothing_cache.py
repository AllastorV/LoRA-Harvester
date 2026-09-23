"""Content-addressed, model-versioned inference cache; never caches failed calls."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from .clothing_io import digest_json
from .clothing_backend import validate_schema

PROMPT_VERSION = 'clothing-v1.1.0'


class ClothingCache:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=15)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses '
                        '(key TEXT PRIMARY KEY, value TEXT NOT NULL, accessed REAL NOT NULL)')
        self.db.commit()

    def get(self, key):
        row = self.db.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
        except ValueError:
            self.db.execute('DELETE FROM responses WHERE key=?', (key,))
            self.db.commit()
            return None
        return value

    def put(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO responses VALUES (?, ?, ?)',
                        (key, json.dumps(value, ensure_ascii=False, allow_nan=False), time.time()))
        self.db.commit()

    def close(self):
        self.db.close()


class CachedVision:
    def __init__(self, backend, cache=None):
        self.backend, self.cache = backend, cache

    def infer(self, prompt, images, schema):
        self.backend.check_cancelled()
        key = digest_json({'version': PROMPT_VERSION, 'model': self.backend.fingerprint,
                           'ctx': self.backend.settings.num_ctx, 'prompt': prompt,
                           'images': images, 'schema': schema})
        value = self.cache.get(key) if self.cache else None
        if value is not None:
            try:
                return validate_schema(value, schema)
            except RuntimeError:
                pass
        value = self.backend.infer(prompt, images, schema)
        validate_schema(value, schema)
        self.backend.check_cancelled()
        if self.cache:
            self.cache.put(key, value)
        return value
