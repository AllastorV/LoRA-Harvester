"""Final audit regressions. All model responses are controlled test doubles."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from test_clothing_core import Fixture, ScriptedBackend, decision
from src.core.clothing_backend import ClothingBackendError, validate_endpoint, validate_schema
from src.core.clothing_captions import CaptionConflict, apply_caption, prepare_caption, undo_caption
from src.core.clothing_io import atomic_json, digest_bytes, FileLock
from src.core.clothing_matcher import parts_schema
from src.core.clothing_service import record_job, undo_last_job, run_clothing_batch
from src.core.dataset_files import transfer_image_pair


class OutfitAuditTests(Fixture):
    def add_reference(self):
        other = self.root / 'second.png'
        Image.new('RGB', (200, 300), 'blue').save(other)
        self.profile.references.append(self.store.import_reference(other))
        self.store.upsert(self.profile)

    def preview(self):
        self.image.with_suffix('.txt').write_text('1girl', encoding='utf-8')
        result, _ = self.result()
        return prepare_caption(result)

    def test_conflicting_reference_color_does_not_choose_lucky_best_match(self):
        self.add_reference()
        result, _ = self.result([decision(), decision(visible_color_match='no', identity_score=.15)])
        self.assertEqual(result.status, 'review')
        self.assertEqual(result.tags, [])

    def test_visibility_disagreement_never_adds_uncertain_part(self):
        self.add_reference()
        result, _ = self.result([decision(), decision(('visible', 'not_visible', 'not_visible'), identity_score=.90)])
        self.assertEqual(result.status, 'matched')
        self.assertEqual(result.tags, ['shhooldress', 'serafaku'])

    def test_new_competing_profile_invalidates_old_preview(self):
        preview = self.preview()
        rival = copy.deepcopy(self.profile)
        rival.id, rival.master_tag = 'rival', 'red_variant'
        rival.parts[1].tag = 'red skirt'
        self.store.upsert(rival)
        with self.assertRaises(CaptionConflict):
            apply_caption(preview, self.store)
        self.assertEqual(self.image.with_suffix('.txt').read_text(), '1girl')

    def test_changed_rival_reference_invalidates_old_preview(self):
        rival = copy.deepcopy(self.profile)
        rival.id, rival.master_tag = 'rival', 'red_variant'
        rival.parts[1].tag = 'red skirt'
        self.store.upsert(rival)
        self.image.with_suffix('.txt').write_text('1girl')
        result, _ = self.result([decision(), decision(verdict='different', identity_score=.05)])
        preview = prepare_caption(result)
        rival.enabled = False
        self.store.upsert(rival)
        with self.assertRaises(CaptionConflict):
            apply_caption(preview, self.store)

    def test_undo_recovers_after_caption_restored_but_state_write_fails(self):
        preview = self.preview()
        journal = apply_caption(preview, self.store)
        import src.core.clothing_captions as mod
        original_restore = mod._restore
        def fail_state(path, data):
            if Path(path).parent.name == 'state':
                raise OSError('simulated interrupted state restore')
            return original_restore(path, data)
        with patch.object(mod, '_restore', side_effect=fail_state):
            with self.assertRaises(OSError):
                undo_caption(journal)
        undo_caption(journal)
        self.assertEqual(self.image.with_suffix('.txt').read_text(), '1girl')

    def test_recovered_journal_does_not_block_undo_job_forever(self):
        journal = apply_caption(self.preview(), self.store)
        record_job(self.store, [journal])
        undo_caption(journal)
        outcome = undo_last_job(self.store)
        self.assertEqual(outcome['errors'], [])
        self.assertEqual(json.loads((self.store.root / 'undo_jobs.json').read_text()), [])

    def test_undo_rejects_journal_targeting_unrelated_filename(self):
        journal = Path(apply_caption(self.preview(), self.store))
        unrelated = self.image.parent / 'unrelated.txt'
        unrelated.write_bytes(self.image.with_suffix('.txt').read_bytes())
        record = json.loads(journal.read_text())
        record['caption_name'] = unrelated.name
        atomic_json(journal, record)
        with self.assertRaises(CaptionConflict):
            undo_caption(journal)

    def test_duplicate_model_ids_fail_before_cache_write(self):
        data = decision()
        data['parts'][1]['id'] = 'p0'
        with self.assertRaises(ClothingBackendError):
            validate_schema(data, parts_schema(self.profile))

    def test_empty_batch_does_not_contact_model(self):
        backend = ScriptedBackend([], self.settings)
        with patch.object(backend, 'check', side_effect=AssertionError('must not connect')):
            stats = run_clothing_batch([], store=self.store, settings=self.settings, backend=backend)
        self.assertEqual(stats['total'], 0)

    def test_disabled_rival_can_reuse_part_tag_but_not_another_master(self):
        rival = copy.deepcopy(self.profile)
        rival.id, rival.master_tag = 'other', 'serafaku'
        rival.parts = [copy.deepcopy(self.profile.parts[1])]
        with self.assertRaises(ValueError):
            self.store.upsert(rival)

    def test_transfer_carries_caption_json_and_clothing_state(self):
        preview = self.preview()
        apply_caption(preview, self.store)
        self.image.with_suffix('.json').write_text('{"origin":"test"}')
        dest = transfer_image_pair(self.image, self.root / 'sorted', copy=False)
        self.assertTrue(dest.exists())
        self.assertFalse(self.image.exists())
        self.assertEqual(dest.with_suffix('.txt').read_bytes(), preview.after)
        self.assertTrue(dest.with_suffix('.json').exists())
        state = dest.parent / '.lh-clothing/state' / (digest_bytes(dest.name.encode()) + '.json')
        self.assertTrue(state.exists())

    def test_transfer_never_overwrites_orphan_caption_or_other_image_extension(self):
        self.image.with_suffix('.txt').write_text('source')
        target = self.root / 'sorted'
        target.mkdir()
        orphan = target / self.image.with_suffix('.txt').name
        orphan.write_text('keep me')
        first = transfer_image_pair(self.image, target)
        self.assertNotEqual(first.stem, self.image.stem)
        self.assertEqual(orphan.read_text(), 'keep me')
        Image.new('RGB', (20, 20)).save(target / self.image.with_suffix('.jpg').name)
        second = transfer_image_pair(self.image, target)
        self.assertNotEqual(second.stem, first.stem)
        self.assertEqual(second.with_suffix('.txt').read_text(), 'source')

    def test_transfer_staging_failure_keeps_complete_original_pair(self):
        self.image.with_suffix('.txt').write_text('keep source')
        import src.core.dataset_files as mod
        real_copy = mod.shutil.copy2
        def fail_caption(src, dst):
            if Path(src).suffix == '.txt':
                raise OSError('test disk full')
            return real_copy(src, dst)
        target = self.root / 'sorted'
        with patch.object(mod.shutil, 'copy2', side_effect=fail_caption):
            with self.assertRaises(OSError):
                transfer_image_pair(self.image, target, copy=False)
        self.assertTrue(self.image.exists())
        self.assertEqual(self.image.with_suffix('.txt').read_text(), 'keep source')
        self.assertEqual(list(target.glob('*.png')), [])

    def test_transfer_publication_failure_rolls_back_image(self):
        self.image.with_suffix('.txt').write_text('keep source')
        import src.core.dataset_files as mod
        real_publish = mod._publish_exclusive
        def fail_caption(src, dst):
            if Path(dst).suffix == '.txt':
                raise OSError('test failed publication')
            return real_publish(src, dst)
        target = self.root / 'sorted'
        with patch.object(mod, '_publish_exclusive', side_effect=fail_caption):
            with self.assertRaises(OSError):
                transfer_image_pair(self.image, target, copy=False)
        self.assertTrue(self.image.exists())
        self.assertEqual(list(target.glob('*.png')), [])


class ProtocolAuditTests(unittest.TestCase):
    def test_port_zero_is_not_silently_changed_to_default(self):
        with self.assertRaises(ValueError):
            validate_endpoint('http://127.0.0.1:0')


if __name__ == '__main__':
    unittest.main()
