"""Behavioral tests use a scripted vision backend, not an accuracy benchmark."""
from __future__ import annotations
import copy
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.clothing_io import (atomic_json, crop_rgb, file_digest, find_images,
                                 load_rgb, validate_box, FileLock)
from src.core.clothing_profiles import (ClothingProfileStore, ClothingSettings,
                                        ClothingProfile, ClothingPart, ClothingReference,
                                        parse_tags, tag_key)
from src.core.clothing_backend import (ClothingBackendError, ClothingCancelled,
                                       OllamaClothingBackend,
                                       validate_endpoint, validate_schema, object_schema,
                                       SCORE, BOOL)
from src.core.clothing_cache import ClothingCache, CachedVision
from src.core.clothing_matcher import ClothingMatcher, ClothingResult
from src.core.clothing_captions import (prepare_caption, apply_caption, undo_caption,
                                        CaptionConflict, merge_caption)
from src.core.clothing_service import (run_clothing_batch, save_selection, load_selection,
                                       undo_last_job, optional_pipeline_pass)


def decision(states=('visible', 'visible', 'not_visible'), **kwargs):
    value = {'reference_single_outfit': True, 'target_single_outfit': True,
             'outfit_visible': True, 'verdict': 'match', 'identity_score': .95,
             'visible_design_match': True, 'visible_color_match': 'yes',
             'evidence': 'Distinctive collar and blue pleated skirt are visible.',
             'parts': [{'id': f'p{i}', 'visibility': s, 'score': .96,
                        'evidence': 'Observed in target pixels.' if s == 'visible' else 'Outside crop.'}
                       for i, s in enumerate(states)]}
    value.update(kwargs)
    return value


class ScriptedBackend:
    def __init__(self, answers=None, settings=None):
        self.settings = settings or ClothingSettings(person_mode='manual', use_cache=False)
        self.fingerprint = 'scripted-only-not-a-real-vision-model'
        self.answers = list(answers or [])
        self.calls = []
        self.cancel_event = threading.Event()
        self.unloaded = False

    def check(self):
        self.check_cancelled()
        return {'model': 'scripted'}

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise ClothingCancelled()

    def infer(self, prompt, images, schema):
        self.calls.append((prompt, images, schema))
        if not self.answers:
            raise AssertionError('Missing scripted response.')
        answer = copy.deepcopy(self.answers.pop(0))
        return validate_schema(answer, schema)

    def unload(self):
        self.unloaded = True


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ClothingProfileStore(self.root / 'library')
        self.image = self.root / 'dataset' / 'örnek.png'
        self.image.parent.mkdir()
        Image.new('RGB', (200, 300), (10, 20, 230)).save(self.image)
        self.reference = self.root / 'reference.png'
        Image.new('RGB', (200, 300), (10, 30, 200)).save(self.reference)
        self.profile = ClothingProfile(
            'Blue uniform', 'shhooldress',
            [ClothingPart('serafaku'), ClothingPart('blue skirt'), ClothingPart('black thighighhs')],
            [self.store.import_reference(self.reference)])
        self.store.upsert(self.profile)
        self.settings = ClothingSettings(person_mode='manual', use_cache=False)

    def tearDown(self):
        self.temp.cleanup()

    def result(self, answers=None, profiles=None, bbox=(0, 0, 1, 1), settings=None):
        backend = ScriptedBackend(answers or [decision()], settings or self.settings)
        matcher = ClothingMatcher(self.store, settings or self.settings, CachedVision(backend))
        return matcher.analyze(self.image, profiles or self.store.load(), bbox), backend


class ProfileTests(Fixture):
    def test_typo_tags_preserved_verbatim(self):
        loaded = self.store.load()[0]
        self.assertEqual(loaded.master_tag, 'shhooldress')
        self.assertEqual(loaded.parts[0].tag, 'serafaku')
        self.assertEqual(loaded.parts[-1].tag, 'black thighighhs')

    def test_twelve_profiles_not_capped_at_eight(self):
        for i in range(11):
            p = copy.deepcopy(self.profile)
            p.id, p.name, p.master_tag = f'variant{i}', f'Variant {i}', f'master{i}'
            self.store.upsert(p)
        self.assertEqual(len(self.store.load()), 12)

    def test_color_variants_require_unique_master(self):
        p = copy.deepcopy(self.profile)
        p.id, p.name = 'red', 'Red uniform'
        p.parts[1].tag = 'red skirt'
        with self.assertRaises(ValueError):
            self.store.upsert(p)
        p.master_tag = 'reddress'
        self.store.upsert(p)
        self.assertEqual(len(self.store.load()), 2)

    def test_atomic_store_corruption_not_silently_erased(self):
        self.store.path.write_text('{ broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.store.upsert(self.profile)
        self.assertEqual(self.store.path.read_text(), '{ broken')

    def test_reference_copy_independent_of_original(self):
        self.reference.unlink()
        self.assertTrue(self.store.reference_path(self.profile.references[0]).is_file())

    def test_reference_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.store.reference_path(ClothingReference('../reference.png'))

    def test_box_validation(self):
        for box in ((.4, 0, .2, 1), (0, 0, 1.1, 1), (0, 0, float('nan'), 1), (True, 0, 1, 1)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box)

    def test_master_not_repeated_as_part(self):
        self.profile.parts.append(ClothingPart('shhooldress'))
        with self.assertRaises(ValueError):
            self.profile.validate()

    def test_comma_tags_are_split_without_spellcheck(self):
        self.assertEqual(parse_tags(' shhooldress,serafaku, blue_skirt, blue skirt '),
                         ['shhooldress', 'serafaku', 'blue_skirt'])

    def test_string_false_cannot_enable_automatic_writes(self):
        with self.assertRaises(ValueError):
            ClothingSettings.from_dict({'auto_after_video': 'false'})

    def test_non_boolean_profile_enabled_is_rejected(self):
        self.profile.enabled = 'false'
        with self.assertRaises(ValueError):
            self.store.upsert(self.profile)

    def test_settings_roundtrip_auto_is_opt_in(self):
        settings = self.store.load_settings()
        self.assertFalse(settings.auto_after_video)
        self.assertFalse(settings.auto_after_caption)
        self.store.save_settings(settings)
        self.assertEqual(asdict(settings), asdict(self.store.load_settings()))

    def test_delete_profile_keeps_reference_files(self):
        path = self.store.reference_path(self.profile.references[0])
        self.store.delete(self.profile.id)
        self.assertEqual(self.store.load(), [])
        self.assertTrue(path.is_file())

    def test_reference_crop_changes_fingerprint(self):
        before = self.store.fingerprint(self.profile)
        self.profile.references[0].bbox = [0, 0, 1, .7]
        self.assertNotEqual(before, self.store.fingerprint(self.profile))


class MatchingTests(Fixture):
    def test_full_outfit(self):
        result, _ = self.result([decision(('visible',) * 3)])
        self.assertEqual(result.tags, ['shhooldress', 'serafaku', 'blue skirt', 'black thighighhs'])

    def test_cropped_legs_does_not_add_socks(self):
        result, _ = self.result()
        self.assertEqual(result.tags, ['shhooldress', 'serafaku', 'blue skirt'])
        self.assertEqual(result.status, 'matched')

    def test_upper_only_keeps_master(self):
        result, _ = self.result([decision(('visible', 'not_visible', 'not_visible'))])
        self.assertEqual(result.tags, ['shhooldress', 'serafaku'])

    def test_face_only_cannot_add_master(self):
        result, _ = self.result([decision(('not_visible',) * 3, outfit_visible=False)])
        self.assertEqual(result.tags, [])
        self.assertEqual(result.status, 'no_match')

    def test_uncertain_color_is_not_visible_part(self):
        result, _ = self.result([decision(('visible', 'uncertain', 'not_visible'))])
        self.assertEqual(result.tags, ['shhooldress', 'serafaku'])

    def test_unseen_color_variant_blocks_master_even_with_score_gap(self):
        other = copy.deepcopy(self.profile)
        other.id, other.name, other.master_tag = 'red', 'Red variant', 'redmaster'
        other.parts[1].tag = 'red skirt'
        self.store.upsert(other)
        result, _ = self.result([decision(('visible', 'not_visible', 'not_visible')),
                                 decision(('visible', 'not_visible', 'not_visible'),
                                          verdict='possible', identity_score=.45)])
        self.assertEqual(result.status, 'review')
        self.assertEqual(result.tags, [])

    def test_visible_color_mismatch_rejected(self):
        result, _ = self.result([decision(visible_color_match='no')])
        self.assertEqual(result.status, 'no_match')

    def test_visible_conflicting_part_rejected(self):
        result, _ = self.result([decision(('visible', 'contradicted', 'not_visible'))])
        self.assertEqual(result.status, 'no_match')

    def test_part_threshold_independent_of_identity(self):
        d = decision()
        d['parts'][1]['score'] = .5
        result, _ = self.result([d])
        self.assertEqual(result.tags, ['shhooldress', 'serafaku'])

    def test_identity_threshold(self):
        result, _ = self.result([decision(identity_score=.50)])
        self.assertEqual(result.status, 'review')

    def test_lookalike_face_or_style_not_enough(self):
        result, _ = self.result([decision(visible_design_match=False)])
        self.assertEqual(result.status, 'review')

    def test_manual_mode_requires_selection(self):
        result, backend = self.result(bbox=None)
        self.assertEqual(result.status, 'review')
        self.assertEqual(backend.calls, [])

    def test_main_person_largest_is_the_only_crop_sent(self):
        settings = ClothingSettings(person_mode='largest', use_cache=False)
        people = {'uncertain': False, 'people': [{'bbox': [0, 0, 300, 500], 'score': .95},
                                                {'bbox': [500, 0, 1000, 1000], 'score': .95}]}
        result, backend = self.result([people, decision()], bbox=None, settings=settings)
        self.assertEqual(result.bbox, [.5, 0, 1, 1])
        import base64
        target = Image.open(io.BytesIO(base64.b64decode(backend.calls[1][1][0])))
        self.assertEqual(target.size, (100, 300))

    def test_center_selection(self):
        settings = ClothingSettings(person_mode='center', use_cache=False)
        people = {'uncertain': False, 'people': [{'bbox': [0, 0, 300, 900], 'score': .95},
                                                {'bbox': [400, 300, 650, 650], 'score': .95}]}
        result, _ = self.result([people, decision()], bbox=None, settings=settings)
        self.assertEqual(result.bbox, [.4, .3, .65, .65])

    def test_ambiguous_person_detection_abstains(self):
        settings = ClothingSettings(person_mode='largest', use_cache=False)
        result, _ = self.result([{'uncertain': True, 'people': []}], bbox=None, settings=settings)
        self.assertEqual(result.status, 'review')

    def test_duplicate_part_ids_rejected(self):
        d = decision()
        d['parts'][1]['id'] = 'p0'
        with self.assertRaises(ClothingBackendError):
            self.result([d])

    def test_unknown_model_tag_rejected(self):
        d = decision()
        d['parts'][0]['id'] = 'invented_tag'
        with self.assertRaises(ClothingBackendError):
            self.result([d])

    def test_two_people_in_final_crop_are_not_merged(self):
        with self.assertRaises(ValueError):
            self.result([decision(target_single_outfit=False)])

    def test_multiple_references_are_all_used(self):
        self.profile.references.append(copy.deepcopy(self.profile.references[0]))
        self.store.upsert(self.profile)
        result, backend = self.result([decision(verdict='different', identity_score=.1), decision()])
        self.assertEqual(len(backend.calls), 2)
        # A lucky reference may not hide a contradictory reference verdict.
        self.assertEqual(result.status, 'review')

    def test_all_twelve_profiles_are_compared(self):
        for i in range(11):
            p = copy.deepcopy(self.profile)
            p.id, p.master_tag = f'other{i}', f'other_master{i}'
            self.store.upsert(p)
        result, backend = self.result([decision()] + [decision(verdict='different', identity_score=.05)] * 11)
        self.assertEqual(len(backend.calls), 12)
        self.assertEqual(result.profile_id, self.profile.id)

    def test_reference_analysis_does_not_change_literal_tags(self):
        response = {'single_outfit': True, 'summary': 'Sailor uniform',
                    'parts': [{'id': f'p{i}', 'observable': i < 2, 'description': text}
                              for i, text in enumerate(['Sailor collar blouse', 'Blue pleated skirt', ''])]}
        backend = ScriptedBackend([response])
        matcher = ClothingMatcher(self.store, self.settings, CachedVision(backend))
        result = matcher.analyze_references(self.profile)
        self.assertEqual(result['suggestions']['serafaku'], 'Sailor collar blouse')
        self.assertEqual(self.store.load()[0].parts[0].tag, 'serafaku')

    def test_reference_with_multiple_outfits_is_rejected(self):
        response = {'single_outfit': False, 'summary': '',
                    'parts': [{'id': f'p{i}', 'observable': False, 'description': ''} for i in range(3)]}
        with self.assertRaises(ValueError):
            ClothingMatcher(self.store, self.settings, CachedVision(ScriptedBackend([response]))).analyze_references(self.profile)


class CaptionTests(Fixture):
    def preview(self, text=None, answers=None):
        if text is not None:
            self.image.with_suffix('.txt').write_text(text, encoding='utf-8')
        result, _ = self.result(answers)
        return prepare_caption(result)

    def test_preview_does_not_write(self):
        preview = self.preview('1girl, outdoors')
        self.assertEqual(self.image.with_suffix('.txt').read_text(), '1girl, outdoors')
        self.assertTrue(preview.new_text.startswith('shhooldress, serafaku, blue skirt'))

    def test_master_first_and_deduplicate_without_changing_user_tags(self):
        preview = self.preview('1girl, blue_skirt, outdoors')
        self.assertEqual(preview.new_text, 'shhooldress, serafaku, blue skirt, 1girl, outdoors\n')
        self.assertNotIn('blue skirt', preview.state_after['owned_tags'])

    def test_apply_then_undo_exact_bytes(self):
        original = b'\xef\xbb\xbf1girl, outdoors\r\n'
        self.image.with_suffix('.txt').write_bytes(original)
        preview = self.preview()
        journal = apply_caption(preview, self.store)
        self.assertEqual(self.image.with_suffix('.txt').read_bytes(), preview.after)
        undo_caption(journal)
        self.assertEqual(self.image.with_suffix('.txt').read_bytes(), original)

    def test_undo_new_caption_removes_only_created_file(self):
        preview = self.preview()
        journal = apply_caption(preview, self.store)
        undo_caption(journal)
        self.assertFalse(self.image.with_suffix('.txt').exists())
        self.assertTrue(self.image.exists())

    def test_reapply_is_idempotent(self):
        preview = self.preview('1girl')
        apply_caption(preview, self.store)
        second = self.preview()
        self.assertIsNone(apply_caption(second, self.store))
        self.assertEqual(second.new_text.count('shhooldress'), 1)

    def test_changed_visibility_removes_only_owned_tags(self):
        apply_caption(self.preview('1girl, custom_manual'), self.store)
        second = self.preview(answers=[decision(('visible', 'not_visible', 'not_visible'))])
        self.assertEqual(second.new_text, 'shhooldress, serafaku, 1girl, custom_manual\n')

    def test_preexisting_invisible_tag_is_preserved_with_warning(self):
        preview = self.preview('1girl, black thighighhs')
        self.assertIn('black thighighhs', preview.new_text)
        self.assertTrue(preview.warnings)

    def test_caption_modified_after_preview_is_never_overwritten(self):
        preview = self.preview('1girl')
        caption = self.image.with_suffix('.txt')
        caption.write_text('manual edit', encoding='utf-8')
        with self.assertRaises(CaptionConflict):
            apply_caption(preview, self.store)
        self.assertEqual(caption.read_text(), 'manual edit')

    def test_image_modified_after_preview_is_never_written(self):
        preview = self.preview('1girl')
        Image.new('RGB', (300, 300), 'red').save(self.image)
        with self.assertRaises(CaptionConflict):
            apply_caption(preview, self.store)

    def test_profile_modified_after_preview_is_never_written(self):
        preview = self.preview('1girl')
        self.profile.master_tag = 'changedmaster'
        self.store.upsert(self.profile)
        with self.assertRaises(CaptionConflict):
            apply_caption(preview, self.store)

    def test_profile_deleted_after_preview_is_never_written(self):
        preview = self.preview('1girl')
        self.store.delete(self.profile.id)
        with self.assertRaises(CaptionConflict):
            apply_caption(preview, self.store)

    def test_shared_caption_stem_is_rejected(self):
        Image.new('RGB', (10, 10)).save(self.image.with_suffix('.jpg'))
        with self.assertRaises(CaptionConflict):
            self.preview()

    def test_undo_refuses_external_edits(self):
        journal = apply_caption(self.preview('1girl'), self.store)
        self.image.with_suffix('.txt').write_text('edited later')
        with self.assertRaises(CaptionConflict):
            undo_caption(journal)
        self.assertEqual(self.image.with_suffix('.txt').read_text(), 'edited later')

    def test_rejected_match_has_no_preview(self):
        result, _ = self.result([decision(identity_score=.1)])
        with self.assertRaises(ValueError):
            prepare_caption(result)

    def test_utf8_errors_do_not_destroy_caption(self):
        self.image.with_suffix('.txt').write_bytes(b'\xff\xfeinvalid')
        with self.assertRaises(UnicodeError):
            self.preview()

    def test_ownership_write_failure_rolls_back(self):
        preview = self.preview('1girl')
        import src.core.clothing_captions as module
        real = module.atomic_json
        def fail_state(path, value):
            if Path(path).parent.name == 'state':
                raise OSError('simulated disk failure')
            real(path, value)
        with patch.object(module, 'atomic_json', side_effect=fail_state):
            with self.assertRaises(OSError):
                apply_caption(preview, self.store)
        self.assertEqual(self.image.with_suffix('.txt').read_text(), '1girl')

    def test_prepared_journal_recovers_interrupted_write(self):
        preview = self.preview('1girl')
        journal = apply_caption(preview, self.store)
        p = Path(journal)
        data = json.loads(p.read_text(encoding='utf-8'))
        data['status'] = 'prepared'
        atomic_json(p, data)
        undo_caption(p)
        self.assertEqual(self.image.with_suffix('.txt').read_text(), '1girl')

    def test_undo_protects_replaced_source_image(self):
        journal = apply_caption(self.preview('1girl'), self.store)
        Image.new('RGB', (200, 300), 'red').save(self.image)
        with self.assertRaises(CaptionConflict):
            undo_caption(journal)

    def test_non_clothing_undo_path_rejected(self):
        with self.assertRaises(CaptionConflict):
            undo_caption(self.root / 'unexpected.json')


class BatchAndCacheTests(Fixture):
    def test_automatic_person_selection_is_reused_across_outfits(self):
        other = copy.deepcopy(self.profile)
        other.id, other.name, other.master_tag = 'other', 'Other', 'othermaster'
        self.store.upsert(other)
        settings = ClothingSettings(person_mode='largest', use_cache=True)
        people = {'uncertain': False, 'people': [
            {'bbox': [0, 0, 1000, 1000], 'score': .95}]}
        backend = ScriptedBackend([
            people,
            decision(verdict='different', visible_color_match='no'),
            decision(),
        ], settings)
        stats = run_clothing_batch([self.image], store=self.store, settings=settings,
                                   sequential=True, backend=backend)
        self.assertEqual(stats['matched'], 1)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(sum('Locate each separate' in prompt
                             for prompt, _, _ in backend.calls), 1)
        resumed = ScriptedBackend([], settings)
        repeat = run_clothing_batch([self.image], store=self.store, settings=settings,
                                    sequential=True, backend=resumed)
        self.assertEqual(repeat['matched'], 1)
        self.assertEqual(repeat['cache_hits'], 2)
        self.assertEqual(resumed.calls, [])

    def test_outfits_run_in_order_and_resume_without_rechecking_matches(self):
        second_image = self.image.parent / 'second.png'
        Image.new('RGB', (200, 300), (230, 30, 20)).save(second_image)
        second_profile = copy.deepcopy(self.profile)
        second_profile.id = 'second_outfit'
        second_profile.name = 'Second outfit'
        second_profile.master_tag = 'secondmaster'
        self.store.upsert(second_profile)
        for image in (self.image, second_image):
            save_selection(self.store, image, [0, 0, 1, 1])
        settings = ClothingSettings(person_mode='manual', use_cache=True)
        backend = ScriptedBackend([
            decision(),
            decision(verdict='different', visible_color_match='no'),
            decision(),
        ], settings)
        seen, stage_logs = [], []
        paths = [self.image, second_image]
        stats = run_clothing_batch(paths, store=self.store, settings=settings,
                                   backend=backend, on_result=lambda result, preview:
                                   seen.append((result, preview)), log=stage_logs.append)
        self.assertEqual(stats['matched'], 2)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual([item[0].profile_id for item in seen],
                         [self.profile.id, second_profile.id])
        self.assertEqual([line.split(' (')[0] for line in stage_logs if 'unmatched images' in line],
                         [self.profile.name, second_profile.name])
        self.assertEqual(seen[0][1].library_digest, self.store.library_fingerprint())
        self.assertFalse(self.image.with_suffix('.txt').exists())

        resumed_backend = ScriptedBackend([], settings)
        resumed = run_clothing_batch(paths, store=self.store, settings=settings,
                                     backend=resumed_backend)
        self.assertEqual(resumed['matched'], 2)
        self.assertEqual(resumed['cache_hits'], 3)
        self.assertEqual(len(resumed_backend.calls), 0)

        applied = run_clothing_batch(paths, store=self.store, settings=settings,
                                     backend=ScriptedBackend([], settings), apply=True)
        self.assertEqual(applied['written'], 2)
        self.assertIn(self.profile.master_tag, self.image.with_suffix('.txt').read_text(encoding='utf-8'))
        self.assertIn(second_profile.master_tag, second_image.with_suffix('.txt').read_text(encoding='utf-8'))

        second_profile.master_tag = 'updatedsecondmaster'
        second_profile.parts[0].description = 'New visible sleeve detail'
        self.store.upsert(second_profile)
        changed_backend = ScriptedBackend([decision()], settings)
        changed = run_clothing_batch(paths, store=self.store, settings=settings,
                                     backend=changed_backend)
        self.assertEqual(changed['matched'], 2)
        self.assertEqual(changed['cache_hits'], 2)
        self.assertEqual(len(changed_backend.calls), 1)

    def test_failed_outfit_comparison_checks_later_profiles_without_writing(self):
        other = copy.deepcopy(self.profile)
        other.id, other.name, other.master_tag = 'other', 'Other', 'othermaster'
        self.store.upsert(other)
        save_selection(self.store, self.image, [0, 0, 1, 1])
        backend = ScriptedBackend([decision()], self.settings)
        infer = backend.infer
        calls = []
        def fail_once(prompt, images, schema):
            calls.append(1)
            if len(calls) == 1:
                raise ClothingBackendError('temporary comparison failure')
            return infer(prompt, images, schema)
        backend.infer = fail_once
        results = []
        stats = run_clothing_batch([self.image], store=self.store, settings=self.settings,
                                   sequential=True, apply=True, backend=backend,
                                   on_result=lambda result, _: results.append(result))
        self.assertEqual(len(calls), 2)
        self.assertEqual(stats['review'], 1)
        self.assertEqual(stats['written'], 0)
        self.assertIn('earlier outfit comparison failed', results[0].reason)
        self.assertFalse(self.image.with_suffix('.txt').exists())

    def test_cancelled_outfit_pass_resumes_at_next_unmatched_image(self):
        second_image = self.image.parent / 'second.png'
        Image.new('RGB', (200, 300), (230, 30, 20)).save(second_image)
        second_profile = copy.deepcopy(self.profile)
        second_profile.id = 'second_outfit'
        second_profile.name = 'Second outfit'
        second_profile.master_tag = 'secondmaster'
        self.store.upsert(second_profile)
        for image in (self.image, second_image):
            save_selection(self.store, image, [0, 0, 1, 1])
        settings = ClothingSettings(person_mode='manual', use_cache=True)
        cancel = threading.Event()
        first = ScriptedBackend([
            decision(), decision(verdict='different', visible_color_match='no')], settings)

        def stop_after_first_outfit(index, total, message):
            if index == total and 'Blue uniform' in message:
                cancel.set()

        interrupted = run_clothing_batch([self.image, second_image], store=self.store,
                                         settings=settings, backend=first,
                                         cancel_event=cancel, on_progress=stop_after_first_outfit)
        self.assertTrue(interrupted['cancelled'])
        self.assertEqual(len(first.calls), 2)
        resumed_backend = ScriptedBackend([decision()], settings)
        resumed = run_clothing_batch([self.image, second_image], store=self.store,
                                     settings=settings, backend=resumed_backend)
        self.assertEqual(resumed['matched'], 2)
        self.assertEqual(resumed['cache_hits'], 2)
        self.assertEqual(len(resumed_backend.calls), 1)

    def test_default_batch_is_preview_only(self):
        settings = self.settings
        backend = ScriptedBackend([decision()], settings)
        save_selection(self.store, self.image, [0, 0, 1, 1])
        results = []
        stats = run_clothing_batch([self.image], store=self.store, settings=settings,
                                   backend=backend, on_result=lambda r, p: results.append((r, p)))
        self.assertEqual(stats['matched'], 1)
        self.assertEqual(stats['written'], 0)
        self.assertFalse(self.image.with_suffix('.txt').exists())
        self.assertIsNotNone(results[0][1])
        self.assertTrue(backend.unloaded)

    def test_batch_apply_and_persisted_undo(self):
        backend = ScriptedBackend([decision()], self.settings)
        save_selection(self.store, self.image, [0, 0, 1, 1])
        stats = run_clothing_batch([self.image], store=self.store, settings=self.settings,
                                   backend=backend, apply=True)
        self.assertEqual(stats['written'], 1)
        self.assertEqual(undo_last_job(self.store)['undone'], 1)
        self.assertFalse(self.image.with_suffix('.txt').exists())

    def test_partial_job_errors_do_not_overwrite_other_files(self):
        bad = self.image.parent / 'broken.png'
        bad.write_bytes(b'not an image')
        backend = ScriptedBackend([decision()], self.settings)
        save_selection(self.store, self.image, [0, 0, 1, 1])
        stats = run_clothing_batch([bad, self.image], store=self.store, settings=self.settings,
                                   backend=backend, apply=True)
        self.assertEqual(stats['errors'], 1)
        self.assertEqual(stats['written'], 1)
        self.assertFalse(bad.with_suffix('.txt').exists())

    def test_cancellation_before_start_never_writes(self):
        backend = ScriptedBackend([decision()], self.settings)
        backend.cancel_event.set()
        stats = run_clothing_batch([self.image], store=self.store, settings=self.settings,
                                   backend=backend, apply=True)
        self.assertTrue(stats['cancelled'])
        self.assertFalse(self.image.with_suffix('.txt').exists())

    def test_ollama_schema_json_in_thinking_field(self):
        backend = OllamaClothingBackend(self.settings)
        raw = {'done': True, 'done_reason': 'stop', 'response': '',
               'thinking': '{"ok": true}'}
        with patch.object(backend, '_request', return_value=raw):
            self.assertEqual(backend.infer('classify', [], object_schema({'ok': BOOL})),
                             {'ok': True})

    def test_cache_keys_include_model_images_and_prompt(self):
        settings = self.settings
        backend = ScriptedBackend([{'ok': True}, {'ok': False}, {'ok': True}, {'ok': False}], settings)
        cache = ClothingCache(self.root / 'cache.sqlite3')
        try:
            vision = CachedVision(backend, cache)
            schema = object_schema({'ok': BOOL})
            self.assertEqual(vision.infer('a', ['imageA'], schema), {'ok': True})
            self.assertEqual(vision.infer('a', ['imageA'], schema), {'ok': True})
            self.assertEqual(len(backend.calls), 1)
            vision.infer('b', ['imageA'], schema)
            vision.infer('b', ['imageB'], schema)
            backend.fingerprint = 'new model digest'
            vision.infer('b', ['imageB'], schema)
            self.assertEqual(len(backend.calls), 4)
        finally:
            cache.close()

    def test_manual_selection_invalidated_on_image_change(self):
        save_selection(self.store, self.image, [.1, .1, .9, .9])
        self.assertEqual(load_selection(self.store, self.image), [.1, .1, .9, .9])
        Image.new('RGB', (150, 150), 'red').save(self.image)
        self.assertIsNone(load_selection(self.store, self.image))

    def test_clear_manual_selection(self):
        save_selection(self.store, self.image, [0, 0, 1, 1])
        save_selection(self.store, self.image, None)
        self.assertIsNone(load_selection(self.store, self.image))

    def test_optional_pass_disabled_does_not_contact_model(self):
        with patch('src.core.clothing_service.run_clothing_batch') as run:
            self.assertIsNone(optional_pipeline_pass([self.image], 'video', store=self.store))
            self.assertIsNone(optional_pipeline_pass([self.image], 'caption', store=self.store))
            run.assert_not_called()

    def test_folder_scan_preserves_uppercase_extensions_and_ignores_metadata(self):
        upper = self.image.parent / 'UPPER.PNG'
        Image.new('RGB', (10, 10)).save(upper)
        hidden = self.image.parent / '.lh-clothing' / 'test.png'
        hidden.parent.mkdir()
        Image.new('RGB', (10, 10)).save(hidden)
        found = find_images(self.image.parent, True)
        self.assertIn(upper, found)
        self.assertNotIn(hidden, found)

    def test_alpha_composited_not_black(self):
        path = self.root / 'alpha.png'
        Image.new('RGBA', (10, 10), (0, 0, 0, 0)).save(path)
        self.assertEqual(load_rgb(path).getpixel((0, 0)), (255, 255, 255))

    def test_lock_prevents_concurrent_writer(self):
        with FileLock(self.root / 'test.lock'):
            with self.assertRaises(RuntimeError):
                with FileLock(self.root / 'test.lock'):
                    pass

    def test_remote_endpoint_and_cloud_model_rejected(self):
        for endpoint in ('https://example.com', 'http://192.168.1.2:11434',
                         'http://127.0.0.1:11434/evil', 'http://user:pass@localhost:11434'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_endpoint(endpoint)
        with self.assertRaises(ValueError):
            ClothingSettings(model='qwen3-vl:cloud').validate()

    def test_schema_rejects_nonfinite_and_bool_scores(self):
        for val in (float('nan'), float('inf'), True, '0.9', -1, 2):
            with self.subTest(value=val), self.assertRaises(ClothingBackendError):
                validate_schema(val, SCORE)

    def test_schema_rejects_missing_and_extra_keys(self):
        schema = object_schema({'ok': BOOL})
        for value in ({}, {'ok': True, 'extra': 1}):
            with self.assertRaises(ClothingBackendError):
                validate_schema(value, schema)


if __name__ == '__main__':
    unittest.main()
