"""Training ZIP contains one flat image and caption folder."""
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.core.kohya_exporter import KohyaExporter, export_training_zip


def prepared_dataset(tmp_path):
    source = tmp_path / 'source' / 'person'
    source.mkdir(parents=True)
    (source / 'frame.png').write_bytes(b'image-bytes')
    (source / 'frame.txt').write_text('mychar, blue skirt', encoding='utf-8')
    prepared = tmp_path / 'prepared'
    KohyaExporter().export(source.parent, prepared, repeats=7)
    return prepared


def test_zip_has_one_folder_with_only_training_pairs(tmp_path):
    prepared = prepared_dataset(tmp_path)
    archive = tmp_path / 'training.zip'
    assert export_training_zip(prepared, archive) == 1
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert set(zipped.namelist()) == {'training/frame.png', 'training/frame.txt'}
        assert zipped.read('training/frame.txt').decode() == 'mychar, blue skirt'
    assert (prepared / '7_person/frame.png').is_file()


def test_zip_keeps_duplicate_names_and_their_captions(tmp_path):
    source = tmp_path / 'source'
    for concept in ('person', 'animal'):
        folder = source / concept
        folder.mkdir(parents=True)
        (folder / 'frame.png').write_bytes((concept + '-image').encode())
        (folder / 'frame.txt').write_text(concept + '-caption', encoding='utf-8')
    prepared = tmp_path / 'prepared'
    KohyaExporter().export(source, prepared, repeats=7)
    archive = tmp_path / 'bundle.zip'
    assert export_training_zip(prepared, archive) == 2
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names)) == 4
        assert all(name.startswith('bundle/') and name.count('/') == 1 for name in names)
        assert {Path(name).suffix for name in names} == {'.png', '.txt'}
        for name in names:
            if name.endswith('.png'):
                caption = Path(name).with_suffix('.txt').as_posix()
                assert caption in names
                concept = zipped.read(name).decode().removesuffix('-image')
                assert zipped.read(caption).decode() == concept + '-caption'


def test_zip_never_overwrites_or_leaves_cancelled_archive(tmp_path):
    prepared = prepared_dataset(tmp_path)
    archive = tmp_path / 'training.zip'
    with pytest.raises(InterruptedError):
        export_training_zip(prepared, archive, cancel=lambda: True)
    assert not archive.exists()
    archive.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        export_training_zip(prepared, archive)
    assert archive.read_bytes() == b'existing'


def test_training_page_exports_selected_prepared_folder(tmp_path, monkeypatch):
    import os
    import time
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from src.ui.training_page import TrainingPage

    app = QApplication.instance() or QApplication([])
    prepared = prepared_dataset(tmp_path)
    archive = tmp_path / 'from_ui.zip'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(archive), 'ZIP (*.zip)'))
    page = TrainingPage('tr')
    page.set_dataset_path(str(prepared))
    assert page._zip_btn.isEnabled()
    page._zip_btn.click()
    deadline = time.monotonic() + 5
    while page._zip_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert page._zip_thread is None
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert set(zipped.namelist()) == {'from_ui/frame.png', 'from_ui/frame.txt'}
    page.close()
