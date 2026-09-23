"""Real Qt integration; never count a skipped test as a rendered UI check."""
import importlib.util
import os
import time
from pathlib import Path
import pytest
from PIL import Image

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytestmark = pytest.mark.skipif(importlib.util.find_spec('PyQt5') is None,
                               reason='PyQt5 unavailable: performance UI integration not executed')


def until(app, condition, timeout=6):
    start = time.monotonic()
    while not condition():
        app.processEvents()
        if time.monotonic() - start > timeout:
            raise AssertionError('Qt job did not complete')
        time.sleep(.005)
    app.processEvents()


@pytest.fixture
def review():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QThread
    from src.ui.review_grid_page import ReviewGridPage
    app = QApplication.instance() or QApplication([])
    page = ReviewGridPage('en')
    yield app, page
    for worker in page.findChildren(QThread):
        worker.requestInterruption()
    until(app, lambda: not any(w.isRunning() for w in page.findChildren(QThread)))
    page.close()
    page.deleteLater()
    app.processEvents()


def make_dataset(root, prefix, number):
    root.mkdir()
    for n in range(number):
        p = root/f'{prefix}{n}.png'
        Image.new('RGB', (320, 220), (n*20 % 255, 80, 140)).save(p)
        p.with_suffix('.txt').write_text('master, blue skirt', encoding='utf-8')


def test_review_real_qimage_delivery_and_background_captions(review, tmp_path):
    app, page = review
    make_dataset(tmp_path/'data', 'a', 3)
    before = {p: p.read_bytes() for p in (tmp_path/'data').iterdir()}
    page._folder = tmp_path/'data'
    page._load_folder()
    until(app, lambda: page._loader is None and page._stats_loader is None)
    assert page._list.count() == 3
    for n in range(3):
        assert not page._list.item(n).icon().isNull()
        assert page._list.item(n).toolTip() == 'master, blue skirt'
    assert page._stat_avg_tags._val_lbl.text() == '2.0'
    assert all(p.read_bytes() == value for p, value in before.items())


def test_review_switch_folder_ignores_old_worker_results(review, tmp_path):
    app, page = review
    make_dataset(tmp_path/'first', 'old', 20)
    make_dataset(tmp_path/'second', 'new', 2)
    page._folder = tmp_path/'first'
    page._load_folder()
    page._folder = tmp_path/'second'
    page._load_folder()
    until(app, lambda: page._loader is None and page._stats_loader is None)
    assert page._list.count() == 2
    assert all(page._list.item(n).text().startswith('new') for n in range(2))
    assert all(not page._list.item(n).icon().isNull() for n in range(2))


def test_review_removed_placeholder_does_not_shift_thumbnail_ownership(review, tmp_path):
    app, page = review
    make_dataset(tmp_path/'data', 'a', 4)
    page._folder = tmp_path/'data'
    page._load_folder()
    # Match the bookkeeping used by rejection, before queued thumbnail delivery.
    removed = page._pairs[0]
    page._items_by_path.pop(str(removed.image))
    page._list.takeItem(0)
    page._pairs = page._pairs[1:]
    until(app, lambda: page._loader is None and page._stats_loader is None)
    assert page._list.count() == 3
    assert all(not page._list.item(n).icon().isNull() for n in range(3))
    assert all(page._list.item(n).toolTip() == 'master, blue skirt' for n in range(3))
