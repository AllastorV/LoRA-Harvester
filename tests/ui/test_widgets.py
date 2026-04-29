import sys
import pytest
from PyQt5.QtWidgets import QApplication, QWidget

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def test_progress_glow_bar_init(qapp):
    from src.ui.animations import ProgressGlowBar
    bar = ProgressGlowBar()
    assert bar.value() == 0
    assert bar.maximum() == 100


def test_drawer_slide_expand(qapp):
    from src.ui.animations import DrawerSlide
    w = QWidget()
    w.setMaximumWidth(0)
    ds = DrawerSlide(w, target_width=380, duration=250)
    ds.expand()
    assert ds._target == 380


def test_hover_lift_installs(qapp):
    from src.ui.animations import install_hover_lift
    w = QWidget()
    install_hover_lift(w)
    assert w.property("_hover_lift") is True


def test_caption_studio_has_tabs(qapp):
    from src.ui.caption_studio_page import CaptionStudioPage
    from PyQt5.QtWidgets import QTabWidget
    page = CaptionStudioPage()
    tab_widget = page.findChild(QTabWidget, "caption_tabs")
    assert tab_widget is not None
    assert tab_widget.count() == 2
    assert tab_widget.tabText(0) == "Generate"
    assert tab_widget.tabText(1) == "Edit"


def test_tag_frequency_page_has_table_and_blacklist(qapp):
    from src.ui.tag_frequency_page import TagFrequencyPage
    from PyQt5.QtWidgets import QTableWidget, QFrame
    page = TagFrequencyPage()
    table = page.findChild(QTableWidget, "freq_table")
    bl_panel = page.findChild(QFrame, "blacklist_panel")
    assert table is not None, "freq_table QTableWidget not found"
    assert bl_panel is not None, "blacklist_panel QFrame not found"
    assert table.columnCount() == 4
