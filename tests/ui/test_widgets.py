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
