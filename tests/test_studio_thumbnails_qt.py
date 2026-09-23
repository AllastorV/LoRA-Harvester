"""Visible file rows render their real image previews without changing item text."""
import os
import time
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_thumbnail_delegate_paints_loaded_image(tmp_path):
    from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
    from src.ui.design_system.media import MediaLoader, attach_image_thumbnails

    path = tmp_path / "sample.png"
    Image.new("RGB", (80, 80), (224, 36, 27)).save(path)
    app = QApplication.instance() or QApplication([])
    media = MediaLoader()
    table = QTableWidget(1, 1)
    table.setColumnWidth(0, 220)
    table.setItem(0, 0, QTableWidgetItem(path.name))
    attach_image_thumbnails(table, media, lambda index: path, 0)
    table.resize(240, 110)
    table.show()
    try:
        deadline = time.monotonic() + 3
        while not media.cache and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert media.cache, "visible thumbnail was not decoded"
        app.processEvents()
        image = table.viewport().grab().toImage()
        red_pixels = sum(
            image.pixelColor(x, y).red() > 180
            and image.pixelColor(x, y).green() < 80
            for x in range(min(70, image.width()))
            for y in range(min(54, image.height()))
        )
        assert red_pixels > 100
        assert table.item(0, 0).text() == "sample.png"
        assert table.verticalHeader().defaultSectionSize() >= 54
    finally:
        table.close()
        media.shutdown()
