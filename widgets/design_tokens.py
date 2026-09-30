"""
Token desain & helper yang dipakai bersama oleh page_monitoring.py dan
page_verifikasi.py. Dipisah ke sini supaya tidak duplikasi & supaya
warna/gradient gampang diubah di satu tempat kalau desain Figma berubah.
"""
import os
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QPainter, QImage
from PyQt5.QtWidgets import QLabel, QWidget, QGraphicsDropShadowEffect

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

FONT_FAMILY = "Inter"  # fallback otomatis ke font sistem kalau Inter belum terinstall

# Drop shadow (QGraphicsDropShadowEffect) itu lumayan berat untuk GPU/CPU
# lemah seperti Jetson Nano, apalagi karena beberapa panel di-redraw tiap
# detik (jam berjalan). Kalau nanti terasa patah-patah/lag di Jetson Nano,
# set ini ke False -- semua shadow otomatis dimatikan tanpa harus ubah
# kode lain.
ENABLE_SHADOWS = True

# ---- warna, diambil persis dari Figma ----
COLOR_BLUE = "#00548c"
COLOR_BLUE_SOFT_BG = "rgba(0, 84, 140, 38)"     # = #00548c dengan opacity 15%
COLOR_NAVY = "#093552"
COLOR_NAVY_SOFT_BG = "#d2e3ed"
COLOR_GRAY_TEXT = "#818181"
COLOR_BG = "#f3f8fc"

CCTV_HEADER_GRADIENT = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8f101b, stop:1 #d90a1a)"
)
CCTV_FOOTER_GRADIENT = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
    "stop:0 rgba(143,16,27,204), stop:1 rgba(217,10,26,204))"
)
CLOCK_PANEL_GRADIENT = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #68899c, stop:1 #cfd9d8)"
)


def load_pixmap(filename: str) -> QPixmap:
    """
    Coba muat gambar hasil export Figma dari folder assets/.
    Kalau belum ada, kembalikan QPixmap kosong (isNull() == True) --
    widget pemanggil sudah punya fallback teks/emoji masing-masing.
    """
    return QPixmap(os.path.join(ASSET_DIR, filename))


def make_icon_label(parent, filename: str, fallback_emoji: str, size: int, font_px: int = 20) -> QLabel:
    """Label ikon: pakai PNG asli kalau sudah diexport, kalau belum pakai emoji placeholder."""
    lbl = QLabel(parent)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("background: transparent;")
    pix = load_pixmap(filename)
    if not pix.isNull():
        lbl.setPixmap(pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    else:
        lbl.setText(fallback_emoji)
        lbl.setStyleSheet(f"background: transparent; font-size: {font_px}px;")
    return lbl


def apply_shadow(widget: QWidget, blur=20, y_offset=6, alpha=70):
    if not ENABLE_SHADOWS:
        return
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, y_offset)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)


class BackgroundPage(QWidget):
    """
    Base class untuk halaman yang punya background dekoratif (garis
    diagonal merah/biru/hijau, hasil export "bg_pattern.png"). Gambar
    digambar langsung lewat paintEvent (bukan QLabel biasa) supaya
    otomatis re-scale "cover" (mengisi penuh & terpotong rapi, tidak
    gepeng) tiap kali window di-resize atau fullscreen -- tanpa perlu
    kode resize manual di tiap halaman.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_pixmap = load_pixmap("bg_pattern.png")

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self._bg_pixmap.isNull():
            scaled = self._bg_pixmap.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(self.rect(), QColor(COLOR_BG))
        super().paintEvent(event)


def cv_to_qpixmap(frame) -> QPixmap:
    """
    Konversi frame OpenCV (ndarray BGR, hasil cv2.VideoCapture/YOLO
    overlay) menjadi QPixmap supaya bisa ditampilkan di QLabel.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    # .copy() penting -- melepas QImage dari buffer memori numpy asli,
    # supaya tidak jadi rusak/corrupt kalau array itu di-reuse/di-GC.
    return QPixmap.fromImage(qimg.copy())


def render_cv_frame(label: QLabel, frame):
    """
    Tampilkan 1 frame OpenCV ke sebuah QLabel, otomatis di-scale
    mengikuti ukuran label saat ini (dipakai untuk video ANPR & Postur).
    Aman dipanggil dengan frame=None (mis. belum ada frame pertama).
    """
    if frame is None:
        return
    pixmap = cv_to_qpixmap(frame)
    label.setPixmap(pixmap.scaled(
        max(1, label.width()), max(1, label.height()),
        Qt.KeepAspectRatio, Qt.SmoothTransformation
    ))
