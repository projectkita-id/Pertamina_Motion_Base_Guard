"""
Halaman 'Monitoring Plat & Postur' - versi REFLOW (lihat catatan di
page_verifikasi.py untuk penjelasan pendekatannya).
"""
import os
from PyQt5.QtCore import Qt, QTimer, QDateTime, QSize, QRectF, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QFontMetricsF
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QFrame, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QDialog
)

from widgets.design_tokens import (
    BackgroundPage, load_pixmap, make_icon_label, apply_shadow, render_cv_frame,
    cv_to_qpixmap,
    FONT_FAMILY, COLOR_NAVY, COLOR_NAVY_SOFT_BG, COLOR_BG,
    CCTV_HEADER_GRADIENT, CCTV_FOOTER_GRADIENT, CLOCK_PANEL_GRADIENT,
)


class ResizableImageLabel(QLabel):
    """
    QLabel yang menyimpan gambar aslinya (resolusi penuh) lalu otomatis
    re-scale mengikuti ukuran label SAAT INI setiap kali di-resize --
    dipakai di dialog preview tag supaya gambar ikut membesar/mengecil
    saat jendela preview ditarik, bukan cuma jendelanya yang berubah
    sementara gambarnya tetap diam di ukuran semula.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 160)

    def set_original_pixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            self.setPixmap(self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))


def draw_overlay_lines_on_pixmap(pixmap: QPixmap, lines: list) -> QPixmap:
    """
    Gambar teks info (Plat/Tag/Waktu/Durasi) di pojok kiri atas GAMBAR
    UNTUK DITAMPILKAN SAJA (bukan ditulis ke file aslinya) -- supaya
    selalu menunjukkan data TERBARU (ikut koreksi plat), tanpa risiko
    "kekunci" ke data lama seperti waktu masih di-bakar permanen ke file.
    """
    if not lines:
        return pixmap
    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)

    line_h = 24
    box_w = 360
    box_h = 16 + line_h * len(lines)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(0, 0, 0, 160))
    painter.drawRect(10, 10, box_w, box_h)

    font = QFont(FONT_FAMILY)
    font.setPixelSize(16)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(255, 230, 0))
    y = 10 + line_h
    for line in lines:
        painter.drawText(22, y, line)
        y += line_h
    painter.end()
    return result


def show_image_preview_dialog(parent, title, image_source, info_text=None, overlay_lines=None):
    """
    Dialog preview gambar yang bisa di-resize DAN di-fullscreen (tombol
    maximize aktif di title bar-nya, klik 2x untuk fullscreen penuh).
    Dipakai bersama oleh preview tag & preview foto plat supaya tidak
    ada kode duplikat.

    image_source  : path file gambar (str) ATAU QPixmap langsung.
    overlay_lines : list teks opsional yang digambar di pojok kiri atas
                    gambar SAAT PREVIEW SAJA (lihat draw_overlay_lines_on_pixmap).
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    # QDialog secara default cuma punya tombol close di title bar --
    # baris ini yang menambahkan tombol minimize & maximize, supaya
    # jendela preview-nya bisa di-fullscreen-kan.
    dlg.setWindowFlags(
        dlg.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint
    )
    dlg.resize(900, 650)
    dlg.setMinimumSize(360, 280)
    dlg.setSizeGripEnabled(True)

    lay = QVBoxLayout(dlg)
    img_lbl = ResizableImageLabel()
    img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    if isinstance(image_source, QPixmap):
        pix = image_source
    elif image_source and os.path.exists(str(image_source)):
        pix = QPixmap(str(image_source))
    else:
        pix = None

    if pix is not None and not pix.isNull():
        if overlay_lines:
            pix = draw_overlay_lines_on_pixmap(pix, overlay_lines)
        img_lbl.set_original_pixmap(pix)
    else:
        img_lbl.setText("Gambar tidak ditemukan.")
    lay.addWidget(img_lbl, 1)

    if info_text:
        info_lbl = QLabel(info_text)
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setStyleSheet(f"font-family: '{FONT_FAMILY}'; padding: 8px;")
        lay.addWidget(info_lbl)

    dlg.exec_()


# Nama baris status AMT (Awak Mobil Tangki) di daftar Status -- ubah di sini kalau ejaan/namanya berubah.
LABEL_NAIK_TANGKI = "AMT Naik Tangki"
LABEL_TURUN_TANGKI = "AMT Turun Tangki"


class TagRowWidget(QFrame):
    """
    Baris 1 tag di daftar. Seluruh area baris bisa diklik untuk preview
    gambar (lewat sinyal `clicked`) -- KECUALI tombol hapus di pojok
    kanan atas, yang menangani klik-nya sendiri (tidak ikut memicu
    preview) karena tombol Qt otomatis 'menyerap' event klik miliknya.
    """
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class FillListWidget(QListWidget):
    """
    QListWidget biasa: tinggi tiap baris ikut ukuran konten (fixed),
    jadi kalau panel-nya melebar/memanjang (mis. saat fullscreen),
    baris-barisnya diam di ukuran kecil dan sisa ruang jadi kosong.

    Kelas ini menghitung ulang tinggi tiap baris setiap kali list
    di-resize, supaya baris-baris ikut memenuhi tinggi panel yang
    tersedia -- dan ikon/font di dalam tiap baris ikut diperbesar
    proporsional juga (bukan cuma baris kosongnya yang melebar),
    supaya tidak terlihat "kosong di tengah" saat baris jadi tinggi.
    Kalau baris lebih banyak dari yang muat di tinggi minimum,
    otomatis balik jadi scrollable seperti biasa.
    """
    MIN_ROW_HEIGHT = 68
    MAX_ROW_HEIGHT = 130  # batas atas -- supaya kalau tag cuma 1-2, baris tidak
                           # "memakan" seluruh tinggi panel (kelihatan nge-tengah)

    # ukuran dasar (dipakai saat row_h == MIN_ROW_HEIGHT, scale = 1.0)
    BASE_ICON = 48
    BASE_NAME_FONT = 13
    BASE_DATE_FONT = 10
    BASE_DURATION_FONT = 12
    BASE_CLOCK_ICON = 14
    MAX_SCALE = 2.6  # batas atas supaya font tidak jadi raksasa tak wajar

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_rows()

    def _relayout_rows(self):
        count = self.count()
        if count == 0:
            return
        available = self.viewport().height()
        row_h = max(self.MIN_ROW_HEIGHT, min(self.MAX_ROW_HEIGHT, available // count))
        scale = min(self.MAX_SCALE, row_h / self.MIN_ROW_HEIGHT)
        row_w = self.viewport().width()

        rows = [self.itemWidget(self.item(i)) for i in range(count)]
        rows = [r for r in rows if r is not None]
        # font tanggal SERAGAM di seluruh daftar: yang terkecil yang masih muat di semua baris
        # (baris Dry Check punya kolom durasi sehingga ruangnya paling sempit)
        common_date_px = min((self._row_date_px(r, scale, row_w) for r in rows), default=None)
        # nama juga diseragamkan (terkecil yang muat di semua baris, dengan tinggi baris yang tersedia)
        common_name_px = min((self._row_name_px(r, scale, row_w, common_date_px, row_h) for r in rows), default=None)
        for i in range(count):
            item = self.item(i)
            row = self.itemWidget(item)
            if row is None:
                continue
            row.setFixedHeight(row_h)
            item.setSizeHint(QSize(row_w, row_h))
            self._scale_row_contents(row, scale, row_w, common_date_px, common_name_px)

    def _text_px_fit(self, text: str, bold: bool, max_width: float, wanted_px: int, min_px: int = 8) -> int:
        """
        Ukuran font (px) terbesar <= wanted_px yang membuat `text` (satu baris) muat dalam max_width.
        Font baris diskalakan mengikuti TINGGI baris, sedangkan LEBAR panel tetap -- jadi saat daftar cuma
        berisi 1-2 baris (baris tinggi, font membesar) tanggal/jam bisa lebih lebar dari ruangnya dan terpotong.
        """
        f = QFont(FONT_FAMILY)
        f.setPixelSize(100)
        f.setBold(bold)
        w100 = QFontMetricsF(f).horizontalAdvance(text)           # lebar teks pada 100 px
        if w100 <= 0 or max_width <= 0:
            return wanted_px
        fit = int(max_width * 0.97 / (w100 / 100.0))              # 3% sisa supaya tidak mepet
        return max(min_px, min(wanted_px, fit))

    def _row_text_avail(self, row: QWidget, scale: float, row_w: int) -> float:
        """
        Lebar yang benar-benar tersedia untuk kolom teks (nama + tanggal): lebar baris dikurangi margin
        kiri/kanan (4+4), ikon, tombol hapus (24), spasi antar-elemen (12 masing-masing), dan kolom
        durasi kalau tampil.
        """
        icon_size = int(self.BASE_ICON * scale)
        avail = row_w - (4 + icon_size + 12 + 12 + 24 + 4)
        if not row.duration_lbl.isHidden():
            f_dur = QFont(FONT_FAMILY)
            f_dur.setPixelSize(int(self.BASE_DURATION_FONT * scale))
            avail -= 12 + int(self.BASE_CLOCK_ICON * scale) + 4 + QFontMetricsF(f_dur).horizontalAdvance(row.duration_lbl.text())
        return avail

    def _row_date_px(self, row: QWidget, scale: float, row_w: int) -> int:
        """Ukuran font tanggal (px) terbesar yang membuat teks tanggal `row` muat di lebar yang tersedia."""
        date_px = int(self.BASE_DATE_FONT * scale)
        if not row_w:
            return date_px
        return self._text_px_fit(row.date_lbl.text(), False, self._row_text_avail(row, scale, row_w), date_px)

    @staticmethod
    def _name_need_h(text: str, px: int, avail_w: float) -> float:
        """Tinggi (px) teks nama tebal berukuran `px` bila dibungkus dalam lebar `avail_w`."""
        f = QFont(FONT_FAMILY)
        f.setPixelSize(px)
        f.setBold(True)
        return QFontMetricsF(f).boundingRect(QRectF(0, 0, max(avail_w * 0.97, 1), 10000), Qt.TextWordWrap, text).height()

    def _row_name_px(self, row: QWidget, scale: float, row_w: int, date_px: int, row_h: int) -> int:
        """
        Ukuran font nama (px) terbesar yang membuat nama (boleh terbungkus beberapa baris) MUAT di tinggi yang
        tersedia: tinggi baris - margin atas/bawah (8+8) - tinggi baris tanggal. Tanpa ini, nama panjang
        seperti "Plat H 6889 IV - Dry Check #1" terbungkus 3 baris dan "#1" terpotong saat font membesar.
        """
        name_px = int(self.BASE_NAME_FONT * scale)
        if not row_w or not row_h:
            return name_px
        avail_w = self._row_text_avail(row, scale, row_w)
        max_h = row_h - 16 - date_px * 1.35 - 4
        text = row.name_lbl.text()
        for px in range(name_px, 8, -1):
            if self._name_need_h(text, px, avail_w) <= max_h:
                return px
        return 9

    def _scale_row_contents(self, row: QWidget, scale: float, row_w: int = None,
                            date_px_override: int = None, name_px_override: int = None):
        icon_size = int(self.BASE_ICON * scale)
        row.icon_lbl.setFixedSize(icon_size, icon_size)
        pix = row.icon_lbl.pixmap()
        if pix is not None and not pix.isNull():
            row.icon_lbl.setPixmap(
                pix.scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        name_px = int(self.BASE_NAME_FONT * scale)
        date_px = int(self.BASE_DATE_FONT * scale)
        dur_px = int(self.BASE_DURATION_FONT * scale)
        clock_px = int(self.BASE_CLOCK_ICON * scale)

        if date_px_override is not None:
            date_px = date_px_override            # ukuran bersama seluruh daftar (lihat _relayout_rows)
        elif row_w:
            date_px = self._row_date_px(row, scale, row_w)
        if name_px_override is not None:
            name_px = name_px_override
        if row_w:
            # Layout hanya menjatah label nama setinggi sizeHint-nya (mis. 2 baris) walau teks yang terbungkus di
            # lebar sempit butuh 3 baris -> baris ketiga (mis. "#1") terpotong. Pasang tinggi minimum yang benar.
            need_h = self._name_need_h(row.name_lbl.text(), name_px, self._row_text_avail(row, scale, row_w))
            row.name_lbl.setMinimumHeight(int(round(need_h)) + 2)

        row.name_lbl.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: {name_px}px; color: #000000;
        """)
        row.date_lbl.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-size: {date_px}px; color: #4b4b4b;
        """)
        row.duration_lbl.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-size: {dur_px}px; color: #000000;
        """)
        row.clock_icon_lbl.setFixedSize(clock_px, clock_px)


class MonitoringPage(BackgroundPage):
    def __init__(self, on_upload, on_kembali=None, on_plate_corrected=None, parent=None):
        super().__init__(parent)
        self._on_upload = on_upload
        self._on_kembali = on_kembali
        self._on_plate_corrected = on_plate_corrected
        self.tag_records = []  # list of dict -- diisi oleh add_tag(), dibaca saat upload
        # capture "Handrail Aktif" -- SENGAJA terpisah dari tag_records karena
        # tag_records-lah yang di-upload ke backend (kontrak API cuma mengenal tag).
        # Handrail hanya ditampilkan di daftar (digabung urut waktu lewat _seq).
        self.handrail_records = []
        # capture "Naik Tangki" (petugas naik ke atas tangki) -- juga terpisah dari tag_records;
        # dikirim ke database lewat parameter sendiri saat upload (kolom di tabel verifikasi).
        self.tank_climb_records = []
        # capture "AMT Turun Tangki" (AMT terlihat turun dari tangki) -- juga terpisah dari tag_records;
        # dikirim ke database lewat parameter sendiri (tank_descend_records) saat upload.
        self.tank_descend_records = []
        self._seq = 0  # penanda urutan masuk (tag, handrail, naik tangki) -- untuk menggabung daftar

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(18)

        outer.addLayout(self._build_top_row())

        content_row = QHBoxLayout()
        content_row.setSpacing(24)
        content_row.addWidget(self._build_cctv_panel(), 68)
        content_row.addLayout(self._build_right_column(), 32)
        outer.addLayout(content_row, 1)

        self._start_clock()

    # ------------------------------------------------------------------
    def _build_top_row(self):
        row = QHBoxLayout()
        row.setSpacing(20)
        row.addWidget(self._build_logo_card())

        # Judul sengaja dikosongkan sesuai permintaan -- widget-nya TETAP
        # ada (disimpan di self.title_label) supaya kapan pun mau
        # ditampilkan lagi, tinggal panggil:
        #     self.title_label.setText("Monitoring Plat & Postur")
        self.title_label = QLabel("")
        self.title_label.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 800; font-size: 26px; color: #000000;
        """)
        row.addWidget(self.title_label)
        row.addStretch(1)
        return row

    def _build_logo_card(self):
        card = QFrame()
        card.setFixedSize(280, 76)
        card.setStyleSheet("background-color: #ffffff; border-radius: 14px;")
        apply_shadow(card, blur=10, y_offset=2, alpha=60)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 6)
        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        pix = load_pixmap("logo_pertamina.png")
        if not pix.isNull():
            logo.setPixmap(pix.scaled(220, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText("PERTAMINA\nPATRA NIAGA")
            logo.setStyleSheet(f"""
                background: transparent; font-family: '{FONT_FAMILY}';
                font-weight: 800; font-size: 14px; color: #ED1C24;
            """)
        lay.addWidget(logo)
        return card

    # ------------------------------------------------------------------
    # PANEL CCTV (kiri) -- figma: node 289:113
    # ------------------------------------------------------------------
    def _build_cctv_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: #ffffff; border-radius: 26px;")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        apply_shadow(panel, blur=26, y_offset=12, alpha=55)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(76)
        header.setStyleSheet(f"""
            background: {CCTV_HEADER_GRADIENT};
            border-top-left-radius: 26px; border-top-right-radius: 26px;
        """)
        h = QHBoxLayout(header)
        h.setContentsMargins(24, 10, 24, 10)
        h.setSpacing(12)

        icon_cctv = make_icon_label(header, "icon_cctv.png", "📹", size=34, font_px=24)
        title_cctv = QLabel("Kamera Deteksi Postur")
        title_cctv.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 21px; color: #ffffff;
        """)
        self.live_dot = QLabel()
        self.live_dot.setFixedSize(16, 16)
        self.live_dot.setStyleSheet("background-color: #22c55e; border-radius: 8px;")
        self.live_text = QLabel("LIVE")
        self.live_text.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 17px; color: #ffffff;
        """)
        # indikator LIVE baru muncul begitu PostureWorker benar-benar
        # mengirim frame pertama (lihat set_live()) -- disembunyikan dulu.
        self.live_dot.hide()
        self.live_text.hide()
        h.addWidget(icon_cctv)
        h.addWidget(title_cctv)
        h.addStretch(1)
        h.addWidget(self.live_dot)
        h.addWidget(self.live_text)

        self.video_label = QLabel("Memuat kamera deteksi postur...\nMohon tunggu sebentar.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumHeight(220)
        self.video_label.setStyleSheet(f"""
            background-color: #ffffff; color: #9aa0a6;
            font-family: '{FONT_FAMILY}'; font-size: 16px;
        """)

        footer = QFrame()
        footer.setFixedHeight(56)
        footer.setStyleSheet(f"""
            background: {CCTV_FOOTER_GRADIENT};
            border-bottom-left-radius: 26px; border-bottom-right-radius: 26px;
        """)
        # catatan: sesuai Figma, footer CCTV di halaman ini sengaja polos (tanpa teks)

        vbox.addWidget(header)
        vbox.addWidget(self.video_label, 1)
        vbox.addWidget(footer)
        return panel

    def update_frame(self, cv_frame):
        """Tampilkan 1 frame hasil PostureWorker (skeleton + status tag)."""
        render_cv_frame(self.video_label, cv_frame)

    def set_live(self, active: bool):
        """Tampilkan/sembunyikan indikator LIVE -- dipanggil dari main.py
        setiap kali PostureWorker benar-benar mengirim frame."""
        self.live_dot.setVisible(active)
        self.live_text.setVisible(active)

    # ------------------------------------------------------------------
    # KOLOM KANAN: panel jam, panel plat, daftar tag, tombol
    # ------------------------------------------------------------------
    def _build_right_column(self):
        col = QVBoxLayout()
        col.setSpacing(18)
        col.addWidget(self._build_clock_panel())
        col.addWidget(self._build_plate_panel())
        col.addWidget(self._build_tag_list_panel(), 1)
        col.addLayout(self._build_buttons_row())
        return col

    def _build_clock_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {CLOCK_PANEL_GRADIENT}; border-radius: 16px;")
        apply_shadow(panel, blur=14, y_offset=8, alpha=55)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(24, 16, 24, 16)
        vbox.setSpacing(2)

        cap_row = QHBoxLayout()
        icon_clock = make_icon_label(panel, "icon_clock.png", "🕐", size=18, font_px=14)
        caption = QLabel("Tanggal & Waktu")
        caption.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-size: 13px; color: #ffffff;
        """)
        cap_row.addWidget(icon_clock)
        cap_row.addWidget(caption)
        cap_row.addStretch(1)
        vbox.addLayout(cap_row)

        self.clock_time_label = QLabel("00 : 00 : 00")
        self.clock_time_label.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 700; font-size: 30px; color: #ffffff;
        """)
        vbox.addWidget(self.clock_time_label)

        self.clock_date_label = QLabel("")
        self.clock_date_label.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 500; font-size: 13px; color: #ffffff;
        """)
        vbox.addWidget(self.clock_date_label)
        return panel

    def _start_clock(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()

    def _tick(self):
        now = QDateTime.currentDateTime()
        self.clock_time_label.setText(now.toString("HH : mm : ss"))
        self.clock_date_label.setText(now.toString("dddd, d MMMM yyyy"))

    def _build_plate_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: #ffffff; border-radius: 16px;")
        apply_shadow(panel, blur=16, y_offset=6, alpha=50)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(24, 18, 24, 18)
        vbox.setSpacing(12)

        label_row = QHBoxLayout()
        label = QLabel("Nomor Plat Kendaraan Terverifikasi")
        label.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 15px; color: #000000;
        """)
        label_row.addWidget(label)
        label_row.addStretch(1)

        # tombol preview pakai ikon yang sama gayanya dengan tombol lain
        # di aplikasi (load_pixmap + fallback emoji), bukan emoji polos.
        preview_pix = load_pixmap("icon_tag_photo.png")
        self.btn_preview_plate = QPushButton(("  " if not preview_pix.isNull() else "🖼  ") + "Preview")
        if not preview_pix.isNull():
            self.btn_preview_plate.setIcon(QIcon(preview_pix))
            self.btn_preview_plate.setIconSize(QSize(20, 20))
        self.btn_preview_plate.setCursor(Qt.PointingHandCursor)
        self.btn_preview_plate.setToolTip("Lihat foto plat hasil capture")
        self.btn_preview_plate.setMinimumHeight(40)
        self.btn_preview_plate.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_FAMILY}'; font-weight: 600; font-size: 14px;
                color: {COLOR_NAVY}; background-color: {COLOR_NAVY_SOFT_BG};
                border: none; border-radius: 8px; padding: 6px 16px;
            }}
            QPushButton:hover {{ background-color: #c3d8e6; }}
            QPushButton:disabled {{ color: #9aa0a6; background-color: #eef0f2; }}
        """)
        self.btn_preview_plate.clicked.connect(self._show_plate_preview)
        self.btn_preview_plate.setEnabled(False)  # aktif begitu ada foto plat (lihat set_plate_image_path)
        label_row.addWidget(self.btn_preview_plate)
        vbox.addLayout(label_row)

        # sengaja BUKAN read-only -- operator boleh merevisi/mengoreksi
        # nilai plat di sini sebelum di-upload (mis. salah baca OCR).
        # Font diperbesar & bold + kotaknya lebih tinggi supaya lebih
        # gampang dibaca dari jarak jauh (mis. dilihat operator gate).
        self.plate_input = QLineEdit()
        self.plate_input.setAlignment(Qt.AlignCenter)
        self.plate_input.setFixedHeight(76)
        self.plate_input.setFocusPolicy(Qt.ClickFocus)
        self.plate_input.setStyleSheet(f"""
            font-family: '{FONT_FAMILY}'; font-weight: 700; font-size: 32px;
            border: 1px solid #cfd9d8; border-radius: 14px; background-color: #ffffff;
        """)
        # setiap kali operator selesai mengoreksi (Enter / pindah fokus),
        # sinkronkan nilai plat yang baru ke SEMUA tag yang sudah tercatat
        # di daftar -- supaya tidak ada data plat yang "ketinggalan" lama.
        self.plate_input.textChanged.connect(self._sync_plate_correction_to_tags)
        vbox.addWidget(self.plate_input)

        self._plate_image_path = None
        self._plate_captured_at = None
        return panel

    def set_verified_plate(self, plate_text: str):
        self.plate_input.setText(plate_text)

    def _sync_plate_correction_to_tags(self):
        """
        Dipanggil SETIAP KALI teks di kolom plat berubah (live, bukan
        tunggu kolomnya kehilangan fokus -- editingFinished ternyata
        tidak selalu terpicu tepat waktu tergantung urutan klik operator).
        Semua tag yang SUDAH tercatat di daftar ikut di-update nomor
        platnya, dan PostureWorker yang sedang jalan (kalau ada) juga
        diberi tahu lewat on_plate_corrected supaya tag BARU yang
        tertangkap setelah ini juga pakai nomor plat yang benar.
        """
        new_plate = self.plate_input.text().strip() or "TANPA_PLAT"

        changed = False
        for r in self.tag_records + self.handrail_records + self.tank_climb_records + self.tank_descend_records:
            if r.get("plate") != new_plate:
                r["plate"] = new_plate
                changed = True
        if changed:
            self._rebuild_tag_list()

        if self._on_plate_corrected:
            self._on_plate_corrected(new_plate)

    def set_plate_image_path(self, path, captured_at=None):
        """Dipanggil dari main.py setelah foto plat berhasil di-capture --
        mengaktifkan tombol Preview di atas kolom plat. captured_at
        (datetime) opsional, ditampilkan di dialog preview."""
        self._plate_image_path = path
        self._plate_captured_at = captured_at
        self.btn_preview_plate.setEnabled(bool(path and os.path.exists(path)))

    def clear_plate_image(self):
        self._plate_image_path = None
        self._plate_captured_at = None
        self.btn_preview_plate.setEnabled(False)

    def _show_plate_preview(self):
        if not self._plate_image_path:
            return
        plate = self.plate_input.text().strip() or "-"
        waktu_str = (
            self._plate_captured_at.strftime("%Y-%m-%d %H:%M:%S")
            if self._plate_captured_at else "-"
        )
        info = f"Plat: {plate}    |    Waktu capture: {waktu_str}"
        show_image_preview_dialog(
            self, "Preview Foto Plat", self._plate_image_path, info,
            overlay_lines=[f"Plat: {plate}", f"Waktu: {waktu_str}"],
        )


    def _build_tag_list_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: #ffffff; border-radius: 16px;")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        apply_shadow(panel, blur=16, y_offset=6, alpha=50)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(20, 16, 20, 16)
        vbox.setSpacing(8)

        header = QLabel("Status")
        header.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 15px; color: #000000;
        """)
        vbox.addWidget(header)

        self.tag_list = FillListWidget()
        self.tag_list.setFrameShape(QFrame.NoFrame)
        self.tag_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tag_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; }
            QListWidget::item { border-bottom: 1px solid #ececec; }
            QListWidget::item:selected { background: #eef5fb; }
        """)
        # daftar dimulai kosong -- baris ditambahkan realtime lewat add_tag()
        # tiap kali PostureWorker berhasil menangkap 1 tag terpasang.
        vbox.addWidget(self.tag_list, 1)
        return panel

    def add_tag(self, tag_data: dict):
        """
        Dipanggil dari main.py setiap kali PostureWorker mengirim sinyal
        new_tag_signal (1 tag baru selesai terpasang). tag_data berisi:
            {"tag_num", "time", "duration", "plate", "thumb" (ndarray BGR
             90x60), "filepath" (path foto tersimpan di captured_tags/),
             "after_filepath" & "after_thumb" (OPSIONAL -- capture KEDUA:
             crop area ROI setelah tag terpasang, lihat
             detection/pose/tag_roi_capture.py)}

        Satu tag = satu record, tapi di daftar tampil sebagai 2 BARIS
        terpisah: "Dry Check #n" (capture pertama, saat memasang) dan
        "Installed #n" (capture kedua, crop ROI setelah terpasang).

        CATATAN: "tag_num" bawaan dari PostureWorker itu penghitung
        GLOBAL (terus naik, tidak tahu-menahu soal tag yang sudah
        dihapus di UI). Supaya nomor yang TAMPIL selalu urut rapi 1..N
        dan otomatis mulai dari 1 lagi kalau daftarnya kosong, nomor
        tampilannya di-hitung ulang di sini berdasarkan isi daftar saat
        ini -- BUKAN dipakai mentah-mentah dari tracker.
        """
        tag_data = dict(tag_data)  # jangan ubah dict asli yang dikirim worker
        existing_nums = [r.get("tag_num", 0) for r in self.tag_records]
        tag_data["tag_num"] = (max(existing_nums) + 1) if existing_nums else 1
        self._seq += 1
        tag_data["_seq"] = self._seq

        self.tag_records.insert(0, tag_data)
        self._rebuild_tag_list()

    def add_handrail(self, handrail_data: dict):
        """
        Dipanggil dari main.py setiap kali PostureWorker mengirim
        handrail_capture_signal (handrail terdeteksi NAIK = aktif).
        handrail_data berisi:
            {"time", "plate", "thumb" (ndarray BGR kecil), "filepath", "prob"}
        "Handrail Aktif" adalah STATUS (bukan item bernomor seperti tag), jadi
        barisnya tampil tanpa "#n". Tidak ikut di tag_records; dikirim ke
        database lewat parameter terpisah (handrail_records) saat upload.
        """
        data = dict(handrail_data)
        self._seq += 1
        data["_seq"] = self._seq
        self.handrail_records.insert(0, data)
        self._rebuild_tag_list()

    def remove_handrail(self, handrail_data: dict):
        """Hapus 1 capture handrail dari daftar."""
        self.handrail_records = [r for r in self.handrail_records if r is not handrail_data]
        self._rebuild_tag_list()

    def add_tank_climb(self, climb_data: dict):
        """
        Dipanggil dari main.py setiap kali PostureWorker mengirim tank_climb_signal
        (petugas terdeteksi NAIK ke atas tangki). climb_data berisi:
            {"time", "plate", "thumb" (ndarray BGR kecil), "filepath",
             "video_time" (OPSIONAL, "mm:ss" -- hanya kalau sumbernya file video)}
        Sama seperti handrail: STATUS (tanpa nomor), tidak masuk tag_records, dan dikirim ke
        database lewat parameter terpisah (tank_climb_records) saat upload.
        """
        data = dict(climb_data)
        self._seq += 1
        data["_seq"] = self._seq
        self.tank_climb_records.insert(0, data)
        self._rebuild_tag_list()

    def remove_tank_climb(self, climb_data: dict):
        """Hapus 1 capture naik-tangki dari daftar."""
        self.tank_climb_records = [r for r in self.tank_climb_records if r is not climb_data]
        self._rebuild_tag_list()

    def add_tank_descend(self, descend_data: dict):
        """
        Dipanggil dari main.py setiap kali PostureWorker mengirim tank_descend_signal
        (AMT terlihat TURUN dari tangki). descend_data berisi isi yang sama dengan add_tank_climb():
            {"time", "plate", "thumb", "filepath", "video_time" (OPSIONAL, hanya sumber file)}
        STATUS tanpa nomor, tidak masuk tag_records; dikirim ke database lewat parameter
        tank_descend_records saat upload (kalau ada beberapa, yang tersimpan yang TERAKHIR).
        """
        data = dict(descend_data)
        self._seq += 1
        data["_seq"] = self._seq
        self.tank_descend_records.insert(0, data)
        self._rebuild_tag_list()

    def remove_tank_descend(self, descend_data: dict):
        """Hapus 1 capture turun-tangki dari daftar."""
        self.tank_descend_records = [r for r in self.tank_descend_records if r is not descend_data]
        self._rebuild_tag_list()

    def clear_tags(self):
        """Kosongkan daftar tag, handrail & naik tangki di layar (dipakai saat Reset & setelah upload berhasil)."""
        self.tag_list.clear()
        self.tag_records.clear()
        self.handrail_records.clear()
        self.tank_climb_records.clear()
        self.tank_descend_records.clear()

    def remove_tag(self, tag_data: dict, kind: str = "dry"):
        """
        Hapus SATU FOTO dari tag (tombol 'x' di baris itu) -- bukan otomatis
        menghapus tag seluruhnya:
            kind "dry"       -> hapus foto Dry Check saja (baris Installed, kalau
                                 ada, tetap ada)
            kind "installed" -> hapus foto Installed saja (baris Dry Check tetap ada)
        Tag itu (dan nomornya) baru dihapus SELURUHNYA kalau setelah ini ia sudah
        tidak punya foto sama sekali (Dry Check maupun Installed) -- baru di situ
        nomor tag yang lebih besar digeser turun 1 supaya tetap urut tanpa lompat.
        Contoh: ada tag 1,2,3,4,5 -> tag 3 (Dry Check & Installed) habis dihapus ->
        tag 4 jadi 3, tag 5 jadi 4 (tag 1 & 2 tidak berubah).
        """
        if kind == "installed":
            tag_data["after_filepath"] = None
            tag_data["after_thumb"] = None
        else:
            tag_data["filepath"] = None
            tag_data["thumb"] = None

        if not tag_data.get("filepath") and not tag_data.get("after_filepath"):
            deleted_num = tag_data.get("tag_num")
            self.tag_records = [r for r in self.tag_records if r is not tag_data]
            if deleted_num is not None:
                for r in self.tag_records:
                    if r.get("tag_num") is not None and r["tag_num"] > deleted_num:
                        r["tag_num"] -= 1
        self._rebuild_tag_list()

    def _rebuild_tag_list(self):
        """Gambar ulang seluruh daftar tag dari self.tag_records (urutan dipertahankan)."""
        self.tag_list.clear()
        # tag, handrail & naik tangki digabung, yang paling baru di atas (urut _seq menurun)
        entries = ([("tag", r) for r in self.tag_records]
                   + [("handrail", r) for r in self.handrail_records]
                   + [("climb", r) for r in self.tank_climb_records]
                   + [("descend", r) for r in self.tank_descend_records])
        entries.sort(key=lambda e: e[1].get("_seq", 0), reverse=True)
        for entry_type, data in entries:
            if entry_type == "handrail":
                kinds = ["handrail"]
            elif entry_type == "climb":
                kinds = ["climb"]
            elif entry_type == "descend":
                kinds = ["descend"]
            else:
                # urutan per tag: Dry Check (capture 1) lalu Installed (capture 2). Baris yang
                # fotonya sudah dihapus (lihat remove_tag) tidak ikut ditampilkan lagi.
                kinds = [k for k, has in (("dry", data.get("thumb") is not None),
                                          ("installed", data.get("after_thumb") is not None)) if has]
            for kind in kinds:
                row = self._build_tag_row(data, kind)
                item = QListWidgetItem()
                item.setSizeHint(row.sizeHint())
                self.tag_list.addItem(item)
                self.tag_list.setItemWidget(item, row)
        self.tag_list._relayout_rows()

    def _show_tag_preview(self, tag_data: dict, kind: str = "dry"):
        """
        Tampilkan foto ukuran penuh saat barisnya diklik.
        kind = "dry"       -> foto Dry Check (capture pertama, saat memasang)
               "installed" -> foto Installed (capture kedua, crop ROI)
               "handrail"  -> foto Handrail Aktif (handrail terdeteksi naik)
               "climb"     -> foto AMT Naik Tangki (AMT terdeteksi naik ke atas tangki)
               "descend"   -> foto AMT Turun Tangki (AMT terlihat turun dari tangki)
        """
        self._sync_plate_correction_to_tags()  # jaring pengaman -- pastikan datanya paling baru
        plate = tag_data.get("plate", "-")
        if kind in ("climb", "descend"):
            label = LABEL_NAIK_TANGKI if kind == "climb" else LABEL_TURUN_TANGKI
            waktu = tag_data.get("time", "-")
            vt = tag_data.get("video_time")
            waktu_txt = f"{waktu} (video {vt})" if vt else waktu
            show_image_preview_dialog(
                self, f"Preview - Plat {plate} - {label}",
                tag_data.get("filepath"),
                f"Plat: {plate}    |    {label}    |    Waktu: {waktu_txt}",
                overlay_lines=[f"Plat: {plate}", f"{label.upper()}", f"Waktu: {waktu}"]
                + ([f"Video: {vt}"] if vt else []),
            )
            return
        if kind == "handrail":
            waktu = tag_data.get("time", "-")
            show_image_preview_dialog(
                self, f"Preview - Plat {plate} - Handrail Aktif",
                tag_data.get("filepath"),
                f"Plat: {plate}    |    Handrail Aktif    |    Waktu: {waktu}",
                overlay_lines=[f"Plat: {plate}", "Handrail: AKTIF", f"Waktu: {waktu}"],
            )
            return
        tag_num = tag_data.get("tag_num", "?")
        info = (
            f"Plat: {plate}    |    Tag #{tag_num}    |    "
            f"Waktu: {tag_data.get('time', '-')}    |    "
            f"Durasi: {tag_data.get('duration', '-')}"
        )
        if kind == "installed":
            # foto Installed = crop ROI kecil -> tanpa kotak overlay (akan menutupi
            # hampir seluruh gambar); info sudah ada di teks bawah dialog. Tanpa "Durasi"
            # (durasi hanya milik Dry Check).
            info_installed = (
                f"Plat: {plate}    |    Installed #{tag_num}    |    "
                f"Waktu: {tag_data.get('time', '-')}"
            )
            show_image_preview_dialog(
                self, f"Preview - Plat {plate} - Installed #{tag_num}",
                tag_data.get("after_filepath"),
                info_installed,
            )
            return

        overlay_lines = [
            f"Plat: {plate}",
            f"Tag: #{tag_num}",
            f"Waktu: {tag_data.get('time', '-')}",
            f"Durasi: {tag_data.get('duration', '-')}",
        ]
        show_image_preview_dialog(
            self, f"Preview - Plat {plate} - Dry Check #{tag_num}",
            tag_data.get("filepath"),
            info.replace(f"Tag #{tag_num}", f"Dry Check #{tag_num}"),
            overlay_lines,
        )

    def _build_tag_row(self, tag_data: dict, kind: str = "dry") -> QWidget:
        """
        1 baris = 1 foto.
        kind "dry"       -> "Dry Check #n"  (capture pertama, saat memasang tag) -- "Pengecekan <waktu>" + durasi
        kind "installed" -> "Installed #n"  (capture kedua, crop ROI setelah terpasang) -- "Terpasang <waktu>", TANPA durasi
        kind "handrail"  -> "Handrail Aktif" (status; handrail terdeteksi naik, tanpa nomor)
        kind "climb"     -> "AMT Naik Tangki"  (status; AMT naik ke atas tangki, tanpa nomor)
        kind "descend"   -> "AMT Turun Tangki" (status; AMT turun dari tangki, tanpa nomor)
        """
        is_installed = (kind == "installed")
        is_handrail = (kind == "handrail")
        is_climb = (kind == "climb")
        is_descend = (kind == "descend")
        is_status = is_handrail or is_climb or is_descend   # baris STATUS: tanpa nomor "#n"
        kind_label = ("Handrail Aktif" if is_handrail else LABEL_NAIK_TANGKI if is_climb
                      else LABEL_TURUN_TANGKI if is_descend
                      else "Installed" if is_installed else "Dry Check")
        plate = tag_data.get("plate", "-")
        tag_num = tag_data.get("tag_num", "?")
        time_str = tag_data.get("time", "-")
        duration = tag_data.get("duration", "-")
        # ndarray BGR kecil dari PostureWorker: thumb (90x60) = Dry Check,
        # after_thumb (proporsi asli crop ROI) = Installed
        thumb = tag_data.get("after_thumb") if is_installed else tag_data.get("thumb")

        row = TagRowWidget()
        row.setCursor(Qt.PointingHandCursor)
        row.setStyleSheet("background: transparent;")
        row.clicked.connect(lambda: self._show_tag_preview(tag_data, kind))

        h = QHBoxLayout(row)
        h.setContentsMargins(4, 8, 4, 8)
        h.setSpacing(12)

        icon = QLabel(row)
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(48, 48)
        icon.setStyleSheet("background-color: #f3f4f6; border-radius: 8px;")
        if thumb is not None:
            pix = cv_to_qpixmap(thumb)
            icon.setPixmap(pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            icon.setText("🖼")
            icon.setStyleSheet(icon.styleSheet() + "font-size: 22px;")

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        name_lbl = QLabel(
            f"Plat {plate} - {kind_label}" if is_status else f"Plat {plate} - {kind_label} #{tag_num}"
        )
        name_lbl.setWordWrap(True)
        name_lbl.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 13px; color: #000000;
        """)
        if is_climb:
            # tanggal + jam naik tangki dalam SATU baris teks (seperti baris Handrail) -- kolom jam terpisah
            # di kanan memakan ~140 px sampai nama & tanggal terpotong di panel yang sempit.
            date_txt = f"Naik {time_str}"
        elif is_descend:
            date_txt = f"Turun {time_str}"
        elif is_handrail:
            date_txt = f"Aktif {time_str}"
        elif is_installed:
            date_txt = f"Terpasang {time_str}"       # foto SETELAH tag terpasang
        else:
            date_txt = f"Pengecekan {time_str}"      # Dry Check = pengecekan saat memasang, BUKAN "terpasang"
        date_lbl = QLabel(date_txt)
        date_lbl.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-size: 10px; color: #4b4b4b;
        """)
        text_col.addWidget(name_lbl)
        text_col.addWidget(date_lbl)

        duration_row = QHBoxLayout()
        duration_row.setSpacing(4)
        clock_icon = make_icon_label(row, "icon_clock_small.png", "🕐", size=14, font_px=12)
        duration_lbl = QLabel(duration)
        duration_lbl.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-size: 12px; color: #000000;
        """)
        duration_row.addWidget(clock_icon)
        duration_row.addWidget(duration_lbl)
        if is_status or is_installed:
            # handrail & naik tangki tidak punya "durasi pemasangan"; foto Installed juga tidak
            # (durasi hanya ditampilkan di baris Dry Check) -> tanpa kolom kanan
            clock_icon.hide()
            duration_lbl.hide()

        # tombol hapus -- pojok kanan atas baris (lihat TagRowWidget di atas
        # untuk penjelasan kenapa klik tombol ini tidak ikut memicu preview)
        delete_btn = QPushButton("✕")
        delete_btn.setFixedSize(24, 24)
        delete_btn.setCursor(Qt.PointingHandCursor)
        # Tombol hapus per BARIS -- menghapus foto baris itu saja (Dry Check atau Installed),
        # bukan seluruh tag. Tag (nomornya) baru hilang kalau kedua fotonya sudah tidak ada.
        delete_btn.setToolTip(
            "Hapus capture handrail ini" if is_handrail else
            "Hapus capture naik tangki ini" if is_climb else
            "Hapus capture turun tangki ini" if is_descend else
            (f"Hapus foto Installed #{tag_num} (Dry Check #{tag_num} tetap ada)" if is_installed else
             f"Hapus foto Dry Check #{tag_num} (Installed #{tag_num} tetap ada)" if tag_data.get("after_thumb") is not None else
             f"Hapus tag #{tag_num}")
        )
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #fdecea; color: #c0392b;
                border-radius: 12px; border: none; font-weight: 700;
            }
            QPushButton:hover { background-color: #f8c9c4; }
        """)
        if is_handrail:
            delete_btn.clicked.connect(lambda: self.remove_handrail(tag_data))
        elif is_climb:
            delete_btn.clicked.connect(lambda: self.remove_tank_climb(tag_data))
        elif is_descend:
            delete_btn.clicked.connect(lambda: self.remove_tank_descend(tag_data))
        else:
            delete_btn.clicked.connect(lambda: self.remove_tag(tag_data, kind))

        h.addWidget(icon, alignment=Qt.AlignVCenter)
        h.addLayout(text_col, 1)
        h.addLayout(duration_row)
        h.addWidget(delete_btn, alignment=Qt.AlignTop)
        h.setAlignment(text_col, Qt.AlignVCenter)
        h.setAlignment(duration_row, Qt.AlignVCenter)

        # simpan referensi supaya FillListWidget bisa scale ulang ukuran
        # ikon & font saat baris ini melebar/memanjang (lihat _relayout_rows)
        if (is_climb or is_descend) and tag_data.get("video_time"):   # sumber file video: waktu di dalam video ada di tooltip & preview
            row.setToolTip(f"Waktu di dalam video: {tag_data['video_time']}")
        row.icon_lbl = icon
        row.name_lbl = name_lbl
        row.date_lbl = date_lbl
        row.clock_icon_lbl = clock_icon
        row.duration_lbl = duration_lbl
        return row

    # ------------------------------------------------------------------
    def _build_buttons_row(self):
        row = QHBoxLayout()
        row.setSpacing(16)

        back_pix = load_pixmap("icon_back.png")
        self.btn_kembali = QPushButton(("  " if not back_pix.isNull() else "‹  ") + "Kembali")
        if not back_pix.isNull():
            self.btn_kembali.setIcon(QIcon(back_pix))
            self.btn_kembali.setIconSize(QSize(22, 22))
        self.btn_kembali.setMinimumHeight(54)
        self.btn_kembali.setCursor(Qt.PointingHandCursor)
        self.btn_kembali.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_FAMILY}'; font-weight: 600; font-size: 18px;
                color: {COLOR_NAVY}; background-color: {COLOR_NAVY_SOFT_BG};
                border: 1px solid {COLOR_NAVY}; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #c3d8e6; }}
        """)
        if self._on_kembali:
            self.btn_kembali.clicked.connect(self._on_kembali)

        verif_pix = load_pixmap("icon_verified.png")
        self.btn_verifikasi = QPushButton(("  " if not verif_pix.isNull() else "✓  ") + "Verifikasi")
        if not verif_pix.isNull():
            self.btn_verifikasi.setIcon(QIcon(verif_pix))
            self.btn_verifikasi.setIconSize(QSize(22, 22))
        self.btn_verifikasi.setToolTip("Upload foto plat & semua tag yang tertangkap ke database")
        self.btn_verifikasi.setMinimumHeight(54)
        self.btn_verifikasi.setCursor(Qt.PointingHandCursor)
        self.btn_verifikasi.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_FAMILY}'; font-weight: 600; font-size: 18px;
                color: #ffffff; background-color: {COLOR_NAVY};
                border-radius: 8px; border: none;
            }}
            QPushButton:hover {{ background-color: #062338; }}
        """)
        self.btn_verifikasi.clicked.connect(self._on_upload)

        row.addWidget(self.btn_kembali, 1)
        row.addWidget(self.btn_verifikasi, 1)
        return row