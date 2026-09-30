"""
Halaman 'Verifikasi Plat Nomor' - versi REFLOW.

Beda dengan versi sebelumnya (yang pakai widget.setGeometry() presisi
1:1 ke Figma), versi ini pakai QVBoxLayout/QHBoxLayout + QSizePolicy
supaya semua elemen ikut menyesuaikan saat window di-resize atau
di-maximize. Proporsi, warna, dan struktur tetap mengikuti desain
Figma (node 290:397) -- tapi angka piksel yang di Figma di sini
diperlakukan sebagai "ukuran minimum/dasar", bukan posisi mutlak.
"""
from PyQt5.QtCore import Qt, QTimer, QDateTime, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QFrame, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QMessageBox
)

from widgets.design_tokens import (
    BackgroundPage, load_pixmap, make_icon_label, apply_shadow, render_cv_frame,
    FONT_FAMILY, COLOR_BLUE, COLOR_BLUE_SOFT_BG, COLOR_GRAY_TEXT, COLOR_BG,
    CCTV_HEADER_GRADIENT, CCTV_FOOTER_GRADIENT,
)


class VerifikasiPage(BackgroundPage):
    def __init__(self, on_verified, on_go_monitoring, on_reset, parent=None):
        super().__init__(parent)
        self._on_verified = on_verified
        self._on_go_monitoring = on_go_monitoring
        self._on_reset = on_reset

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(20)

        outer.addLayout(self._build_top_row())

        content_row = QHBoxLayout()
        content_row.setSpacing(24)
        content_row.addWidget(self._build_cctv_panel(), 65)
        content_row.addWidget(self._build_right_panel(), 35)
        outer.addLayout(content_row, 1)

        self._start_footer_clock()

    # ------------------------------------------------------------------
    def _build_top_row(self):
        row = QHBoxLayout()
        row.addWidget(self._build_logo_card())
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
                background: transparent;
                font-family: '{FONT_FAMILY}'; font-weight: 800;
                font-size: 14px; color: #ED1C24;
            """)
        lay.addWidget(logo)
        return card

    # ------------------------------------------------------------------
    # PANEL CCTV (kiri) -- figma: node 303:3
    # ------------------------------------------------------------------
    def _build_cctv_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: #ffffff; border-radius: 22px;")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        apply_shadow(panel, blur=22, y_offset=10, alpha=55)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # -- header --
        header = QFrame()
        header.setFixedHeight(66)
        header.setStyleSheet(f"""
            background: {CCTV_HEADER_GRADIENT};
            border-top-left-radius: 22px; border-top-right-radius: 22px;
        """)
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 8, 20, 8)
        h.setSpacing(10)

        icon_cctv = make_icon_label(header, "icon_cctv.png", "📹", size=28, font_px=18)
        title_cctv = QLabel("Kamera ANPR")
        title_cctv.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 18px; color: #ffffff;
        """)
        self.live_dot = QLabel()
        self.live_dot.setFixedSize(14, 14)
        self.live_dot.setStyleSheet("background-color: #22c55e; border-radius: 7px;")
        self.live_text = QLabel("LIVE")
        self.live_text.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 15px; color: #ffffff;
        """)
        # indikator LIVE baru muncul begitu ANPRWorker benar-benar mengirim
        # frame pertama (lihat set_live()) -- disembunyikan dulu di awal.
        self.live_dot.hide()
        self.live_text.hide()
        h.addWidget(icon_cctv)
        h.addWidget(title_cctv)
        h.addStretch(1)
        h.addWidget(self.live_dot)
        h.addWidget(self.live_text)

        # -- body video (placeholder teks loading -- diganti frame asli begitu ANPRWorker jalan) --
        self.video_label = QLabel("Memuat kamera ANPR...\nMohon tunggu sebentar.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumHeight(180)
        self.video_label.setStyleSheet(f"""
            background-color: #ffffff; color: #9aa0a6;
            font-family: '{FONT_FAMILY}'; font-size: 16px;
        """)

        # -- footer --
        footer = QFrame()
        footer.setFixedHeight(48)
        footer.setStyleSheet(f"""
            background: {CCTV_FOOTER_GRADIENT};
            border-bottom-left-radius: 22px; border-bottom-right-radius: 22px;
        """)
        f = QHBoxLayout(footer)
        f.setContentsMargins(20, 0, 20, 0)
        self.footer_time_label = QLabel("")
        self.footer_time_label.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-size: 15px; color: #ffffff;
        """)
        f.addWidget(self.footer_time_label)
        f.addStretch(1)

        vbox.addWidget(header)
        vbox.addWidget(self.video_label, 1)
        vbox.addWidget(footer)
        return panel

    def _start_footer_clock(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_footer_time)
        self._timer.start(1000)
        self._update_footer_time()

    def _update_footer_time(self):
        now = QDateTime.currentDateTime()
        self.footer_time_label.setText(now.toString("d MMM yyyy    HH : mm"))

    # ------------------------------------------------------------------
    # PANEL FORM VERIFIKASI (kanan) -- figma: node 290:547
    # ------------------------------------------------------------------
    def _build_right_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: rgba(255, 255, 255, 217); border-radius: 18px;")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(40, 32, 40, 32)
        vbox.setSpacing(10)
        vbox.addStretch(1)

        # -- ikon sertifikat --
        icon_wrap = QLabel()
        icon_wrap.setFixedSize(110, 110)
        icon_wrap.setAlignment(Qt.AlignCenter)
        icon_wrap.setStyleSheet("background-color: #eaf0fb; border-radius: 55px;")
        icon_pix = load_pixmap("icon_certificate.png")
        if not icon_pix.isNull():
            icon_wrap.setPixmap(icon_pix.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            icon_wrap.setText("🪪")
            icon_wrap.setStyleSheet(icon_wrap.styleSheet() + "font-size: 34px;")
        vbox.addWidget(icon_wrap, alignment=Qt.AlignHCenter)

        # -- judul --
        title = QLabel("Verifikasi Plat Nomor")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 800; font-size: 30px; color: #000000;
        """)
        vbox.addWidget(title)

        # -- subjudul --
        subtitle = QLabel("Masukkan nomor plat kendaraan untuk melakukan verifikasi")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"""
            background: transparent; font-family: '{FONT_FAMILY}';
            font-weight: 600; font-size: 18px; color: {COLOR_GRAY_TEXT};
        """)
        vbox.addWidget(subtitle)

        vbox.addSpacing(14)

        # -- kolom input --
        self.plate_input = QLineEdit()
        self.plate_input.setPlaceholderText("Contoh: B 1234 ABC")
        self.plate_input.setAlignment(Qt.AlignCenter)
        self.plate_input.setFixedHeight(64)
        # ClickFocus -- kursor/keyboard fokus HANYA aktif kalau kolom ini
        # benar-benar diklik operator. Kalau tidak diset begini, Qt bisa
        # otomatis kasih fokus ke kolom ini pas window pertama dibuka,
        # dan itu bikin auto_fill_plate() (di bawah) SELALU mengira operator
        # "sedang mengetik manual" sehingga hasil OCR tidak pernah
        # otomatis mengisi kolom ini.
        self.plate_input.setFocusPolicy(Qt.ClickFocus)
        self.plate_input.setStyleSheet(f"""
            font-family: '{FONT_FAMILY}'; font-size: 20px;
            border: 1px solid {COLOR_GRAY_TEXT}; border-radius: 14px;
            background-color: #ffffff;
        """)
        vbox.addWidget(self.plate_input)

        vbox.addSpacing(6)

        # -- tombol Reset + Verifikasi --
        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        reset_pix = load_pixmap("icon_reset.png")
        self.btn_reset = QPushButton(("  " if not reset_pix.isNull() else "↺  ") + "Reset")
        if not reset_pix.isNull():
            self.btn_reset.setIcon(QIcon(reset_pix))
            self.btn_reset.setIconSize(QSize(22, 22))
        self.btn_reset.setMinimumHeight(52)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_FAMILY}'; font-weight: 600; font-size: 18px;
                color: {COLOR_BLUE}; background-color: {COLOR_BLUE_SOFT_BG};
                border: 1px solid {COLOR_BLUE}; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: rgba(0,84,140,55); }}
        """)
        self.btn_reset.clicked.connect(self._handle_reset)

        verif_pix = load_pixmap("icon_verified.png")
        self.btn_verifikasi = QPushButton(("  " if not verif_pix.isNull() else "✓  ") + "Verifikasi")
        if not verif_pix.isNull():
            self.btn_verifikasi.setIcon(QIcon(verif_pix))
            self.btn_verifikasi.setIconSize(QSize(22, 22))
        self.btn_verifikasi.setMinimumHeight(52)
        self.btn_verifikasi.setCursor(Qt.PointingHandCursor)
        self.btn_verifikasi.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_FAMILY}'; font-weight: 600; font-size: 18px;
                color: #ffffff; background-color: {COLOR_BLUE};
                border-radius: 8px; border: none;
            }}
            QPushButton:hover {{ background-color: #003f68; }}
        """)
        self.btn_verifikasi.clicked.connect(self._handle_verify)

        btn_row.addWidget(self.btn_reset, 1)
        btn_row.addWidget(self.btn_verifikasi, 1)
        vbox.addLayout(btn_row)

        # -- tombol Monitoring (kembali ke halaman monitoring) --
        monitor_pix = load_pixmap("icon_monitor.png")
        self.btn_monitoring = QPushButton(("  " if not monitor_pix.isNull() else "🖥  ") + "Monitoring")
        if not monitor_pix.isNull():
            self.btn_monitoring.setIcon(QIcon(monitor_pix))
            self.btn_monitoring.setIconSize(QSize(22, 22))
        self.btn_monitoring.setMinimumHeight(56)
        self.btn_monitoring.setCursor(Qt.PointingHandCursor)
        self.btn_monitoring.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_FAMILY}'; font-weight: 600; font-size: 18px;
                color: #ffffff; background-color: {COLOR_BLUE};
                border-radius: 8px; border: none;
            }}
            QPushButton:hover {{ background-color: #003f68; }}
        """)
        self.btn_monitoring.clicked.connect(self._on_go_monitoring)
        vbox.addSpacing(6)
        vbox.addWidget(self.btn_monitoring)

        vbox.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Dipanggil dari main.py (MainWindow), disambungkan ke ANPRWorker
    # ------------------------------------------------------------------
    def update_frame(self, cv_frame):
        """Tampilkan 1 frame hasil ANPRWorker (sudah ada overlay box+teks plat)."""
        render_cv_frame(self.video_label, cv_frame)

    def set_live(self, active: bool):
        """Tampilkan/sembunyikan indikator LIVE -- dipanggil dari main.py
        setiap kali ANPRWorker benar-benar mengirim frame (jadi bukan
        sekadar 'thread sudah start', tapi 'video sudah benar-benar jalan')."""
        self.live_dot.setVisible(active)
        self.live_text.setVisible(active)

    def auto_fill_plate(self, text: str):
        """
        Auto-isi input plat dari hasil OCR realtime -- TAPI tidak menimpa
        kalau operator sedang aktif mengetik manual di kolom ini
        (dicek lewat hasFocus()), sama seperti perilaku di ui.py.
        """
        if text and text.strip() and not self.plate_input.hasFocus():
            self.plate_input.setText(text)

    def reset_view(self):
        """Dipanggil saat tombol Reset ditekan -- kosongkan input & video."""
        self.plate_input.clear()
        self.video_label.setText("Memuat kamera ANPR...\nMohon tunggu sebentar.")
        self.set_live(False)

    def _handle_reset(self):
        self._on_reset()

    def _handle_verify(self):
        plate = self.plate_input.text().strip()
        if not plate:
            QMessageBox.warning(self, "Peringatan", "Nomor plat belum diisi.")
            return
        self._on_verified(plate)