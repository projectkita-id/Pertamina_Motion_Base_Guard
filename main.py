"""
Entry point aplikasi. Menyatukan:
  - UI (widgets/page_verifikasi.py, widgets/page_monitoring.py)
  - Worker deteksi background (widgets/workers.py: ANPRWorker, PostureWorker)
  - Upload ke database (widgets/db_client.py -- masih stub, lihat file itu)
  - Modul deteksi asli Anda di folder detection/ (anpr_main.py, v6.py, dst.)

ALUR SISTEM (mengikuti ui.py, dengan penyesuaian):
  1. App dibuka -> langsung di halaman VERIFIKASI. ANPRWorker otomatis
     jalan di background: video live + auto-isi kolom plat dari OCR.
  2. Tombol "Verifikasi" (halaman Verifikasi)
     -> validasi kolom plat terisi, simpan foto plat BERSIH (tanpa overlay)
        ke folder captured_plates/, ANPRWorker DIHENTIKAN (hemat resource
        -- cuma 1 model deteksi yang jalan dalam satu waktu), lalu pindah
        ke halaman Monitoring dan mulai PostureWorker dengan nomor plat tsb.
  3. Tombol "Monitoring" (halaman Verifikasi)
     -> CUMA pindah halaman. ANPRWorker TETAP jalan di background (tidak
        dihentikan) -- TANPA menyimpan foto plat & TANPA memulai
        PostureWorker.
  4. Tombol "Kembali" (halaman Monitoring)
     -> balik ke halaman Verifikasi. PostureWorker TETAP jalan di
        background (tidak dihentikan) -- ANPRWorker TIDAK otomatis
        jalan lagi (harus Reset dulu kalau mau verifikasi plat baru).
  5. Tombol "Verifikasi" (halaman Monitoring)
     -> upload foto plat + SEMUA tag yang sudah tertangkap ke database
        (lihat widgets/db_client.py). Kalau BERHASIL: daftar tag &
        kolom plat di layar dikosongkan, DAN file .jpg lokalnya
        (captured_plates/captured_tags) dihapus dari disk (sudah aman
        tersimpan di database, tidak perlu dobel di disk).
  6. Tombol "Reset" (halaman Verifikasi)
     -> reset total: hentikan semua worker, kosongkan video/plat/daftar
        tag, lalu ANPRWorker dijalankan ulang dari nol.

Jalankan:
    python main.py            (mode development, ada title bar)
    python main.py --kiosk    (fullscreen tanpa title bar, utk Jetson Nano dst.)
"""
import os
import sys
from datetime import datetime

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox

from widgets.page_monitoring import MonitoringPage
from widgets.page_verifikasi import VerifikasiPage
from widgets.workers import ANPRWorker, PostureWorker
from widgets.db_client import upload_records

KIOSK_MODE = "--kiosk" in sys.argv or "--fullscreen" in sys.argv
MIN_WIDTH = 800
MIN_HEIGHT = 480

CAPTURED_PLATES_DIR = "captured_plates"


class MainWindow(QMainWindow):
    def __init__(self, kiosk: bool = False):
        super().__init__()
        self.setWindowTitle("Monitoring Plat & Postur - Pertamina Patra Niaga")
        self.setWindowIcon(QIcon("assets/app_icon.ico"))
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        if kiosk:
            self.setWindowFlags(Qt.FramelessWindowHint)
        else:
            self.resize(1366, 850)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_verifikasi = VerifikasiPage(
            on_verified=self.handle_verify,
            on_go_monitoring=self.goto_monitoring_only,
            on_reset=self.handle_reset,
        )
        self.page_monitoring = MonitoringPage(
            on_upload=self.handle_upload,
            on_kembali=self.goto_verifikasi,
            on_plate_corrected=self.handle_plate_corrected,
        )

        self.stack.addWidget(self.page_verifikasi)  # index 0 -- halaman AWAL
        self.stack.addWidget(self.page_monitoring)  # index 1

        self.anpr_thread = None
        self.posture_thread = None
        self.last_clean_anpr_frame = None
        self.last_captured_plate_path = None
        self.last_captured_plate_time = None  # waktu FOTO plat diambil (beda dgn waktu upload)

        # sesuai ui.py: ANPRWorker langsung jalan begitu app dibuka
        self.start_anpr_thread()

    # ------------------------------------------------------------------
    # NAVIGASI ANTAR HALAMAN
    # ------------------------------------------------------------------
    def goto_verifikasi(self):
        """Tombol 'Kembali' di halaman Monitoring -- PostureWorker TETAP jalan."""
        self.stack.setCurrentWidget(self.page_verifikasi)

    def goto_monitoring_only(self):
        """
        Tombol 'Monitoring' di halaman Verifikasi -- CUMA pindah halaman.
        ANPRWorker TIDAK dihentikan di sini (beda dengan tombol
        'Verifikasi') -- cuma render video-nya yang berhenti karena
        halaman Verifikasi sedang tidak dilihat (lihat update_anpr_frame).
        PostureWorker juga TIDAK dimulai di sini -- baru jalan setelah
        plat benar-benar di-'Verifikasi' (lihat handle_verify()).
        """
        self.stack.setCurrentWidget(self.page_monitoring)

    def handle_verify(self, plate_text: str):
        """Tombol 'Verifikasi' di halaman Verifikasi -- simpan foto plat lalu pindah."""
        plate = plate_text.strip() if plate_text else "TANPA_PLAT"

        if self.last_clean_anpr_frame is not None:
            os.makedirs(CAPTURED_PLATES_DIR, exist_ok=True)
            capture_time = datetime.now()
            timestamp = capture_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{plate}_{timestamp}.jpg"
            filepath = os.path.join(CAPTURED_PLATES_DIR, filename)
            cv2.imwrite(filepath, self.last_clean_anpr_frame)
            self.last_captured_plate_path = filepath
            self.last_captured_plate_time = capture_time  # waktu FOTO diambil, disimpan terpisah
            print(f"[INFO] Foto plat disimpan: {filepath} (waktu capture: {capture_time})")

        self.page_monitoring.set_verified_plate(plate)
        self.page_monitoring.set_plate_image_path(
            self.last_captured_plate_path, self.last_captured_plate_time
        )
        self.stop_anpr_thread()  # ANPR selesai tugasnya untuk plat ini -- matikan, hemat resource
        self._ensure_posture_running(plate)
        self.stack.setCurrentWidget(self.page_monitoring)

    def _ensure_posture_running(self, plate: str):
        if self.posture_thread is not None and self.posture_thread.isRunning():
            self.posture_thread.set_plate_number(plate)
        else:
            self.start_posture_thread(plate)

    def handle_plate_corrected(self, new_plate: str):
        """
        Dipanggil dari MonitoringPage setiap kali operator mengoreksi
        kolom plat. Kalau PostureWorker sedang jalan, kasih tahu nomor
        plat yang baru -- supaya TAG BARU yang tertangkap setelah
        koreksi ini juga langsung pakai nomor yang benar (bukan cuma
        tag yang sudah ada di daftar).
        """
        if self.posture_thread is not None and self.posture_thread.isRunning():
            self.posture_thread.set_plate_number(new_plate)

    # ------------------------------------------------------------------
    # UPLOAD KE DATABASE (tombol 'Verifikasi' di halaman Monitoring)
    # ------------------------------------------------------------------
    def handle_upload(self):
        records = self.page_monitoring.tag_records
        if not records:
            QMessageBox.information(self, "Info", "Belum ada tag yang tertangkap untuk diupload.")
            return

        plate_number = self.page_monitoring.plate_input.text().strip() or "TANPA_PLAT"
        handrail_records = list(self.page_monitoring.handrail_records)
        tank_climb_records = list(self.page_monitoring.tank_climb_records)
        tank_descend_records = list(self.page_monitoring.tank_descend_records)
        ok = upload_records(
            plate_number, self.last_captured_plate_path,
            self.last_captured_plate_time, records, handrail_records, tank_climb_records,
            tank_descend_records,
        )
        if ok:
            self._delete_local_capture_files(records)
            extra = ((" + status handrail aktif" if handrail_records else "")
                     + (" + waktu AMT naik tangki" if tank_climb_records else "")
                     + (" + waktu AMT turun tangki" if tank_descend_records else ""))
            QMessageBox.information(
                self, "Berhasil",
                f"{len(records)} data tag{extra} & foto plat berhasil diupload, "
                f"daftar dikosongkan dan file lokal dihapus."
            )
            # sesuai permintaan: setelah upload sukses, otomatis balik ke
            # halaman Verifikasi dan mulai ulang dari nol (sama seperti
            # tombol Reset) -- siap dipakai untuk kendaraan berikutnya.
            self.handle_reset()
            self.stack.setCurrentWidget(self.page_verifikasi)
        else:
            QMessageBox.warning(
                self, "Gagal",
                "Upload gagal (cek koneksi/server). Daftar tag TIDAK dihapus, silakan coba lagi."
            )

    def _delete_local_capture_files(self, tag_records: list):
        """
        Setelah gambar SUKSES tersimpan di database (sebagai LONGBLOB),
        file .jpg lokalnya sudah tidak perlu lagi -- dihapus dari disk
        supaya penyimpanan device tidak penuh seiring waktu.
        """
        paths = (
            [self.last_captured_plate_path]
            + [r.get("filepath") for r in tag_records]
            # capture KEDUA (foto ROI setelah tag terpasang) -- ikut dibersihkan
            + [r.get("after_filepath") for r in tag_records]
            # capture "Handrail Aktif" (tidak ikut di-upload, tapi filenya dibersihkan juga)
            + [r.get("filepath") for r in self.page_monitoring.handrail_records]
            # capture "Naik Tangki" (dikirim ke database sebagai kolom di verifikasi; filenya dibersihkan juga)
            + [r.get("filepath") for r in self.page_monitoring.tank_climb_records]
            # capture "AMT Turun Tangki" (juga dikirim sebagai kolom di verifikasi; filenya dibersihkan juga)
            + [r.get("filepath") for r in self.page_monitoring.tank_descend_records]
        )
        for path in paths:
            if not path:
                continue
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"[WARN] Gagal hapus file lokal '{path}': {e}")

    # ------------------------------------------------------------------
    # KONTROL WORKER (ANPR & POSTUR)
    # ------------------------------------------------------------------
    def start_anpr_thread(self):
        if self.anpr_thread is not None and self.anpr_thread.isRunning():
            self.anpr_thread.stop()
        self.anpr_thread = ANPRWorker()
        self.anpr_thread.frame_signal.connect(self.update_anpr_frame)
        self.anpr_thread.plate_detected_signal.connect(self.page_verifikasi.auto_fill_plate)
        self.anpr_thread.start()

    def stop_anpr_thread(self):
        """
        Matikan ANPRWorker -- dipanggil begitu plat sudah beres diproses
        (Verifikasi/Monitoring ditekan) supaya cuma 1 model deteksi yang
        jalan dalam satu waktu (ANPR ATAU Postur, tidak dua-duanya).
        """
        if self.anpr_thread is not None and self.anpr_thread.isRunning():
            self.anpr_thread.stop()
        self.page_verifikasi.set_live(False)
        self.page_verifikasi.video_label.setText(
            "Kamera ANPR nonaktif.\n(sedang di halaman Monitoring)"
        )

    def start_posture_thread(self, plate_text: str):
        if self.posture_thread is not None and self.posture_thread.isRunning():
            self.posture_thread.stop()
        self.posture_thread = PostureWorker(current_plate=plate_text)
        self.posture_thread.frame_signal.connect(self.update_posture_frame)
        self.posture_thread.new_tag_signal.connect(self.page_monitoring.add_tag)
        # handrail terdeteksi NAIK (aktif) -> foto "Handrail Aktif" masuk daftar
        self.posture_thread.handrail_capture_signal.connect(self.page_monitoring.add_handrail)
        # petugas terdeteksi NAIK ke atas tangki -> foto "Naik Tangki" + waktunya masuk daftar
        self.posture_thread.tank_climb_signal.connect(self.page_monitoring.add_tank_climb)
        # AMT terlihat TURUN dari tangki -> foto "AMT Turun Tangki" + waktunya masuk daftar
        self.posture_thread.tank_descend_signal.connect(self.page_monitoring.add_tank_descend)
        self.posture_thread.start()

    def update_anpr_frame(self, annotated_frame, clean_frame):
        # simpan frame bersih terus-menerus (dipakai saat tombol Verifikasi
        # ditekan), tapi cuma RENDER ke layar kalau halaman ini sedang aktif
        # -- hemat CPU saat operator lagi di halaman Monitoring.
        self.last_clean_anpr_frame = clean_frame
        # indikator LIVE dinyalakan SETIAP ADA FRAME MASUK (murah, cukup
        # toggle visibility) -- terlepas dari halaman mana yang aktif,
        # supaya mencerminkan "ANPRWorker beneran jalan", bukan "halaman
        # ini sedang dilihat".
        self.page_verifikasi.set_live(True)
        if self.stack.currentWidget() is self.page_verifikasi:
            self.page_verifikasi.update_frame(annotated_frame)

    def update_posture_frame(self, annotated_frame):
        self.page_monitoring.set_live(True)
        if self.stack.currentWidget() is self.page_monitoring:
            self.page_monitoring.update_frame(annotated_frame)

    # ------------------------------------------------------------------
    # RESET TOTAL (tombol 'Reset' di halaman Verifikasi)
    # ------------------------------------------------------------------
    def handle_reset(self):
        if self.anpr_thread and self.anpr_thread.isRunning():
            self.anpr_thread.stop()
        if self.posture_thread and self.posture_thread.isRunning():
            self.posture_thread.stop()
        self.posture_thread = None

        self.page_verifikasi.reset_view()
        self.page_monitoring.clear_tags()
        self.page_monitoring.set_verified_plate("")
        self.page_monitoring.clear_plate_image()
        self.page_monitoring.set_live(False)
        self.page_monitoring.video_label.setText(
            "Memuat kamera deteksi postur...\nMohon tunggu sebentar."
        )
        self.last_clean_anpr_frame = None
        self.last_captured_plate_path = None
        self.last_captured_plate_time = None

        self.start_anpr_thread()
        print("[INFO] Reset total selesai -- ANPRWorker dijalankan ulang dari nol.")

    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.anpr_thread and self.anpr_thread.isRunning():
            self.anpr_thread.stop()
        if self.posture_thread and self.posture_thread.isRunning():
            self.posture_thread.stop()
        event.accept()


def load_stylesheet(app: QApplication, path: str = "assets/style.qss"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("assets/app_icon.ico"))
    load_stylesheet(app)

    window = MainWindow(kiosk=KIOSK_MODE)
    if KIOSK_MODE:
        window.showFullScreen()
    else:
        window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()