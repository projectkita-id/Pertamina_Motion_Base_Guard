"""
Deteksi HANDRAIL (naik = aktif / turun = tidak aktif) untuk pipeline Pose,
+ capture foto "Handrail Aktif" begitu handrail terdeteksi NAIK.

Detektornya sendiri adalah file asli detection/handrail/handrail_detector.py
(TIDAK diubah) + modelnya detection/handrail/handrail_model.npz. Modul ini
cuma "pembungkus" supaya bisa ditanam di loop PostureWorker:

  1. Sampling waktu  : detektor dilatih pada 6 sampel/detik (PROCESS_FPS),
                       jadi frame dipilih berdasarkan WAKTU, bukan tiap frame
                       (median 5 sampel & dwell 1,5 dtk di dalamnya bergantung
                       ke laju ini).
  2. Status stabil   : median kausal + Schmitt trigger + dwell (di detektor).
  3. Capture         : tiap kali status berubah TURUN -> NAIK, frame bersih
                       (tanpa overlay) disimpan sebagai
                       "<plat>_handrail_aktif<n>.jpg".
  4. Overlay         : badge "HANDRAIL: NAIK (AKTIF)" / "TURUN" di video.

Pemakaian (lihat widgets/workers.py -> PostureWorker.run):

    hr = HandrailMonitor(save_dir="captured_tags")
    ...
    event = hr.update(frame, t)          # t = waktu dalam detik (monoton)
    if event == "NAIK":
        info = hr.save(clean_frame, plate)   # {"n","filepath","thumb","prob"}
    ...
    hr.draw(annotated)

CATATAN: ROI & model dikalibrasi untuk kamera "C42 - POS GATE KEEPER"
(lihat komentar ROI di handrail_detector.py). Untuk kamera / sudut lain, ROI
harus disesuaikan dan model dilatih ulang (train_handrail.py).
"""
import os
import sys
from pathlib import Path

import cv2

_HANDRAIL_DIR = Path(__file__).resolve().parent.parent / "handrail"
if str(_HANDRAIL_DIR) not in sys.path:
    sys.path.insert(0, str(_HANDRAIL_DIR))

import handrail_detector as hd  # noqa: E402  (file asli, tidak diubah)

# ============================================================
# KONFIGURASI
# ============================================================
SHOW_ROI = True                 # gambar kotak ROI analisis handrail (cyan) di video; False = sembunyikan
ROI_COLOR = (255, 255, 0)       # BGR -- cyan
# Indikator status di video digambar SEPERTI INDIKATOR FPS: teks putih polos, tanpa kotak warna
# (skala/tebal/font sama dengan v6.FPSMeter.draw).
TEXT_SCALE = 0.7
TEXT_THICK = 2
TEXT_COLOR = (255, 255, 255)
TEXT_ROW_H = 30
TEXT_ANCHOR = "left"            # "left" = di bawah teks FPS (kiri atas); "right" = rata kanan di pojok kanan atas
THUMB_SIZE = (96, 54)           # thumbnail untuk daftar di GUI (16:9)
MIN_FRAME_STD = 3.0             # frame dgn simpangan baku piksel < ini (hitam/putih polos, mis. kamera
                                # putus/gelap) DILEWATI: model menganggap frame polos = "naik" (p~1.0),
                                # jadi tanpa pengaman ini bisa memicu capture "Handrail Aktif" palsu.


class HandrailMonitor:
    def __init__(self, model_path=None, save_dir="captured_tags"):
        self.model_path = str(model_path or hd.MODEL_PATH)
        self.det = hd.HandrailDetector(self.model_path)
        self.interval = 1.0 / hd.PROCESS_FPS
        self.save_dir = save_dir
        self.state = 0            # 0 = TURUN (tidak aktif), 1 = NAIK (aktif)
        self.prob = 0.0           # probabilitas median terakhir
        self._next_t = None
        self._last_t = None
        self._count = 0           # jumlah capture "Handrail Aktif" (untuk nama file)
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def update(self, frame, t):
        """
        Panggil tiap frame dengan waktu t (detik, monoton naik). Frame hanya
        dianalisis ~PROCESS_FPS kali per detik.
        Return: "NAIK" / "TURUN" pada frame di mana status stabil berubah;
                selain itu None.
        """
        if self._last_t is not None and t < self._last_t - 1e-6:
            # waktu mundur (video file diputar ulang dari awal) -> mulai urutan baru
            self.det = hd.HandrailDetector(self.model_path)
            self.state, self.prob, self._next_t = 0, 0.0, None
        self._last_t = t

        if self._next_t is None:
            self._next_t = t
        if t + 1e-6 < self._next_t:
            return None
        self._next_t += self.interval
        if self._next_t < t:                      # tertinggal jauh -> kejar
            self._next_t = t + self.interval

        if float(frame[::16, ::16].std()) < MIN_FRAME_STD:
            return None                           # frame polos/rusak -> bukan pemandangan valid, abaikan

        _, med, state, changed = self.det.update(frame)
        self.prob, self.state = med, state
        if changed:
            return "NAIK" if state else "TURUN"
        return None

    # ------------------------------------------------------------------
    def save(self, clean_frame, plate):
        """
        Simpan frame bersih sebagai <save_dir>/<plat>_handrail_aktif<n>.jpg.
        Return {"n", "filepath", "thumb" (ndarray BGR kecil), "prob"} atau {} kalau gagal.
        """
        self._count += 1
        filepath = os.path.join(self.save_dir, f"{plate}_handrail_aktif{self._count}.jpg")
        if not cv2.imwrite(filepath, clean_frame):
            print(f"[ERROR] Gagal menyimpan capture handrail: {filepath}")
            return {}
        thumb = cv2.resize(clean_frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
        print(f"[SUCCESS] Handrail AKTIF #{self._count}! Capture disimpan: {filepath} (p={self.prob:.2f})")
        return {"n": self._count, "filepath": filepath, "thumb": thumb, "prob": self.prob}

    # ------------------------------------------------------------------
    def draw(self, frame):
        """
        Gambar indikator status handrail (+ kotak ROI kalau SHOW_ROI) ke frame ANOTASI, in-place.
        Indikator = teks putih polos seperti indikator FPS (lihat TEXT_ANCHOR); ini baris pertama (row 0).
        """
        h, w = frame.shape[:2]
        if SHOW_ROI:
            x0, y0, x1, y1 = hd.ROI
            cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                          ROI_COLOR, max(1, w // 960))

        text = "HANDRAIL: NAIK (AKTIF)" if self.state else "HANDRAIL: TURUN (TIDAK AKTIF)"
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICK)
        pos = (w - tw - 10, 25) if TEXT_ANCHOR == "right" else (10, 25 + TEXT_ROW_H)   # kiri: baris 0 dipakai teks FPS
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICK, cv2.LINE_AA)