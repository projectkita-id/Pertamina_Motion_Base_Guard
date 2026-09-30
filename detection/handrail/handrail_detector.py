"""
DETEKTOR HANDRAIL TRUK TANGKI  ->  NAIK (aktif)  /  TURUN (tidak aktif)
=======================================================================
Sumber video bisa: webcam, kamera CCTV (RTSP/HTTP), atau file rekaman.
Cukup ubah bagian "KONFIGURASI" di bawah (hardcode), lalu jalankan:

    python handrail_detector.py

Tekan  q  atau  ESC  pada jendela video untuk berhenti.

Cara kerja singkat
------------------
1. Ambil ROI (area di atas tangki) dari frame  -> resize 340x180.
2. Filter garis vertikal tipis (tiang handrail) pada tiga kanal warna,
   dipooling ke grid, + jumlah tiang vertikal panjang  -> 272 fitur.
3. Regresi logistik (bobot di handrail_model.npz) -> probabilitas "naik".
4. Median kausal + Schmitt trigger + dwell time  -> status stabil.

Hanya butuh:  pip install opencv-python numpy
"""
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np

# =====================================================================
#                    KONFIGURASI  (ubah di sini)
# =====================================================================
# --- Sumber video: pilih SATU, hapus tanda '#' pada barisnya ---------
SOURCE = "3.mp4"                                          # file rekaman
# SOURCE = 0                                              # webcam (indeks 0, 1, ...)
# SOURCE = "rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101"   # CCTV RTSP

# --- Model & keluaran ------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handrail_model.npz")
LOG_CSV = "handrail_log.csv"       # catatan tiap perubahan status ("" = tidak dicatat)
SAVE_VIDEO = ""                    # contoh: "hasil.mp4" ("" = tidak menyimpan video)
SHOW_WINDOW = True                 # False jika dijalankan di server tanpa layar

# --- ROI (pecahan lebar/tinggi frame: x0, y0, x1, y1) ----------------
# Nilai ini dikalibrasi untuk kamera "C42 - POS GATE KEEPER" (frame 1920x1080:
# x 200..1050, y 330..780). Untuk kamera / sudut lain, ROI HARUS disesuaikan
# dan model dilatih ulang (lihat train_handrail.py).
ROI = (200 / 1920, 330 / 1080, 1050 / 1920, 780 / 1080)

# --- Pengambilan sampel & pemfilteran status -------------------------
PROCESS_FPS = 6.0      # frame yang dianalisis per detik (sama dengan saat pelatihan)
SMOOTH_WIN = 5         # median kausal atas 5 sampel terakhir
ON_THRESHOLD = 0.60    # status -> NAIK   jika median probabilitas >= nilai ini
OFF_THRESHOLD = 0.30   # status -> TURUN  jika median probabilitas <= nilai ini ...
OFF_DWELL_SEC = 1.5    # ... terus-menerus selama sekian detik (meredam terhalang pekerja)
RECONNECT_SEC = 3.0    # jeda sebelum mencoba menyambung ulang kamera live yang putus
# =====================================================================


# ---------------------------------------------------------------------
#  Ekstraksi fitur  (JANGAN diubah tanpa melatih ulang model)
# ---------------------------------------------------------------------
FEAT_W, FEAT_H = 340, 180      # ukuran ROI untuk analisis
GX, GY = 15, 9                 # grid pooling
XMAX = 290                     # kolom > XMAX (tiang atap area gate) diabaikan


def _gauss_kernel(sig, n, deriv2=False):
    x = np.arange(n) - n // 2
    g = np.exp(-x ** 2 / (2 * sig ** 2))
    g /= g.sum()
    if not deriv2:
        return g.astype(np.float32)
    k = (x ** 2 / sig ** 4 - 1 / sig ** 2) * g
    k -= k.mean()
    k /= np.abs(k).sum()
    return k.astype(np.float32)


_KX = _gauss_kernel(1.3, 9, deriv2=True)   # turunan ke-2 Gaussian arah horizontal (garis tipis)
_KY = _gauss_kernel(7.0, 37)               # perataan memanjang arah vertikal


def _vertical_ridge(bgr):
    f = bgr.astype(np.float32)
    outs = [np.abs(cv2.sepFilter2D(f[:, :, c], cv2.CV_32F,
                                   _KX.reshape(1, -1), _KY.reshape(-1, 1)))
            for c in range(3)]
    return np.max(outs, axis=0)


def crop_roi(frame):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = ROI
    roi = frame[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))]
    return cv2.resize(roi, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)


def extract_features(frame):
    """frame BGR (ukuran bebas) -> vektor 272 fitur."""
    im = crop_roi(frame)
    R = _vertical_ridge(im)[:, :XMAX]
    R = R / (np.percentile(R, 90) + 1e-3)                       # normalisasi kontras
    cell = cv2.resize(R, (GX, GY), interpolation=cv2.INTER_AREA).ravel()
    cmax = cv2.resize(cv2.dilate(R, np.ones((5, 5))), (GX, GY),
                      interpolation=cv2.INTER_AREA).ravel()
    thr = (R > 2.5).astype(np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((18, 1), np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(thr)
    ncomp = np.sum((st[1:, cv2.CC_STAT_HEIGHT] >= 18) & (st[1:, cv2.CC_STAT_WIDTH] <= 8))
    return np.concatenate([cell, cmax, [ncomp, thr.mean() * 100]])


# ---------------------------------------------------------------------
#  Model + penstabil status
# ---------------------------------------------------------------------
class HandrailDetector:
    def __init__(self, model_path=MODEL_PATH):
        m = np.load(model_path)
        self.mean, self.scale = m["mean"], m["scale"]
        self.coef, self.bias = m["coef"], float(m["intercept"])
        self.buf = deque(maxlen=SMOOTH_WIN)
        self.off_needed = max(1, int(math.ceil(OFF_DWELL_SEC * PROCESS_FPS)))
        self.state = 0            # 0 = TURUN, 1 = NAIK
        self._below = 0

    def probability(self, frame):
        x = (extract_features(frame) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(x @ self.coef + self.bias))))

    def update(self, frame):
        """Proses satu sampel frame. Return (prob_mentah, prob_median, status, berubah?)."""
        return self.step(self.probability(frame))

    def step(self, p):
        """Masukkan satu probabilitas mentah ke penstabil status."""
        self.buf.append(p)
        med = float(np.median(self.buf))
        changed = False
        if self.state == 0:
            if med >= ON_THRESHOLD:
                self.state, self._below, changed = 1, 0, True
        else:
            self._below = self._below + 1 if med <= OFF_THRESHOLD else 0
            if self._below >= self.off_needed:
                self.state, self._below, changed = 0, 0, True
        return p, med, self.state, changed


# ---------------------------------------------------------------------
#  Pembaca sumber video
# ---------------------------------------------------------------------
def is_live_source(src):
    return not (isinstance(src, str) and os.path.isfile(src))


def iter_samples(cap, live, process_fps=PROCESS_FPS, want_all_frames=False):
    """Generator (waktu_detik, frame_atau_None, sampel?).
    Frame di-decode hanya bila perlu (sampel analisis / tampilan)."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0
    interval = 1.0 / process_fps
    t_start = time.monotonic()
    next_t, i = 0.0, 0
    while True:
        if not cap.grab():
            return
        t = (time.monotonic() - t_start) if live else i / fps
        i += 1
        is_sample = t + 1e-6 >= next_t
        if is_sample:
            next_t += interval
            if next_t < t:                 # tertinggal jauh (live) -> kejar
                next_t = t + interval
        if is_sample or want_all_frames:
            ok, frame = cap.retrieve()
            if not ok:
                return
            yield t, frame, is_sample


def fmt_time(t):
    return "%02d:%02d.%d" % (int(t // 60), int(t % 60), int((t * 10) % 10))


def draw_overlay(frame, state, prob, t):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = ROI
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 255, 0), 2)
    color = (0, 180, 0) if state else (0, 0, 220)
    text = "HANDRAIL: NAIK (AKTIF)" if state else "HANDRAIL: TURUN (TIDAK AKTIF)"
    bh = max(50, h // 14)
    cv2.rectangle(frame, (0, 0), (w, bh), color, -1)
    cv2.putText(frame, text, (15, int(bh * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                bh / 45.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "p=%.2f  t=%s" % (prob, fmt_time(t)), (w - int(w * 0.28), int(bh * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 60.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def open_capture(src):
    cap = cv2.VideoCapture(src)
    if isinstance(src, str) and src.lower().startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------------------
#  Program utama
# ---------------------------------------------------------------------
def main():
    det = HandrailDetector()
    live = is_live_source(SOURCE)
    writer, log_f, log_w = None, None, None
    if LOG_CSV:
        log_f = open(LOG_CSV, "w", newline="", encoding="utf-8")
        log_w = csv.writer(log_f)
        log_w.writerow(["waktu_video_detik", "jam_sistem", "kejadian", "prob_median"])

    print("Sumber :", SOURCE, "(live)" if live else "(file)")
    last_p, last_state, last_t = 0.0, 0, 0.0
    try:
        while True:
            cap = open_capture(SOURCE)
            if not cap.isOpened():
                if not live:
                    raise SystemExit("Tidak bisa membuka: %s" % SOURCE)
                print("Kamera belum bisa dibuka, coba lagi %.0f dtk ..." % RECONNECT_SEC)
                time.sleep(RECONNECT_SEC)
                continue

            for t, frame, is_sample in iter_samples(cap, live, want_all_frames=SHOW_WINDOW or bool(SAVE_VIDEO)):
                if is_sample:
                    p, med, state, changed = det.update(frame)
                    last_p, last_state, last_t = med, state, t
                    if changed:
                        ev = "HANDRAIL NAIK" if state else "HANDRAIL TURUN"
                        print("[%s] %s (p=%.2f)" % (fmt_time(t), ev, med))
                        if log_w:
                            log_w.writerow(["%.2f" % t, time.strftime("%Y-%m-%d %H:%M:%S"), ev, "%.3f" % med])
                            log_f.flush()
                if SHOW_WINDOW or SAVE_VIDEO:
                    view = draw_overlay(frame.copy(), last_state, last_p, t)
                    view = cv2.resize(view, (960, int(view.shape[0] * 960 / view.shape[1])))
                    if SAVE_VIDEO and writer is None:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        writer = cv2.VideoWriter(SAVE_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (view.shape[1], view.shape[0]))
                    if writer is not None:
                        writer.write(view)
                    if SHOW_WINDOW:
                        cv2.imshow("Deteksi Handrail", view)
                        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                            return
            cap.release()
            if not live:
                print("Selesai. Status akhir:", "NAIK" if last_state else "TURUN")
                return
            print("Koneksi terputus, menyambung ulang ...")
            time.sleep(RECONNECT_SEC)
    finally:
        if writer is not None:
            writer.release()
        if log_f:
            log_f.close()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
