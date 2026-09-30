"""
video_source.py
Pembungkus cv2.VideoCapture yang jalan di thread terpisah supaya pembacaan
frame dari kamera / IP stream tidak memblokir loop inferensi.

Dua mode:
- Sumber LIVE (webcam/RTSP/HTTP): thread selalu overwrite ke frame TERBARU
  (drop frame lama) supaya tidak ada lag menumpuk -- realtime.
- Sumber FILE: thread dibatasi pakai queue kecil (blocking) supaya
  pembacaan file MENGIKUTI kecepatan konsumsi (main loop), bukan asal
  secepat mungkin. Tanpa ini, file pendek bisa habis kebaca duluan di
  background SEBELUM model YOLO/EasyOCR selesai loading, sehingga saat
  main loop mulai jalan, video sudah dianggap "selesai" padahal belum
  pernah diproses sama sekali.
"""

import os
import queue
import threading
import time
import cv2


class VideoSource:
    def __init__(self, source, reconnect_delay: float = 2.0):
        """
        source bisa berupa:
          - int (0, 1, ...)         -> webcam lokal
          - path file video (.mp4, dll) -> rekaman
          - URL RTSP  'rtsp://user:pass@ip:554/stream1'
          - URL HTTP MJPEG 'http://ip:port/video' (mis. app "IP Webcam" Android)
        """
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.is_file = isinstance(source, str) and not source.lower().startswith(
            ("rtsp://", "http://", "https://")
        ) and not source.isdigit()

        if self.is_file and not os.path.isfile(source):
            raise FileNotFoundError(
                f"File video tidak ditemukan: '{source}'\n"
                f"  -> Working directory saat ini: '{os.getcwd()}'\n"
                f"  -> Path lengkap yang dicoba   : '{os.path.abspath(source)}'\n"
                f"  Pastikan file ada di lokasi tsb, atau pakai path absolut, "
                f"mis. SOURCE = r'E:\\Projectkita\\Pertamina\\anpr_indonesia\\2.mp4'"
            )

        self.cap = None
        self._open()

        if not self.cap.isOpened():
            extra = ""
            if self.is_file:
                extra = (
                    " File-nya ada, tapi OpenCV gagal membuka/decode. Kemungkinan "
                    "codec video tidak didukung build OpenCV kamu, atau file corrupt."
                )
            print(f"[WARN] Gagal membuka video source '{source}' saat start.{extra}")

        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self.eof = False  # True kalau file sudah benar-benar habis

        # Queue kecil khusus mode file, untuk backpressure (lihat docstring atas).
        self._file_queue = queue.Queue(maxsize=2) if self.is_file else None

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        # Paksa backend FFMPEG untuk file/URL (lebih konsisten lintas OS
        # dibanding MSMF default di Windows yang kadang gagal di frame pertama).
        # Kalau FFMPEG backend tidak tersedia di build OpenCV kamu, fallback ke default.
        if isinstance(src, str):
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                print("[DEBUG] Backend FFMPEG gagal buka, coba backend default OpenCV...")
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.is_file:
            # Kurangi buffer khusus sumber live, supaya tidak delay.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            backend = self.cap.getBackendName()
            print(f"[DEBUG] Video capture terbuka. Backend={backend}, "
                  f"resolusi={w}x{h}, total_frame={nframes}")
        else:
            print(f"[DEBUG] Video capture GAGAL dibuka untuk source: {src}")

    def _update(self):
        if self.is_file:
            self._update_file()
        else:
            self._update_live()

    def _update_file(self):
        """Mode file: baca lalu taruh ke queue kecil (blocking kalau penuh),
        supaya kecepatan baca mengikuti kecepatan konsumsi main loop."""
        first_frame_retry = 0
        max_first_frame_retries = 100  # ~5 detik, toleransi warm-up decoder
        got_first_frame = False

        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                print(f"[ERROR] Video capture untuk file '{self.source}' tidak terbuka.")
                self.eof = True
                break

            ret, frame = self.cap.read()

            if not ret:
                if not got_first_frame and first_frame_retry < max_first_frame_retries:
                    first_frame_retry += 1
                    time.sleep(0.05)
                    continue
                if not got_first_frame:
                    print(
                        f"[ERROR] Tidak berhasil membaca satupun frame dari "
                        f"'{self.source}' setelah {max_first_frame_retries}x percobaan."
                    )
                self.eof = True
                break

            got_first_frame = True
            # put() blocking -> thread ini otomatis "direm" kalau consumer
            # (main loop) belum sempat ambil frame sebelumnya.
            while not self.stopped:
                try:
                    self._file_queue.put(frame, timeout=0.5)
                    break
                except queue.Full:
                    continue

        self.eof = True

    def _update_live(self):
        """Mode live (webcam/RTSP/HTTP): selalu simpan frame TERBARU,
        drop yang lama, reconnect otomatis kalau putus."""
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._open()
                time.sleep(self.reconnect_delay)
                continue

            ret, frame = self.cap.read()

            if not ret:
                # Sumber live putus -> coba reconnect
                self.cap.release()
                time.sleep(self.reconnect_delay)
                self._open()
                continue

            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self):
        if self.is_file:
            try:
                frame = self._file_queue.get(timeout=0.1)
                return True, frame
            except queue.Empty:
                if self.eof:
                    return False, None
                return False, None  # belum ada frame baru, tapi belum tentu EOF
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def get_fps(self):
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 0 else 25.0
        return 25.0

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()