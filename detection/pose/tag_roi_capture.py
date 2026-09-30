"""
Capture KEDUA pemasangan tag: foto area ROI setelah tag selesai terpasang.

Latar belakang
--------------
Logika capture BAWAAN (v6.TagInstallationTracker) mengambil 1 foto saat
petugas mulai memasang tag (menunduk/jongkok >= ALERT_SECONDS) -- itu
"capture pertama" dan TIDAK diubah sama sekali.

Modul ini menambahkan "capture kedua" yang berjalan PARALEL:

    1. Petugas terdeteksi mulai memasang tag
       -> dibuat ROI sementara = bounding box orang + margin kecil
          (ROI dikunci di posisi itu, digambar di video sebagai kotak
          putus-putus kuning).
    2. Petugas terdeteksi berdiri lagi = pemasangan selesai
       -> setelah jeda CAPTURE_DELAY_AFTER_STAND detik (default 0 =
          langsung), area ROI di-crop dari frame BERSIH (tanpa skeleton/
          box), disimpan sebagai  "<plat>_tag<n>_terpasang.jpg",
          lalu ROI dihapus. Selama jeda, ROI tetap tampil (hijau).
    3. Kembali ke langkah 1 setiap kali ada pemasangan tag berikutnya.

Cara modul ini tahu kapan "mulai memasang" & "selesai": cukup MEMBACA
tag_tracker.tracker_data[tid]["is_installing"] milik v6 (read-only) --
jadi definisi "sedang memasang tag" dijamin SAMA PERSIS dengan capture
bawaan, tanpa menduplikasi logika deteksinya.

Pemakaian (lihat widgets/workers.py -> PostureWorker.run):

    roi = TagROICapture(save_dir=tag_tracker.save_dir)
    ...
    # per orang, SETELAH tag_tracker.update(...) dipanggil:
    roi.update(tid, box, is_installing, now, clean_frame, tag_num_provider)
    ...
    # sekali per frame, setelah loop semua orang (walau tidak ada orang terdeteksi):
    for tag_num, crop in roi.collect(now, clean_frame):   # ROI yang jedanya sudah habis
        info = roi.save(crop, plate, tag_num)   # {"after_filepath", "after_thumb"}
    roi.prune(now)
    roi.draw(annotated, now)
"""
import os

import cv2

# ============================================================
# KONFIGURASI
# ============================================================
ROI_MARGIN = 0.10          # ROI = bounding box + 10% lebar/tinggi di tiap sisi ("lebih sedikit" dari bbox)
ROI_LOST_TIMEOUT = 15.0    # detik -- kalau orangnya hilang dari tracking selama ini
                           # sebelum sempat berdiri, ROI dibuang (supaya kotak tidak "nyangkut" di layar)
CAPTURE_DELAY_AFTER_STAND = 0.0  # detik -- jeda setelah terdeteksi berdiri sebelum ROI di-capture.
                                 # 0 = langsung di frame saat berdiri terdeteksi. Naikkan (mis. 2.0)
                                 # kalau petugas perlu waktu menyingkir supaya tag di ROI terlihat jelas.
ROI_COLOR = (0, 255, 255)  # BGR -- kuning: ROI aktif (sedang memasang)
ROI_COLOR_WAIT = (0, 255, 0)  # BGR -- hijau: sudah berdiri, menunggu jeda capture
ROI_LABEL = "ROI Pemasangan Tag"
AFTER_SUFFIX = "terpasang"  # nama file: <plat>_tag<n>_terpasang.jpg
THUMB_MAX_SIDE = 90        # sisi terpanjang thumbnail untuk daftar di GUI (proporsi asli dipertahankan)


def expand_box(box, margin, frame_w, frame_h):
    """bbox (x1,y1,x2,y2) + margin proporsional, dipotong agar tidak keluar frame."""
    x1, y1, x2, y2 = [float(v) for v in box]
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    x1 = int(max(0, x1 - mx))
    y1 = int(max(0, y1 - my))
    x2 = int(min(frame_w, x2 + mx))
    y2 = int(min(frame_h, y2 + my))
    return x1, y1, x2, y2


def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash=12, gap=8):
    x1, y1 = pt1
    x2, y2 = pt2

    def _hline(y, xa, xb):
        x = xa
        while x < xb:
            cv2.line(img, (x, y), (min(x + dash, xb), y), color, thickness)
            x += dash + gap

    def _vline(x, ya, yb):
        y = ya
        while y < yb:
            cv2.line(img, (x, y), (x, min(y + dash, yb)), color, thickness)
            y += dash + gap

    _hline(y1, x1, x2)
    _hline(y2, x1, x2)
    _vline(x1, y1, y2)
    _vline(x2, y1, y2)


class TagROICapture:
    """Mengelola ROI sementara per-orang (per track_id) + capture area ROI."""

    def __init__(self, save_dir="captured_tags", margin=ROI_MARGIN, lost_timeout=ROI_LOST_TIMEOUT,
                 capture_delay=None):
        self.save_dir = save_dir
        self.margin = margin
        self.lost_timeout = lost_timeout
        self.capture_delay = CAPTURE_DELAY_AFTER_STAND if capture_delay is None else capture_delay
        # tid -> {"roi": (x1,y1,x2,y2), "start": t, "last_seen": t}   (orang sedang memasang tag)
        self._active = {}
        # [{"tid", "roi", "due", "tag_num"}]   (sudah berdiri, menunggu jeda capture selesai)
        self._waiting = []
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def active_rois(self):
        """Salinan {track_id: (x1,y1,x2,y2)} ROI yang sedang aktif (buat debug/tes)."""
        return {tid: st["roi"] for tid, st in self._active.items()}

    def update(self, tid, box, is_installing, now, clean_frame, tag_num_provider=None):
        """
        Panggil SEKALI per orang per frame, SETELAH tag_tracker.update().

        is_installing    : nilai tag_tracker.tracker_data[tid]["is_installing"]
        tag_num_provider : fungsi tanpa argumen, dipanggil TEPAT saat orang ini selesai
                           memasang tag -> nomor tag yang baru dicatat (atau None kalau
                           tidak ada, maka capture dibatalkan).
        Return           : True pada frame di mana orang ini selesai memasang tag
                           (ROI pindah ke antrean tunggu; hasil crop-nya diambil lewat
                           collect() setelah jeda capture_delay habis); selain itu False.
        """
        if tid is None:
            return False

        state = self._active.get(tid)

        if is_installing:
            if state is None:
                h, w = clean_frame.shape[:2]
                roi = expand_box(box, self.margin, w, h)
                self._active[tid] = {"roi": roi, "start": now, "last_seen": now}
                print(f"[INFO] ID{tid} ROI pemasangan tag dibuat: {roi}")
            else:
                state["last_seen"] = now
            return False

        if state is not None:
            # sebelumnya sedang memasang, sekarang tidak lagi = SELESAI (berdiri)
            del self._active[tid]
            tag_num = tag_num_provider() if tag_num_provider is not None else None
            if tag_num_provider is not None and tag_num is None:
                return False
            self._waiting.append({
                "tid": tid, "roi": state["roi"],
                "due": now + self.capture_delay, "tag_num": tag_num,
            })
            print(f"[INFO] ID{tid} selesai memasang tag -> capture ROI dalam {self.capture_delay:.1f}s.")
            return True

        return False

    def collect(self, now, clean_frame):
        """
        Panggil SEKALI per frame (walau tidak ada orang terdeteksi). Kembalikan
        list (tag_num, crop) untuk ROI yang jeda capture-nya sudah habis -- crop
        diambil dari clean_frame SAAT INI, lalu ROI-nya dihapus.
        """
        done, remaining = [], []
        for w in self._waiting:
            if now < w["due"]:
                remaining.append(w)
                continue
            x1, y1, x2, y2 = w["roi"]
            crop = clean_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            print(f"[INFO] ID{w['tid']} capture ROI {crop.shape[1]}x{crop.shape[0]}, ROI dihapus.")
            done.append((w["tag_num"], crop.copy()))
        self._waiting = remaining
        return done

    def is_waiting(self, tag_num):
        """True kalau capture kedua untuk tag_num ini belum diambil (masih menunggu jeda)."""
        return any(w["tag_num"] == tag_num for w in self._waiting)

    def prune(self, now):
        """Buang ROI milik orang yang sudah hilang dari tracking > lost_timeout detik."""
        stale = [tid for tid, st in self._active.items() if now - st["last_seen"] > self.lost_timeout]
        for tid in stale:
            del self._active[tid]
            print(f"[WARN] ID{tid} hilang dari tracking sebelum selesai memasang tag -- ROI dibuang.")

    def draw(self, frame, now=None):
        """Gambar semua ROI ke frame (in-place). Panggil pada frame ANOTASI, bukan frame bersih."""
        for st in self._active.values():
            self._draw_one(frame, st["roi"], ROI_COLOR, ROI_LABEL)
        for w in self._waiting:
            label = ROI_LABEL if now is None else f"Capture ROI {max(0.0, w['due'] - now):.1f}s"
            self._draw_one(frame, w["roi"], ROI_COLOR_WAIT, label)

    @staticmethod
    def _draw_one(frame, roi, color, label):
        x1, y1, x2, y2 = roi
        _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
        ty = y1 - 8 if y1 > 24 else y1 + 20
        cv2.putText(frame, label, (x1 + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def clear(self):
        self._active.clear()
        self._waiting.clear()

    # ------------------------------------------------------------------
    def save(self, crop, plate, tag_num):
        """
        Simpan crop sebagai  <save_dir>/<plat>_tag<n>_terpasang.jpg
        (penamaan mengikuti file capture pertama: <plat>_tag<n>.jpg).

        Return dict untuk digabung ke data tag yang dikirim ke GUI:
            {"after_filepath": path, "after_thumb": ndarray BGR kecil}
        """
        filename = f"{plate}_tag{tag_num}_{AFTER_SUFFIX}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        if not cv2.imwrite(filepath, crop):
            print(f"[ERROR] Gagal menyimpan capture ROI: {filepath}")
            return {}

        h, w = crop.shape[:2]
        scale = THUMB_MAX_SIDE / float(max(h, w))
        thumb = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        print(f"[SUCCESS] Tag #{tag_num} terpasang! Capture ROI disimpan: {filepath}")
        return {"after_filepath": filepath, "after_thumb": thumb}