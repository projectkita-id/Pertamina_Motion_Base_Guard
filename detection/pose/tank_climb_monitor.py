"""
Deteksi AMT (petugas) NAIK KE ATAS TANGKI dan TURUN dari tangki + capture foto + catat waktunya.

Cara kerja
----------
Pose model (YOLOv8-pose) yang sudah dipakai PostureWorker memberi kotak (bounding box)
tiap orang. Titik kaki orang = tengah sisi BAWAH kotaknya.

ROI = SATU GARIS LURUS pembagi layar (TANK_LINE, mode "line" -- bawaan): tangki di sisi
atas/kiri garis, tangga + lantai + lapangan di sisi bawah/kanan. Petugas dianggap NAIK kalau
ia terlihat MENYEBERANGI garis dari bawah ke atas lalu tetap di atas >= DWELL_UP_SEC; TURUN
kalau terlihat menyeberang dari atas ke bawah. Keputusan memakai bukti MENYEBERANG, bukan
sekadar "ada orang di sisi atas", karena satu garis tidak bisa memisahkan tangki dari
orang/objek lain yang kebetulan jatuh di sisi atas (petugas tetangga, objek di pojok layar).
(Mode "zone" -- dua kotak zona -- tetap tersedia lewat ROI_MODE = "zone".)

Aturan mode "zone" (cara lama): titik kaki di dalam ZONA ATAS TANGKI (TANK_ZONE) = ada petugas
di atas tangki.

Aturan status (dengan debounce supaya tidak kedap-kedip):
    DI BAWAH  -> DI ATAS   : (mode zone) titik kaki ada di zona atas TANPA PUTUS selama >= DWELL_UP_SEC
                             (mode line) terbukti menyeberang garis ke atas lalu tetap di atas >= DWELL_UP_SEC
                             => event "NAIK"  (frame bersih di-capture + waktu dicatat)
    DI ATAS   -> DI BAWAH  : petugas TERLIHAT KEMBALI di jalur tangga / lantai (LADDER_ZONE)
                             selama >= DWELL_DOWN_SEC sementara zona atas kosong, ATAU tidak
                             terlihat di mana pun selama >= ABSENT_DOWN_SEC   => event "TURUN"

Kenapa TURUN memakai bukti "terlihat di tangga", bukan sekadar "zona atas kosong"? Petugas
sering berjalan ke ujung tangki yang jauh dari kamera, di mana ia kecil & terhalang sehingga
tidak terdeteksi berdetik-detik -- itu BUKAN berarti ia sudah turun. Tanpa aturan ini muncul
pasangan NAIK/TURUN palsu.

Status dihitung dari "adakah orang di zona" (bukan dari ID pelacak), jadi tetap benar
walau ID pelacak berganti saat petugas terhalang tangga/tangki.

Pemakaian (lihat widgets/workers.py -> PostureWorker.run):

    tc = TankClimbMonitor(save_dir="captured_tags")
    ...
    # tiap frame, SETELAH loop semua orang (walau tidak ada orang terdeteksi):
    event = tc.update(persons, t, frame.shape)    # persons = [(track_id, (x1,y1,x2,y2)), ...]
    if event == "NAIK":
        info = tc.save(clean_frame, plate)        # {"n","filepath","thumb"}
    ...
    tc.draw(annotated)

CATATAN KALIBRASI: TANK_LINE (dan TANK_ZONE/LADDER_ZONE untuk mode "zone") dikalibrasi untuk kamera
"C42 - POS GATE KEEPER" (1920x1080) memakai 11 video rekaman. Untuk kamera / sudut lain, garis
HARUS disesuaikan: geser dua titiknya sampai garis lewat di ujung tangga (tangki di sisi atas/kiri).
"""
import os

import cv2
import numpy as np

# ============================================================
# KONFIGURASI
# ============================================================
# MODE ROI: "line" = SATU GARIS LURUS pembagi layar (sisi atas garis = "di atas tangki");
#           "zone" = dua kotak/zona (TANK_ZONE + LADDER_ZONE) -- cara lama, tetap tersedia.
ROI_MODE = "line"

# --- mode "line" --------------------------------------------------------------------
# SATU garis lurus pembagi layar, ditentukan oleh DUA TITIK di garis itu, masing-masing (x/lebar, y/tinggi)
# = pecahan lebar/tinggi frame (0,0 = pojok kiri-atas; 1,1 = pojok kanan-bawah). Garis diperpanjang
# ke seluruh layar. Sisi ATAS/KIRI garis = sisi tangki; sisi bawah/kanan = tangga, lantai, lapangan.
# Garis bawaan: HORIZONTAL di 76,3% tinggi frame (= 824 px pada 1080p), setinggi ujung atas tangga.
# Tangki + latar di atas garis; tangga + lantai di bawahnya. Dikalibrasi dari 11 video kamera
# "C42 - POS GATE KEEPER": lolos 11/11 video, dan tetap lolos untuk tinggi garis 72% - 78%
# (+-40 px dari nilai bawaan). Di bawah 70% atau di atas 80% mulai gagal.
TANK_LINE = ((0.0, 0.763), (1.0, 0.763))
LINE_COLOR = (0, 200, 255)     # BGR -- oranye
# Keputusan mode garis berbasis BUKTI MELINTAS (bukan sekadar "ada orang di sisi atas garis"):
#   NAIK  = ada orang yang terlihat menyeberangi garis dari BAWAH ke ATAS, lalu tetap di atas >= DWELL_UP_SEC
#   TURUN = ada orang yang terlihat menyeberangi garis dari ATAS ke BAWAH
# Kenapa: garis lurus tidak bisa memisahkan tangki dari orang/objek lain yang kebetulan jatuh di sisi
# "atas" garis (petugas tetangga, lantai jauh, objek parsial di pojok layar). Orang-orang itu tidak pernah
# MENYEBERANGI garis lewat tangga, jadi tidak memicu apa pun.
CROSS_MEM_SEC = 2.0            # bukti melintas berlaku bila posisi sebelumnya di sisi seberang terlihat <= ini
CROSS_RADIUS_FRAC = 0.30       # ...dan jaraknya <= pecahan ini x tinggi frame (kira-kira 1 langkah petugas)

# --- mode "zone" (cara lama) ----------------------------------------------------------
# Zona atas tangki, dalam PECAHAN lebar/tinggi frame (x/W, y/H) -> tidak tergantung resolusi.
# (Nilai piksel pada frame 1920x1080 tertulis di komentar.)
TANK_ZONE = [
    (280 / 1920, 540 / 1080),
    (780 / 1920, 540 / 1080),
    (780 / 1920, 800 / 1080),
    (280 / 1920, 800 / 1080),
]
# Jalur tangga + lantai di ujung kanan tangki (tempat petugas lewat saat naik/turun).
LADDER_ZONE = [
    (560 / 1920, 800 / 1080),
    (1150 / 1920, 800 / 1080),
    (1150 / 1920, 1080 / 1080),
    (560 / 1920, 1080 / 1080),
]
MIN_BOX_H_FRAC = 0.05          # kotak lebih pendek dari 5% tinggi frame dianggap noise
DWELL_UP_SEC = 1.0             # harus ada di zona atas TANPA PUTUS selama ini sebelum dianggap "naik"
GAP_TOL_SEC = 0.7              # jeda deteksi sesingkat ini di dalam zona tidak memutus hitungan
DWELL_DOWN_SEC = 1.0           # terlihat di jalur tangga selama ini (zona atas kosong) -> "turun"
ABSENT_DOWN_SEC = 120.0        # jaring pengaman: tidak terlihat di mana pun selama ini -> dianggap sudah "turun".
                               # SENGAJA panjang: di uji coba, petugas di ujung tangki bisa tidak terdeteksi >30 detik
                               # (gelap / jauh / terhalang) padahal masih di atas.
SHOW_ZONE = True               # gambar zona di video (False = sembunyikan)
ZONE_COLOR = (0, 200, 255)     # BGR -- oranye (zona atas tangki)
LADDER_COLOR = (200, 200, 0)   # BGR -- cyan gelap (jalur tangga)
# Indikator status di video digambar SEPERTI INDIKATOR FPS: teks putih polos, tanpa kotak warna
# (skala/tebal/font sama dengan v6.FPSMeter.draw).
TEXT_SCALE = 0.7
TEXT_THICK = 2
TEXT_COLOR = (255, 255, 255)
TEXT_ROW_H = 30                # jarak antar baris indikator (px)
TEXT_ANCHOR = "left"           # "left" = di bawah teks FPS (kiri atas): FPS / HANDRAIL / AMT
INDICATOR_UP = "AMT: DI ATAS TANGKI"
INDICATOR_DOWN = "AMT: TIDAK DI ATAS TANGKI"
                               # "right" = rata kanan di pojok kanan atas (bisa menimpa jam bawaan kamera)
THUMB_SIZE = (96, 54)          # thumbnail untuk daftar di GUI (16:9)


def foot_point(box):
    """Titik kaki = tengah sisi bawah kotak orang."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, y2


class TankClimbMonitor:
    def __init__(self, save_dir="captured_tags", zone=None, ladder_zone=None, dwell_sec=None,
                 gap_tol_sec=None, dwell_down_sec=None, absent_sec=None,
                 mode=None, line=None):
        self.save_dir = save_dir
        self.mode = ROI_MODE if mode is None else mode          # "line" | "zone"
        self.line = tuple(line or TANK_LINE)                     # ((x1, y1), (x2, y2)) pecahan lebar/tinggi frame
        if self.mode == "line" and abs(self.line[1][0] - self.line[0][0]) < 1e-6:
            raise ValueError("TANK_LINE tidak boleh vertikal persis (x kedua titik sama)")
        self.zone = list(zone or TANK_ZONE)
        self.ladder_zone = list(ladder_zone or LADDER_ZONE)
        self.dwell_sec = DWELL_UP_SEC if dwell_sec is None else dwell_sec
        self.gap_tol_sec = GAP_TOL_SEC if gap_tol_sec is None else gap_tol_sec
        self.dwell_down_sec = DWELL_DOWN_SEC if dwell_down_sec is None else dwell_down_sec
        self.absent_sec = ABSENT_DOWN_SEC if absent_sec is None else absent_sec
        self.state = 0                 # 0 = tidak ada petugas di atas tangki, 1 = ada
        self.n_on_top = 0              # jumlah orang di zona atas pada frame terakhir
        self.n_on_ladder = 0           # jumlah orang di jalur tangga pada frame terakhir
        self._since = None             # mulai kapan zona atas terisi TANPA PUTUS
        self._last_occ = None          # terakhir kali zona atas terisi
        self._ladder_since = None      # mulai kapan jalur tangga terisi TANPA PUTUS (zona atas kosong)
        self._ladder_last = None
        self._last_seen = None         # terakhir kali ada orang di zona atas ATAU tangga
        self._last_t = None
        self._count = {"naik": 0, "turun": 0}   # jumlah capture per jenis (untuk nomor pada nama file)
        # alasan event "TURUN" terakhir: "terlihat" = AMT benar-benar TERLIHAT menyeberang garis ke bawah
        # (ada buktinya di video -> foto capture berguna); "hilang" = jaring pengaman (tidak terlihat sama
        # sekali >= ABSENT_DOWN_SEC -> waktunya hanya perkiraan dan fotonya tidak menunjukkan AMT).
        self.last_turun_reason = None
        self._poly_cache = {}          # nama zona -> (shape, polygon piksel)
        self._hist_above = []          # [(t, x, y)] posisi kaki DI ATAS garis pada sampel-sampel terakhir
        self._hist_below = []          # [(t, x, y)] ... DI BAWAH garis
        self._hist_asc = []            # [(t, x, y)] posisi "pendaki" (sudah terbukti menyeberang ke atas)
        self._hist_desc = []           # [(t, x, y)] posisi "penuruh" (sudah terbukti menyeberang ke bawah)
        self.n_above = 0               # (mode garis) jumlah orang di atas garis pada frame terakhir
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _poly_px(self, shape, which="top"):
        h, w = shape[:2]
        cached = self._poly_cache.get(which)
        if cached is None or cached[0] != (h, w):
            zone = self.zone if which == "top" else self.ladder_zone
            cached = ((h, w), np.array([(x * w, y * h) for x, y in zone], np.float32))
            self._poly_cache[which] = cached
        return cached[1]

    def line_y(self, fx, shape):
        """Tinggi (piksel) garis pembagi pada posisi x = fx (garis diperpanjang ke seluruh layar)."""
        h, w = shape[:2]
        (x1, y1), (x2, y2) = self.line
        return (y1 + (y2 - y1) * ((fx / float(w)) - x1) / (x2 - x1)) * h

    def _count_in(self, persons, shape, which):
        """
        Jumlah orang (mode "zone" saja) yang titik kakinya di dalam zona `which` ("top" / "ladder").
        (Mode "line" memakai _line_counts() -- berbasis bukti melintas.)
        """
        h, _w = shape[:2]
        poly = self._poly_px(shape, which)
        n = 0
        for _tid, box in persons:
            if (float(box[3]) - float(box[1])) < MIN_BOX_H_FRAC * h:
                continue
            fx, fy = foot_point(box)
            if cv2.pointPolygonTest(poly, (fx, fy), False) >= 0:
                n += 1
        return n

    def _line_counts(self, persons, t, shape):
        """
        Mode garis. Return (n_pendaki, n_penurun, n_di_atas_garis):
          pendaki  = orang di ATAS garis yang terbukti menyeberang dari bawah (atau melanjutkan jejak pendaki)
          penurun  = orang di BAWAH garis yang terbukti menyeberang dari atas (atau melanjutkan jejak penurun)
        Bukti dicari di posisi 1-2 detik terakhir dalam radius CROSS_RADIUS_FRAC x tinggi frame -- tanpa ID
        pelacak, jadi tahan terhadap pergantian ID saat petugas terhalang.
        """
        h, _w = shape[:2]
        r = CROSS_RADIUS_FRAC * h
        r_cont = 0.5 * r                                   # jejak yang sama bergerak lebih pelan per sampel
        above, below = [], []
        for _tid, box in persons:
            if (float(box[3]) - float(box[1])) < MIN_BOX_H_FRAC * h:
                continue
            fx, fy = foot_point(box)
            (above if fy < self.line_y(fx, shape) else below).append((fx, fy))

        def near(hist, pt, radius, max_age):
            return any((t - ht) <= max_age and (ht < t) and (hx - pt[0]) ** 2 + (hy - pt[1]) ** 2 <= radius ** 2
                       for ht, hx, hy in hist)

        asc = [p for p in above if near(self._hist_below, p, r, CROSS_MEM_SEC)
               or near(self._hist_asc, p, r_cont, 1.0)]
        desc = [p for p in below if near(self._hist_above, p, r, CROSS_MEM_SEC)
                or near(self._hist_desc, p, r_cont, 1.0)]

        keep = lambda hist: [e for e in hist if (t - e[0]) <= CROSS_MEM_SEC]
        self._hist_above = keep(self._hist_above) + [(t, x, y) for x, y in above]
        self._hist_below = keep(self._hist_below) + [(t, x, y) for x, y in below]
        self._hist_asc = keep(self._hist_asc) + [(t, x, y) for x, y in asc]
        self._hist_desc = keep(self._hist_desc) + [(t, x, y) for x, y in desc]
        return len(asc), len(desc), len(above)

    def persons_in_zone(self, persons, shape):
        """Jumlah orang yang titik kakinya di dalam zona ATAS tangki."""
        return self._count_in(persons, shape, "top")

    # ------------------------------------------------------------------
    def update(self, persons, t, shape):
        """
        Panggil SEKALI per frame (walau tidak ada orang), dengan waktu t (detik, monoton naik).
        persons: [(track_id, (x1,y1,x2,y2)), ...]   shape: frame.shape
        Return: "NAIK" / "TURUN" pada frame di mana status berubah; selain itu None.
        """
        if self._last_t is not None and t < self._last_t - 1e-6:
            # waktu mundur (video file diputar ulang) -> mulai urutan baru
            self.state, self.n_on_top, self.n_on_ladder, self.n_above = 0, 0, 0, 0
            self._since = self._last_occ = self._ladder_since = self._ladder_last = self._last_seen = None
            self._hist_above, self._hist_below, self._hist_asc, self._hist_desc = [], [], [], []
        self._last_t = t

        if self.mode == "line":
            self.n_on_top, self.n_on_ladder, self.n_above = self._line_counts(persons, t, shape)
            # "masih ada petugas di atas?" (untuk membatalkan bukti turun) = ada PENDAKI yang terbukti naik,
            # BUKAN sembarang orang di sisi atas garis -- orang statis di sisi atas (petugas tetangga, dll.)
            # tidak boleh menahan TURUN selamanya.
            top_present = self.n_on_top > 0
        else:
            self.n_on_top = self._count_in(persons, shape, "top")
            self.n_on_ladder = self._count_in(persons, shape, "ladder")
            self.n_above = self.n_on_top
            top_present = self.n_on_top > 0
        if self.n_above + self.n_on_ladder > 0:
            self._last_seen = t

        # --- hitungan "zona atas terisi tanpa putus" ---
        if self.n_on_top > 0:
            if self._last_occ is None or (t - self._last_occ) > self.gap_tol_sec:
                self._since = t                    # hitungan baru (sebelumnya terputus / kosong)
            self._last_occ = t
        elif self._since is not None and self._last_occ is not None and (t - self._last_occ) > self.gap_tol_sec:
            self._since = None                     # zona kosong cukup lama -> hitungan putus

        # --- hitungan "terlihat di jalur tangga sementara zona atas kosong" ---
        if self.n_on_ladder > 0 and not top_present:
            if self._ladder_last is None or (t - self._ladder_last) > self.gap_tol_sec:
                self._ladder_since = t
            self._ladder_last = t
        elif top_present:
            self._ladder_since = self._ladder_last = None       # masih ada orang di atas -> batal
        elif self._ladder_since is not None and self._ladder_last is not None \
                and (t - self._ladder_last) > self.gap_tol_sec:
            self._ladder_since = None

        if self.state == 0:
            # SEDANG ada orang di zona atas (n_on_top > 0) DAN sudah >= dwell tanpa putus.
            # (Tanpa syarat "sedang ada", orang yang melintas sesaat bisa memicu NAIK setelah pergi,
            # karena hitungan tetap berjalan selama toleransi jeda.)
            if self.n_on_top > 0 and self._since is not None and (t - self._since) >= self.dwell_sec:
                self.state = 1
                self._ladder_since = self._ladder_last = None
                return "NAIK"
        else:
            seen_on_ladder = (self.n_on_ladder > 0 and not top_present
                              and self._ladder_since is not None
                              and (t - self._ladder_since) >= self.dwell_down_sec)
            long_absent = self._last_seen is not None and (t - self._last_seen) >= self.absent_sec
            if seen_on_ladder or long_absent:
                self.state = 0
                self._since = self._ladder_since = self._ladder_last = None
                self.last_turun_reason = "terlihat" if seen_on_ladder else "hilang"
                return "TURUN"
        return None

    # ------------------------------------------------------------------
    def save(self, clean_frame, plate, kind="naik"):
        """
        Simpan frame bersih sebagai <save_dir>/<plat>_<kind>_tangki<n>.jpg
        (kind "naik" -> ..._naik_tangki<n>.jpg ; kind "turun" -> ..._turun_tangki<n>.jpg).
        Return {"n", "filepath", "thumb" (ndarray BGR kecil)} atau {} kalau gagal.
        """
        if kind not in self._count:
            raise ValueError(f"kind harus 'naik' atau 'turun', bukan {kind!r}")
        self._count[kind] += 1
        n = self._count[kind]
        filepath = os.path.join(self.save_dir, f"{plate}_{kind}_tangki{n}.jpg")
        if not cv2.imwrite(filepath, clean_frame):
            print(f"[ERROR] Gagal menyimpan capture {kind} tangki: {filepath}")
            return {}
        thumb = cv2.resize(clean_frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
        print(f"[SUCCESS] AMT {kind.upper()} TANGKI #{n}! Capture disimpan: {filepath}")
        return {"n": n, "filepath": filepath, "thumb": thumb}

    # ------------------------------------------------------------------
    def draw(self, frame, row=1):
        """
        Gambar garis pembagi (mode "line") atau zona (mode "zone") + indikator status ke frame
        ANOTASI, in-place. Indikator = teks putih polos seperti indikator FPS (lihat TEXT_ANCHOR);
        row = urutan indikator (0 = handrail, 1 = indikator ini).
        """
        h, w = frame.shape[:2]
        if SHOW_ZONE:
            lw = max(1, w // 960)
            if self.mode == "line":
                y_left, y_right = self.line_y(0, frame.shape), self.line_y(w - 1, frame.shape)
                cv2.line(frame, (0, int(y_left)), (w - 1, int(y_right)), LINE_COLOR, lw * 2, cv2.LINE_AA)
                fs = 0.6; th = 2
                if abs(y_left - y_right) < 0.02 * h:          # garis (nyaris) horizontal: label di tepi kiri
                    ly = int(y_left)
                    cv2.putText(frame, "DI ATAS TANGKI", (10, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, LINE_COLOR, th, cv2.LINE_AA)
                    cv2.putText(frame, "DI BAWAH", (10, ly + 24), cv2.FONT_HERSHEY_SIMPLEX, fs, LINE_COLOR, th, cv2.LINE_AA)
                else:                                          # garis miring: label kedua sisi di ketinggian 45% layar
                    (x1, y1), (x2, y2) = self.line
                    xf = x1 + (0.45 - y1) * (x2 - x1) / (y2 - y1) if abs(y2 - y1) > 1e-6 else 0.5
                    xl = int(min(max(xf, 0.05), 0.95) * w); yl = int(0.45 * h)
                    left, right = "DI ATAS TANGKI <-", "-> DI BAWAH"
                    (tw, _), _ = cv2.getTextSize(left, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                    cv2.putText(frame, left, (xl - tw - 14, yl), cv2.FONT_HERSHEY_SIMPLEX, fs, LINE_COLOR, th, cv2.LINE_AA)
                    cv2.putText(frame, right, (xl + 14, yl), cv2.FONT_HERSHEY_SIMPLEX, fs, LINE_COLOR, th, cv2.LINE_AA)
            else:
                cv2.polylines(frame, [self._poly_px(frame.shape, "top").astype(np.int32)], True, ZONE_COLOR, lw)
                cv2.polylines(frame, [self._poly_px(frame.shape, "ladder").astype(np.int32)], True, LADDER_COLOR, lw)

        text = INDICATOR_UP if self.state else INDICATOR_DOWN
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICK)
        if TEXT_ANCHOR == "right":
            pos = (w - tw - 10, 25 + row * TEXT_ROW_H)
        else:                                              # kiri: baris 0 di kiri dipakai teks FPS
            pos = (10, 25 + (row + 1) * TEXT_ROW_H)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICK, cv2.LINE_AA)