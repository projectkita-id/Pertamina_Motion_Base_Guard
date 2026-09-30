"""
Deteksi orang + klasifikasi postur (Berdiri/Menunduk/Jongkok) real-time.
SUMBER VIDEO DI-HARDCODE DI SINI — tinggal ganti nilai SOURCE di bawah,
tidak perlu argumen command line.

Cara ganti sumber:
    - Video rekaman   -> SOURCE = "14.mp4"
    - Webcam lokal    -> SOURCE = 0
    - CCTV IP (RTSP)  -> SOURCE = "rtsp://user:password@192.168.1.10:554/stream1"
    - CCTV IP (HTTP)  -> SOURCE = "http://192.168.1.10:8080/video"

Lalu jalankan:
    python3 realtime_posture_single.py
"""

import os
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from posture_detect import classify_posture, COLORS, SKELETON, CONF_THRESHOLD

# ============================================================
# KONFIGURASI — TINGGAL GANTI DI SINI
# ============================================================

SOURCE = str(Path(__file__).resolve().parent / "14.mp4")   # taruh 14.mp4 di folder detection/pose/
# SOURCE = "1.mp4"
# SOURCE = "3.mp4"
# SOURCE = 0                                  # <- webcam lokal (index 0)
# SOURCE = "rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101"  # <- CCTV IP

OUTPUT_PATH = None                  # None = tidak usah simpan file, cukup tampilkan live saja
SHOW_WINDOW = True            # True = tampilkan jendela GUI live (perlu display, tidak bisa headless)
PROGRESS_EVERY = 30          # cetak progress tiap N frame (biar kelihatan jalan, tidak "diam")

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "model" / "yolov8m-pose.pt")   # taruh di detection/model/
DEVICE = "auto"                   # "auto" | "cuda:0" | "cpu" | "mps"
CONF_THRESHOLD_DETECT = 0.25
FRAME_SKIP = 1                    # proses 1 dari N frame (>1 = lebih cepat, kurang halus)
ALERT_SECONDS = 5.0               # durasi jongkok/menunduk utk dianggap memasang tag (detik)
CONFIRM_FRAMES = 4                # label baru harus muncul N frame berturut sblm ditampilkan
                                    # (naikkan kalau masih 'lompat-lompat', turunkan kalau kerasa lag)
MAX_FRAMES = 0                    # 0 = jalan terus/sampai video habis, angka > 0 utk batasi (testing)
USE_HALF_PRECISION = True         # FP16 di GPU -> signifikan lebih cepat, akurasi nyaris sama

# --- Pengaturan performa & tampilan ---
INFER_IMGSZ = 640            # resolusi internal untuk inferensi (640 jauh lebih cepat dari 1080p asli)
DISPLAY_MAX_WIDTH = 960      # lebar maksimum jendela tampilan (piksel). None = tampilkan resolusi asli.

# --- Kalibrasi tinggi per-orang (utk kasus kaki tertutup saat jongkok) ---
RATIO_JONGKOK = 0.55         # rasio tinggi bbox / tinggi_maks_org_ini di bawah ini -> dianggap Jongkok
RATIO_MENUNDUK = 0.80        # di bawah ini (tapi >= RATIO_JONGKOK) -> dianggap Menunduk

# ============================================================
# KODE UTAMA
# ============================================================


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def draw_detection(frame, box, track_id, label, detail, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    id_txt = f"ID{track_id} " if track_id is not None else ""
    text = f"{id_txt}{label if label else 'Tidak jelas'}"
    cv2.putText(frame, text, (x1, max(y1 - 25, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, detail, (x1, max(y1 - 5, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_skeleton(frame, kpts, confs, color):
    for p1, p2 in SKELETON:
        if confs[p1] > CONF_THRESHOLD and confs[p2] > CONF_THRESHOLD:
            pt1 = tuple(kpts[p1].astype(int))
            pt2 = tuple(kpts[p2].astype(int))
            cv2.line(frame, pt1, pt2, color, 2)
    for j, (x, y) in enumerate(kpts):
        if confs[j] > CONF_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


class FPSMeter:
    """
    Kalkulator FPS ringan (exponential moving average). Dipakai baik oleh
    main() di file ini (mode CLI berdiri sendiri) maupun oleh
    PostureWorker di widgets/workers.py (mode dijalankan lewat aplikasi
    PyQt) -- supaya logika hitung FPS cuma ada di satu tempat.
    """

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = None
        self._prev_t = time.time()

    def tick(self) -> float:
        """Panggil ini SEKALI tiap frame selesai diproses. Return nilai FPS terbaru."""
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - self._prev_t))
        self._prev_t = now
        self.fps = inst_fps if self.fps is None else (
            self.smoothing * self.fps + (1 - self.smoothing) * inst_fps
        )
        return self.fps

    def draw(self, frame, pos=(10, 25), color=(255, 255, 255)):
        """Cetak angka FPS ke pojok kiri atas frame (in-place)."""
        text = f"FPS: {self.fps:.1f}" if self.fps is not None else "FPS: --"
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


class HeightCalibrator:
    """
    Kalibrasi tinggi badan per-orang secara OTOMATIS & DINAMIS (bukan
    angka tetap) -- penting karena tinggi tiap petugas beda-beda.

    Setiap kali seseorang (track_id tertentu) terdeteksi 'Berdiri' dengan
    yakin (tier 1 -- kaki+lutut+pinggul semua terlihat), catat tinggi
    bounding box-nya sebagai baseline "tinggi penuh" orang itu.

    Begitu nanti kaki orang itu tertutup (tier 2, tidak bisa dihitung
    sudut lutut), tinggi bbox SAAT INI dibandingkan terhadap baseline
    tadi (rasio, bukan piksel absolut) -- kalau jauh lebih pendek, itu
    indikasi orang tsb jongkok/menekuk rendah walau torso terlihat tegak.
    """

    def __init__(self, ratio_jongkok=0.55, ratio_menunduk=0.80):
        self.max_height = {}
        self.ratio_jongkok = ratio_jongkok
        self.ratio_menunduk = ratio_menunduk

    def observe(self, track_id, raw_label, tier, bbox_height):
        if track_id is None:
            return
        if tier == 1 and raw_label == "Berdiri":
            current_max = self.max_height.get(track_id, 0)
            if bbox_height > current_max:
                self.max_height[track_id] = bbox_height

    def refine(self, track_id, raw_label, tier, bbox_height):
        if track_id is None or tier != 2:
            return raw_label, None

        baseline = self.max_height.get(track_id)
        if not baseline:
            return raw_label, None

        ratio = bbox_height / baseline
        if ratio < self.ratio_jongkok:
            return "Jongkok (kaki tak terlihat)", ratio
        elif ratio < self.ratio_menunduk:
            return "Menunduk/Membungkuk (kaki tak terlihat)", ratio
        else:
            return raw_label, ratio


class PostureSmoother:
    """
    Mencegah label 'lompat-lompat' karena noise per-frame. Label yang
    DITAMPILKAN baru berubah kalau label mentah yang baru muncul
    CONFIRM_FRAMES kali berturut-turut.
    """

    def __init__(self, confirm_frames=4):
        self.confirm_frames = confirm_frames
        self.confirmed = {}
        self.candidate = {}

    def update(self, track_id, raw_label):
        if track_id is None:
            return raw_label

        cand_label, cand_count = self.candidate.get(track_id, (raw_label, 0))
        if raw_label == cand_label:
            cand_count += 1
        else:
            cand_label, cand_count = raw_label, 1
        self.candidate[track_id] = (cand_label, cand_count)

        confirmed = self.confirmed.get(track_id)
        if confirmed is None or cand_count >= self.confirm_frames:
            confirmed = cand_label
            self.confirmed[track_id] = confirmed
        return confirmed


class TagInstallationTracker:
    """
    Melacak alur pemasangan tag:
    1. Menunduk/Jongkok >= hold_threshold (5 detik) -> 'Sedang Memasang Tag' (Simpan frame bersih ke buffer)
    2. Berdiri setelah 'Sedang Memasang Tag' -> Cetak info di pojok kiri atas lalu simpan ke file (tag1.jpg, dst)
    """
    def __init__(self, hold_threshold=5.0, status_display_time=3.0, save_dir="captured_tags"):
        self.hold_threshold = hold_threshold
        self.status_display_time = status_display_time
        self.tracker_data = {}
        self.global_tag_counter = 1  # Penomoran sekuensial tag1, tag2, dst.
        self.save_dir = save_dir
        self.tag_history = []        # Penampung data histori untuk antarmuka UI
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def update(self, track_id, current_label, now, clean_frame):
        if track_id is None:
            return current_label, COLORS.get(current_label, (200, 200, 200)), ""

        if track_id not in self.tracker_data:
            self.tracker_data[track_id] = {
                "last_posture": current_label,
                "start_time": now,
                "is_installing": False,
                "done_until": 0.0,
                "captured_frame": None,
                "install_start_time": 0.0
            }

        data = self.tracker_data[track_id]

        # Reset timer jika postur berubah
        if current_label != data["last_posture"]:
            # Transisi: Jika sebelumnya sedang memasang tag lalu sekarang BERDIRI -> Tag Terpasang & Simpan File
            if data["is_installing"] and current_label == "Berdiri":
                data["done_until"] = now + self.status_display_time
                data["is_installing"] = False
                
                # Total durasi pemasangan tag dari awal jongkok/menunduk hingga berdiri
                total_duration = now - data["install_start_time"]
                
                # Simpan gambar BERSIH tanpa overlay teks apa pun -- overlay
                # info (Plat/Tag/Time/Duration) ditambahkan belakangan oleh
                # widgets/workers.py (PostureWorker) saat file di-rename,
                # supaya info cuma dicetak SEKALI (tidak menumpuk 2x).
                if data["captured_frame"] is not None:
                    save_img = data["captured_frame"].copy()

                    filename = f"tag{self.global_tag_counter}.jpg"
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, save_img)
                    print(f"[SUCCESS] Tag #{self.global_tag_counter} terpasang! Gambar disimpan: {filepath} (Durasi: {total_duration:.1f}s)")
                    
                    # Simpan data ke list histori agar bisa dibaca oleh ui.py
                    thumb = cv2.resize(data["captured_frame"], (90, 60))
                    self.tag_history.append({
                        "tag_num": self.global_tag_counter,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # tanggal + jam
                        "duration": f"{total_duration:.1f}s",
                        "thumb": thumb
                    })

                    self.global_tag_counter += 1
                    data["captured_frame"] = None

            data["last_posture"] = current_label
            data["start_time"] = now

        duration = now - data["start_time"]

        # Cek durasi jongkok/menunduk
        if current_label in ("Jongkok", "Menunduk/Membungkuk"):
            if duration >= self.hold_threshold:
                if not data["is_installing"]:
                    data["is_installing"] = True
                    data["install_start_time"] = data["start_time"]  # Catat waktu awal mulai menunduk/jongkok
                    data["captured_frame"] = clean_frame.copy()      # Tangkap frame bersih tanpa skeleton
                    print(f"[INFO] ID{track_id} sedang memasang tag. Menangkap gambar bersih...")

        display_label = current_label
        color = COLORS.get(current_label, (200, 200, 200))
        extra_status = ""

        if data["is_installing"]:
            display_label = f"{current_label} (Sedang Memasang Tag)"
            color = (0, 165, 255)  # Warna Oranye
            extra_status = f" | MEMASANG TAG ({duration:.0f}s)"

        elif now < data["done_until"]:
            display_label = "Berdiri (Tag Terpasang)"
            color = (255, 0, 0)    # Warna Biru
            extra_status = " | TAG TERPASANG!"

        return display_label, color, extra_status


def main():
    device = get_device(DEVICE)
    print(f"[INFO] Sumber : {SOURCE}")
    print(f"[INFO] Device : {device}")

    model = YOLO(MODEL_PATH)
    model.to(device)
    if device.startswith("cuda"):
        print(f"[INFO] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
        if USE_HALF_PRECISION:
            model.model.half()
            print("[INFO] FP16 (half precision) aktif untuk mempercepat inferensi")

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka sumber video: {SOURCE}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Resolusi: {width}x{height} | FPS sumber: {src_fps:.1f} | "
          f"Total frame (estimasi): {total_frames if total_frames > 0 else 'live/tidak diketahui'}")

    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc,
                                  src_fps / max(FRAME_SKIP, 1), (width, height))

    tag_tracker = TagInstallationTracker(hold_threshold=ALERT_SECONDS, status_display_time=3.0, save_dir="captured_tags")
    smoother = PostureSmoother(confirm_frames=CONFIRM_FRAMES)
    height_calib = HeightCalibrator(ratio_jongkok=RATIO_JONGKOK, ratio_menunduk=RATIO_MENUNDUK)

    if SHOW_WINDOW:
        cv2.namedWindow("Posture Detection", cv2.WINDOW_NORMAL)
        if DISPLAY_MAX_WIDTH:
            disp_h = int(height * (DISPLAY_MAX_WIDTH / width))
            cv2.resizeWindow("Posture Detection", DISPLAY_MAX_WIDTH, disp_h)

    frame_idx = 0
    processed = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Sumber video berhenti (habis / koneksi putus).")
            break
        frame_idx += 1

        if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
            continue

        # Simpan frame asli (bersih) sebelum dicoret-coret dengan skeleton/box
        clean_frame = frame.copy()

        t0 = time.time()
        results = model.track(
            frame, persist=True, conf=CONF_THRESHOLD_DETECT, verbose=False,
            device=device, tracker="bytetrack.yaml", imgsz=INFER_IMGSZ,
        )[0]
        infer_ms = (time.time() - t0) * 1000

        annotated = frame.copy()
        now = time.time()

        if results.keypoints is not None and results.boxes is not None:
            kpts_all = results.keypoints.xy.cpu().numpy()
            confs_all = (results.keypoints.conf.cpu().numpy()
                         if results.keypoints.conf is not None
                         else np.ones(kpts_all.shape[:2]))
            track_ids = (results.boxes.id.cpu().numpy().astype(int)
                         if results.boxes.id is not None else [None] * len(kpts_all))
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()

            for i in range(len(kpts_all)):
                kpts, confs = kpts_all[i], confs_all[i]
                raw_label, angle, tier, detail = classify_posture(kpts, confs)
                tid = track_ids[i]
                box = boxes_xyxy[i]
                bbox_height = float(box[3] - box[1])

                # 1) Catat baseline tinggi badan tiap kali orang ybs berdiri yakin
                height_calib.observe(tid, raw_label, tier, bbox_height)
                # 2) Kalau kaki tertutup (tier2), coba pertajam pakai rasio tinggi
                refined_label, ratio = height_calib.refine(tid, raw_label, tier, bbox_height)
                if ratio is not None:
                    detail += f" | rasio_tinggi:{ratio:.2f}"

                label = smoother.update(tid, refined_label)  # label yang sudah di-smooth

                if label is None:
                    base_label = "Tidak jelas"
                else:
                    base_label = label.replace(" (kaki tak terlihat)", "")

                # 3) Update status pemasangan tag & kirim clean_frame untuk ditangkap jika perlu
                display_label, color, extra_status = tag_tracker.update(tid, base_label, now, clean_frame)
                detail += extra_status

                draw_skeleton(annotated, kpts, confs, color)
                draw_detection(annotated, box, tid, display_label, detail, color)

        processed += 1
        elapsed = time.time() - t_start
        live_fps = processed / elapsed if elapsed > 0 else 0

        if processed % PROGRESS_EVERY == 0:
            if total_frames > 0:
                pct = 100 * frame_idx / total_frames
                remaining = (total_frames - frame_idx) / max(live_fps, 0.01)
                print(f"[PROGRESS] frame {frame_idx}/{total_frames} ({pct:.0f}%) "
                      f"| {live_fps:.2f} FPS | perkiraan sisa {remaining/60:.1f} menit", flush=True)
            else:
                print(f"[PROGRESS] frame {frame_idx} diproses | {live_fps:.2f} FPS", flush=True)

        cv2.putText(annotated, f"FPS:{live_fps:.1f} infer:{infer_ms:.0f}ms dev:{device}",
                    (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer:
            writer.write(annotated)
        if SHOW_WINDOW:
            display_frame = annotated
            if DISPLAY_MAX_WIDTH and width > DISPLAY_MAX_WIDTH:
                disp_h = int(height * (DISPLAY_MAX_WIDTH / width))
                display_frame = cv2.resize(annotated, (DISPLAY_MAX_WIDTH, disp_h))
            cv2.imshow("Posture Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if MAX_FRAMES and processed >= MAX_FRAMES:
            break

    cap.release()
    if writer:
        writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print(f"[SELESAI] {processed} frame dalam {total_time:.1f}s "
          f"({processed/total_time:.2f} FPS rata-rata)")
    if OUTPUT_PATH:
        print(f"[SELESAI] Video hasil tersimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()