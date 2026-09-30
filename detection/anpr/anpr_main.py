"""
ANPR Indonesia — Realtime License Plate Detection + OCR
=========================================================

Pipeline:
    1. Deteksi lokasi plat nomor pakai model YOLOv8 kamu (model.pt, class "plate").
    2. Crop area plat, preprocessing gambar.
    3. Baca teks pakai EasyOCR.
    4. Normalisasi teks ke format plat Indonesia (mis. "B 1234 ABC").
    5. Voting antar-frame supaya teks yang ditampilkan stabil (tidak kedip).
    6. Overlay bounding box + teks ke frame, tampilkan realtime + opsional simpan hasil.

Sumber video yang didukung (argumen --source):
    0                                   -> webcam laptop/PC (default index 0)
    1, 2, ...                           -> webcam lain / capture card
    /path/ke/video.mp4                  -> file rekaman
    rtsp://user:pass@192.168.1.5:554/.. -> kamera IP / CCTV
    http://192.168.1.5:8080/video       -> HP Android dengan app "IP Webcam"
                                            (buka app -> Start server -> pakai URL /video)

Contoh pemakaian:
    python main.py --source 0
    python main.py --source rekaman.mp4 --save output.mp4
    python main.py --source http://192.168.43.1:8080/video
    python main.py --source rtsp://admin:admin@192.168.1.10:554/stream1
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from video_source import VideoSource
from plate_utils import normalize_plate, PlateVoteTracker

MODEL_PATH_DEFAULT = str(Path(__file__).resolve().parent.parent / "model" / "model.pt")


# ============================================================================
# KONFIGURASI HARDCODE — edit bagian ini sesuai kamera kamu, lalu langsung
# jalankan cukup dengan: python main.py   (tidak perlu argumen apapun)
# ============================================================================

# Pilih SALAH SATU baris SOURCE di bawah, yang lain kasih komentar (#):

# SOURCE = 0
SOURCE = str(Path(__file__).resolve().parent / "3.mp4")   # file video (taruh 3.mp4 di folder detection/anpr/)
# SOURCE = "http://192.168.1.33:8080/video"          # HP Android app "IP Webcam"
# SOURCE = "rtsp://192.168.1.10:554/stream1"         # kamera IP / CCTV
# SOURCE = "rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101"

MODEL_PATH = MODEL_PATH_DEFAULT   # path ke model.pt
CONF_THRESHOLD = 0.4              # confidence threshold deteksi plat
IMG_SIZE = 640                    # ukuran input YOLO
DEVICE = None                     # None=auto, atau "cpu" / "cuda:0"
OCR_LANG = "en"                   # bahasa EasyOCR (cukup untuk A-Z0-9)
OCR_EVERY_N_FRAME = 2             # jalankan OCR tiap N frame per objek plat
SAVE_VIDEO_PATH = None            # mis. "hasil.mp4", atau None untuk tidak simpan
SHOW_WINDOW = True                # False = jalan tanpa window (headless)
LOG_CSV_PATH = None               # mis. "deteksi.csv", atau None untuk tidak log
# ============================================================================


def parse_args():
    """Argumen CLI bersifat opsional — kalau tidak diisi, nilai di atas
    (KONFIGURASI HARDCODE) yang dipakai. Jadi cukup jalankan `python main.py`
    tanpa argumen apapun."""
    p = argparse.ArgumentParser(description="ANPR Indonesia Realtime")
    p.add_argument("--source", default=str(SOURCE),
                    help="0=webcam, path video, atau URL rtsp/http kamera IP HP")
    p.add_argument("--model", default=MODEL_PATH, help="Path ke model.pt YOLO plat")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold deteksi plat")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Ukuran input YOLO")
    p.add_argument("--device", default=DEVICE,
                    help="'cpu', 'cuda:0', dst. Default: auto (GPU jika tersedia)")
    p.add_argument("--ocr-lang", default=OCR_LANG,
                    help="Bahasa EasyOCR untuk karakter plat (default 'en' sudah cukup untuk A-Z0-9)")
    p.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAME,
                    help="Jalankan OCR tiap N frame per objek (biar tetap realtime, default 2)")
    p.add_argument("--save", default=SAVE_VIDEO_PATH, help="Path output video hasil overlay (opsional)")
    p.add_argument("--no-display", action="store_true", default=not SHOW_WINDOW,
                    help="Jangan buka window (mis. untuk jalan di server tanpa GUI)")
    p.add_argument("--log-csv", default=LOG_CSV_PATH,
                    help="Simpan log deteksi (waktu, plat, confidence) ke file CSV")
    return p.parse_args()


def build_ocr_reader(lang: str):
    import easyocr
    return easyocr.Reader([lang], gpu=True, verbose=False)


def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Preprocessing ringan supaya OCR lebih akurat: grayscale, upscale, CLAHE, threshold."""
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 3 if max(h, w) < 200 else 2
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    return gray


def read_plate_text(reader, crop: np.ndarray) -> str:
    processed = preprocess_plate_crop(crop)
    if processed.size == 0:
        return ""
    results = reader.readtext(
        processed,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        paragraph=False,
    )
    if not results:
        return ""
    # Gabungkan semua potongan teks terbaca di dalam crop, urut kiri->kanan
    results.sort(key=lambda r: r[0][0][0])
    text = "".join(r[1] for r in results)
    return text


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FPSMeter:
    """
    Kalkulator FPS ringan (exponential moving average, biar angkanya
    tidak lompat-lompat tiap frame). Dipakai baik oleh main() di file
    ini (mode CLI berdiri sendiri) maupun oleh ANPRWorker di
    widgets/workers.py (mode dijalankan lewat aplikasi PyQt) -- supaya
    logika hitung FPS cuma ada di satu tempat.
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


class SimpleTracker:
    """Tracker super ringan berbasis IoU antar-frame (bukan Kalman/DeepSORT)
    -- cukup untuk menjaga identitas plat yang sama supaya voting teks stabil."""

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.tracks = {}  # id -> {"box": (...), "missed": int}
        self.next_id = 0

    def update(self, boxes):
        assigned = {}
        used_track_ids = set()
        for box in boxes:
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in used_track_ids:
                    continue
                score = iou(box, t["box"])
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= self.iou_thresh:
                tid = best_id
            else:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"box": box, "missed": 0}
            used_track_ids.add(tid)
            assigned[tid] = box

        # tandai track yang tidak terdeteksi frame ini
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        return assigned  # {track_id: box}


def draw_overlay(frame, box, text, is_valid, conf):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (0, 200, 0) if is_valid else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{text}" if text else "..."
    label = f"{label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()

    print(f"[INFO] Membuka sumber video: {args.source}")
    src = args.source
    try:
        vs = VideoSource(src)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Tunggu sampai benar-benar dapat 1 frame (bukan cuma delay tetap), supaya
    # decoder FFmpeg OpenCV sudah ke-load & aktif SEBELUM torch/ultralytics
    # diimport (menghindari konflik DLL PyTorch vs OpenCV di Windows).
    print("[INFO] Menunggu frame pertama...")
    warmup_deadline = time.time() + 8.0
    got_frame = False
    while time.time() < warmup_deadline:
        ret, frame = vs.read()
        if ret and frame is not None:
            got_frame = True
            print("[INFO] Frame pertama berhasil dibaca, decoder video OK.")
            break
        time.sleep(0.05)
    if not got_frame:
        if vs.is_file:
            print(
                "[ERROR] Tidak berhasil membaca frame apapun dari file video dalam "
                "8 detik. Cek apakah file corrupt / codec tidak didukung."
            )
            vs.stop()
            return
        print(
            "[WARN] Belum dapat frame dari sumber live dalam 8 detik, lanjut "
            "menunggu di background (mis. kamera IP sedang connect)..."
        )

    # PENTING: import ultralytics/torch SESUDAH video capture terbukti bisa
    # baca frame. Di Windows, PyTorch bisa memuat DLL (mis. zlib/libpng) yang
    # bentrok dengan DLL FFmpeg bawaan OpenCV; kalau decoder FFmpeg OpenCV
    # sudah ke-load lebih dulu, ia tidak akan digantikan oleh DLL dari torch.
    from ultralytics import YOLO
    model = YOLO(args.model)

    print("[INFO] Menyiapkan OCR reader (EasyOCR)...")
    ocr_reader = build_ocr_reader(args.ocr_lang)

    tracker = SimpleTracker()
    vote_tracker = PlateVoteTracker(history_len=15)
    frame_counter_per_track = {}

    writer = None
    log_file = None
    if args.log_csv:
        log_file = open(args.log_csv, "w", encoding="utf-8")
        log_file.write("timestamp,plate_text,valid_format,confidence\n")

    fps_smooth = None
    prev_t = time.time()
    display_available = True

    print("[INFO] Mulai deteksi realtime. Tekan 'q' untuk keluar.")
    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                if vs.is_file:
                    if vs.thread.is_alive():
                        # Thread masih coba baca (warm-up/retry) -> jangan buru-buru
                        # dianggap video selesai.
                        time.sleep(0.05)
                        continue
                    print("[INFO] Video selesai.")
                    break
                # sumber live sedang reconnect, tunggu sebentar
                time.sleep(0.05)
                continue

            results = model.predict(
                frame, conf=args.conf, imgsz=args.imgsz,
                device=args.device, verbose=False
            )[0]

            boxes = []
            confs = []
            if results.boxes is not None:
                for b in results.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    boxes.append(tuple(xyxy))
                    confs.append(float(b.conf[0]))

            track_map = tracker.update(boxes)
            box_to_conf = {tuple(b): c for b, c in zip(boxes, confs)}

            for tid, box in track_map.items():
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]

                frame_counter_per_track[tid] = frame_counter_per_track.get(tid, 0) + 1
                do_ocr = (frame_counter_per_track[tid] % max(1, args.ocr_every)) == 0

                display_text = vote_tracker.best(tid)
                if do_ocr and crop.size > 0:
                    raw_text = read_plate_text(ocr_reader, crop)
                    normalized, is_valid = normalize_plate(raw_text)
                    if normalized:
                        display_text = vote_tracker.update(tid, normalized)
                else:
                    is_valid = False

                conf = box_to_conf.get(box, 0.0)
                _, is_valid_display = normalize_plate(display_text) if display_text else ("", False)
                draw_overlay(frame, box, display_text, is_valid_display, conf)

                if log_file and display_text:
                    log_file.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')},{display_text},"
                        f"{is_valid_display},{conf:.3f}\n"
                    )
                    log_file.flush()

            vote_tracker.cleanup(set(track_map.keys()))
            for tid in list(frame_counter_per_track.keys()):
                if tid not in track_map:
                    del frame_counter_per_track[tid]

            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            fps_smooth = inst_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * inst_fps
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, vs.get_fps(), (w, h))
                writer.write(frame)

            if not args.no_display and display_available:
                try:
                    cv2.imshow("ANPR Indonesia - Realtime", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    display_available = False
                    print(
                        "[WARN] OpenCV tidak punya backend GUI (headless build). "
                        "Lanjut jalan TANPA window. Untuk fix: "
                        "pip uninstall opencv-python-headless -y && pip install opencv-python\n"
                        f"[WARN] Detail error: {e}"
                    )
                    if not args.save and not args.log_csv:
                        print("[INFO] Tidak ada --save/--log-csv, tidak ada output untuk dilihat. Menghentikan.")
                        break

    finally:
        vs.stop()
        if writer is not None:
            writer.release()
        if log_file is not None:
            log_file.close()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("[INFO] Selesai.")


if __name__ == "__main__":
    main()