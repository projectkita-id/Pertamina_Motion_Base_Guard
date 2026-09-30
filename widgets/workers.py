"""
Worker threads yang menjalankan pipeline ANPR (deteksi + OCR plat) dan
Posture/Tag (deteksi postur + tracking pemasangan tag) di BACKGROUND
(QThread terpisah), supaya jendela UI tidak freeze selama model YOLO/
EasyOCR jalan.

Logika deteksinya sendiri TIDAK diubah dari ui.py (kode uji coba Anda)
-- file ini cuma "membungkusnya" rapi supaya nyambung ke struktur
project ini (widgets/page_verifikasi.py & widgets/page_monitoring.py).

Modul-modul asli (anpr_main.py [=main.py lama], v6.py, posture_detect.py,
video_source.py, plate_utils.py) ada di folder detection/ persis seperti
yang Anda kirim -- TIDAK diubah isinya sama sekali, supaya kalau Anda
tuning parameter di sana (mis. SOURCE, CONF_THRESHOLD, ALERT_SECONDS),
cukup edit file itu langsung, tidak perlu sentuh file ini.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

# Folder detection/ ditambahkan ke sys.path supaya modul-modul di dalamnya
# (anpr_main, v6, posture_detect, video_source, plate_utils) bisa diimport
# langsung dengan nama filenya -- sama seperti waktu semua file itu masih
# rata di satu folder (flat import, tidak pakai "detection.xxx").
# Folder detection/anpr dan detection/pose ditambahkan ke sys.path supaya
# modul-modul di dalamnya (anpr_main, plate_utils, video_source, v6,
# posture_detect) bisa diimport langsung dengan nama filenya -- flat
# import, tidak pakai "detection.anpr.xxx".
#
# Struktur yang didukung:
#   detection/
#     anpr/   -> anpr_main.py, plate_utils.py, video_source.py, 3.mp4
#     pose/   -> v6.py, posture_detect.py, 14.mp4
#     model/  -> model.pt, yolov8n-pose.pt (dibaca lewat path absolut
#                 dari dalam anpr_main.py & v6.py, lihat file itu)
_DETECTION_DIR = Path(__file__).resolve().parent.parent / "detection"
for _sub in ("anpr", "pose"):
    _p = str(_DETECTION_DIR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anpr_main as anpr_module  # noqa: E402  (main.py asli, di-rename biar tidak bentrok sama main.py punya app ini)
import v6  # noqa: E402
from posture_detect import classify_posture  # noqa: E402
from tag_roi_capture import TagROICapture  # noqa: E402  (capture KEDUA: ROI setelah tag terpasang)
from handrail_monitor import HandrailMonitor  # noqa: E402  (deteksi handrail naik/turun + capture "Handrail Aktif")
from tank_climb_monitor import TankClimbMonitor  # noqa: E402  (deteksi petugas naik ke atas tangki + capture + waktu)
from video_source import VideoSource  # noqa: E402
from plate_utils import normalize_plate, PlateVoteTracker  # noqa: E402


class ANPRWorker(QThread):
    """Deteksi + OCR plat nomor, untuk video & auto-isi input di halaman Verifikasi."""

    frame_signal = pyqtSignal(np.ndarray, np.ndarray)  # (annotated_frame, clean_frame)
    plate_detected_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        try:
            vs = VideoSource(anpr_module.SOURCE)
            model = YOLO(anpr_module.MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] ANPRWorker gagal start: {e}")
            return

        ocr_reader = anpr_module.build_ocr_reader(anpr_module.OCR_LANG)
        tracker = anpr_module.SimpleTracker()
        vote_tracker = PlateVoteTracker(history_len=15)
        frame_counter_per_track = {}
        fps_meter = anpr_module.FPSMeter()

        while self._run_flag:
            ret, frame = vs.read()
            if not ret or frame is None:
                if vs.is_file:
                    # video pendek (mis. rekaman uji coba) -> otomatis ulang dari awal
                    vs.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            clean_frame = frame.copy()
            annotated_frame = frame.copy()

            results = model.predict(
                frame, conf=anpr_module.CONF_THRESHOLD, imgsz=anpr_module.IMG_SIZE,
                device=anpr_module.DEVICE, verbose=False
            )[0]

            boxes, confs = [], []
            if results.boxes is not None:
                for b in results.boxes:
                    boxes.append(tuple(b.xyxy[0].cpu().numpy().tolist()))
                    confs.append(float(b.conf[0]))

            track_map = tracker.update(boxes)
            box_to_conf = {tuple(b): c for b, c in zip(boxes, confs)}

            latest_detected_text = ""
            for tid, box in track_map.items():
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]

                frame_counter_per_track[tid] = frame_counter_per_track.get(tid, 0) + 1
                do_ocr = (frame_counter_per_track[tid] % max(1, anpr_module.OCR_EVERY_N_FRAME)) == 0

                display_text = vote_tracker.best(tid)
                if do_ocr and crop.size > 0:
                    raw_text = anpr_module.read_plate_text(ocr_reader, crop)
                    normalized, is_valid = normalize_plate(raw_text)
                    if normalized:
                        display_text = vote_tracker.update(tid, normalized)

                conf = box_to_conf.get(box, 0.0)
                _, is_valid_display = normalize_plate(display_text) if display_text else ("", False)
                anpr_module.draw_overlay(annotated_frame, box, display_text, is_valid_display, conf)

                if display_text:
                    latest_detected_text = display_text

            if latest_detected_text.strip():
                self.plate_detected_signal.emit(latest_detected_text)

            fps_meter.tick()
            fps_meter.draw(annotated_frame)

            self.frame_signal.emit(annotated_frame, clean_frame)
            time.sleep(0.01)

        vs.stop()

    def stop(self):
        self._run_flag = False
        self.wait()


class PostureWorker(QThread):
    """Deteksi postur + tracking pemasangan tag, untuk halaman Monitoring."""

    frame_signal = pyqtSignal(np.ndarray)
    new_tag_signal = pyqtSignal(dict)
    # handrail terdeteksi NAIK (aktif) -> capture "Handrail Aktif" untuk GUI:
    #   {"n", "filepath", "thumb", "prob", "time", "plate"}
    handrail_capture_signal = pyqtSignal(dict)
    # petugas terdeteksi NAIK ke atas tangki -> capture "Naik Tangki" + waktunya untuk GUI:
    #   {"n", "filepath", "thumb", "time", "plate", "video_time" (hanya utk sumber file)}
    tank_climb_signal = pyqtSignal(dict)
    # AMT terlihat TURUN dari tangki (menyeberang garis dari atas ke bawah) -> capture "AMT Turun Tangki"
    # + waktunya; isi dict sama dengan tank_climb_signal.
    tank_descend_signal = pyqtSignal(dict)

    def __init__(self, current_plate="TANPA_PLAT"):
        super().__init__()
        self._run_flag = True
        self.current_plate = current_plate or "TANPA_PLAT"

    def set_plate_number(self, plate_number):
        self.current_plate = plate_number if plate_number else "TANPA_PLAT"

    def run(self):
        # SOURCE bisa berupa: path file video (mis. "14.mp4"), index webcam
        # (0, 1, ...), atau URL kamera live (rtsp://... / http://...).
        # is_live menentukan cara menangani putus koneksi/EOF di bawah --
        # PENTING dibedakan supaya nanti tinggal ganti v6.SOURCE ke kamera
        # asli tanpa perlu ubah kode ini lagi.
        source = v6.SOURCE
        is_live = str(source).isdigit() or str(source).lower().startswith(("rtsp://", "http://", "https://"))

        try:
            device = v6.get_device(v6.DEVICE)
            model = YOLO(v6.MODEL_PATH).to(device)
            if device.startswith("cuda") and torch.cuda.is_available() and v6.USE_HALF_PRECISION:
                model.model.half()
            cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        except Exception as e:
            print(f"[ERROR] PostureWorker gagal start: {e}")
            return

        if not cap.isOpened():
            print(f"[ERROR] PostureWorker: tidak bisa membuka sumber video '{source}'")
            return

        tag_tracker = v6.TagInstallationTracker(hold_threshold=v6.ALERT_SECONDS, status_display_time=3.0)
        smoother = v6.PostureSmoother(confirm_frames=v6.CONFIRM_FRAMES)
        height_calib = v6.HeightCalibrator(ratio_jongkok=v6.RATIO_JONGKOK, ratio_menunduk=v6.RATIO_MENUNDUK)
        last_history_len = 0
        fps_meter = v6.FPSMeter()

        # ---- capture KEDUA (ROI area pemasangan tag) -- ditambahkan PARALEL,
        # logika capture pertama (v6.TagInstallationTracker) tidak diubah. ----
        roi_capture = TagROICapture(save_dir=tag_tracker.save_dir)
        pending_after = {}  # tag_num -> {"after_filepath", "after_thumb"}, menunggu digabung ke data tag

        # ---- deteksi HANDRAIL (naik = aktif / turun = tidak aktif), berjalan paralel
        # dengan deteksi postur. Kalau model/detektornya gagal dimuat, fitur ini
        # dilewati saja -- deteksi postur & tag tetap jalan normal. ----
        try:
            handrail = HandrailMonitor(save_dir=tag_tracker.save_dir)
        except Exception as e:
            print(f"[WARN] Deteksi handrail dimatikan (gagal dimuat): {e}")
            handrail = None

        # ---- deteksi PETUGAS NAIK KE ATAS TANGKI (dari kotak orang hasil pose model yang
        # sama), + capture & waktu. Gagal dimuat -> dilewati, fitur lain tetap jalan. ----
        try:
            tank_climb = TankClimbMonitor(save_dir=tag_tracker.save_dir)
        except Exception as e:
            print(f"[WARN] Deteksi naik tangki dimatikan (gagal dimuat): {e}")
            tank_climb = None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_no = 0   # dipakai sebagai jam untuk sumber FILE (live memakai time.monotonic)

        while self._run_flag:
            ok, frame = cap.read()
            if not ok:
                if is_live:
                    # sumber live (webcam/RTSP/HTTP) putus -- JANGAN di-seek
                    # ke frame 0 (itu cuma valid untuk file), tapi coba
                    # sambungkan ulang dari nol setelah jeda singkat.
                    print(f"[WARN] PostureWorker: koneksi ke '{source}' putus, mencoba reconnect...")
                    cap.release()
                    time.sleep(2.0)
                    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video pendek -> otomatis ulang dari awal
                    frame_no = 0
                continue

            frame_no += 1
            clean_frame = frame.copy()
            now = time.time()

            # jam untuk deteksi berbasis waktu (handrail & naik tangki): sumber live ->
            # time.monotonic; sumber FILE -> waktu video (nomor frame / fps)
            t_clock = time.monotonic() if is_live else frame_no / src_fps

            # ---- handrail: analisis ~6 sampel/detik; begitu status stabil berubah
            # jadi NAIK (aktif) -> capture frame bersih & kirim ke GUI ----
            if handrail is not None:
                if handrail.update(frame, t_clock) == "NAIK":
                    info = handrail.save(clean_frame, self.current_plate)
                    if info:
                        info["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        info["plate"] = self.current_plate
                        self.handrail_capture_signal.emit(info)

            results = model.track(
                frame, persist=True, conf=v6.CONF_THRESHOLD_DETECT, verbose=False,
                device=device, tracker="bytetrack.yaml", imgsz=v6.INFER_IMGSZ
            )[0]

            annotated = frame.copy()
            persons_now = []   # (track_id, kotak) semua orang di frame ini -> deteksi naik tangki

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
                    persons_now.append((tid, box))
                    bbox_height = float(box[3] - box[1])

                    height_calib.observe(tid, raw_label, tier, bbox_height)
                    refined_label, ratio = height_calib.refine(tid, raw_label, tier, bbox_height)
                    label = smoother.update(tid, refined_label)
                    base_label = label.replace(" (kaki tak terlihat)", "") if label else "Tidak jelas"

                    hist_before = len(tag_tracker.tag_history)
                    display_label, color, extra_status = tag_tracker.update(tid, base_label, now, clean_frame)
                    detail += extra_status

                    # ---- capture KEDUA: ROI dibuat saat mulai memasang tag, di-capture
                    # begitu orang ini berdiri lagi (tag selesai terpasang), lalu dihapus ----
                    if tid is not None:
                        is_installing = tag_tracker.tracker_data.get(tid, {}).get("is_installing", False)
                        roi_capture.update(
                            tid, box, is_installing, now, clean_frame,
                            # nomor tag = tag yang BARU SAJA dicatat tag_tracker untuk orang ini
                            tag_num_provider=lambda: (
                                tag_tracker.tag_history[-1]["tag_num"]
                                if len(tag_tracker.tag_history) > hist_before else None
                            ),
                        )

                    v6.draw_skeleton(annotated, kpts, confs, color)
                    v6.draw_detection(annotated, box, tid, display_label, detail, color)

            # ROI yang jeda capture-nya sudah habis -> crop & simpan (tetap dicek tiap
            # frame walau tidak ada orang terdeteksi), lalu buang ROI yang orangnya
            # hilang & gambar sisanya di video
            for tag_num, roi_crop in roi_capture.collect(now, clean_frame):
                pending_after[tag_num] = roi_capture.save(roi_crop, self.current_plate, tag_num)
            roi_capture.prune(now)
            roi_capture.draw(annotated, now)
            if handrail is not None:
                handrail.draw(annotated)

            # ---- petugas naik ke atas tangki: dipanggil TIAP frame (juga saat tidak ada orang
            # terdeteksi, supaya hitungan waktu "zona kosong" tetap jalan) ----
            if tank_climb is not None:
                tc_event = tank_climb.update(persons_now, t_clock, frame.shape)
                # NAIK -> selalu di-capture. TURUN -> di-capture hanya kalau AMT benar-benar TERLIHAT menuruni
                # tangga; TURUN karena jaring pengaman ("tidak terlihat >= 120 dtk") tidak punya foto berarti.
                if tc_event == "NAIK" or (tc_event == "TURUN" and tank_climb.last_turun_reason == "terlihat"):
                    kind = "naik" if tc_event == "NAIK" else "turun"
                    info = tank_climb.save(clean_frame, self.current_plate, kind=kind)
                    if info:
                        info["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        info["plate"] = self.current_plate
                        if not is_live:   # sumber file: sertakan juga waktu di dalam video
                            info["video_time"] = "%02d:%02d" % (int(t_clock // 60), int(t_clock % 60))
                        (self.tank_climb_signal if kind == "naik" else self.tank_descend_signal).emit(info)
                elif tc_event == "TURUN":
                    print("[INFO] AMT dianggap sudah turun (tidak terlihat lama) -- tidak ada foto capture turun.")
                tank_climb.draw(annotated, row=1)

            # ---- ada tag baru yang barusan "selesai terpasang"? ----
            while last_history_len < len(tag_tracker.tag_history):
                new_item = tag_tracker.tag_history[last_history_len]
                if roi_capture.is_waiting(new_item["tag_num"]):
                    break  # tahan dulu -- tag ini baru dikirim ke GUI setelah capture kedua-nya diambil
                orig_file = os.path.join(tag_tracker.save_dir, f"tag{new_item['tag_num']}.jpg")
                new_filename = f"{self.current_plate}_tag{new_item['tag_num']}.jpg"
                new_filepath = os.path.join(tag_tracker.save_dir, new_filename)

                # File CUMA di-rename (bukan ditulis ulang dengan teks
                # overlay Plat/Tag/Time/Duration) -- info itu sengaja
                # TIDAK di-"bake" jadi piksel permanen di gambar, supaya
                # kalau operator mengoreksi nomor plat belakangan di
                # halaman Monitoring, koreksinya tetap nyambung (data teks
                # ada di tag_records / database, bukan terkunci di foto).
                if os.path.exists(orig_file):
                    try:
                        os.rename(orig_file, new_filepath)
                    except OSError:
                        # kalau rename gagal (mis. beda drive di Windows), fallback copy+hapus
                        import shutil
                        shutil.copyfile(orig_file, new_filepath)
                        os.remove(orig_file)

                new_item["plate"] = self.current_plate
                # path file foto tag ini -- dipakai nanti waktu tombol
                # "Verifikasi" di halaman Monitoring diklik (upload ke DB)
                new_item["filepath"] = new_filepath
                # capture KEDUA (foto area ROI setelah tag terpasang), kalau ada:
                #   after_filepath / after_thumb -> ditampilkan di GUI (page_monitoring.py)
                new_item.update(pending_after.pop(new_item["tag_num"], {}))
                self.new_tag_signal.emit(new_item)
                last_history_len += 1

            fps_meter.tick()
            fps_meter.draw(annotated)

            self.frame_signal.emit(annotated)
            time.sleep(0.01)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()