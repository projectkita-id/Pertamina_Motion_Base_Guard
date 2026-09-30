"""
Deteksi orang + klasifikasi postur (Berdiri / Menunduk / Jongkok)
menggunakan YOLOv8-pose (pretrained, tanpa training tambahan).

Pendekatan:
1. YOLOv8-pose mendeteksi orang + 17 keypoint tubuh (format COCO).
2. Hitung sudut lutut (hip-knee-ankle) dan sudut pinggang (shoulder-hip-knee).
3. Klasifikasikan postur berdasarkan kombinasi kedua sudut tsb.

Index keypoint COCO (urutan output YOLOv8-pose):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import cv2
import numpy as np
from ultralytics import YOLO

KP = {
    "l_shoulder": 5, "r_shoulder": 6,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

CONF_THRESHOLD = 0.35  # keypoint dianggap valid jika confidence di atas ini


def angle_between(a, b, c):
    """Sudut di titik b, dibentuk oleh garis b-a dan b-c (dalam derajat)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def pick_side(kpts, confs, side):
    """Pilih sisi kiri/kanan yang keypoint-nya lebih confident & lengkap."""
    idxs = [KP[f"{side}_shoulder"], KP[f"{side}_hip"], KP[f"{side}_knee"], KP[f"{side}_ankle"]]
    if all(confs[i] > CONF_THRESHOLD for i in idxs):
        return [kpts[i] for i in idxs]
    return None


def classify_posture(kpts, confs):
    """
    Klasifikasi bertingkat (graceful degradation) tergantung keypoint
    mana saja yang berhasil terdeteksi dengan confidence cukup:

    Tier 1 (paling akurat): bahu+pinggul+lutut+pergelangan kaki terlihat
        -> pakai sudut lutut & sudut pinggang
    Tier 2 (kaki tertutup/occluded, umum terjadi di CCTV):
        bahu+pinggul terlihat -> pakai sudut kemiringan badan
        (bahu-pinggul) terhadap garis vertikal
    Tier 3: keypoint tidak cukup -> None ("tidak jelas"), lebih baik
        jujur tidak tahu daripada menebak dari data yang tidak lengkap.

    Return: (label, angle_utama, tier, detail_str)
    """
    side_data = pick_side(kpts, confs, "l") or pick_side(kpts, confs, "r")
    if side_data is not None:
        shoulder, hip, knee, ankle = side_data
        knee_angle = angle_between(hip, knee, ankle)
        hip_angle = angle_between(shoulder, hip, knee)

        if knee_angle < 130:
            label = "Jongkok"
        elif hip_angle < 145:
            label = "Menunduk/Membungkuk"
        else:
            label = "Berdiri"
        return label, knee_angle, 1, f"lutut:{knee_angle:.0f} pinggang:{hip_angle:.0f}"

    # Tier 2: hanya bahu+pinggul yang terlihat (kaki occluded)
    for side in ("l", "r"):
        s_idx, h_idx = KP[f"{side}_shoulder"], KP[f"{side}_hip"]
        if confs[s_idx] > CONF_THRESHOLD and confs[h_idx] > CONF_THRESHOLD:
            shoulder, hip = kpts[s_idx], kpts[h_idx]
            dx = shoulder[0] - hip[0]
            dy = shoulder[1] - hip[1]  # y ke bawah positif di gambar
            # sudut torso terhadap garis vertikal (0 = tegak lurus berdiri)
            torso_tilt = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
            if torso_tilt > 40:
                label = "Menunduk/Membungkuk"
            else:
                label = "Berdiri (kaki tak terlihat)"
            return label, torso_tilt, 2, f"kemiringan torso:{torso_tilt:.0f}"

    return None, None, 3, "keypoint tidak cukup"


COLORS = {
    "Berdiri": (0, 200, 0),
    "Menunduk/Membungkuk": (0, 165, 255),
    "Jongkok": (0, 0, 255),
}

SKELETON = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def process_frame(model, frame, conf=0.25):
    results = model(frame, conf=conf, verbose=False)[0]
    annotated = frame.copy()

    if results.keypoints is None:
        return annotated, []

    detections = []
    kpts_all = results.keypoints.xy.cpu().numpy()
    confs_all = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None
    boxes = results.boxes

    for i in range(len(kpts_all)):
        kpts = kpts_all[i]
        confs = confs_all[i] if confs_all is not None else np.ones(len(kpts))
        label, main_angle, tier, detail = classify_posture(kpts, confs)

        box_conf = float(boxes.conf[i]) if boxes is not None else None
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy() if boxes is not None else (0, 0, 0, 0)

        detections.append({
            "label": label,
            "angle": main_angle,
            "tier": tier,
            "detail": detail,
            "bbox": (x1, y1, x2, y2),
            "conf": box_conf,
        })

        color = COLORS.get(label, (200, 200, 200))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # gambar skeleton
        for p1, p2 in SKELETON:
            if confs[p1] > CONF_THRESHOLD and confs[p2] > CONF_THRESHOLD:
                pt1 = tuple(kpts[p1].astype(int))
                pt2 = tuple(kpts[p2].astype(int))
                cv2.line(annotated, pt1, pt2, color, 2)
        for j, (x, y) in enumerate(kpts):
            if confs[j] > CONF_THRESHOLD:
                cv2.circle(annotated, (int(x), int(y)), 3, color, -1)

        text = f"{label if label else 'Tidak jelas'} ({detail})"
        cv2.putText(annotated, text, (int(x1), max(int(y1) - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated, detections


if __name__ == "__main__":
    import sys

    model_name = sys.argv[2] if len(sys.argv) > 2 else "yolov8m-pose.pt"
    model = YOLO(model_name)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "frames/f03.jpg"
    frame = cv2.imread(image_path)
    annotated, detections = process_frame(model, frame, conf=0.15)

    out_path = "posture_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved: {out_path}")
    for d in detections:
        print(d["label"], d["angle"], f"tier={d['tier']}", d["detail"])