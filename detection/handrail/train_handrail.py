"""
PELATIHAN / PELATIHAN ULANG MODEL HANDRAIL
==========================================
Menghasilkan handrail_model.npz dari video yang sudah diberi label waktu.
Fitur diambil dengan fungsi yang SAMA seperti pada handrail_detector.py.

Pakai:
    pip install scikit-learn opencv-python numpy
    python train_handrail.py                # ekstrak fitur semua video, validasi, simpan model
    python train_handrail.py 1,2_1,3_1      # (opsional) ekstrak fitur beberapa video saja

Fitur di-cache di folder feature_cache/, jadi aman dijalankan berulang.

Menambah video baru: tambahkan satu baris di GT (lihat penjelasan di bawah), taruh
<nama>.mp4 di VIDEO_DIR, lalu jalankan lagi.
"""
import os
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from handrail_detector import extract_features, iter_samples, PROCESS_FPS

VIDEO_DIR = os.environ.get("VIDEO_DIR", ".")      # folder berisi <nama>.mp4
CACHE_DIR = "feature_cache"
OUT_MODEL = "handrail_model.npz"

# ---------------------------------------------------------------------
# GROUND TRUTH hasil pemeriksaan visual (detik).
#   R    = handrail mulai naik (titik tengah gerakan naik)
#   L    = handrail mulai turun (None = tetap naik sampai video habis)
#   zone = separuh lebar zona transisi di sekitar R yang diabaikan saat melatih/menguji
# Video yang berisi kejadian sama diberi 'group' sama agar validasi tidak bocor.
# ---------------------------------------------------------------------
GT = {
    "1":    dict(R=9.0,  L=None,  zone=3.5, group="1"),
    "2_1":  dict(R=18.2, L=117.5, zone=2.5, group="2_1"),
    "3_1":  dict(R=8.8,  L=None,  zone=2.5, group="3_1"),
    "4_1":  dict(R=5.5,  L=None,  zone=2.5, group="4_1"),
    "5":    dict(R=2.5,  L=None,  zone=2.5, group="5"),
    "6_1":  dict(R=4.5,  L=149.0, zone=2.5, group="6_1"),
    "7_1":  dict(R=5.5,  L=160.5, zone=2.5, group="7_1"),
    "8":    dict(R=20.5, L=77.0,  zone=2.5, group="8"),
    "9_1":  dict(R=13.0, L=77.0,  zone=2.5, group="9_1"),
    "10_1": dict(R=4.5,  L=124.8, zone=2.5, group="10_1"),
    "14_1": dict(R=5.0,  L=161.0, zone=2.5, group="7_1"),   # kejadian sama dengan 7_1
}


def labels_for(name, t):
    """1 = naik, 0 = turun, -1 = zona transisi (diabaikan)."""
    g = GT[name]
    y = np.zeros(len(t), dtype=int)
    y[(t >= g["R"]) & ((t < g["L"]) if g["L"] else True)] = 1
    ign = np.abs(t - g["R"]) <= g["zone"]
    if g["L"]:
        ign |= np.abs(t - g["L"]) <= 2.5
    y[ign] = -1
    return y


def extract_video(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".npz")
    if os.path.isfile(path):
        d = np.load(path)
        return d["X"], d["t"]
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, name + ".mp4"))
    if not cap.isOpened():
        raise SystemExit("Video tidak ditemukan: %s" % os.path.join(VIDEO_DIR, name + ".mp4"))
    X, T = [], []
    for t, frame, is_sample in iter_samples(cap, live=False):
        if is_sample:
            X.append(extract_features(frame))
            T.append(t)
    X, T = np.array(X), np.array(T)
    np.savez(path, X=X, t=T)
    print("  fitur %-5s : %d sampel" % (name, len(T)), flush=True)
    return X, T


def make_clf():
    return LogisticRegression(C=0.02, max_iter=1000, class_weight="balanced")


def main():
    if len(sys.argv) > 1:                       # mode ekstraksi sebagian
        for n in sys.argv[1].split(","):
            extract_video(n)
        return

    data = {n: extract_video(n) for n in GT}
    Y = {n: labels_for(n, data[n][1]) for n in GT}

    # ---- Validasi leave-one-group-out --------------------------------
    print("\nValidasi leave-one-video-out (model diuji pada video yang tidak dilihat saat latihan)")
    dn_ok, up_ok = [], []
    for grp in sorted({g["group"] for g in GT.values()}):
        tr = [n for n in GT if GT[n]["group"] != grp]
        Xtr = np.vstack([data[n][0][Y[n] >= 0][::2] for n in tr])
        ytr = np.concatenate([Y[n][Y[n] >= 0][::2] for n in tr])
        sc = StandardScaler().fit(Xtr)
        clf = make_clf().fit(sc.transform(Xtr), ytr)
        for n in [n for n in GT if GT[n]["group"] == grp]:
            p = clf.predict_proba(sc.transform(data[n][0]))[:, 1]
            dn_ok += list(p[Y[n] == 0] < 0.5)
            up_ok += list(p[Y[n] == 1] >= 0.5)
    print("  recall TURUN = %.3f (n=%d)   recall NAIK = %.3f (n=%d)"
          % (np.mean(dn_ok), len(dn_ok), np.mean(up_ok), len(up_ok)))

    # ---- Model akhir: semua video -------------------------------------
    Xa = np.vstack([data[n][0][Y[n] >= 0] for n in GT])
    ya = np.concatenate([Y[n][Y[n] >= 0] for n in GT])
    sc = StandardScaler().fit(Xa)
    clf = make_clf().fit(sc.transform(Xa), ya)
    np.savez(OUT_MODEL, mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]))
    print("\nModel disimpan ke %s  (sampel latih: %d)" % (OUT_MODEL, len(ya)))


if __name__ == "__main__":
    main()
