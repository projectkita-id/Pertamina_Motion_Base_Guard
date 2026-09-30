"""
VERIFIKASI END-TO-END pada video rekaman
========================================
Menjalankan KODE PRODUKSI (handrail_detector.py) langsung pada file .mp4, lalu
membandingkan status yang dihasilkan dengan ground truth di train_handrail.py.

Agar jujur (tidak menguji model pada video yang ikut melatihnya), setiap video
diuji dengan model "held-out": dilatih tanpa video itu (dan tanpa video yang
merekam kejadian sama). Model held-out dibuat dari feature_cache/ hasil train_handrail.py.

Pakai:
    python train_handrail.py                 # sekali, untuk membuat feature_cache/
    python verify_videos.py 1,2_1,3_1        # jalankan sebagian video (tersimpan di results/)
    python verify_videos.py                  # semua video yang belum ada hasilnya, lalu laporan
    python verify_videos.py report           # hanya tampilkan laporan + grafik
"""
import os
import sys

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

import handrail_detector as hd
from train_handrail import GT, VIDEO_DIR, CACHE_DIR, labels_for, make_clf

RES_DIR, MODEL_DIR = "results", "models_lovo"


def heldout_model(name):
    grp = GT[name]["group"]
    tr = [n for n in GT if GT[n]["group"] != grp]
    Xs, ys = [], []
    for n in tr:
        d = np.load(os.path.join(CACHE_DIR, n + ".npz"))
        y = labels_for(n, d["t"])
        Xs.append(d["X"][y >= 0]); ys.append(y[y >= 0])
    X, y = np.vstack(Xs), np.concatenate(ys)
    sc = StandardScaler().fit(X)
    clf = make_clf().fit(sc.transform(X), y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name + ".npz")
    np.savez(path, mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]))
    return path


def run_video(name):
    det = hd.HandrailDetector(heldout_model(name))
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, name + ".mp4"))
    T, P, M, S = [], [], [], []
    for t, frame, is_sample in hd.iter_samples(cap, live=False):
        if is_sample:
            p, med, state, _ = det.update(frame)
            T.append(t); P.append(p); M.append(med); S.append(state)
    os.makedirs(RES_DIR, exist_ok=True)
    np.savez(os.path.join(RES_DIR, name + ".npz"), t=T, p=P, med=M, state=S)
    print("  selesai", name, flush=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(GT)
    fig, axs = plt.subplots(len(names), 1, figsize=(13, 2.0 * len(names)))
    print("\n%-6s %-8s %-30s %-34s %s" % ("video", "akurasi", "GT (detik)", "terdeteksi (detik)", "latensi"))
    all_dn, all_up, extra = [], [], 0
    for ax, n in zip(axs, names):
        r = np.load(os.path.join(RES_DIR, n + ".npz"))
        t, p, s = r["t"], r["p"], r["state"]
        y = labels_for(n, t); m = y >= 0
        all_dn += list(s[(y == 0)] == 0); all_up += list(s[(y == 1)] == 1)
        ch = np.where(np.diff(s) != 0)[0] + 1
        ev = [(t[i], int(s[i])) for i in ch]
        g = GT[n]
        exp = [("NAIK", g["R"])] + ([("TURUN", g["L"])] if g["L"] else [])
        extra += abs(len(ev) - len(exp))
        lat = []
        rs = [e for e in ev if e[1] == 1]; fs = [e for e in ev if e[1] == 0]
        if rs: lat.append(rs[0][0] - g["R"])
        if g["L"] and fs: lat.append(fs[-1][0] - g["L"])
        print("%-6s %-8.3f %-30s %-34s %s" % (
            n, (s[m] == y[m]).mean(),
            ", ".join("%s %.1f" % e for e in exp),
            ", ".join("%s %.1f" % ("NAIK" if e[1] else "TURUN", e[0]) for e in ev),
            ", ".join("%+.1f" % l for l in lat)))
        ax.plot(t, p, lw=0.8, color="tab:blue", label="prob. mentah")
        ax.step(t, s, where="post", color="k", lw=1.5, label="status (1=NAIK)")
        up_end = g["L"] if g["L"] else g_end(n, t)
        ax.axvspan(g["R"], up_end, color="green", alpha=0.12)
        ax.axvspan(0, g["R"], color="red", alpha=0.08)
        if g["L"]:
            ax.axvspan(g["L"], t[-1], color="red", alpha=0.08)
        ax.set_ylabel(n, rotation=0, labelpad=22); ax.set_ylim(-0.05, 1.05)
    axs[0].legend(loc="lower right", fontsize=7)
    plt.tight_layout(); plt.savefig("timeline_verifikasi.png", dpi=70)
    print("\nRingkasan (frame di luar zona transisi):")
    print("  recall TURUN = %.4f (n=%d)   recall NAIK = %.4f (n=%d)" % (
        np.mean(all_dn), len(all_dn), np.mean(all_up), len(all_up)))
    print("  perubahan status berlebih/kurang (tidak sesuai GT) = %d" % extra)
    print("Grafik: timeline_verifikasi.png")


def g_end(n, t):
    return t[-1]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "report":
        report()
    else:
        todo = arg.split(",") if arg else [n for n in GT if not os.path.isfile(os.path.join(RES_DIR, n + ".npz"))]
        for n in todo:
            run_video(n)
        if all(os.path.isfile(os.path.join(RES_DIR, n + ".npz")) for n in GT):
            report()
