# Monitoring Plat & Postur — Pertamina Patra Niaga

Aplikasi desktop (PyQt5) untuk memantau kendaraan yang masuk lewat
**deteksi plat nomor otomatis (ANPR)** dan **deteksi postur + pemasangan
tag keselamatan** secara realtime, memakai YOLOv8 + OpenCV.

<p align="center">
  <img src="assets/app_icon.png" width="96" alt="App icon">
</p>

---

## ✨ Fitur

- **ANPR (Automatic Number Plate Recognition)**
  Deteksi lokasi plat nomor pakai YOLOv8 custom, dibaca teksnya pakai
  EasyOCR, dinormalisasi ke format plat Indonesia (`B 1234 ABC`), dan
  distabilkan pakai voting antar-frame supaya teks yang ditampilkan tidak
  "kedip-kedip" berubah tiap frame.

- **Deteksi Postur & Tracking Tag**
  Deteksi orang + 17 keypoint tubuh (YOLOv8-pose), klasifikasi postur
  (Berdiri / Menunduk / Jongkok) berdasarkan sudut lutut & pinggang, lalu
  melacak durasi tag keselamatan terpasang — otomatis meng-capture foto
  begitu proses pemasangan selesai.

- **Indikator FPS realtime** di kedua kamera (ANPR & Postur), supaya
  gampang lihat apakah model masih jalan lancar di perangkat yang dipakai.

- **UI Desktop 2 Halaman**
  - **Verifikasi** — video live kamera ANPR, auto-isi nomor plat, koreksi
    manual kalau OCR salah baca.
  - **Monitoring** — video live kamera deteksi postur, daftar tag yang
    tertangkap (bisa di-preview ukuran penuh & dihapus dengan renumbering
    otomatis), nomor plat bisa direvisi lagi sebelum diproses lebih lanjut.

- **Responsif** — layout reflow, bisa di-resize/maximize/fullscreen bebas,
  mendukung mode kios (`--kiosk`) untuk layar kecil (mis. Jetson Nano +
  panel 7").

- **Ringan** — cuma 1 model AI yang aktif dalam satu waktu. ANPR otomatis
  berhenti begitu plat sudah diproses (tombol "Verifikasi"/"Monitoring" di
  halaman Verifikasi), baru setelah itu deteksi postur mulai jalan —
  tidak pernah dua-duanya menyala bersamaan.

---

## 📁 Struktur Proyek

```
pertamina_gui/
├── main.py                     # entry point aplikasi
├── assets/                     # logo, ikon, background (hasil export Figma)
│   ├── style.qss
│   ├── app_icon.ico / .png     # ikon title bar & taskbar aplikasi
│   └── *.png
│
├── widgets/                    # kode UI (PyQt5)
│   ├── design_tokens.py        # warna, gradient, helper bersama
│   ├── page_verifikasi.py      # halaman "Verifikasi Plat Nomor"
│   ├── page_monitoring.py      # halaman "Monitoring Plat & Postur"
│   └── workers.py              # ANPRWorker & PostureWorker (QThread)
│
└── detection/                  # inti sistem deteksi
    ├── anpr/
    │   ├── anpr_main.py        # konfigurasi & pipeline ANPR (SOURCE, model, dst)
    │   ├── plate_utils.py      # normalisasi teks plat Indonesia
    │   └── video_source.py     # wrapper VideoCapture (file & live-camera)
    ├── pose/
    │   ├── v6.py                # konfigurasi & tracking tag/postur (SOURCE, model, dst)
    │   └── posture_detect.py    # klasifikasi postur dari keypoint
    └── model/
        ├── model.pt              # (taruh sendiri, tidak ikut di repo)
        └── yolov8n-pose.pt       # (taruh sendiri, tidak ikut di repo)
```



---

## 🔧 Requirements

- Python 3.9+
- PyQt5
- OpenCV (`opencv-python`)
- Ultralytics (YOLOv8)
- PyTorch
- EasyOCR
- NumPy

Instal semuanya:

```bash
pip install PyQt5 opencv-python ultralytics torch easyocr numpy
```

> Kalau punya GPU NVIDIA, install PyTorch versi CUDA-nya dulu (lihat
> [pytorch.org](https://pytorch.org)) supaya inferensi jauh lebih cepat.
> Tanpa GPU, aplikasi tetap jalan di CPU tapi lebih lambat.

---

## 🚀 Instalasi

1. **Clone repo ini**
   ```bash
   git clone <url-repo-ini>
   cd pertamina_gui
   ```

2. **Install dependency**
   ```bash
   pip install -r requirements.txt
   ```

3. **Taruh model & sumber video Anda sendiri:**
   | File | Lokasi |
   |---|---|
   | Model deteksi plat (`model.pt`) | `detection/model/model.pt` |
   | Model pose (`yolov8n-pose.pt` / `yolov8s-pose.pt` / dst.) | `detection/model/` |
   | Video uji ANPR (opsional, kalau belum pakai kamera asli) | `detection/anpr/` |
   | Video uji Postur (opsional) | `detection/pose/` |

4. **Sesuaikan sumber video** di dua file konfigurasi:

   `detection/anpr/anpr_main.py`
   ```python
   SOURCE = str(Path(__file__).resolve().parent / "3.mp4")   # file video uji
   # SOURCE = 0                                                # webcam
   # SOURCE = "rtsp://user:pass@192.168.1.10:554/stream1"      # kamera IP/CCTV
   ```

   `detection/pose/v6.py`
   ```python
   SOURCE = str(Path(__file__).resolve().parent / "14.mp4")  # file video uji
   # SOURCE = 0
   # SOURCE = "rtsp://user:pass@192.168.1.11:554/stream1"
   ```

---

## ▶️ Menjalankan

```bash
python main.py
```

Mode kios (fullscreen tanpa title bar, cocok untuk perangkat tertanam /
layar kecil):

```bash
python main.py --kiosk
```

Tekan `Esc` untuk keluar dari mode kios saat testing.

---

## 🧭 Alur Penggunaan

1. Aplikasi terbuka di halaman **Verifikasi** — kamera ANPR otomatis
   jalan, kolom nomor plat terisi otomatis dari hasil OCR. Indikator
   **LIVE** & FPS muncul begitu video benar-benar mengalir.
2. Operator boleh mengoreksi manual kalau bacaan OCR salah.
3. Tekan **Verifikasi** → foto plat disimpan, ANPR **berhenti**, aplikasi
   pindah ke halaman **Monitoring** dan deteksi postur mulai jalan.
   Tekan **Monitoring** (tanpa Verifikasi) → cuma pindah halaman, ANPR
   tetap jalan di background, deteksi postur belum dimulai.
4. Di halaman Monitoring, setiap tag keselamatan yang selesai terpasang
   otomatis masuk ke daftar (foto, waktu, durasi) — bisa diklik untuk
   preview foto ukuran penuh (bisa di-fullscreen), atau dihapus kalau
   salah tangkap (nomor tag sisanya otomatis urut ulang). Nomor plat
   juga masih bisa direvisi di halaman ini.
5. Tombol **Kembali** membawa balik ke halaman Verifikasi TANPA
   menghentikan deteksi postur di background (ANPR tidak otomatis
   menyala lagi — perlu Reset kalau mau verifikasi plat baru).
6. Tombol **Reset** menghentikan semua proses deteksi dan memulai ulang
   dari nol untuk kendaraan berikutnya.

---

## ⚙️ Parameter Penting

| Parameter | Lokasi | Fungsi |
|---|---|---|
| `SOURCE` | `anpr_main.py` / `v6.py` | Sumber video (file / webcam / RTSP) |
| `CONF_THRESHOLD` | `anpr_main.py` | Ambang keyakinan deteksi plat |
| `OCR_EVERY_N_FRAME` | `anpr_main.py` | OCR dijalankan tiap N frame (biar tetap realtime) |
| `CONF_THRESHOLD_DETECT` | `v6.py` | Ambang keyakinan deteksi orang |
| `ALERT_SECONDS` | `v6.py` | Lama tag harus "diam" sebelum dianggap selesai terpasang |
| `RATIO_JONGKOK` / `RATIO_MENUNDUK` | `v6.py` | Kalibrasi rasio tinggi tubuh untuk klasifikasi postur |
| `MODEL_PATH` | `anpr_main.py` / `v6.py` | Lokasi file model `.pt` |
| `ENABLE_SHADOWS` | `widgets/design_tokens.py` | Matikan efek shadow UI kalau lag di perangkat lemah |

---

## 🖥️ Deployment ke Perangkat Tertanam (Jetson)

### ⚠️ Soal hardware: Jetson Nano (2019) vs Jetson Orin Nano

Jetson Nano generasi awal (2019, chip Maxwell) sudah **end-of-life**,
terkunci di JetPack 4.6 lama, dan sistem ini (2x YOLOv8 + EasyOCR + GUI
PyQt5) akan **sangat berat** di situ — terutama EasyOCR.

**Rekomendasi: Jetson Orin Nano 8GB** (chip Ampere, ~15-40x lebih
kencang dari Nano lama). Varian 4GB bisa dipakai tapi RAM-nya pas-pasan
kalau 2 model + OCR + GUI jalan bergantian.

### Checklist optimasi (urut dari paling berdampak)

1. **Export model ke TensorRT** (dampak paling besar, 3-5x lebih cepat)
   — WAJIB dijalankan langsung di Jetson-nya, hasil export tidak bisa
   dipindah dari PC biasa:
   ```bash
   yolo export model=detection/model/model.pt format=engine half=True
   yolo export model=detection/model/yolov8n-pose.pt format=engine half=True
   ```
   Lalu ganti `MODEL_PATH` di `anpr_main.py` dan `v6.py` ke file
   `.engine` hasil export — `ultralytics.YOLO()` otomatis bisa
   membacanya, tidak perlu ubah kode lain.

2. **Install PyTorch versi resmi NVIDIA**, bukan `pip install torch`
   biasa (itu tidak teroptimasi CUDA untuk ARM64 Jetson, jalan di CPU
   meski ada GPU). Install lewat JetPack SDK Manager atau wheel khusus
   Jetson dari forum NVIDIA (sesuaikan versi JetPack-nya). Cek dengan:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"  # harus True
   ```

3. **Kecilkan resolusi inferensi** — turunkan `IMG_SIZE` di
   `anpr_main.py` dan `INFER_IMGSZ` di `v6.py` (mis. dari 640 ke 480
   atau 416), sambil dicek akurasinya masih cukup.

4. **EasyOCR paling berat** — naikkan `OCR_EVERY_N_FRAME` di
   `anpr_main.py` (mis. dari 2 ke 4-5; akurasi tetap terjaga karena ada
   voting antar-frame). Kalau masih berat, pertimbangkan ganti ke OCR
   yang lebih ringan.

5. **Matikan efek shadow UI**: set `ENABLE_SHADOWS = False` di
   `widgets/design_tokens.py`.

6. **Setup OS Jetson-nya:**
   - Power mode maksimal: `sudo nvpmodel -m 0`
   - Pasang kipas aktif (performa maksimal = panas → throttling kalau
     tanpa pendingin)
   - Boot & simpan project dari SSD (NVMe/USB3), **bukan microSD** —
     microSD jadi bottleneck I/O
   - Tambahkan swap file kalau RAM sering penuh (terutama varian 4GB)

7. **Install PyQt5 lewat `apt`** (bukan `pip`) untuk hindari waktu
   compile yang lama di ARM:
   ```bash
   sudo apt-get install python3-pyqt5
   ```

8. Jalankan dengan `python3 main.py --kiosk`. Cek resolusi layar dengan
   `xrandr`, sesuaikan `MIN_WIDTH`/`MIN_HEIGHT` di `main.py` kalau perlu.

> **Catatan desain:** sistem ini sudah dirancang hemat resource dari
> awal — ANPR dan deteksi Postur sengaja **tidak pernah jalan
> bersamaan** (cuma 1 model AI aktif dalam satu waktu), jadi beban
> puncaknya sudah jauh lebih ringan dibanding menjalankan keduanya
> terus-menerus.

---

## 📄 Lisensi

Internal project — Pertamina Patra Niaga.
