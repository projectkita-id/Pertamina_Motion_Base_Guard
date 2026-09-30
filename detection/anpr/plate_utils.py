"""
plate_utils.py
Utilitas untuk membersihkan & memvalidasi hasil OCR plat nomor kendaraan Indonesia.

Format umum plat Indonesia:
    [1-2 huruf kode wilayah] [1-4 angka] [1-3 huruf seri]
    Contoh: B 1234 ABC, D 9 AB, AB 1234 CD, BE 1 A

Beberapa daerah (mis. mulai 2022) juga memakai format 5 digit angka.
"""

import re
from collections import deque

# Karakter yang sering tertukar oleh OCR
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"}

# Regex longgar: huruf-angka-huruf, dipisah spasi opsional
PLATE_REGEX = re.compile(
    r"^([A-Z]{1,2})\s?([0-9]{1,5})\s?([A-Z]{0,3})$"
)


def clean_raw_text(text: str) -> str:
    """Buang karakter selain huruf/angka, ubah ke uppercase."""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _fix_segment(segment: str, want_digit: bool) -> str:
    """Perbaiki satu segmen (huruf atau angka) sesuai posisi yang diharapkan."""
    fixed = []
    for ch in segment:
        if want_digit and ch in LETTER_TO_DIGIT:
            fixed.append(LETTER_TO_DIGIT[ch])
        elif not want_digit and ch in DIGIT_TO_LETTER:
            fixed.append(DIGIT_TO_LETTER[ch])
        else:
            fixed.append(ch)
    return "".join(fixed)


def _segment_cost(segment: str, want_digit: bool) -> int:
    """Hitung berapa banyak karakter di segmen ini yang 'tidak wajar' untuk
    posisinya (butuh dikoreksi). Skor lebih kecil = lebih meyakinkan."""
    cost = 0
    for ch in segment:
        if want_digit:
            if ch.isdigit():
                continue
            if ch in LETTER_TO_DIGIT:
                cost += 1  # bisa dikoreksi, tapi tetap ada biaya kecil
            else:
                cost += 3  # huruf yang tidak mirip angka sama sekali
        else:
            if ch.isalpha():
                continue
            if ch in DIGIT_TO_LETTER:
                cost += 1
            else:
                cost += 3
    return cost


def normalize_plate(raw_text: str):
    """
    Coba parse string OCR mentah (biasanya tanpa spasi, mis. 'B1234ABC')
    menjadi format plat Indonesia yang rapi: '<wilayah> <nomor> <seri>'.

    Strategi: karena panjang teks plat pendek, coba semua kemungkinan titik
    potong [wilayah(1-2 huruf)] [nomor(1-5 angka)] [seri(0-3 huruf)] yang
    jumlah panjangnya pas dengan teks, lalu pilih kombinasi yang paling sedikit
    butuh koreksi karakter (paling "wajar").

    Mengembalikan (plat_terformat, is_valid: bool).
    """
    cleaned = clean_raw_text(raw_text)
    if not cleaned or len(cleaned) < 3:
        return cleaned, False

    n = len(cleaned)
    best = None  # (cost, prefix, middle, suffix)

    for prefix_len in (1, 2):
        for suffix_len in (0, 1, 2, 3):
            middle_len = n - prefix_len - suffix_len
            if middle_len < 1 or middle_len > 5:
                continue
            prefix = cleaned[:prefix_len]
            middle = cleaned[prefix_len:prefix_len + middle_len]
            suffix = cleaned[prefix_len + middle_len:]

            cost = (
                _segment_cost(prefix, want_digit=False)
                + _segment_cost(middle, want_digit=True)
                + _segment_cost(suffix, want_digit=False)
            )
            if best is None or cost < best[0]:
                best = (cost, prefix, middle, suffix)

    if best is None:
        return cleaned, False

    _, prefix, middle, suffix = best
    prefix = _fix_segment(prefix, want_digit=False)
    middle = _fix_segment(middle, want_digit=True)
    suffix = _fix_segment(suffix, want_digit=False)

    candidate = f"{prefix} {middle} {suffix}".strip()
    candidate = re.sub(r"\s+", " ", candidate)

    is_valid = bool(
        re.match(r"^[A-Z]{1,2}$", prefix)
        and re.match(r"^[0-9]{1,5}$", middle)
        and re.match(r"^[A-Z]{0,3}$", suffix)
    )
    return candidate, is_valid


class PlateVoteTracker:
    """
    Melacak hasil OCR per objek plat (berdasarkan track id sederhana / posisi)
    dan mengambil hasil paling sering muncul (voting) dalam beberapa frame terakhir
    agar output realtime tidak "kedip-kedip" berubah tiap frame.
    """

    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self._buffers = {}

    def update(self, track_key, text: str):
        if not text:
            return self.best(track_key)
        buf = self._buffers.setdefault(track_key, deque(maxlen=self.history_len))
        buf.append(text)
        return self.best(track_key)

    def best(self, track_key):
        buf = self._buffers.get(track_key)
        if not buf:
            return ""
        counts = {}
        for t in buf:
            counts[t] = counts.get(t, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def cleanup(self, active_keys):
        """Hapus track yang sudah tidak terlihat lagi."""
        for k in list(self._buffers.keys()):
            if k not in active_keys:
                del self._buffers[k]