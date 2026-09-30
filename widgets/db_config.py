"""
db_config.py

SATU-SATUNYA FILE yang perlu diubah kalau Anda pindah database (MySQL ->
SQL Server -> Azure SQL Database, atau sebaliknya). Kode lain
(widgets/db_client.py, main.py) tidak perlu disentuh sama sekali --
mereka cuma manggil upload_records() tanpa peduli database-nya apa.

CARA PINDAH DATABASE / METODE UPLOAD:
    1. Ganti nilai DB_BACKEND di bawah ini ke salah satu:
           "rest_api"   -> kirim data lewat REST API (backend tim lain,
                            mis. .NET) -- TIDAK connect database langsung
           "mysql"      -> MySQL / MariaDB (XAMPP, dsb)
           "sqlserver"  -> Microsoft SQL Server lokal/on-premise
           "azure_sql"  -> Azure SQL Database (cloud)
    2. Isi konfigurasi yang sesuai di bagian bawah (URL API, atau
       host/user/password/dst untuk koneksi database langsung).
    3. Install driver/library Python yang dibutuhkan (lihat komentar
       tiap backend di bawah).
    4. Khusus mysql/sqlserver/azure_sql: jalankan schema SQL yang sesuai
       (db/schema.sql untuk MySQL, db/schema_sqlserver.sql untuk SQL
       Server/Azure SQL). Untuk rest_api, database-nya dikelola sendiri
       oleh tim backend -- tidak perlu jalankan schema apa pun di sini.
    Tidak ada kode lain (main.py, dst) yang perlu diubah.
"""

# ============================================================
# GANTI INI SAJA untuk pindah database / metode upload
# ============================================================
DB_BACKEND = "mysql"   # "mysql" | "sqlserver" | "azure_sql" | "rest_api"


# ------------------------------------------------------------
# 0) REST API (backend .NET/tim lain) -- butuh: pip install requests
#    Ini BUKAN koneksi database langsung -- data dikirim lewat HTTP
#    (multipart/form-data) ke endpoint yang sudah disediakan tim backend.
#    Tidak perlu pymysql/pyodbc/ODBC Driver sama sekali kalau pakai ini.
# ------------------------------------------------------------
REST_API_CONFIG = {
    "upload_url": "http://192.168.1.6:5136/api/VehicleInspections/upload",
    "timeout": 30,  # detik, batas waktu tunggu respons server

    # Perkecil gambar sebelum dikirim -- berguna kalau server backend
    # (mis. MySQL di baliknya) menolak dengan error "packet bigger than
    # max_allowed_packet". Idealnya masalah itu diperbaiki di server
    # (naikkan max_allowed_packet), tapi ini jaga-jaga dari sisi aplikasi
    # supaya tetap jalan walau server belum sempat dibenahi.
    "image_max_dimension": 1280,  # px -- sisi terpanjang gambar, None = ukuran asli
    "image_quality": 80,           # 0-100, makin kecil makin terkompresi
}


# ------------------------------------------------------------
# 1) MySQL / MariaDB (XAMPP, dsb) -- butuh: pip install pymysql
# ------------------------------------------------------------
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "pertamina_monitoring",
}

# ------------------------------------------------------------
# 2) Microsoft SQL Server LOKAL (on-premise / server kantor sendiri)
#    Butuh: pip install pyodbc
#    DAN install "ODBC Driver 17 (atau 18) for SQL Server" dari Microsoft
#    (bukan cuma pip install -- ini driver sistem, download terpisah).
#    host bisa berupa "localhost", "NAMA_SERVER", atau
#    "NAMA_SERVER\\NAMA_INSTANCE" (mis. "localhost\\SQLEXPRESS" -- ini
#    yang dipakai kalau Anda install pakai edisi Express, cek nama
#    instance-nya di layar akhir instalasi).
#
#    trusted_connection = True artinya pakai WINDOWS AUTHENTICATION
#    (login pakai akun Windows yang sedang aktif, TIDAK perlu isi
#    user/password) -- ini yang aktif secara default begitu SQL Server
#    baru diinstall. Set False kalau mau pakai SQL Server Authentication
#    (user/password, mis. "sa") -- itu perlu diaktifkan manual dulu di
#    SSMS (Server Properties > Security > SQL Server and Windows
#    Authentication mode), tidak aktif dari awal.
# ------------------------------------------------------------
SQLSERVER_CONFIG = {
    "host": "localhost\\SQLEXPRESS",
    "port": 1433,
    "trusted_connection": True,   # True = Windows Authentication (tidak perlu user/password)
    "user": "sa",                  # dipakai HANYA kalau trusted_connection = False
    "password": "",                # dipakai HANYA kalau trusted_connection = False
    "database": "pertamina_monitoring",
    "driver": "{ODBC Driver 17 for SQL Server}",
    "encrypt": False,  # jaringan lokal biasanya tidak wajib SSL
}

# ------------------------------------------------------------
# 3) Azure SQL Database (cloud) -- butuh: pip install pyodbc
#    DAN "ODBC Driver 17/18 for SQL Server" (sama seperti di atas).
#    host = alamat server Azure, formatnya:
#           "namaserver.database.windows.net"
#    (dapat dilihat di halaman resource SQL Database di Azure Portal)
#    Jangan lupa: di Azure Portal, tambahkan IP Address komputer Anda
#    ke "Firewall rules" server itu, kalau tidak koneksi akan ditolak.
# ------------------------------------------------------------
AZURE_SQL_CONFIG = {
    "host": "namaserver.database.windows.net",
    "port": 1433,
    "user": "youruser",
    "password": "",
    "database": "pertamina_monitoring",
    "driver": "{ODBC Driver 17 for SQL Server}",
    "encrypt": True,  # Azure SQL WAJIB pakai koneksi terenkripsi
}