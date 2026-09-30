-- ============================================================
-- Skema database untuk Microsoft SQL Server / Azure SQL Database
-- (versi T-SQL dari db/schema.sql yang MySQL)
-- ============================================================
--
-- CARA PAKAI -- SQL Server LOKAL (on-premise):
--   1. Buka SQL Server Management Studio (SSMS), connect ke server Anda.
--   2. Klik kanan "Databases" > "New Database..." > nama:
--      pertamina_monitoring > OK.
--   3. Klik kanan database itu > "New Query", paste SEMUA isi file ini
--      (kecuali baris CREATE DATABASE paling atas, lihat catatan di
--      bawah), klik Execute (F5).
--
-- CARA PAKAI -- Azure SQL Database (cloud):
--   Azure SQL Database TIDAK BISA dibuat lewat perintah "CREATE DATABASE"
--   dari sini -- database-nya harus sudah dibuat lebih dulu lewat Azure
--   Portal (Create a resource > SQL Database). Setelah database-nya ada
--   dan Anda sudah connect ke situ (lewat SSMS/Azure Data Studio), jalankan
--   file ini TANPA baris "CREATE DATABASE ... / GO / USE ..." di paling
--   atas (hapus atau skip 4 baris pertama), langsung mulai dari
--   "CREATE TABLE dbo.verifikasi".
--
-- Struktur & timestamp-nya SAMA PERSIS konsepnya dengan db/schema.sql
-- (MySQL) -- lihat komentar di file itu untuk penjelasan lengkap soal
-- captured_at vs verified_at, dan relasi verifikasi -> tags.
-- ============================================================

-- --- baris ini CUMA untuk SQL Server lokal, SKIP untuk Azure SQL ---
IF DB_ID('pertamina_monitoring') IS NULL
BEGIN
    CREATE DATABASE pertamina_monitoring;
END
GO

USE pertamina_monitoring;
GO
-- --- akhir bagian yang di-skip untuk Azure SQL ---

-- ------------------------------------------------------------
-- Tabel utama: 1 baris = 1 kali verifikasi plat
-- ------------------------------------------------------------
IF OBJECT_ID('dbo.verifikasi', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.verifikasi (
        id            INT IDENTITY(1,1) PRIMARY KEY,
        plate_number  NVARCHAR(20)   NOT NULL,     -- mis. "B 1234 ABC"
        plate_image   VARBINARY(MAX) NULL,         -- ISI gambar foto plat (bukan path)
        captured_at   DATETIME2      NULL,         -- waktu FOTO PLAT diambil
        verified_at   DATETIME2      NOT NULL,     -- waktu diupload/diverifikasi FINAL
        created_at    DATETIME2      DEFAULT SYSDATETIME()
    );
END
GO

-- ------------------------------------------------------------
-- Tabel anak: 1 baris = 1 foto tag, terhubung ke 1 baris verifikasi
-- ------------------------------------------------------------
IF OBJECT_ID('dbo.tags', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.tags (
        id             INT IDENTITY(1,1) PRIMARY KEY,
        verifikasi_id  INT            NOT NULL,
        tag_num        INT            NOT NULL,    -- nomor urut tag (1, 2, 3, ...)
        captured_time  NVARCHAR(20)   NULL,        -- TANGGAL + jam tag terpasang
        duration       NVARCHAR(20)   NULL,        -- lama proses, mis. "6.2s"
        tag_image      VARBINARY(MAX) NULL,        -- ISI gambar foto tag (bukan path)
        created_at     DATETIME2      DEFAULT SYSDATETIME(),
        CONSTRAINT FK_tags_verifikasi FOREIGN KEY (verifikasi_id)
            REFERENCES dbo.verifikasi(id) ON DELETE CASCADE
    );
END
GO

-- Index supaya pencarian per plat / per verifikasi lebih cepat
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_verifikasi_plate')
    CREATE INDEX idx_verifikasi_plate ON dbo.verifikasi (plate_number);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_tags_verifikasi')
    CREATE INDEX idx_tags_verifikasi ON dbo.tags (verifikasi_id);
GO
