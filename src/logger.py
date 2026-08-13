"""Modul untuk logging aplikasi.

File ini berfungsi untuk menyiapkan sistem pencatatan aktivitas program.
Dengan logger, developer bisa melihat informasi penting saat program berjalan,
seperti mulai menjalankan proses, berhasil membaca data, atau terjadi error.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Buat folder utama logs terlebih dahulu.
# Lalu buat subfolder berdasarkan tanggal agar file log tersusun rapi.
base_log_dir = PROJECT_ROOT / "logs"
date_folder = base_log_dir / datetime.now().strftime("%Y-%m-%d")
date_folder.mkdir(parents=True, exist_ok=True)

# Nama file log dibuat berdasarkan tanggal dan waktu saat program dijalankan.
# Contoh: 2026-08-08_12-05-53.log
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = date_folder / LOG_FILE

# Konfigurasi dasar logger.
# Semua log akan ditulis ke file yang telah ditentukan.
logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Contoh penggunaan logger saat file ini dijalankan langsung.
    logging.info("logging has started")
