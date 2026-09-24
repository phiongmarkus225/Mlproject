# End-to-End Machine Learning Project

Proyek ini adalah contoh project machine learning yang disusun dengan struktur modular agar lebih mudah dipahami dan dikembangkan. Tujuan utamanya adalah memisahkan proses data ingestion, preprocessing, dan training menjadi bagian-bagian yang jelas.

## Tujuan proyek

Project ini membantu Anda memahami:
- alur kerja machine learning dari awal sampai training,
- penggunaan logging untuk memantau proses,
- penanganan error yang lebih informatif,
- struktur folder yang umum dipakai pada project ML.

## Struktur folder

- artifacts/: folder hasil output proses seperti data train/test dan model preprocessing
- logs/: file log runtime aplikasi
- notebook/: notebook eksplorasi data dan eksperimen model
- src/: folder utama kode program
  - components/: modul inti project
    - data_ingestion.py: membaca data, menyimpan salinan raw data, dan membagi data menjadi train/test
    - data_transformation.py: memproses fitur numerik dan kategorikal sebelum model dilatih
    - model_trainer.py: tempat untuk membangun dan melatih model (masih dapat dikembangkan)
  - pipeline/: tempat alur training dan prediksi
  - exception.py: custom exception untuk memberi informasi error yang lebih detail
  - logger.py: konfigurasi logging aplikasi
  - utils.py: helper fungsi yang sering dipakai seperti menyimpan objek model/preprocessor

## Alur kerja proyek

1. Data ingestion
   - File [src/components/data_ingestion.py](src/components/data_ingestion.py) membaca dataset dari file CSV.
   - Data disimpan ke folder artifacts sebagai raw data, train data, dan test data.
   - Tujuannya adalah mempersiapkan data sebelum diproses.

2. Data transformation
   - File [src/components/data_transformation.py](src/components/data_transformation.py) menentukan fitur numerik dan kategorikal.
   - Fitur numerik diberi imputation dan scaling.
   - Fitur kategorikal diberi imputation, encoding, dan scaling.
   - Hasil preprocessing disimpan sebagai file preprocessor.pkl di folder artifacts.

3. Logging dan error handling
   - [src/logger.py](src/logger.py) mencatat aktivitas aplikasi.
   - [src/exception.py](src/exception.py) membantu menangkap error dengan informasi file dan baris yang terjadi.

4. Utility
   - [src/utils.py](src/utils.py) menyediakan fungsi sederhana untuk menyimpan objek seperti preprocessor atau model.

## Penjelasan modul utama

### 1. Data Ingestion
Modul ini bertanggung jawab untuk:
- membaca dataset dari path yang diberikan,
- menyimpan salinan data mentah,
- membagi data menjadi train dan test dengan rasio 80:20.

### 2. Data Transformation
Modul ini bertanggung jawab untuk:
- memilih kolom yang akan diproses,
- menangani missing value,
- mengubah fitur kategorikal menjadi format numerik,
- menormalisasi fitur agar model lebih stabil.

### 3. Logger
Logger digunakan untuk mencatat langkah-langkah penting seperti:
- data berhasil dibaca,
- data berhasil dibagi,
- preprocessing diterapkan,
- error terjadi.

### 4. Custom Exception
Custom exception membantu developer melihat lokasi error secara lebih jelas ketika proses gagal.

## Cara menjalankan

1. Buat environment virtual
   ```bash
   python -m venv venv
   ```

2. Aktifkan environment
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependency
   ```bash
   pip install -r requirements.txt
   ```

4. Training model (menghasilkan artifacts/model.pkl & preprocessor.pkl)
   ```bash
   python src/pipeline/train_pipeline.py
   ```

5. Jalankan web app + frontend
   ```bash
   python app.py
   ```
   Buka http://localhost:5000

6. Test prediction pipeline secara langsung (tanpa web)
   ```bash
   python src/pipeline/predict_pipeline.py
   ```

## Menjalankan dengan Docker

```bash
# Build & jalankan dengan docker compose (paling mudah)
docker compose up --build

# atau manual
docker build -t ml-project .
docker run -p 5000:5000 ml-project
```

Buka http://localhost:5000. Volume `./artifacts` dan `./logs` di-bind ke
container, sehingga jika model dilatih ulang secara lokal, container langsung
memakai model baru tanpa perlu rebuild.

## Arsitektur aplikasi web

- `app.py` -> entry point Flask (route `/` untuk form, `/predictdata` untuk prediksi, `/health` untuk health check)
- `templates/index.html` -> frontend form input fitur siswa
- `static/` -> CSS styling
- `src/pipeline/predict_pipeline.py` -> memuat `model.pkl` + `preprocessor.pkl` lalu menghasilkan prediksi
- `src/pipeline/train_pipeline.py` -> orkestrasi ingestion -> transformation -> training

## Catatan penting

Proyek ini sudah memiliki alur end-to-end: ingestion -> preprocessing -> training
(termasuk hyperparameter tuning dengan Optuna) -> serving via web API. Struktur ini
mengikuti pola modular yang umum dipakai di industri (component + pipeline + service layer).

Dengan dokumentasi ini, diharapkan orang lain dapat lebih cepat memahami apa yang dilakukan setiap file dan bagaimana alur proyek berjalan.