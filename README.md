# Analisis Sentimen Komentar YouTube: Lexicon-based vs LLM-based

Proyek ini membandingkan kinerja dua metode analisis sentimen (Lexicon-based dan LLM-based) pada 100 komentar pertama dari sebuah video YouTube/Shorts, kemudian mengevaluasinya terhadap **Ground Truth** yang Anda tentukan sendiri secara manual.

## Metode yang Digunakan

1. **Lexicon-based (Kamus Sentimen):**
   * **Preprocessing:** Menggunakan **PySastrawi** untuk menghapus *stopword* dan melakukan *stemming* (pengubahan kata berimbuhan menjadi kata dasar).
   * **Normalisasi:** Kamus slang buatan sendiri untuk mengubah singkatan chat internet (misal: *yg* menjadi *yang*, *bgt* menjadi *banget*) agar pencocokan kata di kamus lebih akurat.
   * **Skoring:** Menggunakan **InSet Lexicon** (Indonesian Sentiment Lexicon oleh Fajri Koto & Gemala Y. Rahmaningtyas, 2017) berisi kata positif dan negatif dengan bobot $-5$ sampai $+5$.
   
2. **LLM-based (Large Language Model):**
   * Menggunakan **NVIDIA NIM API** dengan model **`meta/llama-3.1-70b-instruct`**.
   * Pemrosesan secara batch (20 komentar per batch) untuk mempercepat runtime dan menghemat kuota API.
   * Prompt dioptimalkan khusus untuk memahami bahasa Indonesia slang/gaul serta bahasa daerah (seperti Jawa).

---

## Prasyarat & Instalasi

Pastikan Anda telah menginstal Python (rekomendasi versi 3.8 ke atas) di komputer Anda.

1. Buka PowerShell atau Command Prompt pada direktori proyek ini:
   ```powershell
   cd "c:\Users\luthf\OneDrive\Desktop\KULIAH\semester 6\STKI\tugas-sentimen-analisis"
   ```

2. Instal dependensi library yang dibutuhkan:
   ```powershell
   pip install -r requirements.txt
   ```

---

## Cara Penggunaan

### Metode Utama: Menggunakan Dashboard Web Interaktif (Sangat Direkomendasikan)

Untuk menjalankan antarmuka web interaktif di mana Anda dapat mengubah Ground Truth secara langsung di browser dan melihat hasil akurasi secara instan:

1. Jalankan aplikasi Streamlit:
   ```powershell
   streamlit run app.py
   ```
2. Aplikasi akan otomatis membuka browser Anda di alamat `http://localhost:8501`.
3. Di panel sebelah kiri (Sidebar):
   * Masukkan **URL Video YouTube/Shorts** yang ingin Anda analisis (atau biarkan default).
   * Tentukan batas komentar (default: 100).
   * Klik **🚀 Ambil Data & Mulai Analisis**.
4. Di halaman utama:
   * Anda akan melihat **Judul Video** dan **Tabel Komentar**.
   * Klik dua kali pada kolom **Ground Truth** untuk memilih sentimen asli (`positif`, `negatif`, `netral`) menggunakan dropdown list.
   * Setiap kali Anda mengubah Ground Truth, data akan otomatis disimpan ke `sentiment_results.csv` dan grafik performa/Confusion Matrix di bagian bawah akan ter-update secara real-time!

---

### Metode Alternatif: Menggunakan Command Line (CLI)

Jika Anda lebih menyukai penggunaan via terminal:

1. **Ambil Data & Jalankan Analisis:**
   ```powershell
   python main.py
   ```
   Hasil analisis awal akan disimpan di file `sentiment_results.csv` dengan kolom Ground Truth kosong.

2. **Isi Ground Truth:**
   Buka file `sentiment_results.csv` dengan Excel, isi kolom `Ground Truth` secara manual dengan tulisan `positif`, `negatif`, atau `netral`, lalu simpan kembali sebagai CSV.

3. **Hitung Evaluasi & Grafik:**
   Jalankan script evaluasi:
   ```powershell
   python evaluate.py
   ```
   Metrik performa akan dicetak ke terminal dan visualisasi disimpan di file `evaluation_metrics.png`.

---

## File Konfigurasi (`.env`)
Pengaturan default disimpan di file `.env`. Anda dapat mengedit file ini jika ingin mengubah:
* `NVIDIA_API_KEY`: API Key untuk NVIDIA NIM API.
* `YOUTUBE_VIDEO_URL`: Tautan video YouTube/Shorts yang ingin diambil komentarnya.
* `MAX_COMMENTS`: Batas maksimal komentar yang akan diambil (default: 100).
* `NVIDIA_MODEL`: Model LLM yang digunakan dari NVIDIA NIM API.
