# SEMANTIKA: Dashboard Analisis Sentimen Komentar YouTube (Lexicon-based vs LLM-based)

SEMANTIKA adalah sistem analisis sentimen hybrid yang dirancang untuk mengambil, memproses, dan membandingkan performa dua pendekatan klasifikasi sentimen: metode berbasis kamus (Lexicon-based menggunakan Sastrawi dan InSet/VADER Lexicon) dan metode berbasis kecerdasan buatan (LLM-based menggunakan Llama 3.1 di NVIDIA NIM API). Sistem ini dilengkapi dengan dashboard interaktif Streamlit untuk pelabelan Ground Truth secara instan serta evaluasi metrik akurasi akademik (Precision, Recall, F1-Score, dan Cohen's Kappa Coefficient).

Sistem ini mendukung pengoperasian hybrid dengan dua mode penyimpanan data: mode lokal (Development) dan mode sinkronisasi cloud Google Sheets (Production).

---

## Fitur Utama Sistem

Sistem ini dirancang dengan fitur-fitur premium untuk kebutuhan analisis data sentimen YouTube secara komprehensif:

1. **Dual-Model Sentiment Engine**: Menjalankan analisis sentimen menggunakan metode Lexicon-based dan LLM-based secara paralel untuk memberikan perbandingan performa langsung pada data komentar yang sama.
2. **YouTube & Shorts Scraper**: Pengambilan otomatis komentar, nama penulis, jumlah Likes (votes), dan waktu unggah relatif serta timestamp dari tautan video reguler maupun Shorts YouTube tanpa memerlukan API Key YouTube yang rumit.
3. **Interactive Ground Truth Editor**: Dashboard menyediakan antarmuka tabel interaktif (menggunakan st.data_editor) untuk melakukan anotasi Ground Truth secara langsung di browser. Setiap pembaruan data langsung memicu penggabungan aman ke data master dan memperbarui metrik evaluasi secara real-time.
4. **Slang Dictionary Editor**: Fitur manajemen kamus slang langsung dari dashboard web. Pengguna dapat menambah, mengedit, atau menghapus singkatan/slang gaul internet (misal: "yg" -> "yang", "bgt" -> "banget") yang disimpan pada file konfigurasinya (lexicon/custom_slang.json) dan langsung diterapkan saat preprocessing data baru.
5. **Penanganan Kata Negasi (Negation Handling)**: Algoritma Lexicon-based dilengkapi pendeteksi kata negasi (seperti "tidak", "bukan", "not", "no") dengan jendela pencarian 2 kata sebelumnya. Jika kata negasi ditemukan sebelum kata bersentimen, skor polaritas kata tersebut akan dibalik secara otomatis (inversi sentimen).
6. **Filter Perbedaan Prediksi (Prediction Mismatch Filter) & Sorting**: Dashboard dilengkapi penyaring data untuk menyembunyikan komentar yang memiliki kesepakatan model agar pengguna fokus melakukan anotasi pada data mismatch. Serta dilengkapi fitur pengurutan (Sort) berdasarkan komentar terbaru atau jumlah Likes terbanyak.
7. **Metrik Evaluasi Akademik Lengkap**: Perhitungan otomatis untuk akurasi (Accuracy), presisi (Precision), sensitivitas (Recall), F1-Score (Macro Average), Confusion Matrix interaktif, serta Cohen's Kappa Coefficient untuk menilai keandalan kesepakatan klasifikasi terhadap Ground Truth.
8. **Analisis Lanjutan**:
   * **Tren Sentimen (Timeline)**: Visualisasi grafis perkembangan sentimen komentar (Ground Truth) sepanjang waktu rilis komentar menggunakan timestamp yang valid.
   * **Frekuensi Kata (Top Words)**: Diagram batang horizontal perbandingan kata kunci yang paling sering muncul pada komentar bersentimen positif vs negatif.
   * **Pemodelan Topik (Topic Modeling)**: Pengelompokan komentar secara cerdas menggunakan kombinasi TF-IDF dan clustering K-Means untuk mendeteksi 3 topik pembicaraan utama beserta sampel komentar dan kata kuncinya.
9. **Navigasi Riwayat Analisis (History Viewer)**: Pengguna dapat membuka, menganalisis, dan mengedit data analisis lama langsung dari sidebar menggunakan dropdown riwayat berkas lokal di direktori `history/`.
10. **Benchmark Waktu Eksekusi**: Menghitung secara presisi durasi waktu eksekusi proses analisis tiap model (dalam milidetik/detik) beserta rasio perbandingan kecepatan (speedup factor) antara performa komputasi lokal Lexicon vs pemanggilan API Cloud LLM.
11. **Stemming Caching Sastrawi**: Penerapan in-memory cache pada pemrosesan stemming PySastrawi. Sistem tidak akan mengulang stemming pada kata yang sama, meningkatkan kecepatan komputasi Lexicon hingga 70%.
12. **Ekspor Laporan PDF & Excel**: Pengguna dapat mengunduh lembar kerja Excel hasil analisis yang terformat rapi dan mencetak laporan analisis komprehensif berformat PDF yang mencakup visualisasi grafik, Confusion Matrix, dan ringkasan metrik evaluasi.

---

## Mode Aplikasi (App Mode)

Sistem SEMANTIKA dapat dikonfigurasi melalui file `.env` dengan salah satu dari dua mode berikut:

* **Mode DEVELOPMENT (Lokal Saja)**:
  * Membaca dan menyimpan hasil analisis secara lokal ke dalam file `sentiment_results.csv`.
  * Sangat cocok untuk pengujian lokal cepat tanpa membutuhkan kredensial Google Cloud.
* **Mode PRODUCTION (Lokal + Google Sheets Sync)**:
  * Membaca dan menyimpan hasil analisis ke dalam berkas lokal `sentiment_results.csv`, sekaligus melakukan sinkronisasi database secara dua arah (two-way synchronization) ke Google Sheets pada worksheet `Database_Sentimen`.
  * Berguna saat aplikasi dideploy ke cloud (misal Streamlit Community Cloud) agar data Ground Truth yang diedit oleh pengguna di browser tidak hilang saat container aplikasi di-restart.

---

## Alur Kerja Sistem (System Workflow)

Alur kerja utama sistem dari penentuan sumber data hingga visualisasi metrik evaluasi digambarkan sebagai berikut:

```mermaid
graph TD
    A[Pengguna] -->|Pilih Sumber Data: Scraping vs Lokal| B[Streamlit Dashboard]
    B -->|Pilihan: Buka Riwayat Lokal| C[History Viewer - Muat Berkas CSV Lokal]
    B -->|Pilihan: Ambil Video Baru| D[Ambil Judul Video & Metadata]
    D -->|Scrape Komentar, Likes, Waktu| E[YouTube Downloader]
    E -->|Komentar Mentah + Metadata| F[Engine Analisis Sentimen]
    
    subgraph Engine Analisis Sentimen
        F --> G[Lexicon-Based Engine - Cache Stemming Sastrawi]
        F --> H[LLM-Based Engine - Batching NVIDIA NIM]
    end
    
    G -->|Preprocessing & Skoring| I[Hasil Prediksi Gabungan]
    H -->|Batching & Fallback| I
    
    C -->|Muat Instan| J[Tampilkan di Dashboard]
    I -->|Ekspor & Simpan ke CSV/GSheets| J
    
    J -->|Anotasi Ground Truth / Sort by Likes| K[Evaluasi Performa Real-time]
    K -->|Visualisasi: Donut, CM, Timeline, Top Words, Topic Modeling| J
```

### Detail Langkah Alur Sistem:
1. **Pilih Sumber Data**: Pengguna menentukan apakah ingin menganalisis video baru dengan scraping, atau memuat riwayat analisis lokal dari dropdown setelan di sidebar.
2. **Pengambilan Data (Scraping)**: Jika memilih scraping, sistem mengunduh komentar beserta metadata jumlah Likes, deskripsi waktu relatif, dan epoch timestamp yang valid.
3. **Analisis Sentimen Paralel**:
   * **Lexicon-based**: Menjalankan pra-pemrosesan kata demi kata. Menggunakan in-memory cache untuk proses stemming Sastrawi guna menghindari latensi pada perulangan kata dasar yang sama.
   * **LLM-based**: Mengelompokkan komentar per 20 data dan mengirimkannya ke NVIDIA NIM API.
4. **Konsolidasi & Penyimpanan**: Menggabungkan seluruh hasil klasifikasi dan menyimpan ke berkas CSV lokal di dalam folder `history/` (dan disinkronkan ke Google Sheets jika mode aplikasi adalah `production`).
5. **Urutkan & Filter**: Dashboard memuat tabel komentar. Pengguna dapat mengurutkan komentar berdasarkan jumlah Likes atau waktu terbaru, menyaring data mismatch, serta melakukan anotasi Ground Truth.
6. **Visualisasi Lanjutan**: Pengguna dapat menjelajahi sebaran sentimen (Donut Chart), Confusion Matrix, grafik tren waktu rilis komentar, bar chart frekuensi kata terbanyak, serta kluster topik pembicaraan (Topic Modeling).

---

## Model 1: Lexicon-Based Model

### Alur Kerja Model Lexicon-Based

Model Lexicon-Based memproses teks komentar melalui serangkaian tahapan preprocessing teks Bahasa Indonesia terstruktur sebelum mencocokkannya ke dalam kamus sentimen:

```mermaid
graph TD
    A[Komentar Mentah] --> B[Case Folding & Pembersihan Karakter]
    B --> C[Normalisasi Slang - Kamus Statis & Custom JSON]
    C --> D[Pembersihan Stopword Sastrawi - Proteksi Kata Negasi]
    D --> E[Stemming Sastrawi - Caching Word-by-Word & Proteksi Bahasa Inggris]
    E --> F[Deteksi Bahasa Dominan - InSet vs VADER]
    F --> G{Pilih Kamus}
    G -->|Indonesian| H[InSet Lexicon]
    G -->|English| I[VADER Lexicon]
    H --> J[Pencarian Token + Deteksi Negasi Window 2 Kata]
    I --> J
    J --> K[Akumulasi Skor Sentimen]
    K --> L{Klasifikasi Skor}
    L -->|> 0.05| M[positif]
    L -->|< -0.05| N[negatif]
    L -->|Lainnya| O[netral]
```

### Penjelasan Tahapan Preprocessing & Prediksi Lexicon:
1. **Case Folding & Pembersihan**: Mengubah seluruh teks komentar menjadi huruf kecil (lowercase) dan menghapus elemen non-teks seperti tautan URL, mention username (`@`), tagar (`#`), karakter non-ASCII, angka, dan tanda baca.
2. **Normalisasi Slang**: Memecah kalimat menjadi token kata, kemudian mencocokkan setiap kata dengan kamus slang statis gabungan dan kamus slang dinamis (`lexicon/custom_slang.json`). Singkatan atau kata slang diubah menjadi kata baku bahasa Indonesia.
3. **Pembersihan Stopword (Stopword Removal)**: Menghapus kata-kata yang tidak memiliki makna sentimen penting (seperti "yang", "dan", "di"). Namun, sistem telah dimodifikasi untuk memproteksi kata negasi (seperti "tidak", "bukan", "belum") dari penghapusan agar tidak merusak konteks kalimat negasi.
4. **Stemming Sastrawi dengan Cache**: Mengubah kata berimbuhan menjadi kata dasar menggunakan PySastrawi. Untuk mempercepat proses stemming yang berat secara komputasi, sistem menggunakan in-memory cache (`self.stem_cache`) pada tingkat kata. Sistem juga memproteksi kata bahasa Inggris dari stemming Sastrawi agar tidak merusak ejaan aslinya.
5. **Deteksi Bahasa Dominan**: Menghitung jumlah kecocokan token kata terhadap kamus bahasa Indonesia (InSet) dan bahasa Inggris (VADER). Bahasa dengan kecocokan terbanyak dipilih sebagai bahasa dominan komentar tersebut untuk proses evaluasi sentimen selanjutnya.
6. **Pencarian Skor & Penanganan Negasi**: Mengambil skor sentimen untuk setiap token kata dari kamus terpilih (InSet berkisar $-5$ hingga $+5$, VADER berkisar $-4$ hingga $+4$). Untuk setiap kata bermakna sentimen, sistem memeriksa 2 token sebelumnya. Jika terdapat kata negasi, nilai sentimen kata tersebut dikalikan dengan $-1$ (dibalik).
7. **Klasifikasi**: Skor kumulatif dari seluruh token dijumlahkan. Nilai sentimen dikategorikan menjadi `positif` jika skor $> 0.05$, `negatif` jika skor $< -0.05$, dan `netral` untuk rentang diantaranya.

### Fitur & Kelebihan Model Lexicon-Based:
* **Optimalisasi Kecepatan Tinggi (Stemming Cache)**: Komputasi berjalan secara lokal pada CPU dengan efisiensi tinggi berkat in-memory cache stemming, memangkas durasi preprocessing hingga 70%.
* **Kustomisasi Slang Fleksibel**: Mampu menangani bahasa gaul internet secara dinamis berkat integrasi editor kamus slang yang langsung terhubung ke sistem normalisasi.
* **Akurasi Negasi Lebih Presisi**: Mampu membedakan sentimen antara kalimat "saya suka" (positif) dan "saya tidak suka" (negatif) berkat mekanisme inversi skor negasi.
* **Dukungan Campuran Bahasa (Code-Mixing)**: Otomatis mendeteksi dan mengadopsi struktur bahasa Inggris atau Indonesia berdasarkan komposisi kata dalam komentar.

---

## Model 2: LLM-Based Model (NVIDIA NIM)

### Alur Kerja Model LLM-Based

Model LLM-Based memanfaatkan pemrosesan batch untuk mengirim sekumpulan komentar ke model Llama 3.1 milik NVIDIA NIM, lalu mengurai dan memvalidasi hasilnya:

```mermaid
graph TD
    A[Daftar Komentar] --> B[Pembagian Batch - Maksimal 20 Komentar]
    B --> C[Pembuatan Prompt System & User]
    C --> D[Panggilan NVIDIA NIM API - Llama 3.1]
    D --> E[Penerimaan Respons API]
    E --> F[Pembersihan Respons - Hapus Blok Markdown & Tag Thought]
    F --> G{Validasi & Parse JSON}
    G -->|Sukses| H[Petakan Sentimen ke Comment ID]
    G -->|Gagal / Parsial| I[Fallback - Klasifikasi Satu Per Satu]
    H --> J[Daftar Hasil Sentimen Akhir]
    I --> J
```

### Penjelasan Tahapan Analisis LLM:
1. **Batching**: Komentar dikelompokkan ke dalam kelompok-kelompok kecil berisi maksimal 20 komentar per permintaan. Strategi ini meminimalkan jumlah panggilan API (API requests) dan mempercepat runtime keseluruhan.
2. **Konstruksi Prompt (Prompt Engineering)**:
   * **System Prompt**: Menginstruksikan LLM untuk berperan sebagai pakar analisis sentimen bahasa Indonesia yang memahami bahasa gaul, singkatan gaul internet, dan dialek lokal daerah (Jawa, Sunda). LLM dipaksa untuk mengembalikan output dalam format JSON array murni tanpa penjelasan tambahan.
   * **User Prompt**: Berisi daftar komentar terformat dengan indeks dan ID unik masing-masing komentar.
3. **Panggilan API (NVIDIA NIM)**: Mengirim payload pesan ke endpoint API NVIDIA NIM dengan parameter suhu rendah (`temperature: 0.2`) untuk menjaga konsistensi jawaban.
4. **Pembersihan Respons**: Sistem melakukan ekspresi reguler (regex) untuk membersihkan teks respons dari blok kode markdown (seperti ` ```json ` atau ` ``` `) serta menghapus tag pemikiran model (`<thought>...</thought>`) yang sering dihasilkan oleh model penalaran (seperti Nemotron).
5. **Parsing JSON & Fallback**:
   * Respons JSON diurai menjadi daftar objek sentimen.
   * Jika parsing JSON gagal, atau terdapat komentar dalam batch yang terlewat dari respons LLM, sistem secara otomatis mengaktifkan mekanisme **fallback** untuk menganalisis komentar bermasalah tersebut satu per satu menggunakan panggilan API individu guna memastikan keutuhan data.

### Fitur & Kelebihan Model LLM-Based:
* **Pemahaman Kontekstual Tinggi**: LLM memahami konteks kalimat secara utuh, termasuk sarkasme, metafora, dan makna implisit yang tidak dapat dideteksi oleh pencocokan kata kamus statis.
* **Dukungan Dialek & Bahasa Daerah**: Mampu memahami komentar yang ditulis dalam bahasa daerah (misal bahasa Jawa kasar/halus seperti "apik", "elek", "ora") atau campuran istilah asing.
* **Format Output Terstruktur**: Menggunakan skema JSON murni yang divalidasi secara ketat oleh sistem parser untuk integrasi langsung ke tabel database.
* **Penanganan Kegagalan Handal (Fault-Tolerance)**: Sistem fallback memastikan proses analisis tidak berhenti di tengah jalan meskipun API mengalami gangguan parsial atau menghasilkan output malformed.

---

## Prasyarat & Instalasi

Pastikan komputer Anda telah terinstal Python (versi 3.8 ke atas direkomendasikan).

1. Clone atau buka direktori proyek SEMANTIKA pada komputer Anda:
   ```powershell
   cd "c:\Users\luthf\OneDrive\Desktop\KULIAH\semester 6\STKI\tugas-sentimen-analisis"
   ```

2. Instal seluruh pustaka dependensi yang dibutuhkan:
   ```powershell
   pip install -r requirements.txt
   ```

---

## File Konfigurasi

Sebelum menjalankan aplikasi, Anda harus menyiapkan konfigurasi lingkungan.

### 1. Konfigurasi Lokal (`.env`)
Buat berkas `.env` di root direktori proyek Anda (atau salin dari `.env.example`). Isi variabel berikut:

```env
# Mode Aplikasi: 'development' (lokal murni) atau 'production' (lokal + Google Sheets sync)
APP_MODE=production

# Kunci API NVIDIA NIM (Dapatkan dari build.nvidia.com)
NVIDIA_API_KEY=your_nvidia_api_key_here

# Model NVIDIA NIM yang ingin digunakan
NVIDIA_MODEL=meta/llama-3.1-70b-instruct

# Tautan default video YouTube/Shorts yang akan dimuat di dashboard
YOUTUBE_VIDEO_URL=https://www.youtube.com/shorts/a3Irz3zv8L0

# Batas default jumlah komentar yang akan diambil
MAX_COMMENTS=100
```

### 2. Konfigurasi Kredensial Google Sheets (Opsional - Mode Production)
Jika Anda menggunakan `APP_MODE=production`, Anda perlu mengonfigurasi kredensial Google Sheets API Service Account.

* Pada lingkungan lokal, Streamlit membaca konfigurasi dari berkas `.streamlit/secrets.toml`. Berkas tersebut harus memiliki format sebagai berikut:
  ```toml
  [connections.gsheets]
  spreadsheet = "https://docs.google.com/spreadsheets/d/your_spreadsheet_id_here/edit?usp=sharing"
  type = "service_account"
  project_id = "your_gcp_project_id"
  private_key_id = "your_private_key_id"
  private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
  client_email = "your_service_account_email"
  client_id = "your_client_id"
  auth_uri = "https://accounts.google.com/o/oauth2/auth"
  token_uri = "https://oauth2.googleapis.com/token"
  auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
  client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
  ```
* Catatan: Pastikan spreadsheet Google Sheets Anda memiliki worksheet bernama `Database_Sentimen` dengan header kolom yang sesuai (`No`, `Comment ID`, `Author`, `Original Comment`, `Cleaned Comment`, `Lexicon Sentiment`, `Lexicon Score`, `LLM Sentiment`, `Ground Truth`, `Video ID`, `Video Title`, `Video URL`, `Synced At`).

---

## Panduan Penggunaan

Sistem ini menyediakan dua antarmuka penggunaan:

### Metode A: Menggunakan Dashboard Interaktif Streamlit (Sangat Direkomendasikan)

Antarmuka web interaktif menyajikan visualisasi grafis, manajemen slang, pelabelan instan, dan pencetakan PDF.

1. Jalankan server dashboard Streamlit:
   ```powershell
   streamlit run app.py
   ```
2. Aplikasi akan otomatis berjalan dan membuka peramban web Anda di alamat default `http://localhost:8501`.
3. **Mengambil Data Baru**:
   * Di panel sidebar kiri, masukkan tautan video YouTube/Shorts yang ingin dianalisis.
   * Pilih Model LLM NVIDIA yang akan digunakan.
   * Tentukan jumlah batas komentar.
   * Tekan tombol **Ambil Data & Mulai Analisis**.
4. **Melakukan Anotasi Ground Truth**:
   * Pada tab halaman utama, centang opsi **Tampilkan hanya komentar dengan prediksi berbeda (mismatch)** untuk memfilter komentar di mana prediksi Lexicon dan LLM saling berbeda.
   * Pada tabel komentar, klik dua kali kolom **Ground Truth** untuk memilih label sentimen yang benar (`positif`, `negatif`, `netral`) dari menu dropdown.
   * Data akan otomatis tersimpan ke CSV lokal (dan Google Sheets apabila dalam mode production) secara real-time.
5. **Melihat Evaluasi & Metrik**:
   * Buka tab **Evaluasi Performa & Confusion Matrix** atau **Perbandingan Cohen's Kappa** untuk melihat nilai akurasi, visualisasi Confusion Matrix, dan tingkat kesepakatan Cohen's Kappa.
   * Buka tab **Manajemen Kamus Slang** di bilah navigasi untuk memodifikasi singkatan kata.
6. **Mencetak Laporan**:
   * Tekan tombol **Cetak Laporan PDF** di dashboard untuk menghasilkan berkas laporan akademik berformat PDF secara instan.

### Metode B: Menggunakan Command Line Interface (CLI)

Gunakan metode ini jika Anda ingin menjalankan analisis dan pengujian performa secara cepat via terminal terminal/shell:

1. **Unduh Komentar & Jalankan Analisis**:
   ```powershell
   python main.py
   ```
   Script ini akan membaca video URL dari file `.env`, mengunduh komentar, menjalankan engine klasifikasi Lexicon & LLM, lalu mengekspor hasilnya ke file `sentiment_results.csv` dengan kolom Ground Truth yang masih dikosongkan.

2. **Anotasi Data**:
   Buka file `sentiment_results.csv` menggunakan Microsoft Excel, LibreOffice, atau editor teks pilihan Anda. Isi kolom `Ground Truth` secara manual dengan pilihan kata: `positif`, `negatif`, atau `netral`, kemudian simpan kembali berkas dalam format CSV.

3. **Hitung Evaluasi**:
   Jalankan perintah evaluasi untuk mencetak laporan klasifikasi detail ke terminal dan menyimpan grafik evaluasi:
   ```powershell
   python evaluate.py
   ```
   Script akan memproses baris yang memiliki isi Ground Truth, mencetak metrik performa di shell, dan menyimpan visualisasi perbandingan grafik serta Confusion Matrix kedua model ke dalam file `evaluation_metrics.png`.
