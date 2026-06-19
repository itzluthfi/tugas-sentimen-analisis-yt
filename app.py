import os
import re
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.config import OUTPUT_FILE, YOUTUBE_VIDEO_URL, MAX_COMMENTS
from src.downloader import fetch_youtube_comments, get_video_title, extract_video_id
from src.lexicon_analyzer import LexiconSentimentAnalyzer
from src.llm_analyzer import LLMSentimentAnalyzer

# Set Streamlit Page Config
st.set_page_config(
    page_title="SEMANTIKA - YouTube Sentiment Analysis Dashboard",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #fdfdfd;
    }
    
    /* Title styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Cards styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    .metric-title {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Point board styling */
    .point-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .lexicon-card {
        background: linear-gradient(135deg, #3498db, #2980b9);
    }
    .llm-card {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 15px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "meta/llama-3.1-8b-instruct"
if "youtube_url_widget" not in st.session_state:
    st.session_state.youtube_url_widget = YOUTUBE_VIDEO_URL
if "loaded_history_file" not in st.session_state:
    st.session_state.loaded_history_file = ""

# Auto-load existing results if CSV exists
if st.session_state.df is None and os.path.exists(OUTPUT_FILE):
    try:
        df_loaded = pd.read_csv(OUTPUT_FILE)
        required_cols = ["No", "Comment ID", "Author", "Original Comment", "Cleaned Comment", "Lexicon Sentiment", "Lexicon Score", "LLM Sentiment", "Ground Truth"]
        if all(col in df_loaded.columns for col in required_cols):
            df_loaded["Ground Truth"] = df_loaded["Ground Truth"].fillna("")
            st.session_state.df = df_loaded
            st.session_state.video_url = YOUTUBE_VIDEO_URL
            st.session_state.video_title = get_video_title(YOUTUBE_VIDEO_URL)
            st.session_state.youtube_url_widget = YOUTUBE_VIDEO_URL
    except Exception:
        pass

# Helper to convert DataFrame to a beautifully styled Excel file in memory
def convert_df_to_excel(df):
    output = io.BytesIO()
    
    # Rename columns to match local labels
    df_export = df.copy()
    df_export = df_export.rename(columns={
        "No": "No",
        "Author": "Penulis",
        "Original Comment": "Komentar Asli",
        "Cleaned Comment": "Komentar Bersih (Stemmed)",
        "Lexicon Sentiment": "Sentimen Lexicon",
        "LLM Sentiment": "Sentimen LLM",
        "Ground Truth": "Ground Truth"
    })
    
    # Keep only target columns in the exported spreadsheet
    cols_to_keep = ["No", "Penulis", "Komentar Asli", "Komentar Bersih (Stemmed)", "Sentimen Lexicon", "Sentimen LLM", "Ground Truth"]
    df_export = df_export[[col for col in cols_to_keep if col in df_export.columns]]
    
    # Write to Excel in memory using openpyxl
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Analisis Sentimen', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Analisis Sentimen']
        
        # Color palettes & Font settings
        header_fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid') # Slate Blue
        header_font = Font(name='Arial', size=11, bold=True, color='FFFFFF')
        
        zebra_fill = PatternFill(start_color='F2F4F8', end_color='F2F4F8', fill_type='solid') # Alternating light gray/blue
        white_fill = PatternFill(start_color='FFFFFF', end_color='FFFFFF', fill_type='solid')
        
        thin_border = Border(
            left=Side(style='thin', color='D3D3D3'),
            right=Side(style='thin', color='D3D3D3'),
            top=Side(style='thin', color='D3D3D3'),
            bottom=Side(style='thin', color='D3D3D3')
        )
        
        align_center = Alignment(horizontal='center', vertical='center')
        align_left = Alignment(horizontal='left', vertical='top', wrap_text=True)
        
        # Style Header
        for col_idx in range(1, len(df_export.columns) + 1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
            
        # Style Data Rows
        for row_idx in range(2, worksheet.max_row + 1):
            # Apply zebra striping
            row_fill = zebra_fill if row_idx % 2 == 0 else white_fill
            
            for col_idx, col_name in enumerate(df_export.columns, start=1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.fill = row_fill
                cell.border = thin_border
                
                # Column specific alignments
                if col_name in ["No", "Sentimen Lexicon", "Sentimen LLM", "Ground Truth"]:
                    cell.alignment = align_center
                else:
                    cell.alignment = align_left
                    
        # Auto-fit Column Widths based on maximum content length
        for col_idx, col_name in enumerate(df_export.columns, start=1):
            col_letter = get_column_letter(col_idx)
            
            # Start with the length of the header name
            max_len = len(str(col_name))
            
            # Find the longest value in the rows of this column
            for row_idx in range(2, worksheet.max_row + 1):
                val = str(worksheet.cell(row=row_idx, column=col_idx).value or "")
                max_len = max(max_len, len(val))
                
            # Apply dynamic sizing constraints:
            # For comment text columns, we cap the width at 60 characters so it fits on screens nicely.
            if col_name in ["Komentar Asli", "Komentar Bersih (Stemmed)"]:
                width = min(max(max_len + 3, 15), 60)
            else:
                width = max(max_len + 4, 10)
                
            worksheet.column_dimensions[col_letter].width = width
            
        # Set Row Heights
        worksheet.row_dimensions[1].height = 28 # Header taller
        for r in range(2, worksheet.max_row + 1):
            worksheet.row_dimensions[r].height = 24 # Data rows slightly taller for text wrapping room
            
    return output.getvalue()

# Sidebar Config
st.sidebar.title(":material/settings: Setelan SEMANTIKA")
st.sidebar.markdown("---")

url_input = st.sidebar.text_input(
    "URL Video YouTube / Shorts",
    key="youtube_url_widget",
    help="Masukkan URL video YouTube atau Shorts yang ingin dianalisis."
)

limit_input = st.sidebar.slider(
    "Jumlah Komentar Maksimal",
    min_value=10,
    max_value=100,
    value=MAX_COMMENTS,
    step=10,
    help="Batasi jumlah komentar yang akan ditarik."
)

model_input = st.sidebar.selectbox(
    "Model LLM NVIDIA",
    options=[
        "meta/llama-3.1-8b-instruct", 
        "meta/llama-3.1-70b-instruct", 
        "nvidia/llama-3.1-nemotron-70b-instruct"
    ],
    index=0,
    help="Llama-3.1-8b (Cepat) atau Llama-3.1-70b (Akurat)."
)

force_refresh = st.sidebar.checkbox(
    "Paksa Ambil Baru (Force Refresh)",
    value=False,
    help="Centang ini untuk mengabaikan riwayat lokal dan mengambil data baru dari YouTube & NVIDIA API."
)

# Load existing Ground Truths mapping
def get_existing_ground_truths():
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            if "Comment ID" in df_existing.columns and "Ground Truth" in df_existing.columns:
                df_existing["Ground Truth"] = df_existing["Ground Truth"].fillna("")
                return dict(zip(df_existing["Comment ID"], df_existing["Ground Truth"]))
        except Exception:
            pass
    return {}

def make_safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

btn_analyze = st.sidebar.button(":material/play_circle: Mulai Analisis Data", use_container_width=True)

# History Folder Configuration
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

if btn_analyze:
    st.session_state.loaded_history_file = ""
    if not url_input.strip():
        st.sidebar.error("Silakan masukkan URL YouTube terlebih dahulu!")
    else:
        video_id = extract_video_id(url_input)
        if not video_id:
            st.sidebar.error("Gagal mengekstrak Video ID dari URL!")
        else:
            video_title = get_video_title(url_input)
            safe_title = make_safe_filename(video_title)
            history_filename = f"[{video_id}] {safe_title}.csv"
            history_path = os.path.join(HISTORY_DIR, history_filename)

            if not force_refresh and os.path.exists(history_path):
                st.sidebar.info("Hasil analisis ditemukan di riwayat lokal. Memuat...")
                try:
                    df_loaded = pd.read_csv(history_path)
                    df_loaded["Ground Truth"] = df_loaded["Ground Truth"].fillna("")
                    df_loaded.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                    
                    st.session_state.df = df_loaded
                    st.session_state.video_title = video_title
                    st.session_state.video_url = url_input
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Gagal memuat file riwayat: {e}")
            else:
                with st.status("Menjalankan Analisis Sentimen...", expanded=True) as status:
                    # 1. Fetch Title
                    status.write("Langkah 1/5: Mengambil informasi video YouTube...")
                    video_title = get_video_title(url_input)
                    safe_title = make_safe_filename(video_title)
                    history_filename = f"[{video_id}] {safe_title}.csv"
                    history_path = os.path.join(HISTORY_DIR, history_filename)
                    
                    # 2. Fetch Comments
                    status.write("Langkah 2/5: Mengunduh komentar dari YouTube...")
                    comments = fetch_youtube_comments(url_input, limit=limit_input)
                    
                    if not comments:
                        status.update(label="Gagal mengambil komentar!", state="error", expanded=True)
                        st.error("Gagal mendapatkan komentar dari video ini.")
                    else:
                        status.write(f"   - Sukses mengunduh {len(comments)} komentar.")
                        
                        # 3. Analyze Lexicon
                        status.write("Langkah 3/5: Menjalankan pemrosesan Sastrawi & skoring Lexicon...")
                        lexicon_analyzer = LexiconSentimentAnalyzer()
                        processed_comments = []
                        
                        for idx, c in enumerate(comments):
                            sentiment, score, cleaned_text = lexicon_analyzer.analyze_sentiment(c["text"])
                            processed_comments.append({
                                "comment_id": c["comment_id"],
                                "author": c["author"],
                                "original_comment": c["text"],
                                "cleaned_comment": cleaned_text,
                                "lexicon_sentiment": sentiment,
                                "lexicon_score": score
                            })
                            if (idx + 1) % 10 == 0 or (idx + 1) == len(comments):
                                status.write(f"   - Selesai memproses Lexicon: {idx + 1}/{len(comments)} komentar...")
                        
                        # 4. Analyze LLM
                        status.write(f"Langkah 4/5: Menghubungi NVIDIA NIM API ({model_input})...")
                        llm_analyzer = LLMSentimentAnalyzer(model=model_input)
                        batch_size = 20
                        llm_sentiment_map = {}
                        num_batches = (len(comments) - 1) // batch_size + 1
                        
                        for batch_idx, i in enumerate(range(0, len(comments), batch_size)):
                            batch = comments[i:i+batch_size]
                            status.write(f"   - Mengirim LLM Batch {batch_idx + 1}/{num_batches}...")
                            batch_results = llm_analyzer.analyze_batch(batch)
                            for r in batch_results:
                                llm_sentiment_map[r["comment_id"]] = r["llm_sentiment"]
                        
                        # 5. Combine results and map existing ground truths
                        status.write("Langkah 5/5: Menyimpan berkas hasil...")
                        existing_gts = get_existing_ground_truths()
                        
                        final_data = []
                        for idx, c in enumerate(processed_comments):
                            cid = c["comment_id"]
                            gt = existing_gts.get(cid, "")
                            final_data.append({
                                "No": idx + 1,
                                "Comment ID": cid,
                                "Author": c["author"],
                                "Original Comment": c["original_comment"],
                                "Cleaned Comment": c["cleaned_comment"],
                                "Lexicon Sentiment": c["lexicon_sentiment"],
                                "Lexicon Score": c["lexicon_score"],
                                "LLM Sentiment": llm_sentiment_map.get(cid, "netral"),
                                "Ground Truth": gt
                            })
                            
                        df = pd.DataFrame(final_data)
                        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                        df.to_csv(history_path, index=False, encoding="utf-8-sig")
                        
                        st.session_state.df = df
                        st.session_state.video_title = video_title
                        st.session_state.video_url = url_input
                        st.session_state.llm_model = model_input
                        
                        status.update(label="Analisis sentimen berhasil diselesaikan!", state="complete", expanded=False)
                        st.rerun()

# Sidebar: History Loading Section
st.sidebar.markdown("---")
st.sidebar.subheader(":material/history: Riwayat Analisis")

history_files = []
if os.path.exists(HISTORY_DIR):
    history_files = sorted(
        [f for f in os.listdir(HISTORY_DIR) if f.endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIR, x)),
        reverse=True
    )

if history_files:
    history_options = ["-- Pilih untuk memuat --"] + history_files
    selected_history = st.sidebar.selectbox(
        "Muat Hasil Sebelumnya",
        options=history_options,
        index=0,
        help="Muat hasil analisis secara instan dari lokal disk."
    )
    
    if selected_history != "-- Pilih untuk memuat --" and selected_history != st.session_state.loaded_history_file:
        history_path = os.path.join(HISTORY_DIR, selected_history)
        try:
            df_loaded = pd.read_csv(history_path)
            df_loaded["Ground Truth"] = df_loaded["Ground Truth"].fillna("")
            df_loaded.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            
            filename_clean = selected_history[:-4]
            match = re.match(r"^\[(.*?)\] (.*)$", filename_clean)
            if match:
                vid_id = match.group(1)
                vid_title = match.group(2)
                new_url = f"https://www.youtube.com/watch?v={vid_id}"
            else:
                new_url = YOUTUBE_VIDEO_URL
                vid_title = filename_clean
                
            st.session_state.video_url = new_url
            st.session_state.video_title = vid_title
            st.session_state.df = df_loaded
            st.session_state.llm_model = "Riwayat"
            st.session_state.youtube_url_widget = new_url  # Update sidebar widget
            st.session_state.loaded_history_file = selected_history  # Mark file as loaded
            st.sidebar.success("Berhasil memuat data riwayat!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Gagal memuat: {e}")
else:
    st.sidebar.info("Belum ada riwayat analisis.")

# Main Dashboard Area (SEMANTIKA)
st.markdown("<h1><span style='color:#3498db'>SEMAN</span><span style='color:#2ecc71'>TIKA</span> : Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Aplikasi perbandingan performa analisis sentimen berbasis **Lexicon-based (Sastrawi + InSet)** dan **LLM-based (NVIDIA NIM Llama 3.1)**.")
st.markdown("---")

if st.session_state.df is not None:
    # Header: Video Info
    st.markdown(f"### :material/movie: **{st.session_state.video_title}**")
    st.markdown(f":material/link: **Link Video:** [{st.session_state.video_url}]({st.session_state.video_url})")
    
    st.markdown("---")
    
    st.subheader(":material/table_chart: Tabel Komentar & Penentuan Ground Truth")
    st.info("Instruksi: Silakan klik dua kali pada kolom Ground Truth untuk menentukan sentimen sebenarnya (positif, negatif, netral) menggunakan dropdown. Grafik dan poin performa di bawah akan ter-update secara otomatis.", icon=":material/info:")
    
    display_df = st.session_state.df.copy()
    
    edited_df = st.data_editor(
        display_df,
        column_config={
            "Ground Truth": st.column_config.SelectboxColumn(
                "Ground Truth",
                help="Sentimen sebenarnya yang ditentukan oleh Anda",
                options=["positif", "negatif", "netral", ""],
                required=False
            ),
            "No": st.column_config.NumberColumn("No", width="small", disabled=True),
            "Author": st.column_config.TextColumn("Penulis", width="medium", disabled=True),
            "Original Comment": st.column_config.TextColumn("Komentar Asli", width="large", disabled=True),
            "Cleaned Comment": st.column_config.TextColumn("Komentar Bersih (Stemmed)", width="medium", disabled=True),
            "Lexicon Sentiment": st.column_config.TextColumn("Sentimen Lexicon", width="small", disabled=True),
            "LLM Sentiment": st.column_config.TextColumn("Sentimen LLM", width="small", disabled=True),
        },
        column_order=["No", "Author", "Original Comment", "Cleaned Comment", "Lexicon Sentiment", "LLM Sentiment", "Ground Truth"],
        use_container_width=True,
        key="data_editor",
        num_rows="fixed"
    )
    
    # Auto-save changes
    if not edited_df.equals(st.session_state.df):
        st.session_state.df = edited_df.copy()
        edited_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        
        # Sync with history
        video_id = extract_video_id(st.session_state.video_url)
        if video_id:
            safe_title = make_safe_filename(st.session_state.video_title)
            history_filename = f"[{video_id}] {safe_title}.csv"
            history_path = os.path.join(HISTORY_DIR, history_filename)
            edited_df.to_csv(history_path, index=False, encoding="utf-8-sig")
            
        st.rerun()

    # Download Button Section
    st.markdown(" ")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        # Generate styled Excel file
        excel_data = convert_df_to_excel(st.session_state.df)
        st.download_button(
            label=":material/download: Ekspor Laporan Excel Berwarna (.xlsx)",
            data=excel_data,
            file_name=f"semantika_hasil_{extract_video_id(st.session_state.video_url)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col_dl2:
        csv_download = st.session_state.df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label=":material/download: Ekspor Data CSV Standar (.csv)",
            data=csv_download,
            file_name=f"semantika_hasil_{extract_video_id(st.session_state.video_url)}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Section 3: Live Evaluation
    st.markdown("---")
    st.subheader(":material/trending_up: Evaluasi Performa Real-Time")
    
    # Filter rows with Ground Truth
    df_eval = st.session_state.df.dropna(subset=["Ground Truth"]).copy()
    df_eval = df_eval[df_eval["Ground Truth"].astype(str).str.strip().str.lower().isin(["positif", "negatif", "netral"])]
    
    if len(df_eval) == 0:
        st.warning("Belum ada Ground Truth yang diisi. Silakan isi beberapa baris pada kolom Ground Truth di tabel atas untuk memunculkan evaluasi metrik akurasi.", icon=":material/warning:")
    else:
        st.success(f"Menghitung performa berdasarkan **{len(df_eval)}** komentar yang telah dilabeli Ground Truth.", icon=":material/check_circle:")
        
        y_true = df_eval["Ground Truth"].str.strip().str.lower()
        y_lexicon = df_eval["Lexicon Sentiment"].str.strip().str.lower()
        y_llm = df_eval["LLM Sentiment"].str.strip().str.lower()
        
        # Standard ML Scores
        lex_acc = accuracy_score(y_true, y_lexicon)
        lex_prec, lex_rec, lex_f1, _ = precision_recall_fscore_support(y_true, y_lexicon, average='macro', zero_division=0)
        
        llm_acc = accuracy_score(y_true, y_llm)
        llm_prec, llm_rec, llm_f1, _ = precision_recall_fscore_support(y_true, y_llm, average='macro', zero_division=0)
        
        # Calculate SEMANTIKA Points System
        # Rules:
        # - If Ground Truth is "netral": point change is 0 (tidak terjadi apa-apa)
        # - If Ground Truth is "positif" or "negatif":
        #   - If Model == Ground Truth: +1 point
        #   - If Model != Ground Truth: -1 point
        lex_points = 0
        llm_points = 0
        
        for idx, row in df_eval.iterrows():
            gt = str(row["Ground Truth"]).strip().lower()
            lex = str(row["Lexicon Sentiment"]).strip().lower()
            llm = str(row["LLM Sentiment"]).strip().lower()
            
            if gt in ["positif", "negatif"]:
                # Lexicon
                if lex == gt:
                    lex_points += 1
                else:
                    lex_points -= 1
                
                # LLM
                if llm == gt:
                    llm_points += 1
                else:
                    llm_points -= 1

        # Render Points Comparison UI Cards
        col_pts1, col_pts2 = st.columns(2)
        with col_pts1:
            st.markdown(
                f"""
                <div class="point-card lexicon-card">
                    <h3>POIN PERFORMA LEXICON</h3>
                    <div style="font-size: 3rem; font-weight: 800; margin: 10px 0;">{lex_points}</div>
                    <p>Metode Sastrawi + InSet Lexicon</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_pts2:
            st.markdown(
                f"""
                <div class="point-card llm-card">
                    <h3>POIN PERFORMA LLM</h3>
                    <div style="font-size: 3rem; font-weight: 800; margin: 10px 0;">{llm_points}</div>
                    <p>NVIDIA NIM ({st.session_state.llm_model.split('/')[-1]})</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Point Rules Explanation
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Sistem Poin Komparasi SEMANTIKA:</strong><br/>
            Sistem poin di atas dihitung dengan ketentuan:
            <ul>
                <li>Setiap data Ground Truth yang bernilai <strong>positif</strong> atau <strong>negatif</strong> akan dievaluasi.</li>
                <li>Jika tebakan model <strong>Benar (sesuai Ground Truth)</strong> $\rightarrow$ Model mendapatkan <strong>+1 Poin</strong>.</li>
                <li>Jika tebakan model <strong>Salah</strong> $\rightarrow$ Model dikurangi <strong>-1 Poin</strong>.</li>
                <li>Jika Ground Truth bernilai <strong>netral</strong> $\rightarrow$ <strong>0 Poin (tidak terjadi apa-apa)</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Tab Layout for Visualizations
        tab1, tab2 = st.tabs([":material/donut_large: Sebaran Sentimen (Donut Charts)", ":material/grid_on: Matrik Kebingungan (Confusion Matrices)"])
        
        # Tab 1: Donut Charts
        with tab1:
            st.markdown("### Perbandingan Sebaran Sentimen (Donut Charts)")
            
            # Count sentiment frequencies for each category
            sentiment_labels = ["positif", "negatif", "netral"]
            color_map = {
                "positif": "#2ecc71",  # Green
                "negatif": "#e74c3c",  # Red
                "netral": "#95a5a6"   # Gray
            }
            
            # Helper to calculate sizes
            def get_sizes_and_colors(series):
                counts = series.value_counts()
                sizes = []
                colors = []
                labels = []
                for label in sentiment_labels:
                    count = counts.get(label, 0)
                    if count > 0:
                        sizes.append(count)
                        colors.append(color_map[label])
                        labels.append(label.capitalize())
                return sizes, colors, labels
                
            gt_sizes, gt_colors, gt_labels = get_sizes_and_colors(y_true)
            lex_sizes, lex_colors, lex_labels = get_sizes_and_colors(y_lexicon)
            llm_sizes, llm_colors, llm_labels = get_sizes_and_colors(y_llm)
            
            # Plot the 3 donut charts side-by-side
            fig_donut, (ax_d1, ax_d2, ax_d3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Donut 1: Ground Truth
            if gt_sizes:
                wedges, texts, autotexts = ax_d1.pie(gt_sizes, labels=gt_labels, autopct='%1.1f%%', startangle=90, colors=gt_colors, pctdistance=0.75, textprops=dict(color="black", weight="bold"))
                centre_circle = plt.Circle((0,0), 0.50, fc='white')
                ax_d1.add_artist(centre_circle)
                ax_d1.set_title("Sebaran Ground Truth", fontsize=12, weight="bold")
            else:
                ax_d1.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
                
            # Donut 2: Lexicon
            if lex_sizes:
                wedges, texts, autotexts = ax_d2.pie(lex_sizes, labels=lex_labels, autopct='%1.1f%%', startangle=90, colors=lex_colors, pctdistance=0.75, textprops=dict(color="black", weight="bold"))
                centre_circle = plt.Circle((0,0), 0.50, fc='white')
                ax_d2.add_artist(centre_circle)
                ax_d2.set_title("Sebaran Lexicon-based", fontsize=12, weight="bold")
            else:
                ax_d2.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
                
            # Donut 3: LLM
            if llm_sizes:
                wedges, texts, autotexts = ax_d3.pie(llm_sizes, labels=llm_labels, autopct='%1.1f%%', startangle=90, colors=llm_colors, pctdistance=0.75, textprops=dict(color="black", weight="bold"))
                centre_circle = plt.Circle((0,0), 0.50, fc='white')
                ax_d3.add_artist(centre_circle)
                ax_d3.set_title(f"Sebaran LLM-based\n({st.session_state.llm_model.split('/')[-1]})", fontsize=12, weight="bold")
            else:
                ax_d3.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
                
            plt.tight_layout()
            st.pyplot(fig_donut)
            plt.close()

        # Tab 2: Confusion Matrices & Standard Metrics
        with tab2:
            st.markdown("### Confusion Matrices & Performa Klasifikasi")
            
            # Display metrics columns
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### 📘 Performa Lexicon (Sastrawi + InSet)")
                st.write(f"- **Accuracy (Akurasi)**: {lex_acc * 100:.2f}%")
                st.write(f"- **Precision (Presisi)**: {lex_prec * 100:.2f}%")
                st.write(f"- **Recall (Sensitivitas)**: {lex_rec * 100:.2f}%")
                st.write(f"- **F1-Score**: {lex_f1 * 100:.2f}%")
                
            with col_m2:
                st.markdown(f"#### 🟢 Performa LLM ({st.session_state.llm_model.split('/')[-1]})")
                st.write(f"- **Accuracy (Akurasi)**: {llm_acc * 100:.2f}%")
                st.write(f"- **Precision (Presisi)**: {llm_prec * 100:.2f}%")
                st.write(f"- **Recall (Sensitivitas)**: {llm_rec * 100:.2f}%")
                st.write(f"- **F1-Score**: {llm_f1 * 100:.2f}%")
            
            # Plot Confusion Matrices Side by Side
            fig_cm, (ax_cm1, ax_cm2) = plt.subplots(1, 2, figsize=(15, 6))
            labels_present = sorted(list(set(y_true.unique()) | set(y_lexicon.unique()) | set(y_llm.unique())))
            
            # Lexicon CM
            cm_lex = confusion_matrix(y_true, y_lexicon, labels=labels_present)
            disp_lex = ConfusionMatrixDisplay(confusion_matrix=cm_lex, display_labels=labels_present)
            disp_lex.plot(ax=ax_cm1, cmap=plt.cm.Blues, values_format='d')
            ax_cm1.set_title("Confusion Matrix: Lexicon-based", weight="bold")
            
            # LLM CM
            cm_llm = confusion_matrix(y_true, y_llm, labels=labels_present)
            disp_llm = ConfusionMatrixDisplay(confusion_matrix=cm_llm, display_labels=labels_present)
            disp_llm.plot(ax=ax_cm2, cmap=plt.cm.Greens, values_format='d')
            ax_cm2.set_title("Confusion Matrix: LLM-based", weight="bold")
            
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

else:
    # Welcome message with clean CSS styling
    st.markdown(
        """
        <div class="info-box" style="border-left-color: #3498db;">
            <h3>Selamat datang di SEMANTIKA!</h3>
            <p>Silakan gunakan panel konfigurasi di sidebar kiri untuk menghubungkan dashboard dengan YouTube dan memulai analisis sentimen.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    ### Langkah Memulai:
    1. Pastikan file `.env` Anda sudah terisi dengan **NVIDIA API Key** yang valid.
    2. Masukkan URL video YouTube/Shorts di panel konfigurasi sebelah kiri.
    3. Tentukan batas jumlah komentar (contoh: 100).
    4. Pilih model LLM yang ingin digunakan (disarankan: `meta/llama-3.1-8b-instruct` untuk pemrosesan cepat).
    5. Klik tombol **Mulai Analisis Data**.
    
    Aplikasi akan mengunduh komentar, menganalisis dengan kedua metode, dan menampilkan tabel interaktif untuk pengisian **Ground Truth**.
    """)
