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

from src.config import OUTPUT_FILE, YOUTUBE_VIDEO_URL, MAX_COMMENTS, HISTORY_DIR
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
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = "id"
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
            if "LLM Model" in df_loaded.columns:
                st.session_state.llm_model = str(df_loaded["LLM Model"].iloc[0])
            else:
                df_loaded["LLM Model"] = "meta/llama-3.1-8b-instruct"
                st.session_state.llm_model = "meta/llama-3.1-8b-instruct"
            if "Language" in df_loaded.columns:
                st.session_state.detected_lang = str(df_loaded["Language"].iloc[0])
            else:
                df_loaded["Language"] = "id"
                st.session_state.detected_lang = "id"
            st.session_state.df = df_loaded
            st.session_state.video_url = YOUTUBE_VIDEO_URL
            st.session_state.video_title = get_video_title(YOUTUBE_VIDEO_URL)
            st.session_state.youtube_url_widget = YOUTUBE_VIDEO_URL
    except Exception:
        pass

# Helper to convert DataFrame to a beautifully styled Excel file in memory
def convert_df_to_excel(df, video_title, video_url):
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
    
    # Column Widths mapping (defined early for row height estimates)
    col_widths = {
        "No": 6,
        "Penulis": 18,
        "Komentar Asli": 50,
        "Komentar Bersih (Stemmed)": 40,
        "Sentimen Lexicon": 18,
        "Sentimen LLM": 18,
        "Ground Truth": 18
    }
    
    # Write to Excel in memory using openpyxl, start data at Row 7 (startrow=6)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Analisis Sentimen', index=False, startrow=6)
        workbook = writer.book
        worksheet = writer.sheets['Analisis Sentimen']
        
        # Write Report Title Headers at the top (Rows 1-5)
        worksheet.cell(row=1, column=1, value="SEMANTIKA - Laporan Analisis Sentimen Komentar YouTube")
        worksheet.cell(row=1, column=1).font = Font(name='Arial', size=15, bold=True, color='1F4E79')
        
        worksheet.cell(row=2, column=1, value=f"Judul Video: {video_title}")
        worksheet.cell(row=2, column=1).font = Font(name='Arial', size=10, bold=True)
        
        worksheet.cell(row=3, column=1, value=f"Link Video: {video_url}")
        worksheet.cell(row=3, column=1).font = Font(name='Arial', size=10, color='2563EB', underline='single')
        
        worksheet.cell(row=4, column=1, value=f"Model LLM: {st.session_state.llm_model}")
        worksheet.cell(row=4, column=1).font = Font(name='Arial', size=10, bold=True)
        
        lang_label = "Inggris (EN)" if st.session_state.detected_lang == "en" else "Indonesia (ID)"
        worksheet.cell(row=5, column=1, value=f"Bahasa Terdeteksi: {lang_label}")
        worksheet.cell(row=5, column=1).font = Font(name='Arial', size=10, bold=True)
        
        # Color palettes & Font settings for Table
        header_fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid') # Slate Blue
        header_font = Font(name='Arial', size=11, bold=True, color='FFFFFF')
        
        zebra_fill = PatternFill(start_color='F2F4F8', end_color='F2F4F8', fill_type='solid') # Alternating light gray/blue
        white_fill = PatternFill(start_color='FFFFFF', end_color='FFFFFF', fill_type='solid')
        
        # Soft fills and colors for sentiment states
        positif_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid') # Light Green
        positif_font = Font(name='Arial', size=10, color='006100', bold=True)
        
        negatif_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid') # Light Red
        negatif_font = Font(name='Arial', size=10, color='9C0006', bold=True)
        
        netral_fill = PatternFill(start_color='E2E3E5', end_color='E2E3E5', fill_type='solid')  # Light Gray
        netral_font = Font(name='Arial', size=10, color='383D41', bold=False)
        
        thin_border = Border(
            left=Side(style='thin', color='D3D3D3'),
            right=Side(style='thin', color='D3D3D3'),
            top=Side(style='thin', color='D3D3D3'),
            bottom=Side(style='thin', color='D3D3D3')
        )
        
        align_center = Alignment(horizontal='center', vertical='center')
        align_left = Alignment(horizontal='left', vertical='top', wrap_text=True)
        
        # Style Table Header (Row 7)
        for col_idx in range(1, len(df_export.columns) + 1):
            cell = worksheet.cell(row=7, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
            
        # Style Table Data Rows (Row 8 onwards) & Calculate Row Heights dynamically
        for row_idx in range(8, worksheet.max_row + 1):
            # Apply zebra striping
            row_fill = zebra_fill if row_idx % 2 == 0 else white_fill
            
            max_lines = 1
            for col_idx, col_name in enumerate(df_export.columns, start=1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.fill = row_fill
                cell.border = thin_border
                
                # Column specific alignments
                if col_name in ["No", "Sentimen Lexicon", "Sentimen LLM", "Ground Truth"]:
                    cell.alignment = align_center
                else:
                    cell.alignment = align_left
                    
                # Apply conditional formatting for sentiment columns
                if col_name in ["Sentimen Lexicon", "Sentimen LLM", "Ground Truth"]:
                    val_lower = str(cell.value or "").strip().lower()
                    if val_lower == "positif":
                        cell.fill = positif_fill
                        cell.font = positif_font
                    elif val_lower == "negatif":
                        cell.fill = negatif_fill
                        cell.font = negatif_font
                    elif val_lower == "netral":
                        cell.fill = netral_fill
                        cell.font = netral_font
                
                # Estimate necessary row height dynamically by analyzing wrapped lines
                val = str(cell.value or "")
                val_lines = val.split('\n')
                width = col_widths.get(col_name, 15)
                # Count wrapped lines for this cell
                lines_in_cell = sum(max(1, int(np.ceil(len(l) / width))) for l in val_lines)
                max_lines = max(max_lines, lines_in_cell)
            
            # Set dynamic height: 14pt per line + 12pt padding (min height 20pt)
            worksheet.row_dimensions[row_idx].height = max(20, max_lines * 14 + 12)
            
        # Set Column Widths
        for col_idx, col_name in enumerate(df_export.columns, start=1):
            col_letter = get_column_letter(col_idx)
            width = col_widths.get(col_name, 15)
            worksheet.column_dimensions[col_letter].width = width
            
        # Set Header Row Heights
        worksheet.row_dimensions[1].height = 24
        worksheet.row_dimensions[2].height = 18
        worksheet.row_dimensions[3].height = 18
        worksheet.row_dimensions[4].height = 18 # Model LLM
        worksheet.row_dimensions[5].height = 18 # Language
        worksheet.row_dimensions[6].height = 12 # Empty spacer row
        worksheet.row_dimensions[7].height = 28 # Table header
            
    return output.getvalue()

# Helper to convert DataFrame to a beautifully styled landscape A4 PDF report in memory
def convert_df_to_pdf(df, video_title, video_url):
    output = io.BytesIO()
    
    # Setup landscape A4 document
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom heading & metadata styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#1F4E79'),
        spaceAfter=12
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )
    
    meta_link_style = ParagraphStyle(
        'MetaLink',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=8
    )
    
    # Append report title & metadata headers
    story.append(Paragraph("SEMANTIKA - Laporan Analisis Sentimen Komentar YouTube", title_style))
    story.append(Paragraph(f"Judul Video: {video_title}", meta_style))
    story.append(Paragraph(f"Link Video: <font color='#2563EB'><u>{video_url}</u></font>", meta_link_style))
    story.append(Paragraph(f"Model LLM: {st.session_state.llm_model}", meta_style))
    lang_label = "Inggris (EN)" if st.session_state.detected_lang == "en" else "Indonesia (ID)"
    story.append(Paragraph(f"Bahasa Terdeteksi: {lang_label}", meta_style))
    story.append(Spacer(1, 15))
    
    # Column specific table styles (with wrap_text and alignment)
    th_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.white,
        alignment=1 # Centered
    )
    
    td_center_style = ParagraphStyle(
        'TableCellCenter',
        fontName='Helvetica',
        fontSize=8,
        textColor=colors.HexColor('#0F172A'),
        alignment=1 # Centered
    )
    
    td_left_style = ParagraphStyle(
        'TableCellLeft',
        fontName='Helvetica',
        fontSize=8,
        textColor=colors.HexColor('#0F172A'),
        alignment=0 # Left-aligned
    )
    
    # Table headers row
    headers = [
        Paragraph("<b>No</b>", th_style),
        Paragraph("<b>Penulis</b>", th_style),
        Paragraph("<b>Komentar Asli</b>", th_style),
        Paragraph("<b>Komentar Bersih (Stemmed)</b>", th_style),
        Paragraph("<b>Sentimen Lexicon</b>", th_style),
        Paragraph("<b>Sentimen LLM</b>", th_style),
        Paragraph("<b>Ground Truth</b>", th_style)
    ]
    
    table_data = [headers]
    
    # Format and append rows as Paragraph cells to allow word wrapping
    for idx, row in df.iterrows():
        no_p = Paragraph(str(idx + 1), td_center_style)
        author_p = Paragraph(str(row.get("Author", "")), td_left_style)
        orig_p = Paragraph(str(row.get("Original Comment", "")), td_left_style)
        clean_p = Paragraph(str(row.get("Cleaned Comment", "")), td_left_style)
        lex_p = Paragraph(str(row.get("Lexicon Sentiment", "")).capitalize(), td_center_style)
        llm_p = Paragraph(str(row.get("LLM Sentiment", "")).capitalize(), td_center_style)
        
        gt_val = str(row.get("Ground Truth", ""))
        gt_p = Paragraph(gt_val.capitalize() if gt_val else "-", td_center_style)
        
        table_data.append([no_p, author_p, orig_p, clean_p, lex_p, llm_p, gt_p])
        
    # Printable landscape A4 width is 781.89 points. Sum of columns: 780 pt.
    col_widths = [30, 95, 245, 210, 65, 65, 70]
    
    # Create Table object
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Add table styling (background fills, margins, lines)
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F4E79')), # Header fill Slate Blue
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')), # Light gray gridlines
    ]
    
    # Apply zebra-striping rows and conditional background colors for sentiment columns
    for r in range(1, len(table_data)):
        bg_color = colors.HexColor('#F8FAFC') if r % 2 == 0 else colors.white
        t_style.append(('BACKGROUND', (0, r), (-1, r), bg_color))
        t_style.append(('TOPPADDING', (0, r), (-1, r), 5))
        t_style.append(('BOTTOMPADDING', (0, r), (-1, r), 5))
        
        # Get matching DataFrame row (header is row 0 in table_data)
        row = df.iloc[r - 1]
        
        # Lexicon Sentiment (Column index 4)
        lex_val = str(row.get("Lexicon Sentiment", "")).strip().lower()
        if lex_val == "positif":
            t_style.append(('BACKGROUND', (4, r), (4, r), colors.HexColor('#C6EFCE')))
        elif lex_val == "negatif":
            t_style.append(('BACKGROUND', (4, r), (4, r), colors.HexColor('#FFC7CE')))
        elif lex_val == "netral":
            t_style.append(('BACKGROUND', (4, r), (4, r), colors.HexColor('#E2E3E5')))
            
        # LLM Sentiment (Column index 5)
        llm_val = str(row.get("LLM Sentiment", "")).strip().lower()
        if llm_val == "positif":
            t_style.append(('BACKGROUND', (5, r), (5, r), colors.HexColor('#C6EFCE')))
        elif llm_val == "negatif":
            t_style.append(('BACKGROUND', (5, r), (5, r), colors.HexColor('#FFC7CE')))
        elif llm_val == "netral":
            t_style.append(('BACKGROUND', (5, r), (5, r), colors.HexColor('#E2E3E5')))
            
        # Ground Truth (Column index 6)
        gt_val = str(row.get("Ground Truth", "")).strip().lower()
        if gt_val == "positif":
            t_style.append(('BACKGROUND', (6, r), (6, r), colors.HexColor('#C6EFCE')))
        elif gt_val == "negatif":
            t_style.append(('BACKGROUND', (6, r), (6, r), colors.HexColor('#FFC7CE')))
        elif gt_val == "netral":
            t_style.append(('BACKGROUND', (6, r), (6, r), colors.HexColor('#E2E3E5')))
        
    t.setStyle(TableStyle(t_style))
    story.append(t)
    
    # Compile document
    doc.build(story)
    return output.getvalue()

# Sidebar Navigation Menu
st.sidebar.title(":material/explore: Navigasi Menu")
menu_selection = st.sidebar.radio(
    "Pilih Halaman:",
    options=["Analisis Video Tunggal", "Analisis Perbandingan Global"],
    index=0
)
st.sidebar.markdown("---")

def get_existing_ground_truths():
    """
    Mengambil ground truth yang sudah diisi sebelumnya dari file hasil aktif (OUTPUT_FILE)
    maupun seluruh berkas riwayat di folder history, agar label manual tidak hilang saat refresh.
    """
    gts = {}
    
    # 1. Baca dari sentiment_results.csv jika ada
    if os.path.exists(OUTPUT_FILE):
        try:
            df_old = pd.read_csv(OUTPUT_FILE)
            if "Comment ID" in df_old.columns and "Ground Truth" in df_old.columns:
                df_old = df_old.dropna(subset=["Comment ID"])
                for _, row in df_old.iterrows():
                    cid = str(row["Comment ID"]).strip()
                    gt = str(row["Ground Truth"]).strip() if pd.notna(row["Ground Truth"]) else ""
                    if cid and gt:
                        gts[cid] = gt
        except Exception:
            pass

    # 2. Baca dari folder history
    if os.path.exists(HISTORY_DIR):
        try:
            for f in os.listdir(HISTORY_DIR):
                if f.endswith(".csv"):
                    fpath = os.path.join(HISTORY_DIR, f)
                    df_hist = pd.read_csv(fpath)
                    if "Comment ID" in df_hist.columns and "Ground Truth" in df_hist.columns:
                        df_hist = df_hist.dropna(subset=["Comment ID"])
                        for _, row in df_hist.iterrows():
                            cid = str(row["Comment ID"]).strip()
                            gt = str(row["Ground Truth"]).strip() if pd.notna(row["Ground Truth"]) else ""
                            if cid and gt:
                                gts[cid] = gt
        except Exception:
            pass
            
    return gts

def make_safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def detect_language_from_title(title):
    """
    Detects if the video title is primarily Indonesian ('id') or English ('en') using NVIDIA NIM.
    """
    detector = LLMSentimentAnalyzer(model="meta/llama-3.1-8b-instruct")
    system_prompt = (
        "You are a language detection assistant. Detect whether the following YouTube video title is primarily "
        "in Indonesian (or Indonesian slang/slang) or English.\n"
        "Respond with only 'id' for Indonesian/slang or 'en' for English. Do not write any other words or characters."
    )
    user_prompt = f"Video Title: {title}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = detector._call_nvidia_api(messages)
        res_clean = response.strip().lower()
        if "en" in res_clean:
            return "en"
        return "id"
    except Exception:
        return "id"

if menu_selection == "Analisis Video Tunggal":
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
    
    options_list = [
        "meta/llama-3.1-8b-instruct", 
        "meta/llama-3.1-70b-instruct", 
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "deepseek-ai/deepseek-r1",
        "deepseek-ai/deepseek-r1-distill-qwen-32b",
        "qwen/qwen-2.5-72b-instruct",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it"
    ]
    default_index = 0
    if st.session_state.llm_model in options_list:
        default_index = options_list.index(st.session_state.llm_model)
    
    model_input = st.sidebar.selectbox(
        "Model LLM NVIDIA",
        options=options_list,
        index=default_index,
        help="Pilih model NVIDIA NIM yang ingin digunakan untuk klasifikasi."
    )
    
    force_refresh = st.sidebar.checkbox(
        "Paksa Ambil Baru (Force Refresh)",
        value=False,
        help="Centang ini untuk mengabaikan riwayat lokal dan mengambil data baru dari YouTube & NVIDIA API."
    )
    
    btn_analyze = st.sidebar.button(":material/play_circle: Mulai Analisis Data", use_container_width=True)
    
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
                        if "LLM Model" in df_loaded.columns:
                            st.session_state.llm_model = str(df_loaded["LLM Model"].iloc[0])
                        else:
                            df_loaded["LLM Model"] = "meta/llama-3.1-8b-instruct"
                            st.session_state.llm_model = "meta/llama-3.1-8b-instruct"
                        if "Language" in df_loaded.columns:
                            st.session_state.detected_lang = str(df_loaded["Language"].iloc[0])
                        else:
                            df_loaded["Language"] = "id"
                            st.session_state.detected_lang = "id"
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
                        
                        # Language Detection
                        status.write("   - Mendeteksi bahasa konten video...")
                        detected_lang = detect_language_from_title(video_title)
                        status.write(f"   - Bahasa terdeteksi: {detected_lang.upper()}")
                        
                        # 2. Fetch Comments
                        status.write("Langkah 2/5: Mengunduh komentar dari YouTube...")
                        comments = fetch_youtube_comments(url_input, limit=limit_input)
                        
                        if not comments:
                            status.update(label="Gagal mengambil komentar!", state="error", expanded=True)
                            st.error("Gagal mendapatkan komentar dari video ini.")
                        else:
                            status.write(f"   - Sukses mengunduh {len(comments)} komentar.")
                            
                            # 3. Analyze Lexicon
                            status.write("Langkah 3/5: Menjalankan pemrosesan & skoring Lexicon...")
                            lexicon_analyzer = LexiconSentimentAnalyzer()
                            processed_comments = []
                            
                            for idx, c in enumerate(comments):
                                sentiment, score, cleaned_text = lexicon_analyzer.analyze_sentiment(c["text"], lang=detected_lang)
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
                                    "LLM Model": model_input,
                                    "Language": detected_lang,
                                    "Ground Truth": gt
                                })
                                
                            df = pd.DataFrame(final_data)
                            df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                            df.to_csv(history_path, index=False, encoding="utf-8-sig")
                            
                            st.session_state.df = df
                            st.session_state.video_title = video_title
                            st.session_state.video_url = url_input
                            st.session_state.llm_model = model_input
                            st.session_state.detected_lang = detected_lang
                            
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
                if "LLM Model" in df_loaded.columns:
                    st.session_state.llm_model = str(df_loaded["LLM Model"].iloc[0])
                else:
                    df_loaded["LLM Model"] = "meta/llama-3.1-8b-instruct"
                    st.session_state.llm_model = "meta/llama-3.1-8b-instruct"
                if "Language" in df_loaded.columns:
                    st.session_state.detected_lang = str(df_loaded["Language"].iloc[0])
                else:
                    df_loaded["Language"] = "id"
                    st.session_state.detected_lang = "id"
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
                st.session_state.youtube_url_widget = new_url  # Update sidebar widget
                st.session_state.loaded_history_file = selected_history  # Mark file as loaded
                st.sidebar.success("Berhasil memuat data riwayat!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Gagal memuat: {e}")
    
        # Kelola Riwayat Expander
        with st.sidebar.expander(":material/delete: Kelola Riwayat"):
            st.markdown("<small>Centang riwayat yang ingin dihapus:</small>", unsafe_allow_html=True)
            select_all = st.checkbox("Pilih Semua", key="select_all_del")
            
            to_delete = []
            for h_file in history_files:
                clean_name = h_file.replace(".csv", "")
                match = re.match(r"^\[(.*?)\] (.*)$", clean_name)
                display_name = match.group(2) if match else clean_name
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
                
                checked = st.checkbox(display_name, value=select_all, key=f"del_{h_file}")
                if checked:
                    to_delete.append(h_file)
                    
            col_del1, col_del2 = st.columns(2)
            with col_del1:
                if st.button("Hapus Terpilih", type="primary", use_container_width=True):
                    if to_delete:
                        for h_file in to_delete:
                            file_path = os.path.join(HISTORY_DIR, h_file)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        st.sidebar.success(f"Berhasil menghapus {len(to_delete)} riwayat!")
                        if st.session_state.loaded_history_file in to_delete:
                            st.session_state.df = None
                            st.session_state.loaded_history_file = ""
                            if os.path.exists(OUTPUT_FILE):
                                os.remove(OUTPUT_FILE)
                        st.rerun()
                    else:
                        st.sidebar.warning("Pilih riwayat dulu!")
            with col_del2:
                if st.button("Hapus Semua", use_container_width=True):
                    for h_file in history_files:
                        file_path = os.path.join(HISTORY_DIR, h_file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    if os.path.exists(OUTPUT_FILE):
                        os.remove(OUTPUT_FILE)
                    st.session_state.df = None
                    st.session_state.loaded_history_file = ""
                    st.sidebar.success("Semua riwayat berhasil dihapus!")
                    st.rerun()
    else:
        st.sidebar.info("Belum ada riwayat analisis.")
else:
    # menu_selection == "Analisis Perbandingan Global"
    st.sidebar.title(":material/analytics: Perbandingan Global")
    st.sidebar.markdown("---")
    st.sidebar.info("Filter video dan toggle riwayat dikelola langsung pada panel di halaman utama.")

# Info Metodologi di Sidebar (Pojok Halaman)
st.sidebar.markdown("---")
methodology_md = """
### :material/info: Metodologi: Lexicon vs LLM

#### 1. Lexicon-based (Kamus Kata)
*   **Cara kerja:** Menjumlahkan skor/bobot sentimen kata demi kata berdasarkan kamus kosakata (*InSet* untuk ID / *VADER* untuk EN).
*   **Kelebihan:** Sangat cepat, transparan, dan tidak bergantung pada API eksternal.
*   **Kekurangan:** Tidak memahami konteks kalimat, sindiran (sarkasme), kata negasi (contoh: *"tidak jelek"* dideteksi negatif karena kata *"jelek"*), dan rentan salah jika ada kesalahan ejaan (typo) atau slang yang tidak terdaftar di kamus.

#### 2. LLM-based (Konteks AI / Semantik)
*   **Cara kerja:** Memahami keseluruhan kalimat secara utuh menggunakan kecerdasan buatan (NVIDIA NIM).
*   **Kelebihan:** Sangat pintar memahami konteks, sindiran, negasi, slang internet terbaru, singkatan ekstrim, bahasa daerah, dan bahasa campuran.
*   **Kekurangan:** Memerlukan kuota API, bergantung pada koneksi internet, dan pemrosesan sedikit lebih lambat dibanding Lexicon.
"""

if hasattr(st, "popover"):
    with st.sidebar.popover(":material/info: Info Metodologi (Lexicon vs LLM)", use_container_width=True):
        st.markdown(methodology_md)
else:
    with st.sidebar.expander(":material/info: Info Metodologi (Lexicon vs LLM)"):
        st.markdown(methodology_md)

if menu_selection == "Analisis Perbandingan Global":
    st.markdown("<h1><span style='color:#3498db'>SEMAN</span><span style='color:#2ecc71'>TIKA</span> : Perbandingan Global</h1>", unsafe_allow_html=True)
    st.markdown("Halaman analisis akumulatif yang menggabungkan seluruh atau sebagian riwayat video untuk perbandingan akurasi jangka panjang.")
    st.markdown("---")
    
    # Ambil daftar file riwayat yang ada
    history_files = sorted(
        [f for f in os.listdir(HISTORY_DIR) if f.endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIR, x)),
        reverse=True
    )
    
    # Inisialisasi daftar file aktif di session state
    if "active_global_files" not in st.session_state:
        st.session_state.active_global_files = list(history_files)
        
    # Bersihkan file yang sudah dihapus dari session state
    st.session_state.active_global_files = [f for f in st.session_state.active_global_files if f in history_files]
    
    # Tampilkan expander filter video di halaman utama
    with st.expander(":material/settings: Filter Pilihan Video (Toggle Aktivasi)", expanded=True):
        st.markdown("<small>Pilih video riwayat yang ingin Anda sertakan dalam analisis dan grafik perbandingan global:</small>", unsafe_allow_html=True)
        
        # Tombol pintasan Cepat
        col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 7])
        with col_btn1:
            if st.button("Pilih Semua", key="select_all_global_btn", use_container_width=True):
                st.session_state.active_global_files = list(history_files)
                st.rerun()
        with col_btn2:
            if st.button("Kosongkan Semua", key="deselect_all_global_btn", use_container_width=True):
                st.session_state.active_global_files = []
                st.rerun()
                
        st.markdown(" ")
        
        selected_global_files = []
        if history_files:
            cols = st.columns(3)
            for idx, h_file in enumerate(history_files):
                clean_name = h_file.replace(".csv", "")
                match = re.match(r"^\[(.*?)\] (.*)$", clean_name)
                display_name = match.group(2) if match else clean_name
                
                # Potong nama jika terlalu panjang
                if len(display_name) > 35:
                    display_name = display_name[:32] + "..."
                    
                col_idx = idx % 3
                is_checked = h_file in st.session_state.active_global_files
                
                with cols[col_idx]:
                    checked = st.checkbox(f":material/movie: {display_name}", value=is_checked, key=f"chk_glob_{h_file}")
                    if checked:
                        selected_global_files.append(h_file)
            
            st.session_state.active_global_files = selected_global_files
        else:
            st.info("Belum ada riwayat analisis untuk dibandingkan.")
            
    if not selected_global_files:
        st.warning("Silakan aktifkan minimal satu file riwayat pada filter di atas untuk memulai perbandingan global.")
    else:
        dfs = []
        video_accuracies = []
        
        for h_file in selected_global_files:
            file_path = os.path.join(HISTORY_DIR, h_file)
            try:
                df_temp = pd.read_csv(file_path)
                clean_name = h_file.replace(".csv", "")
                match = re.match(r"^\[(.*?)\] (.*)$", clean_name)
                vid_title = match.group(2) if match else clean_name
                
                df_temp["Video Title"] = vid_title
                dfs.append(df_temp)
                
                # Calculate accuracy for this video individually (only if Ground Truth is filled)
                df_eval_temp = df_temp.dropna(subset=["Ground Truth"]).copy()
                df_eval_temp = df_eval_temp[df_eval_temp["Ground Truth"].astype(str).str.strip().str.lower().isin(["positif", "negatif", "netral"])]
                if len(df_eval_temp) > 0:
                    y_true_temp = df_eval_temp["Ground Truth"].str.strip().str.lower()
                    y_lexicon_temp = df_eval_temp["Lexicon Sentiment"].str.strip().str.lower()
                    y_llm_temp = df_eval_temp["LLM Sentiment"].str.strip().str.lower()
                    
                    lex_acc_temp = accuracy_score(y_true_temp, y_lexicon_temp)
                    llm_acc_temp = accuracy_score(y_true_temp, y_llm_temp)
                    
                    model_temp = "meta/llama-3.1-8b-instruct"
                    if "LLM Model" in df_temp.columns:
                        model_temp = df_temp["LLM Model"].iloc[0]
                    model_short = model_temp.split("/")[-1] if "/" in model_temp else model_temp
                    
                    video_accuracies.append({
                        "Video": vid_title[:30] + "..." if len(vid_title) > 30 else vid_title,
                        "Lexicon Accuracy": lex_acc_temp * 100,
                        "LLM Accuracy": llm_acc_temp * 100,
                        "LLM Model": model_short
                    })
            except Exception as e:
                st.error(f"Gagal membaca {h_file}: {e}")
                
        if dfs:
            df_global = pd.concat(dfs, ignore_index=True)
            total_comments = len(df_global)
            
            # Filter for rows with Ground Truth for evaluation
            df_global_eval = df_global.dropna(subset=["Ground Truth"]).copy()
            df_global_eval = df_global_eval[df_global_eval["Ground Truth"].astype(str).str.strip().str.lower().isin(["positif", "negatif", "netral"])]
            total_eval = len(df_global_eval)
            
            # Render KPI metrics
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            with col_kpi1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Komentar Terkumpul</div>
                        <div class="metric-value">{total_comments}</div>
                        <small>Dari {len(selected_global_files)} video yang dipilih</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_kpi2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Ground Truth Terisi</div>
                        <div class="metric-value">{total_eval}</div>
                        <small>Persentase: {(total_eval / total_comments * 100) if total_comments > 0 else 0:.1f}% dari total</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_kpi3:
                # Display average accuracies if eval data is present
                if total_eval > 0:
                    y_true_g = df_global_eval["Ground Truth"].str.strip().str.lower()
                    y_lex_g = df_global_eval["Lexicon Sentiment"].str.strip().str.lower()
                    y_llm_g = df_global_eval["LLM Sentiment"].str.strip().str.lower()
                    
                    global_lex_acc = accuracy_score(y_true_g, y_lex_g) * 100
                    global_llm_acc = accuracy_score(y_true_g, y_llm_g) * 100
                    st.markdown(
                       f"""
                       <div class="metric-card">
                           <div class="metric-title">Rata-rata Akurasi Global</div>
                           <div class="metric-value" style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">
                               Lexicon: {global_lex_acc:.1f}%<br/>
                               LLM: {global_llm_acc:.1f}%
                           </div>
                       </div>
                       """,
                       unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="metric-card">
                            <div class="metric-title">Rata-rata Akurasi Global</div>
                            <div class="metric-value" style="font-size: 1.3rem; color: #64748b;">N/A</div>
                            <small>Isi Ground Truth terlebih dahulu</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            if total_eval == 0:
                st.warning("Belum ada data Ground Truth yang diisi di seluruh video yang dipilih. Metrik komparasi tidak dapat ditampilkan.")
            else:
                # Plot global donut charts
                st.subheader(":material/pie_chart: Sebaran Sentimen Akumulatif")
                
                y_true_g = df_global_eval["Ground Truth"].str.strip().str.lower()
                y_lex_g = df_global_eval["Lexicon Sentiment"].str.strip().str.lower()
                y_llm_g = df_global_eval["LLM Sentiment"].str.strip().str.lower()
                
                sentiment_labels = ["positif", "negatif", "netral"]
                color_map = {
                    "positif": "#2ecc71",
                    "negatif": "#e74c3c",
                    "netral": "#95a5a6"
                }
                
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
                    
                gt_sizes, gt_colors, gt_labels = get_sizes_and_colors(y_true_g)
                lex_sizes, lex_colors, lex_labels = get_sizes_and_colors(y_lex_g)
                llm_sizes, llm_colors, llm_labels = get_sizes_and_colors(y_llm_g)
                
                fig_donut_g, (ax_g1, ax_g2, ax_g3) = plt.subplots(1, 3, figsize=(18, 6))
                
                if gt_sizes:
                    ax_g1.pie(gt_sizes, labels=gt_labels, autopct='%1.1f%%', startangle=90, colors=gt_colors, pctdistance=0.75, textprops=dict(color="black", weight="bold"))
                    ax_g1.add_artist(plt.Circle((0,0), 0.50, fc='white'))
                    ax_g1.set_title("Global Ground Truth", fontsize=12, weight="bold")
                else:
                    ax_g1.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
                    
                if lex_sizes:
                    ax_g2.pie(lex_sizes, labels=lex_labels, autopct='%1.1f%%', startangle=90, colors=lex_colors, pctdistance=0.75, textprops=dict(color="black", weight="bold"))
                    ax_g2.add_artist(plt.Circle((0,0), 0.50, fc='white'))
                    ax_g2.set_title("Global Lexicon-based", fontsize=12, weight="bold")
                else:
                    ax_g2.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
                    
                if llm_sizes:
                    ax_g3.pie(llm_sizes, labels=llm_labels, autopct='%1.1f%%', startangle=90, colors=llm_colors, pctdistance=0.75, textprops=dict(color="black", weight="bold"))
                    ax_g3.add_artist(plt.Circle((0,0), 0.50, fc='white'))
                    ax_g3.set_title("Global LLM-based", fontsize=12, weight="bold")
                else:
                    ax_g3.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
                    
                plt.tight_layout()
                st.pyplot(fig_donut_g)
                plt.close()
                
                # Global metrics and individual video comparison
                tab_glob1, tab_glob2 = st.tabs([":material/bar_chart: Global Metrics Comparison", ":material/analytics: Per Video Accuracy Comparison"])
                
                with tab_glob1:
                    st.markdown("### Perbandingan Metrik Evaluasi Akumulatif (Global)")
                    
                    global_lex_acc = accuracy_score(y_true_g, y_lex_g)
                    global_lex_prec, global_lex_rec, global_lex_f1, _ = precision_recall_fscore_support(y_true_g, y_lex_g, average='macro', zero_division=0)
                    
                    global_llm_acc = accuracy_score(y_true_g, y_llm_g)
                    global_llm_prec, global_llm_rec, global_llm_f1, _ = precision_recall_fscore_support(y_true_g, y_llm_g, average='macro', zero_division=0)
                    
                    metrics_g = ['Akurasi', 'Presisi', 'Sensitivitas (Recall)', 'F1-Score']
                    lex_scores_g = [global_lex_acc * 100, global_lex_prec * 100, global_lex_rec * 100, global_lex_f1 * 100]
                    llm_scores_g = [global_llm_acc * 100, global_llm_prec * 100, global_llm_rec * 100, global_llm_f1 * 100]
                    
                    x_g = np.arange(len(metrics_g))
                    width_g = 0.35
                    
                    fig_metrics_g, ax_mg = plt.subplots(figsize=(10, 5))
                    rects_g1 = ax_mg.bar(x_g - width_g/2, lex_scores_g, width_g, label='Lexicon-based', color='#3498db')
                    rects_g2 = ax_mg.bar(x_g + width_g/2, llm_scores_g, width_g, label='LLM-based (NVIDIA NIM)', color='#2ecc71')
                    
                    ax_mg.set_ylabel('Skor (%)', weight="bold")
                    ax_mg.set_title('Perbandingan Metrik Akurasi Akumulatif (Global)', weight="bold", fontsize=12)
                    ax_mg.set_xticks(x_g)
                    ax_mg.set_xticklabels(metrics_g, weight="bold")
                    ax_mg.set_ylim(0, 110)
                    ax_mg.legend()
                    
                    # Autolabel
                    def autolabel_g(rects):
                        for rect in rects:
                            height = rect.get_height()
                           # Keep a clean presentation
                            ax_mg.annotate(f'{height:.1f}%',
                                        xy=(rect.get_x() + rect.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', weight="bold", fontsize=9)
                    autolabel_g(rects_g1)
                    autolabel_g(rects_g2)
                    
                    plt.tight_layout()
                    st.pyplot(fig_metrics_g)
                    plt.close()
                    
                with tab_glob2:
                    if video_accuracies:
                        st.markdown("### Perbandingan Akurasi antara Lexicon dan LLM untuk Setiap Video")
                        df_vid_acc = pd.DataFrame(video_accuracies)
                        
                        # Plot a line or grouped bar chart comparing accuracies
                        fig_line, ax_l = plt.subplots(figsize=(12, 5))
                        x_indices = np.arange(len(df_vid_acc))
                        
                        # We draw lines
                        ax_l.plot(x_indices, df_vid_acc["Lexicon Accuracy"], marker='o', linewidth=2, color='#3498db', label='Lexicon Accuracy')
                        ax_l.plot(x_indices, df_vid_acc["LLM Accuracy"], marker='s', linewidth=2, color='#2ecc71', label='LLM Accuracy')
                        
                        # Add value labels
                        for i, val in enumerate(df_vid_acc["Lexicon Accuracy"]):
                            ax_l.annotate(f'{val:.1f}%', (x_indices[i], val), textcoords="offset points", xytext=(0,10), ha='center', color='#1e3a8a', weight="bold")
                        for i, val in enumerate(df_vid_acc["LLM Accuracy"]):
                            ax_l.annotate(f'{val:.1f}%', (x_indices[i], val), textcoords="offset points", xytext=(0,-15), ha='center', color='#166534', weight="bold")
                            
                        ax_l.set_xticks(x_indices)
                        ax_l.set_xticklabels(df_vid_acc["Video"], rotation=30, ha='right', weight="bold", fontsize=9)
                        ax_l.set_ylabel('Akurasi (%)', weight="bold")
                        ax_l.set_ylim(0, 115)
                        ax_l.set_title('Akurasi Metode Lexicon vs LLM per Video', weight="bold", fontsize=12)
                        ax_l.grid(True, linestyle='--', alpha=0.5)
                        ax_l.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig_line)
                        plt.close()
                        
                        # Display table
                        st.dataframe(
                            df_vid_acc,
                            column_config={
                                "Video": st.column_config.TextColumn("Judul Video", width="large"),
                                "Lexicon Accuracy": st.column_config.NumberColumn("Akurasi Lexicon", format="%.2f%%"),
                                "LLM Accuracy": st.column_config.NumberColumn("Akurasi LLM", format="%.2f%%"),
                                "LLM Model": st.column_config.TextColumn("Model LLM yang Digunakan")
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("Belum ada video dengan Ground Truth terisi untuk dibandingkan.")
    st.stop()

# Main Dashboard Area (SEMANTIKA)
st.markdown("<h1><span style='color:#3498db'>SEMAN</span><span style='color:#2ecc71'>TIKA</span> : Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Aplikasi perbandingan performa analisis sentimen berbasis **Lexicon-based (Sastrawi + InSet)** dan **LLM-based (NVIDIA NIM Llama 3.1)**.")
st.markdown("---")

if st.session_state.df is not None:
    # Header: Video Info
    st.markdown(f"### :material/movie: **{st.session_state.video_title}**")
    col_hdr1, col_hdr2, col_hdr3 = st.columns(3)
    with col_hdr1:
        st.markdown(f":material/link: **Link Video:** [{st.session_state.video_url}]({st.session_state.video_url})")
    with col_hdr2:
        st.markdown(f":material/neurology: **Model LLM Aktif:** `{st.session_state.llm_model}`")
    with col_hdr3:
        lang_label = "Inggris (EN)" if st.session_state.detected_lang == "en" else "Indonesia (ID)"
        st.markdown(f":material/translate: **Bahasa Terdeteksi:** `{lang_label}`")
    
    st.markdown("---")
    
    tab_view, tab_edit = st.tabs([":material/visibility: Tampilan Tabel Berwarna", ":material/edit: Edit Ground Truth"])
    
    display_df = st.session_state.df.copy()
    
    with tab_view:
        def style_sentiment(val):
            val_lower = str(val).strip().lower()
            if val_lower == "positif":
                return "background-color: #C6EFCE; color: #006100; font-weight: bold;"
            elif val_lower == "negatif":
                return "background-color: #FFC7CE; color: #9C0006; font-weight: bold;"
            elif val_lower == "netral":
                return "background-color: #E2E3E5; color: #383D41;"
            return ""
            
        if hasattr(display_df.style, "map"):
            styled_df = display_df.style.map(style_sentiment, subset=["Lexicon Sentiment", "LLM Sentiment", "Ground Truth"])
        else:
            styled_df = display_df.style.applymap(style_sentiment, subset=["Lexicon Sentiment", "LLM Sentiment", "Ground Truth"])
            
        st.dataframe(
            styled_df,
            column_config={
                "No": st.column_config.NumberColumn("No", width="small"),
                "Author": st.column_config.TextColumn("Penulis", width="medium"),
                "Original Comment": st.column_config.TextColumn("Komentar Asli", width="large"),
                "Cleaned Comment": st.column_config.TextColumn("Komentar Bersih (Stemmed)", width="medium"),
                "Lexicon Sentiment": st.column_config.TextColumn("Sentimen Lexicon", width="small"),
                "LLM Sentiment": st.column_config.TextColumn("Sentimen LLM", width="small"),
                "Ground Truth": st.column_config.TextColumn("Ground Truth", width="small"),
            },
            column_order=["No", "Author", "Original Comment", "Cleaned Comment", "Lexicon Sentiment", "LLM Sentiment", "Ground Truth"],
            use_container_width=True,
            hide_index=True
        )
        
    with tab_edit:
        st.info("Gunakan tabel di bawah ini untuk menentukan sentimen sebenarnya (Ground Truth) lewat dropdown pilihan.", icon=":material/info:")
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
        excel_data = convert_df_to_excel(st.session_state.df, st.session_state.video_title, st.session_state.video_url)
        st.download_button(
            label=":material/download: Ekspor Laporan Excel Berwarna (.xlsx)",
            data=excel_data,
            file_name=f"semantika_hasil_{extract_video_id(st.session_state.video_url)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col_dl2:
        # Generate landscape PDF report
        pdf_data = convert_df_to_pdf(st.session_state.df, st.session_state.video_title, st.session_state.video_url)
        st.download_button(
            label=":material/download: Ekspor Laporan PDF (.pdf)",
            data=pdf_data,
            file_name=f"semantika_hasil_{extract_video_id(st.session_state.video_url)}.pdf",
            mime="application/pdf",
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
            <strong>Sistem Poin Komparasi SEMANTIKA:</strong><br/>
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
        tab1, tab2, tab3 = st.tabs([
            ":material/donut_large: Sebaran Sentimen (Donut Charts)", 
            ":material/grid_on: Matrik Kebingungan (Confusion Matrices)",
            ":material/bar_chart: Perbandingan Metrik (Bar Chart)"
        ])
        
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
                st.markdown("#### :material/book: Performa Lexicon (Sastrawi + InSet)")
                st.write(f"- **Accuracy (Akurasi)**: {lex_acc * 100:.2f}%")
                st.write(f"- **Precision (Presisi)**: {lex_prec * 100:.2f}%")
                st.write(f"- **Recall (Sensitivitas)**: {lex_rec * 100:.2f}%")
                st.write(f"- **F1-Score**: {lex_f1 * 100:.2f}%")
                
            with col_m2:
                st.markdown(f"#### :material/psychology: Performa LLM ({st.session_state.llm_model.split('/')[-1]})")
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

        # Tab 3: Metrics Comparison Bar Chart
        with tab3:
            st.markdown("### Perbandingan Metrik Evaluasi (Lexicon vs LLM)")
            
            metrics = ['Akurasi', 'Presisi', 'Sensitivitas (Recall)', 'F1-Score']
            lex_scores = [lex_acc * 100, lex_prec * 100, lex_rec * 100, lex_f1 * 100]
            llm_scores = [llm_acc * 100, llm_prec * 100, llm_rec * 100, llm_f1 * 100]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig_metrics, ax_m = plt.subplots(figsize=(10, 5))
            rects1 = ax_m.bar(x - width/2, lex_scores, width, label='Lexicon-based', color='#3498db')
            rects2 = ax_m.bar(x + width/2, llm_scores, width, label=f'LLM-based ({st.session_state.llm_model.split("/")[-1]})', color='#2ecc71')
            
            ax_m.set_ylabel('Skor (%)', weight="bold")
            ax_m.set_title('Perbandingan Metrik Akurasi, Presisi, Recall, dan F1-Score', weight="bold", fontsize=12)
            ax_m.set_xticks(x)
            ax_m.set_xticklabels(metrics, weight="bold")
            ax_m.set_ylim(0, 110)
            ax_m.legend()
            
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax_m.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', weight="bold", fontsize=9)
                                
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            st.pyplot(fig_metrics)
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
