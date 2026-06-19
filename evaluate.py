import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.config import OUTPUT_FILE, EVALUATION_IMAGE

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan 'python main.py' terlebih dahulu.")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    return df

def clean_labels(series):
    # Strip spaces and convert to lowercase
    return series.astype(str).str.strip().str.lower()

def evaluate():
    print("=== Mengevaluasi Kinerja Analisis Sentimen terhadap Ground Truth ===\n")
    
    # 1. Load data
    df = load_data(OUTPUT_FILE)
    
    # Check if Ground Truth column exists
    if "Ground Truth" not in df.columns:
        print("Error: Kolom 'Ground Truth' tidak ditemukan di file CSV.")
        sys.exit(1)
        
    # Remove rows where Ground Truth is empty or NaN
    df_eval = df.dropna(subset=["Ground Truth"]).copy()
    # Filter empty strings
    df_eval = df_eval[df_eval["Ground Truth"].astype(str).str.strip() != ""]
    
    if len(df_eval) == 0:
        print("Peringatan: Kolom 'Ground Truth' masih kosong!")
        print(f"Silakan buka file '{OUTPUT_FILE}' dan isi kolom 'Ground Truth' minimal pada beberapa baris.")
        print("Gunakan kata: 'positif', 'negatif', atau 'netral'.")
        sys.exit(0)
        
    # Clean the labels
    y_true = clean_labels(df_eval["Ground Truth"])
    y_lexicon = clean_labels(df_eval["Lexicon Sentiment"])
    y_llm = clean_labels(df_eval["LLM Sentiment"])
    
    # Valid labels
    valid_labels = ["negatif", "netral", "positif"]
    
    # Filter only rows that have valid ground truth labels
    mask = y_true.isin(valid_labels)
    if not mask.any():
        print("Error: Tidak ada data Ground Truth dengan label valid ('positif', 'negatif', 'netral').")
        print("Harap periksa kembali pengisian Ground Truth di file CSV Anda.")
        sys.exit(1)
        
    y_true = y_true[mask]
    y_lexicon = y_lexicon[mask]
    y_llm = y_llm[mask]
    
    df_eval_filtered = df_eval[mask]
    
    print(f"Menghitung metrik performa berdasarkan {len(df_eval_filtered)} data Ground Truth...")
    
    # 2. Compute Metrics
    # Lexicon metrics
    lex_acc = accuracy_score(y_true, y_lexicon)
    lex_prec, lex_rec, lex_f1, _ = precision_recall_fscore_support(y_true, y_lexicon, average='macro', zero_division=0)
    
    # LLM metrics
    llm_acc = accuracy_score(y_true, y_llm)
    llm_prec, llm_rec, llm_f1, _ = precision_recall_fscore_support(y_true, y_llm, average='macro', zero_division=0)
    
    # 3. Print Results in CLI
    print("\n" + "="*60)
    print("HASIL EVALUASI METODE ANALISIS SENTIMEN")
    print("="*60)
    
    metrics_summary = pd.DataFrame({
        "Metrik (Macro Average)": ["Accuracy (Akurasi)", "Precision (Presisi)", "Recall (Sensitivitas)", "F1-Score"],
        "Lexicon-based": [lex_acc, lex_prec, lex_rec, lex_f1],
        "LLM-based (Llama-3.1-70b)": [llm_acc, llm_prec, llm_rec, llm_f1]
    })
    
    # Format to percentages
    for col in ["Lexicon-based", "LLM-based (Llama-3.1-70b)"]:
        metrics_summary[col] = metrics_summary[col].apply(lambda x: f"{x * 100:.2f}%")
        
    print(metrics_summary.to_string(index=False))
    print("="*60)
    
    print("\n--- Laporan Klasifikasi Detail: Lexicon-based ---")
    print(classification_report(y_true, y_lexicon, target_names=np.intersect1d(valid_labels, y_true.unique()), zero_division=0))
    
    print("--- Laporan Klasifikasi Detail: LLM-based ---")
    print(classification_report(y_true, y_llm, target_names=np.intersect1d(valid_labels, y_true.unique()), zero_division=0))
    
    # 4. Generate Visualizations (Save to evaluation_metrics.png)
    print(f"Membuat diagram evaluasi dan menyimpannya ke '{EVALUATION_IMAGE}'...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Chart 1: Bar Chart Perbandingan Akurasi & F1-Score
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    categories = ['Accuracy', 'F1-Score (Macro)']
    lex_vals = [lex_acc, lex_f1]
    llm_vals = [llm_acc, llm_f1]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, lex_vals, width, label='Lexicon-based', color='#3498db')
    rects2 = ax1.bar(x + width/2, llm_vals, width, label='LLM-based (Llama 3.1 70B)', color='#2ecc71')
    
    ax1.set_ylabel('Skor (0.0 - 1.0)')
    ax1.set_title('Perbandingan Performa: Lexicon vs LLM')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Label the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height * 100:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    # Chart 2: Confusion Matrix Lexicon
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    # Align labels with valid labels
    labels_present = sorted(list(set(y_true.unique()) | set(y_lexicon.unique())))
    cm_lex = confusion_matrix(y_true, y_lexicon, labels=labels_present)
    disp_lex = ConfusionMatrixDisplay(confusion_matrix=cm_lex, display_labels=labels_present)
    disp_lex.plot(ax=ax2, cmap=plt.cm.Blues, values_format='d')
    ax2.set_title('Confusion Matrix: Lexicon-based')
    
    # Chart 3: Confusion Matrix LLM
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    labels_present_llm = sorted(list(set(y_true.unique()) | set(y_llm.unique())))
    cm_llm = confusion_matrix(y_true, y_llm, labels=labels_present_llm)
    disp_llm = ConfusionMatrixDisplay(confusion_matrix=cm_llm, display_labels=labels_present_llm)
    disp_llm.plot(ax=ax3, cmap=plt.cm.Greens, values_format='d')
    ax3.set_title('Confusion Matrix: LLM-based')
    
    plt.suptitle(f'Evaluasi Perbandingan Analisis Sentimen (Data Ground Truth: {len(y_true)})', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(EVALUATION_IMAGE, dpi=300)
    plt.close()
    
    print(f"Visualisasi berhasil disimpan di: {EVALUATION_IMAGE}")
    print("\nEvaluasi selesai dengan sukses!")

if __name__ == "__main__":
    evaluate()
