import os
import sys
import logging
import pandas as pd

# Configure logging to show standard output clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from src.config import YOUTUBE_VIDEO_URL, MAX_COMMENTS, OUTPUT_FILE
from src.downloader import fetch_youtube_comments
from src.lexicon_analyzer import LexiconSentimentAnalyzer
from src.llm_analyzer import LLMSentimentAnalyzer

def main():
    logger.info("=== Memulai Analisis Sentimen Komentar YouTube ===")
    
    # 1. Fetch comments
    comments = fetch_youtube_comments(YOUTUBE_VIDEO_URL, limit=MAX_COMMENTS)
    if not comments:
        logger.error("Gagal mendapatkan komentar. Program dihentikan.")
        return
        
    logger.info(f"Berhasil mengambil {len(comments)} komentar untuk dianalisis.")
    
    # 2. Run Lexicon-based analysis (and pre-processing)
    logger.info("Memulai analisis sentimen metode Lexicon (Sastrawi + InSet)...")
    lexicon_analyzer = LexiconSentimentAnalyzer()
    
    processed_comments = []
    for c in comments:
        sentiment, score, cleaned_text, _ = lexicon_analyzer.analyze_sentiment(c["text"])
        processed_comments.append({
            "comment_id": c["comment_id"],
            "author": c["author"],
            "original_comment": c["text"],
            "cleaned_comment": cleaned_text,
            "lexicon_sentiment": sentiment,
            "lexicon_score": score
        })
        
    # 3. Run LLM-based analysis
    logger.info("Memulai analisis sentimen metode LLM (NVIDIA NIM API)...")
    llm_analyzer = LLMSentimentAnalyzer()
    
    # Process LLM in batches of 20 to avoid rate limits and minimize requests
    batch_size = 20
    llm_sentiment_map = {}
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        logger.info(f"Memproses LLM untuk komentar {i+1} sampai {min(i+batch_size, len(comments))}...")
        batch_results = llm_analyzer.analyze_batch(batch)
        for r in batch_results:
            llm_sentiment_map[r["comment_id"]] = r["llm_sentiment"]
            
    # 4. Combine results into a dataframe
    logger.info("Menggabungkan seluruh hasil analisis...")
    final_data = []
    for idx, c in enumerate(processed_comments):
        cid = c["comment_id"]
        final_data.append({
            "No": idx + 1,
            "Comment ID": cid,
            "Author": c["author"],
            "Original Comment": c["original_comment"],
            "Cleaned Comment": c["cleaned_comment"],
            "Lexicon Sentiment": c["lexicon_sentiment"],
            "Lexicon Score": c["lexicon_score"],
            "LLM Sentiment": llm_sentiment_map.get(cid, "netral"),
            "Ground Truth": ""  # Left blank for the user to edit
        })
        
    df = pd.DataFrame(final_data)
    
    # 5. Export to CSV (using utf-8-sig so Excel can read Indonesian accents/emojis cleanly)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"Hasil analisis berhasil disimpan di: {OUTPUT_FILE}")
    
    print("\n" + "="*50)
    print("ANALISIS SENTIMEN SELESAI!")
    print(f"Total data diproses: {len(df)}")
    print(f"File hasil          : {OUTPUT_FILE}")
    print("\nLangkah selanjutnya:")
    print("1. Buka file 'sentiment_results.csv' menggunakan Excel atau Editor CSV.")
    print("2. Isi kolom 'Ground Truth' secara manual dengan kata: 'positif', 'negatif', atau 'netral'.")
    print("3. Simpan file tersebut.")
    print("4. Jalankan script 'evaluate.py' untuk melihat performa akurasi perbandingan.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
