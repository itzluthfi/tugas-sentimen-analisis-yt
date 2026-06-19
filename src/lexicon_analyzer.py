import os
import re
import urllib.request
import logging
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from src.config import LEXICON_DIR
from src.normalizer import normalize_slang

logger = logging.getLogger(__name__)

# URLs to download the lexicon files
POSITIVE_LEXICON_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
NEGATIVE_LEXICON_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
VADER_LEXICON_URL = "https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt"

class LexiconSentimentAnalyzer:
    def __init__(self):
        self.id_lexicon = {}
        self.en_lexicon = {}
        self.stemmer = None
        self.stopword_remover = None
        
        # Ensure lexicon directory exists and files are downloaded
        self._ensure_lexicon_files()
        # Load lexicons into memory
        self._load_lexicons()
        # Initialize Sastrawi tools for Indonesian
        self._init_sastrawi()

    def _ensure_lexicon_files(self):
        """
        Downloads positive, negative, and English VADER lexicon files if they do not exist.
        """
        os.makedirs(LEXICON_DIR, exist_ok=True)
        
        pos_path = os.path.join(LEXICON_DIR, "positive.tsv")
        neg_path = os.path.join(LEXICON_DIR, "negative.tsv")
        en_path = os.path.join(LEXICON_DIR, "vader_lexicon.txt")
        
        if not os.path.exists(pos_path):
            logger.info("Mengunduh positive.tsv dari fajri91/InSet...")
            urllib.request.urlretrieve(POSITIVE_LEXICON_URL, pos_path)
            
        if not os.path.exists(neg_path):
            logger.info("Mengunduh negative.tsv dari fajri91/InSet...")
            urllib.request.urlretrieve(NEGATIVE_LEXICON_URL, neg_path)

        if not os.path.exists(en_path):
            logger.info("Mengunduh vader_lexicon.txt dari cjhutto/vaderSentiment...")
            urllib.request.urlretrieve(VADER_LEXICON_URL, en_path)

    def _load_lexicons(self):
        """
        Loads the TSV and TXT lexicon files into lookup dictionaries.
        """
        pos_path = os.path.join(LEXICON_DIR, "positive.tsv")
        neg_path = os.path.join(LEXICON_DIR, "negative.tsv")
        en_path = os.path.join(LEXICON_DIR, "vader_lexicon.txt")
        
        # Load Indonesian positive lexicon
        try:
            df_pos = pd.read_csv(pos_path, sep="\t")
            for _, row in df_pos.iterrows():
                word = str(row["word"]).strip().lower()
                weight = float(row["weight"])
                self.id_lexicon[word] = abs(weight)
        except Exception as e:
            logger.error(f"Gagal memuat positive.tsv: {e}")
            
        # Load Indonesian negative lexicon
        try:
            df_neg = pd.read_csv(neg_path, sep="\t")
            for _, row in df_neg.iterrows():
                word = str(row["word"]).strip().lower()
                weight = float(row["weight"])
                self.id_lexicon[word] = -abs(weight)
        except Exception as e:
            logger.error(f"Gagal memuat negative.tsv: {e}")
            
        # Load English VADER lexicon
        try:
            with open(en_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        word = parts[0].strip().lower()
                        weight = float(parts[1])
                        self.en_lexicon[word] = weight
            logger.info(f"Berhasil memuat {len(self.en_lexicon)} kata ke dalam kamus English lexicon.")
        except Exception as e:
            logger.error(f"Gagal memuat vader_lexicon.txt: {e}")
            
        logger.info(f"Berhasil memuat {len(self.id_lexicon)} kata ke dalam kamus Indonesian lexicon.")

    def _init_sastrawi(self):
        """
        Initializes PySastrawi stemmer and stopword remover.
        """
        logger.info("Menginisialisasi Sastrawi Stemmer dan StopWordRemover...")
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
        logger.info("Sastrawi berhasil diinisialisasi.")

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and preprocesses Indonesian text.
        """
        if not text:
            return ""
        
        # Case folding
        text = text.lower()
        
        # Cleaning URLs, mentions, hashtags, non-ascii, numbers, punctuation
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Slang normalization
        text = normalize_slang(text)
        
        # Stopword removal
        text = self.stopword_remover.remove(text)
        
        # Stemming
        text = self.stemmer.stem(text)
        
        return text

    def preprocess_text_en(self, text: str) -> str:
        """
        Cleans and preprocesses English text.
        """
        if not text:
            return ""
        
        # Case folding
        text = text.lower()
        
        # Cleaning URLs, mentions, hashtags, non-ascii, numbers, punctuation
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def analyze_sentiment(self, text: str, lang: str = "id") -> tuple[str, float, str]:
        """
        Analyzes the sentiment of a text based on lexicon matching.
        lang can be "id" or "en".
        """
        if lang == "en":
            cleaned_text = self.preprocess_text_en(text)
            lexicon = self.en_lexicon
        else:
            cleaned_text = self.preprocess_text(text)
            lexicon = self.id_lexicon
            
        tokens = cleaned_text.split()
        score = 0.0
        
        for token in tokens:
            if token in lexicon:
                score += lexicon[token]
                
        if score > 0.05:  # small positive threshold
            sentiment = "positif"
        elif score < -0.05:  # small negative threshold
            sentiment = "negatif"
        else:
            sentiment = "netral"
            
        return sentiment, score, cleaned_text

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = LexiconSentimentAnalyzer()
    
    test_id = "Keren banget videonya! Penjelasannya sangat jelas."
    test_en = "This video is absolutely amazing! The explanation is very clear."
    
    print(analyzer.analyze_sentiment(test_id, "id"))
    print(analyzer.analyze_sentiment(test_en, "en"))
