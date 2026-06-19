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

# URLs to download the InSet lexicon files
POSITIVE_LEXICON_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
NEGATIVE_LEXICON_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"

class LexiconSentimentAnalyzer:
    def __init__(self):
        self.lexicon = {}
        self.stemmer = None
        self.stopword_remover = None
        
        # Ensure lexicon directory exists and files are downloaded
        self._ensure_lexicon_files()
        # Load lexicon into memory
        self._load_lexicon()
        # Initialize Sastrawi tools
        self._init_sastrawi()

    def _ensure_lexicon_files(self):
        """
        Downloads InSet positive and negative lexicon files if they do not exist.
        """
        os.makedirs(LEXICON_DIR, exist_ok=True)
        
        pos_path = os.path.join(LEXICON_DIR, "positive.tsv")
        neg_path = os.path.join(LEXICON_DIR, "negative.tsv")
        
        if not os.path.exists(pos_path):
            logger.info("Mengunduh positive.tsv dari fajri91/InSet...")
            urllib.request.urlretrieve(POSITIVE_LEXICON_URL, pos_path)
            
        if not os.path.exists(neg_path):
            logger.info("Mengunduh negative.tsv dari fajri91/InSet...")
            urllib.request.urlretrieve(NEGATIVE_LEXICON_URL, neg_path)

    def _load_lexicon(self):
        """
        Loads the TSV lexicon files into a single lookup dictionary.
        """
        pos_path = os.path.join(LEXICON_DIR, "positive.tsv")
        neg_path = os.path.join(LEXICON_DIR, "negative.tsv")
        
        # Load positive lexicon
        try:
            df_pos = pd.read_csv(pos_path, sep="\t")
            for _, row in df_pos.iterrows():
                word = str(row["word"]).strip().lower()
                weight = float(row["weight"])
                # Positive weights should be positive
                self.lexicon[word] = abs(weight)
        except Exception as e:
            logger.error(f"Gagal memuat positive.tsv: {e}")
            
        # Load negative lexicon
        try:
            df_neg = pd.read_csv(neg_path, sep="\t")
            for _, row in df_neg.iterrows():
                word = str(row["word"]).strip().lower()
                weight = float(row["weight"])
                # Negative weights should be negative
                self.lexicon[word] = -abs(weight)
        except Exception as e:
            logger.error(f"Gagal memuat negative.tsv: {e}")
            
        logger.info(f"Berhasil memuat {len(self.lexicon)} kata ke dalam kamus lexicon.")

    def _init_sastrawi(self):
        """
        Initializes PySastrawi stemmer and stopword remover.
        """
        logger.info("Menginisialisasi Sastrawi Stemmer dan StopWordRemover (ini memerlukan waktu beberapa detik)...")
        
        # Create stemmer
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        # Create stopword remover
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
        
        logger.info("Sastrawi berhasil diinisialisasi.")

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and preprocesses Indonesian text:
        1. Case folding (lowercase)
        2. Remove links, usernames, hashtags, emojis, punctuation, numbers
        3. Slang normalization
        4. Sastrawi stopword removal
        5. Sastrawi stemming
        """
        if not text:
            return ""
            
        # 1. Case folding
        text = text.lower()
        
        # 2. Cleaning:
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove mentions/usernames (e.g. @username)
        text = re.sub(r"@\w+", "", text)
        # Remove hashtags
        text = re.sub(r"#\w+", "", text)
        # Remove emojis and non-ascii characters
        text = text.encode("ascii", "ignore").decode("ascii")
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        # Remove extra whitespaces and punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # 3. Slang normalization
        text = normalize_slang(text)
        
        # 4. Stopword removal (Sastrawi)
        text = self.stopword_remover.remove(text)
        
        # 5. Stemming (Sastrawi)
        text = self.stemmer.stem(text)
        
        return text

    def analyze_sentiment(self, text: str) -> tuple[str, float, str]:
        """
        Analyzes the sentiment of a text based on lexicon matching.
        Returns:
            - sentiment: "positif", "negatif", or "netral"
            - score: the sum of lexicon weights
            - cleaned_text: the text after preprocessing
        """
        cleaned_text = self.preprocess_text(text)
        tokens = cleaned_text.split()
        
        score = 0.0
        matched_words = []
        
        for token in tokens:
            if token in self.lexicon:
                score += self.lexicon[token]
                matched_words.append(f"{token}({self.lexicon[token]})")
                
        if score > 0:
            sentiment = "positif"
        elif score < 0:
            sentiment = "negatif"
        else:
            sentiment = "netral"
            
        # Logging details for debug if needed
        # logger.debug(f"Tokens: {tokens} | Matched: {matched_words} | Score: {score}")
        
        return sentiment, score, cleaned_text

if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)
    analyzer = LexiconSentimentAnalyzer()
    
    test_sentences = [
        "Keren banget videonya! Penjelasannya sangat jelas dan mudah dipahami.",
        "Jelek sekali editannya, bikin pusing dan buang-buang waktu saja.",
        "Biasa saja sih, tidak ada yang spesial dari konten ini.",
        "Gpp lah yg penting udah berusaha bikin konten kreatif mantap."
    ]
    
    for sentence in test_sentences:
        sent, score, clean = analyzer.analyze_sentiment(sentence)
        print(f"\nOriginal: {sentence}")
        print(f"Cleaned : {clean}")
        print(f"Hasil   : {sent} (Score: {score})")
