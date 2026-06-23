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
        self.stem_cache = {}
        
        # Negation lists for dynamic sentiment inversion
        self.negation_words_id = {"tidak", "g", "ga", "gak", "kaga", "kagak", "tdk", "bukan", "kurang", "belum", "blm"}
        self.negation_words_en = {"not", "no", "never", "dont", "doesnt", "didnt", "cant", "cannot", "isnt", "arent", "wasnt", "werent", "without", "lack"}
        
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
        Initializes PySastrawi stemmer and stopword remover with negation word protection.
        """
        logger.info("Menginisialisasi Sastrawi Stemmer dan StopWordRemover...")
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        # Initialize Stopword Remover with negation words excluded
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
        stopword_factory = StopWordRemoverFactory()
        default_stopwords = stopword_factory.get_stop_words()
        
        # Exclude negation words from being removed as stopwords
        custom_stopwords = [w for w in default_stopwords if w not in self.negation_words_id]
        dictionary = ArrayDictionary(custom_stopwords)
        self.stopword_remover = StopWordRemover(dictionary)
        logger.info("Sastrawi berhasil diinisialisasi.")

    def stem_word(self, word: str) -> str:
        """
        Stems a single word with in-memory caching to optimize Sastrawi performance.
        """
        if word in self.stem_cache:
            return self.stem_cache[word]
        
        stemmed = self.stemmer.stem(word)
        self.stem_cache[word] = stemmed
        return stemmed

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
        
        # Stemming with cache and English word protection
        words = text.split()
        stemmed_words = []
        for word in words:
            if word in self.en_lexicon:
                stemmed_words.append(word)
            else:
                stemmed_words.append(self.stem_word(word))
        text = " ".join(stemmed_words)
        
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

    def analyze_sentiment(self, text: str, default_lang: str = "id") -> tuple[str, float, str, str]:
        """
        Analyzes the sentiment of a text based on lexicon matching.
        It dynamically determines the language based on which lexicon matches more tokens.
        Supports negation handling (reversing score if negation word is found within preceding window of 2 words).
        Returns: (sentiment, score, cleaned_text, detected_lang)
        """
        negation_window = 2

        # 1. Evaluate as English
        cleaned_en = self.preprocess_text_en(text)
        tokens_en = cleaned_en.split()
        score_en = 0.0
        matches_en = 0
        for idx, token in enumerate(tokens_en):
            if token in self.en_lexicon:
                token_score = self.en_lexicon[token]
                # Check for negation in preceding window
                negated = False
                start_win = max(0, idx - negation_window)
                for j in range(start_win, idx):
                    if tokens_en[j] in self.negation_words_en:
                        negated = True
                        break
                if negated:
                    token_score = -token_score
                score_en += token_score
                matches_en += 1

        # 2. Evaluate as Indonesian
        cleaned_id = self.preprocess_text(text)
        tokens_id = cleaned_id.split()
        score_id = 0.0
        matches_id = 0
        for idx, token in enumerate(tokens_id):
            if token in self.id_lexicon:
                token_score = self.id_lexicon[token]
                # Check for negation in preceding window
                negated = False
                start_win = max(0, idx - negation_window)
                for j in range(start_win, idx):
                    if tokens_id[j] in self.negation_words_id:
                        negated = True
                        break
                if negated:
                    token_score = -token_score
                score_id += token_score
                matches_id += 1

        # Determine dominant language
        if matches_en > matches_id:
            detected_lang = "en"
            score = score_en
            cleaned_text = cleaned_en
        elif matches_id > matches_en:
            detected_lang = "id"
            score = score_id
            cleaned_text = cleaned_id
        else:
            # Fallback for ties or no matches
            detected_lang = default_lang
            if default_lang == "en":
                score = score_en
                cleaned_text = cleaned_en
            else:
                score = score_id
                cleaned_text = cleaned_id

        # Classify score
        if score > 0.05:
            sentiment = "positif"
        elif score < -0.05:
            sentiment = "negatif"
        else:
            sentiment = "netral"

        return sentiment, score, cleaned_text, detected_lang

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = LexiconSentimentAnalyzer()
    
    test_id = "Keren banget videonya! Penjelasannya sangat jelas."
    test_en = "This video is absolutely amazing! The explanation is very clear."
    test_mixed = "iPhone ini very cheap dan keren bgt"
    
    print(analyzer.analyze_sentiment(test_id, "id"))
    print(analyzer.analyze_sentiment(test_en, "en"))
    print(analyzer.analyze_sentiment(test_mixed, "id"))
