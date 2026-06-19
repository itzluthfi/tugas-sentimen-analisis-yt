import os
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
YOUTUBE_VIDEO_URL = os.getenv("YOUTUBE_VIDEO_URL", "https://www.youtube.com/shorts/a3Irz3zv8L0")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", 100))
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
APP_MODE = os.getenv("APP_MODE", "development").strip().lower()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_DIR = os.path.join(BASE_DIR, "lexicon")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
OUTPUT_FILE = os.path.join(BASE_DIR, "sentiment_results.csv")
EVALUATION_IMAGE = os.path.join(BASE_DIR, "evaluation_metrics.png")

# Auto-create directories
os.makedirs(HISTORY_DIR, exist_ok=True)
