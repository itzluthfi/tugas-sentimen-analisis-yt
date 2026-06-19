import re
import logging
import requests
from youtube_comment_downloader import YoutubeCommentDownloader

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a standard YouTube watch URL, embed URL, or shorts URL.
    """
    if not url:
        return None
    
    # Regex to cover multiple YouTube URL formats (shorts, watch, share links, etc.)
    pattern = r"(?:v=|\/shorts\/|\/embed\/|\/v\/|youtu\.be\/|\/watch\?v=|\/watch\?.+&v=)([^#\&\?]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    
    # Fallback: if url is just the video id
    if len(url) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
        
    return None

def get_video_title(url: str) -> str:
    """
    Fetches the video title from the YouTube page HTML.
    If it fails, returns a default fallback string.
    """
    if not url:
        return "Video YouTube"
        
    try:
        # Standard headers to avoid bot detection blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            # Look for <title> tag
            title_match = re.search(r"<title>(.*?)</title>", response.text)
            if title_match:
                title = title_match.group(1).strip()
                # Clean YouTube suffix if present
                if " - YouTube" in title:
                    title = title.replace(" - YouTube", "")
                return title
    except Exception as e:
        logger.error(f"Gagal mengambil judul video: {e}")
        
    return "Video YouTube"

def fetch_youtube_comments(url: str, limit: int = 100) -> list:
    """
    Fetches up to `limit` comments for the given YouTube video/shorts URL.
    Returns a list of dictionaries with comment details.
    """
    video_id = extract_video_id(url)
    if not video_id:
        logger.error(f"Gagal mengekstrak Video ID dari URL: {url}")
        return []

    logger.info(f"Memulai pengambilan komentar untuk Video ID: {video_id} (Limit: {limit})")
    
    try:
        downloader = YoutubeCommentDownloader()
        # get_comments_from_url can handle shorts as well, or we can use get_comments(video_id)
        comments_generator = downloader.get_comments_from_url(url)
        
        comments_list = []
        count = 0
        
        for comment in comments_generator:
            if count >= limit:
                break
                
            comments_list.append({
                "comment_id": comment.get("cid"),
                "author": comment.get("author"),
                "text": comment.get("text"),
                "likes": comment.get("votes"),
                "time": comment.get("time")
            })
            count += 1
            if count % 20 == 0:
                logger.info(f"Berhasil mengambil {count} komentar...")
                
        logger.info(f"Selesai! Berhasil mengambil {len(comments_list)} komentar dari YouTube.")
        return comments_list
        
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat mengunduh komentar: {e}")
        return []

if __name__ == "__main__":
    # Test script locally
    test_url = "https://www.youtube.com/shorts/a3Irz3zv8L0"
    comments = fetch_youtube_comments(test_url, limit=5)
    for i, c in enumerate(comments):
        print(f"\n[{i+1}] {c['author']}: {c['text'][:100]}")
