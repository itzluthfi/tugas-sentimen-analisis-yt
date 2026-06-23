import json
import logging
import requests
from src.config import NVIDIA_API_KEY, NVIDIA_MODEL

logger = logging.getLogger(__name__)

class LLMSentimentAnalyzer:
    def __init__(self, model: str = None):
        self.url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.api_key = NVIDIA_API_KEY
        self.model = model if model else NVIDIA_MODEL

        if not self.api_key:
            logger.warning("Peringatan: NVIDIA_API_KEY tidak ditemukan di file .env. Analisis LLM akan dilewati atau gagal.")

    def _call_nvidia_api(self, messages: list) -> str:
        """
        Makes a POST request to the NVIDIA NIM API with the given messages payload.
        """
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 1024,
            "stream": False
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"NVIDIA NIM API Error {response.status_code}: {response.text}")
            
        response_json = response.json()
        
        # Extract content from response
        choices = response_json.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    def analyze_batch(self, comments: list[dict]) -> list[dict]:
        """
        Analyzes a list of comments in a single API call (batching) to optimize speed and API quota.
        Each comment in the input list should be a dict with at least 'comment_id' and 'text'.
        Returns a list of dicts with 'comment_id' and 'llm_sentiment'.
        """
        if not self.api_key:
            raise ValueError("NVIDIA API Key tidak ditemukan. Silakan konfigurasi file .env Anda.")

        # Structure the batch content for the LLM
        formatted_comments = []
        for i, c in enumerate(comments):
            formatted_comments.append(f"Index: {i} | ID: {c['comment_id']} | Teks: {c['text']}")
            
        comments_payload = "\n".join(formatted_comments)
        
        system_prompt = (
            "Anda adalah asisten AI yang ahli dalam analisis sentimen teks Bahasa Indonesia, "
            "termasuk bahasa daerah (seperti Jawa, Sunda) dan singkatan/slang gaul internet.\n"
            "Tugas Anda adalah menentukan sentimen dari daftar komentar YouTube yang diberikan.\n\n"
            "Kategori sentimen wajib berupa salah satu dari: 'positif', 'negatif', atau 'netral'.\n"
            "Aturan Sentimen:\n"
            "- 'positif': Komentar berisi pujian, apresiasi, rasa senang, dukungan, kelucuan positif, atau rekomendasi bagus.\n"
            "- 'negatif': Komentar berisi kritik, keluhan, cacian, kekecewaan, ketidakpuasan, atau hujatan.\n"
            "- 'netral': Komentar berupa pertanyaan biasa, pernyataan umum, tidak menunjukkan emosi kuat, atau di luar konteks video.\n\n"
            "Format Output harus berupa JSON ARRAY murni yang berisi objek dengan format:\n"
            "[\n"
            "  {\"comment_id\": \"ID_KOMENTAR\", \"sentiment\": \"positif/negatif/netral\"},\n"
            "  ...\n"
            "]\n"
            "Jangan menambahkan teks penjelasan, pengantar, atau penutup apapun di luar JSON array tersebut."
        )
        
        user_prompt = f"Analisis sentimen untuk komentar-komentar berikut:\n\n{comments_payload}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.info(f"Mengirim batch analisis sentimen berisi {len(comments)} komentar ke NVIDIA LLM...")
        raw_response = self._call_nvidia_api(messages)
        
        if not raw_response:
            raise RuntimeError("Respons dari LLM kosong.")

        try:
            import re
            # Clean response from markdown blocks and thought blocks if any
            clean_response = raw_response.strip()
            # Strip reasoning process tags (<thought>...</thought>)
            clean_response = re.sub(r'<thought>.*?</thought>', '', clean_response, flags=re.DOTALL).strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            results = json.loads(clean_response)
            
            # Map results to original comment_ids
            sentiment_map = {}
            for item in results:
                cid = item.get("comment_id")
                sent = item.get("sentiment", "netral").lower().strip()
                if sent not in ["positif", "negatif", "netral"]:
                    sent = "netral"
                sentiment_map[cid] = sent
                
            # Verify we got a result for each comment, otherwise fill default
            output_results = []
            for c in comments:
                cid = c["comment_id"]
                sentiment = sentiment_map.get(cid)
                if not sentiment:
                    logger.warning(f"Komentar ID {cid} tidak ditemukan dalam output LLM. Menganalisis secara individu.")
                    sentiment = self.analyze_single(c["text"])
                output_results.append({
                    "comment_id": cid,
                    "llm_sentiment": sentiment
                })
                
            return output_results
            
        except Exception as e:
            logger.error(f"Gagal parse JSON dari respons LLM: {e}. Raw response: {raw_response[:200]}...")
            logger.info("Menjalankan fallback ke analisis satu per satu untuk batch ini.")
            return self._fallback_single(comments)

    def analyze_single(self, text: str) -> str:
        """
        Analyzes a single comment. Useful for fallback.
        """
        system_prompt = (
            "Anda adalah ahli analisis sentimen teks Bahasa Indonesia.\n"
            "Tentukan sentimen komentar YouTube ini menjadi: 'positif', 'negatif', atau 'netral'.\n"
            "Output Anda harus HANYA kata kategorinya saja ('positif', 'negatif', atau 'netral') tanpa penjelasan."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Komentar: {text}"}
        ]
        
        response = self._call_nvidia_api(messages)
        import re
        clean_response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL).strip()
        sentiment = clean_response.lower().strip()
        
        if sentiment in ["positif", "negatif", "netral"]:
            return sentiment
        # Regex search in case of extra words
        if "positif" in sentiment: return "positif"
        if "negatif" in sentiment: return "negatif"
        return "netral"

    def _fallback_single(self, comments: list[dict]) -> list[dict]:
        """
        Fallback method that analyzes comments one by one.
        """
        results = []
        for c in comments:
            sentiment = self.analyze_single(c["text"])
            results.append({
                "comment_id": c["comment_id"],
                "llm_sentiment": sentiment
            })
        return results

if __name__ == "__main__":
    # Test script locally
    logging.basicConfig(level=logging.INFO)
    analyzer = LLMSentimentAnalyzer()
    
    test_comments = [
        {"comment_id": "c1", "text": "Keren banget bang! Kontennya sangat mendidik."},
        {"comment_id": "c2", "text": "Halah konten sampah gini mending dihapus aja bikin rugi kuota."},
        {"comment_id": "c3", "text": "Ini lokasi syutingnya di mana ya kalau boleh tau?"}
    ]
    
    results = analyzer.analyze_batch(test_comments)
    for r in results:
        print(r)
