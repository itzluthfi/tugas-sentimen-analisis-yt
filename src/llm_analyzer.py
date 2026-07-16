import json
import logging
import requests
from src.config import (
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    CEREBRAS_API_KEY,
    CEREBRAS_MODEL,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_MODEL,
    CLOUDFLARE_FALLBACK_MODEL,
)

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
        Includes automatic retry with exponential backoff on rate limit (429) and server errors.
        """
        import time
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
        
        max_retries = 1
        backoff_factor = 2.0  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    response_json = response.json()
                    choices = response_json.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
                    return ""
                
                elif response.status_code in [429, 500, 502, 503, 504]:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Kesalahan API ({response.status_code}). Menunggu {wait_time:.1f} detik sebelum mencoba kembali (Percobaan {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"NVIDIA NIM API Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                # Handle network timeout/issues
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Koneksi error ({e}). Menunggu {wait_time:.1f} detik sebelum mencoba kembali (Percobaan {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                
        raise RuntimeError(f"Gagal menghubungi NVIDIA NIM API setelah {max_retries} kali percobaan.")

    def _call_cerebras_api(self, messages: list) -> str:
        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        cerebras_model_fallback = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cerebras_key}",
            "Content-Type": "application/json"
        }
        cerebras_model = self.model if self.model.startswith("gpt-") else cerebras_model_fallback
        payload = {
            "model": cerebras_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                res_json = response.json()
                choices = res_json.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            raise RuntimeError(f"Cerebras API Error {response.status_code}: {response.text}")
        except Exception as e:
            raise RuntimeError(f"Gagal memanggil Cerebras API: {e}")

    def _call_cloudflare_api(self, messages: list) -> str:
        cf_token = os.getenv("CLOUDFLARE_API_TOKEN")
        cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        cf_model_default = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        cf_model_fallback = os.getenv("CLOUDFLARE_FALLBACK_MODEL", "@cf/meta/llama-3.1-8b-instruct")
        
        if self.model.startswith("@cf/"):
            model_id = self.model
        elif self.model.startswith("meta/"):
            model_id = f"@cf/{self.model}"
        else:
            model_id = cf_model_default

        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/{model_id}"
        headers = {
            "Authorization": f"Bearer {cf_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                res_json = response.json()
                result = res_json.get("result", {})
                if "response" in result:
                    resp = result["response"]
                    if isinstance(resp, str):
                        return resp.strip()
                    else:
                        return json.dumps(resp)
            
            # Fallback ke model Cloudflare lain jika model utama tidak tersedia
            if response.status_code in [400, 404] and model_id != cf_model_fallback:
                logger.warning(f"Cloudflare model {model_id} tidak tersedia. Menghubungi fallback model {cf_model_fallback}...")
                url_fallback = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/{cf_model_fallback}"
                response_fallback = requests.post(url_fallback, json=payload, headers=headers, timeout=15)
                if response_fallback.status_code == 200:
                    res_json_fallback = response_fallback.json()
                    result_fallback = res_json_fallback.get("result", {})
                    if "response" in result_fallback:
                        resp_fallback = result_fallback["response"]
                        if isinstance(resp_fallback, str):
                            return resp_fallback.strip()
                        else:
                            return json.dumps(resp_fallback)
            
            raise RuntimeError(f"Cloudflare Workers AI Error {response.status_code}: {response.text}")
        except Exception as e:
            raise RuntimeError(f"Gagal memanggil Cloudflare API: {e}")

    def _call_api_with_fallbacks(self, messages: list) -> str:
        """
        Tries to call NVIDIA NIM API. If it fails, falls back to Cerebras AI.
        If Cerebras AI also fails, falls back to Cloudflare Workers AI.
        """
        # Dynamic reload of .env to get any newly saved secrets/keys instantly
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        api_key = os.getenv("NVIDIA_API_KEY")
        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        cf_token = os.getenv("CLOUDFLARE_API_TOKEN")
        cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        
        errors = []

        # 1. Coba NVIDIA NIM API jika tersedia
        if api_key:
            try:
                logger.info(f"Mencoba menghubungi NVIDIA NIM API (Model: {self.model})...")
                self.api_key = api_key
                return self._call_nvidia_api(messages)
            except Exception as e_nvidia:
                err_msg = f"NVIDIA NIM API gagal: {e_nvidia}"
                logger.warning(err_msg)
                errors.append(err_msg)
        else:
            errors.append("NVIDIA NIM dilewati (NVIDIA_API_KEY tidak dikonfigurasi).")

        # 2. Fallback Pertama: Cerebras AI
        if cerebras_key and cerebras_key.strip() != "":
            try:
                logger.info("Mencoba menghubungi Cerebras AI...")
                return self._call_cerebras_api(messages)
            except Exception as e_cerebras:
                err_msg = f"Cerebras AI API gagal: {e_cerebras}"
                logger.warning(err_msg)
                errors.append(err_msg)
        else:
            errors.append("Cerebras AI dilewati (CEREBRAS_API_KEY tidak dikonfigurasi).")

        # 3. Fallback Kedua: Cloudflare Workers AI
        if cf_token and cf_account:
            try:
                logger.info("Mencoba menghubungi Cloudflare Workers AI...")
                return self._call_cloudflare_api(messages)
            except Exception as e_cf:
                err_msg = f"Cloudflare Workers AI API gagal: {e_cf}"
                logger.error(err_msg)
                errors.append(err_msg)
        else:
            errors.append("Cloudflare Workers AI dilewati (CLOUDFLARE_API_TOKEN atau CLOUDFLARE_ACCOUNT_ID tidak dikonfigurasi).")

        # Jika semua gagal
        err_detail = " | ".join(errors)
        raise RuntimeError(f"Semua API gagal merespon. Detail: {err_detail}")

    def analyze_batch(self, comments: list[dict], video_context: str = None) -> list[dict]:
        """
        Analyzes a list of comments in a single API call (batching) to optimize speed and API quota.
        Each comment in the input list should be a dict with at least 'comment_id' and 'text'.
        Returns a list of dicts with 'comment_id', 'llm_sentiment', and 'llm_reason'.
        """
        # Dynamic check
        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("NVIDIA_API_KEY")
        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        cf_token = os.getenv("CLOUDFLARE_API_TOKEN")
        
        if not any([api_key, cerebras_key, cf_token]):
            raise ValueError(
                "Tidak ada API key LLM yang dikonfigurasi. Silakan tambahkan NVIDIA_API_KEY, CEREBRAS_API_KEY, atau CLOUDFLARE_API_TOKEN di file .env Anda."
            )

        # Structure the batch content for the LLM
        formatted_comments = []
        for i, c in enumerate(comments):
            formatted_comments.append(f"Index: {i} | ID: {c['comment_id']} | Teks: {c['text']}")
            
        comments_payload = "\n".join(formatted_comments)
        
        if video_context:
            system_prompt = (
                "Anda adalah asisten AI yang ahli dalam analisis sentimen teks Bahasa Indonesia, "
                "termasuk bahasa daerah (seperti Jawa, Sunda) dan singkatan/slang gaul internet.\n"
                f"Tugas Anda adalah menentukan sentimen beserta alasannya dari daftar komentar YouTube khusus dalam kaitannya dengan isi/konten video berikut:\n"
                f"=== KONTEKS VIDEO ===\n"
                f"{video_context}\n"
                f"=====================\n\n"
                "Kategori sentimen wajib berupa salah satu dari: 'positif', 'negatif', atau 'netral'.\n"
                "Aturan Sentimen Berdasarkan Konteks Video:\n"
                "- 'positif': Komentar yang menunjukkan rasa suka, apresiasi, pujian, dukungan, kesepakatan, ketertarikan, atau pujian terhadap isi video, kreator, pembicaraan, atau topik yang dibahas di video tersebut.\n"
                "- 'negatif': Komentar yang berisi kritik, keluhan, cacian, kekecewaan, ketidakpuasan, ketidaksepakatan, atau sentimen buruk terhadap isi video, kreator, pembicaraan, atau topik yang dibahas di video.\n"
                "- 'netral': Komentar yang bersifat general, pertanyaan biasa tanpa sentimen, spam link, percakapan di luar topik video, atau tidak menunjukkan sentimen positif/negatif yang jelas terhadap isi/topik video tersebut.\n\n"
                "Alasan (reason) harus berupa penjelasan singkat (1 kalimat pendek) dalam Bahasa Indonesia mengapa komentar tersebut dikategorikan ke dalam sentimen tersebut berdasarkan kaitannya dengan video.\n\n"
                "Format Output harus berupa JSON ARRAY murni yang berisi objek dengan format:\n"
                "[\n"
                "  {\"comment_id\": \"ID_KOMENTAR\", \"sentiment\": \"positif/negatif/netral\", \"reason\": \"alasan singkat\"},\n"
                "  ...\n"
                "]\n"
                "Jangan menambahkan teks penjelasan, pengantar, atau penutup apapun di luar JSON array tersebut."
            )
        else:
            system_prompt = (
                "Anda adalah asisten AI yang ahli dalam analisis sentimen teks Bahasa Indonesia, "
                "termasuk bahasa daerah (seperti Jawa, Sunda) dan singkatan/slang gaul internet.\n"
                "Tugas Anda adalah menentukan sentimen beserta alasannya dari daftar komentar YouTube yang diberikan secara global.\n\n"
                "Kategori sentimen wajib berupa salah satu dari: 'positif', 'negatif', atau 'netral'.\n"
                "Aturan Sentimen:\n"
                "- 'positif': Komentar berisi pujian, apresiasi, rasa senang, dukungan, kelucuan positif, atau rekomendasi bagus.\n"
                "- 'negatif': Komentar berisi kritik, keluhan, cacian, kekecewaan, ketidakpuasan, atau hujatan.\n"
                "- 'netral': Komentar berupa pertanyaan biasa, pernyataan umum, tidak menunjukkan emosi kuat, atau di luar konteks video.\n\n"
                "Alasan (reason) harus berupa penjelasan singkat (1 kalimat pendek) dalam Bahasa Indonesia mengapa komentar tersebut dikategorikan ke dalam sentimen tersebut.\n\n"
                "Format Output harus berupa JSON ARRAY murni yang berisi objek dengan format:\n"
                "[\n"
                "  {\"comment_id\": \"ID_KOMENTAR\", \"sentiment\": \"positif/negatif/netral\", \"reason\": \"alasan singkat\"},\n"
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
        raw_response = self._call_api_with_fallbacks(messages)
        
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
                reason = item.get("reason", "").strip()
                if sent not in ["positif", "negatif", "netral"]:
                    sent = "netral"
                sentiment_map[cid] = (sent, reason)
                
            # Verify we got a result for each comment, otherwise fill default
            output_results = []
            for c in comments:
                cid = c["comment_id"]
                res = sentiment_map.get(cid)
                if not res:
                    logger.warning(f"Komentar ID {cid} tidak ditemukan dalam output LLM. Menganalisis secara individu.")
                    sentiment, reason = self.analyze_single(c["text"], video_context)
                else:
                    sentiment, reason = res
                output_results.append({
                    "comment_id": cid,
                    "llm_sentiment": sentiment,
                    "llm_reason": reason
                })
                
            return output_results
            
        except Exception as e:
            logger.error(f"Gagal parse JSON dari respons LLM: {e}. Raw response: {raw_response[:200]}...")
            logger.info("Menjalankan fallback ke analisis satu per satu untuk batch ini.")
            return self._fallback_single(comments, video_context)

    def analyze_single(self, text: str, video_context: str = None) -> tuple[str, str]:
        """
        Analyzes a single comment. Useful for fallback.
        Returns: (sentiment, reason)
        """
        if video_context:
            system_prompt = (
                "Anda adalah ahli analisis sentimen teks Bahasa Indonesia.\n"
                f"Tentukan sentimen komentar YouTube ini khusus dalam kaitannya dengan isi/konten video berikut:\n"
                f"=== KONTEKS VIDEO ===\n"
                f"{video_context}\n"
                f"=====================\n\n"
                "Tentukan sentimen komentar YouTube ini menjadi: 'positif', 'negatif', atau 'netral' beserta alasan singkatnya (1 kalimat).\n"
                "Format Output harus berupa JSON murni dengan format: {\"sentiment\": \"positif/negatif/netral\", \"reason\": \"alasan singkat\"}"
            )
        else:
            system_prompt = (
                "Anda adalah ahli analisis sentimen teks Bahasa Indonesia.\n"
                "Tentukan sentimen komentar YouTube ini secara global menjadi: 'positif', 'negatif', atau 'netral' beserta alasan singkatnya (1 kalimat).\n"
                "Format Output harus berupa JSON murni dengan format: {\"sentiment\": \"positif/negatif/netral\", \"reason\": \"alasan singkat\"}"
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Komentar: {text}"}
        ]
        
        response = self._call_api_with_fallbacks(messages)
        import re
        clean_response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL).strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        try:
            res_json = json.loads(clean_response)
            sentiment = res_json.get("sentiment", "netral").lower().strip()
            reason = res_json.get("reason", "").strip()
        except Exception:
            sentiment = "netral"
            reason = "Gagal memproses alasan dari model."
            if "positif" in clean_response.lower():
                sentiment = "positif"
            elif "negatif" in clean_response.lower():
                sentiment = "negatif"
                
        if sentiment not in ["positif", "negatif", "netral"]:
            sentiment = "netral"
            
        return sentiment, reason

    def _fallback_single(self, comments: list[dict], video_context: str = None) -> list[dict]:
        """
        Fallback method that analyzes comments one by one.
        """
        import time
        results = []
        for c in comments:
            sentiment, reason = self.analyze_single(c["text"], video_context)
            results.append({
                "comment_id": c["comment_id"],
                "llm_sentiment": sentiment,
                "llm_reason": reason
            })
            time.sleep(0.5)
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
