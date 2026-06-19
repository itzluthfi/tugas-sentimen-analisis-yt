import os
import requests
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
url = "https://integrate.api.nvidia.com/v1/chat/completions"

payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [
        {"role": "user", "content": "Halo, apakah kamu bisa merespons? Jawab dengan 'Ya, NVIDIA API berfungsi!' jika berhasil."}
    ],
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 50,
    "stream": False
}

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

print("=== Mengetes Koneksi NVIDIA NIM API ===")
if not api_key:
    print("Error: NVIDIA_API_KEY tidak ditemukan di file .env!")
    exit(1)
    
print(f"Menggunakan API Key: {api_key[:10]}...{api_key[-10:]}")
print(f"Menghubungi: {url}")
print(f"Model: {payload['model']}")

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"].strip()
        print("\nKONEKSI BERHASIL!")
        print(f"Respons LLM: {content}")
    else:
        print("\nKONEKSI GAGAL!")
        print(f"Detail Error:\n{response.text}")
except Exception as e:
    print(f"\nTerjadi kesalahan saat request: {e}")
