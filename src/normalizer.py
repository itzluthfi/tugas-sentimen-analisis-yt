import re

# Dictionary mapping common Indonesian slang and abbreviations to standard words.
SLANG_DICT = {
    "yg": "yang",
    "bgt": "banget",
    "dgn": "dengan",
    "krn": "karena",
    "aj": "saja",
    "aja": "saja",
    "gk": "tidak",
    "g": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "kaga": "tidak",
    "kagak": "tidak",
    "tdk": "tidak",
    "tp": "tapi",
    "sdh": "sudah",
    "uda": "sudah",
    "udah": "sudah",
    "kl": "kalau",
    "kalo": "kalau",
    "skrg": "sekarang",
    "blm": "belum",
    "belom": "belum",
    "bikin": "buat",
    "utk": "untuk",
    "sm": "sama",
    "jg": "juga",
    "lu": "kamu",
    "lo": "kamu",
    "loe": "kamu",
    "gw": "saya",
    "gue": "saya",
    "adl": "adalah",
    "bs": "bisa",
    "jd": "jadi",
    "dr": "dari",
    "pd": "pada",
    "bener": "benar",
    "beneran": "benar",
    "mantap": "bagus",
    "mantab": "bagus",
    "mantul": "bagus",
    "keren": "bagus",
    "kece": "bagus",
    "gokil": "bagus",
    "ngakak": "tawa",
    "makasih": "terima kasih",
    "thx": "terima kasih",
    "thanks": "terima kasih",
    "trims": "terima kasih",
    "kocak": "lucu",
    "gpp": "tidak apa-apa",
    "gpapa": "tidak apa-apa",
    "oke": "ok",
    "sngt": "sangat",
    "dapet": "dapat",
    "dpt": "dapat",
    "knp": "kenapa",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "tau": "tahu",
    "taw": "tahu",
    "gatau": "tidak tahu",
    "luar biasa": "hebat",
    "top": "bagus",
    "nih": "",
    "tuh": "",
    "kok": "",
    "sih": "",
    "lah": "",
    "dong": "",
    "deh": ""
}

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_SLANG_PATH = os.path.join(BASE_DIR, "lexicon", "custom_slang.json")

def load_slang_dict() -> dict:
    """
    Loads static slang dict merged with any custom slang from lexicon/custom_slang.json.
    """
    merged_dict = SLANG_DICT.copy()
    if os.path.exists(CUSTOM_SLANG_PATH):
        try:
            with open(CUSTOM_SLANG_PATH, "r", encoding="utf-8") as f:
                custom = json.load(f)
                if isinstance(custom, dict):
                    for k, v in custom.items():
                        merged_dict[str(k).strip().lower()] = str(v).strip().lower()
        except Exception:
            pass
    return merged_dict

def save_custom_slang(custom_dict: dict) -> bool:
    """
    Saves the custom slang mapping to lexicon/custom_slang.json.
    """
    try:
        os.makedirs(os.path.dirname(CUSTOM_SLANG_PATH), exist_ok=True)
        clean_dict = {}
        for k, v in custom_dict.items():
            k_str = str(k).strip().lower()
            v_str = str(v).strip().lower()
            if k_str:
                clean_dict[k_str] = v_str
        with open(CUSTOM_SLANG_PATH, "w", encoding="utf-8") as f:
            json.dump(clean_dict, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

def normalize_slang(text: str) -> str:
    """
    Normalizes slang words and abbreviations in a text to standard Indonesian words.
    Uses dynamically loaded slang dictionary.
    """
    if not text:
        return ""
    
    current_slang = load_slang_dict()
    
    # Tokenize by splitting on whitespace
    words = text.split()
    normalized_words = []
    
    for word in words:
        # Clean word from any trailing punctuation (e.g. "bgt!" -> "bgt")
        clean_word = re.sub(r"[^\w\s]", "", word)
        
        # Check if the lowercase version of the word is in our dictionary
        normalized = current_slang.get(clean_word.lower(), clean_word)
        
        # If normalized is empty string (e.g. for particles like 'sih', 'lah'), skip it
        if normalized == "":
            continue
            
        normalized_words.append(normalized)
        
    return " ".join(normalized_words)

if __name__ == "__main__":
    # Test normalization
    test_comment = "Keren bgt video nya, tp sayangnya agak lambat gk kayak biasanya. Gpp jg sih yg penting bermanfaat"
    print("Sebelum:", test_comment)
    print("Sesudah:", normalize_slang(test_comment))
