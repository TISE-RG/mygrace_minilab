import json
import requests


DEFAULT_OLLAMA_URL = "http://100.112.221.112:11434"
DEFAULT_OLLAMA_MODEL = "gemma4"


def call_ollama(
    prompt: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL
) -> str:
    """
    Memanggil Ollama generate API.
    """
    ollama_url = ollama_url.rstrip("/")
    url = f"{ollama_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"[ERROR_OLLAMA] Tidak dapat menghubungi Ollama: {e}"
    except Exception as e:
        return f"[ERROR_OLLAMA] Terjadi kesalahan: {e}"


def get_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """
    Mengambil daftar model dari Ollama /api/tags.
    """
    ollama_url = ollama_url.rstrip("/")
    url = f"{ollama_url}/api/tags"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        models = []
        for item in data.get("models", []):
            name = item.get("name")
            if name:
                models.append(name)

        return models
    except Exception:
        return []


def build_story_prompt(raw_story: str, story_type: str, elder_name: str = "Bapak/Ibu") -> str:
    return f"""
Anda adalah GRACE Life Story Companion, pendamping naratif yang lembut untuk lansia.

Tugas Anda:
1. Membaca cerita singkat dari lansia.
2. Merapikan cerita menjadi narasi yang hangat, indah, jujur, dan mudah dipahami.
3. Tidak menambah fakta palsu.
4. Tidak menggurui.
5. Tidak membuat diagnosis psikologis.
6. Menjaga martabat lansia sebagai tokoh utama dan penulis kisah hidupnya sendiri.

Nama pengguna: {elder_name}
Jenis cerita: {story_type}

Cerita asli:
\"\"\"
{raw_story}
\"\"\"

Buat jawaban dalam format berikut:

JUDUL:
[beri judul pendek dan menyentuh]

NARASI BERMAKNA:
[tulis 1-3 paragraf naratif yang hangat dan sederhana]

MAKNA HIDUP:
[tulis 2-4 butir makna atau nilai hidup yang terlihat]

PERTANYAAN LANJUTAN:
[tulis 1 pertanyaan lembut untuk membantu pengguna melanjutkan cerita]

SARAN STIMULUS:
[pilih satu: naratif / relasional / spiritual / kontribusi / komunitas / praktis]

CATATAN KEHATI-HATIAN:
[tulis singkat bila cerita mengandung kesedihan, kesepian, konflik, kehilangan, atau kebutuhan dukungan manusia. Jika tidak ada, tulis: Tidak ada catatan khusus.]
"""


def generate_simple_story(
    raw_story: str,
    story_type: str,
    elder_name: str = "Bapak/Ibu",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL
) -> str:
    """
    Menghasilkan narasi bermakna dengan Ollama.
    """
    prompt = build_story_prompt(raw_story, story_type, elder_name)
    result = call_ollama(prompt, ollama_url=ollama_url, model=model)

    if result.startswith("[ERROR_OLLAMA]"):
        return f"""
JUDUL:
Cerita Hidup yang Berharga

NARASI BERMAKNA:
{elder_name} menceritakan: "{raw_story}"

Cerita ini adalah bagian dari kisah hidup yang berharga. Di dalamnya ada pengalaman, perasaan,
dan makna yang dapat disimpan sebagai bagian dari perjalanan hidup.

MAKNA HIDUP:
- Hidup tetap memiliki cerita.
- Pengalaman sehari-hari dapat menjadi sumber makna.
- Cerita ini dapat menjadi bagian dari warisan hidup.

PERTANYAAN LANJUTAN:
Apa hal paling penting yang ingin {elder_name} ingat dari cerita ini?

SARAN STIMULUS:
naratif

CATATAN KEHATI-HATIAN:
{result}
"""
    return result


def build_grace_tagging_prompt(raw_story: str, narrative: str) -> str:
    return f"""
Anda adalah analis GRACE Index untuk cerita lansia.

Berdasarkan cerita asli dan narasi bermakna berikut, beri skor 0-100 untuk enam dimensi:

N = Narrative Continuity: apakah cerita menunjukkan kesinambungan identitas dan makna diri.
S = Spiritual Grounding: apakah ada ketenangan batin, iman, syukur, penerimaan, atau nilai hidup.
R = Relational Vitality: apakah ada hubungan bermakna dengan keluarga, teman, komunitas.
C = Contribution and Usefulness: apakah pengguna merasa berguna, memberi, mengajar, menolong, mendoakan, atau mewariskan.
E = Community and World Engagement: apakah ada keterlibatan dengan dunia luar, komunitas, layanan, kegiatan.
A = Adaptive Resilience: apakah ada kemampuan bertahan, pulih, menerima, mencari bantuan, atau tetap berharap.

Cerita asli:
\"\"\"
{raw_story}
\"\"\"

Narasi bermakna:
\"\"\"
{narrative}
\"\"\"

Kembalikan HANYA JSON valid, tanpa markdown, tanpa penjelasan tambahan, dengan format:

{{
  "N": 70,
  "S": 70,
  "R": 70,
  "C": 70,
  "E": 70,
  "A": 70,
  "values": ["keluarga", "iman"],
  "emotion": ["tenang", "rindu"],
  "risk_notes": "singkat",
  "recommended_stimulus": "naratif"
}}
"""


def analyze_grace_with_llm(
    raw_story: str,
    narrative: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL
) -> dict:
    """
    Meminta Ollama memberi skor GRACE.
    Jika JSON gagal, pakai fallback rule-based ringan.
    """
    prompt = build_grace_tagging_prompt(raw_story, narrative)
    result = call_ollama(prompt, ollama_url=ollama_url, model=model)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        json_text = result[start:end]
        data = json.loads(json_text)

        for key in ["N", "S", "R", "C", "E", "A"]:
            data[key] = max(0, min(100, float(data.get(key, 70))))

        data.setdefault("values", [])
        data.setdefault("emotion", [])
        data.setdefault("risk_notes", "Tidak ada catatan khusus.")
        data.setdefault("recommended_stimulus", "naratif")

        return data

    except Exception:
        values = detect_values(raw_story)
        return {
            "N": 70,
            "S": 70 if "iman" in values else 60,
            "R": 75 if "keluarga" in values else 60,
            "C": 75 if "kontribusi" in values else 55,
            "E": 55,
            "A": 65 if "ketahanan" in values else 60,
            "values": values,
            "emotion": [],
            "risk_notes": "JSON dari LLM tidak valid; memakai fallback sederhana.",
            "recommended_stimulus": recommend_prompt(values)
        }


def detect_values(raw_story: str):
    text = raw_story.lower()
    values = []

    if "anak" in text or "cucu" in text or "keluarga" in text:
        values.append("keluarga")
    if "doa" in text or "tuhan" in text or "gereja" in text:
        values.append("iman")
    if "kerja" in text or "mengajar" in text or "nasihat" in text:
        values.append("kontribusi")
    if "sakit" in text or "sulit" in text or "sedih" in text or "sepi" in text:
        values.append("ketahanan")
    if "senang" in text or "syukur" in text:
        values.append("syukur")

    return values or ["makna hidup"]


def recommend_prompt(values):
    if "keluarga" in values:
        return "relasional"
    if "iman" in values:
        return "spiritual"
    if "kontribusi" in values:
        return "kontribusi"
    if "ketahanan" in values:
        return "naratif"
    return "naratif"