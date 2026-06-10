import json
import time
import requests


DEFAULT_OLLAMA_URL = "http://100.112.221.112:11434"
DEFAULT_OLLAMA_MODEL = "gemma4"


# =========================
# Ollama utilities
# =========================

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


def benchmark_ollama(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    prompt: str | None = None
) -> dict:
    """
    Mengukur kecepatan Ollama untuk model aktif.

    Ollama mengembalikan durasi dalam nanosecond.
    Output token/sec = eval_count / eval_duration * 1e9
    Prompt token/sec = prompt_eval_count / prompt_eval_duration * 1e9
    """
    if prompt is None:
        prompt = (
            "Tuliskan satu paragraf pendek dalam bahasa Indonesia tentang "
            "seorang lansia yang menemukan makna dari hari yang sederhana."
        )

    ollama_url = ollama_url.rstrip("/")
    url = f"{ollama_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 120
        }
    }

    start_time = time.time()

    try:
        response = requests.post(url, json=payload, timeout=180)
        wall_time = time.time() - start_time

        response.raise_for_status()
        data = response.json()

        eval_count = data.get("eval_count", 0) or 0
        eval_duration = data.get("eval_duration", 0) or 0

        prompt_eval_count = data.get("prompt_eval_count", 0) or 0
        prompt_eval_duration = data.get("prompt_eval_duration", 0) or 0

        total_duration = data.get("total_duration", 0) or 0
        load_duration = data.get("load_duration", 0) or 0

        output_tokens_per_sec = (
            eval_count / eval_duration * 1_000_000_000
            if eval_count > 0 and eval_duration > 0
            else 0
        )

        prompt_tokens_per_sec = (
            prompt_eval_count / prompt_eval_duration * 1_000_000_000
            if prompt_eval_count > 0 and prompt_eval_duration > 0
            else 0
        )

        return {
            "ok": True,
            "server": ollama_url,
            "model": model,
            "wall_time_sec": round(wall_time, 3),
            "total_duration_sec": round(total_duration / 1_000_000_000, 3),
            "load_duration_sec": round(load_duration / 1_000_000_000, 3),
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_sec": round(prompt_eval_duration / 1_000_000_000, 3),
            "prompt_tokens_per_sec": round(prompt_tokens_per_sec, 2),
            "eval_count": eval_count,
            "eval_duration_sec": round(eval_duration / 1_000_000_000, 3),
            "output_tokens_per_sec": round(output_tokens_per_sec, 2),
            "response_preview": data.get("response", "")[:500],
            "raw": data
        }

    except Exception as e:
        return {
            "ok": False,
            "server": ollama_url,
            "model": model,
            "error": str(e)
        }


# =========================
# Story generation
# =========================

def build_story_prompt(
    raw_story: str,
    story_type: str,
    elder_name: str = "Bapak/Ibu"
) -> str:
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


# =========================
# GRACE scoring
# =========================

def build_grace_tagging_prompt(raw_story: str, narrative: str) -> str:
    return f"""
You are the GRACE Index scoring engine for elder wellbeing narratives.

Your task is to score a story across six GRACE dimensions from 0 to 100.

IMPORTANT RULES:
- Do NOT copy the example numbers.
- Do NOT give all dimensions the same score unless the story truly supports it.
- Use the full range of scores.
- Scores should reflect evidence found in the story.
- If a dimension is not mentioned or weakly supported, score it lower, typically 35-60.
- If a dimension is clearly present, score it 65-80.
- If a dimension is strongly present, score it 80-95.
- Avoid defaulting to 70.
- Return only valid JSON.

Dimension guide:

N = Narrative Continuity
High when the story shows identity, life meaning, memory continuity, self-understanding, or ability to connect events into a meaningful life story.
Low when the story is flat, confused, fragmented, or only reports events without meaning.

S = Spiritual Grounding
High when the story mentions faith, prayer, God, gratitude, acceptance, peace, moral values, or spiritual reflection.
Low when there is no spiritual or value-based grounding.

R = Relational Vitality
High when the story includes warm connection with family, friends, neighbors, community, or longing for meaningful contact.
Low when the story shows isolation, conflict, abandonment, or lack of connection.

C = Contribution and Usefulness
High when the person gives advice, helps, teaches, prays for others, serves, mentors, creates legacy, or feels useful.
Low when the story shows no contribution or feeling useless.

E = Community and World Engagement
High when the person is connected to church, neighborhood, events, services, marketplace, public space, or outside activities.
Low when the story is confined, withdrawn, or disconnected from the wider world.

A = Adaptive Resilience
High when the person copes, recovers, reframes difficulty, seeks help, remains hopeful, or finds strength.
Low when the story shows distress, helplessness, despair, or inability to respond.

Scoring calibration:
- 0-29: very weak or severely disrupted
- 30-49: weak / disrupted
- 50-64: vulnerable or limited
- 65-79: stable
- 80-100: strong / flourishing

Story text:
\"\"\"
{raw_story}
\"\"\"

Generated narrative:
\"\"\"
{narrative}
\"\"\"

Return only JSON with this exact structure:
{{
  "N": <number 0-100>,
  "S": <number 0-100>,
  "R": <number 0-100>,
  "C": <number 0-100>,
  "E": <number 0-100>,
  "A": <number 0-100>,
  "values": ["value1", "value2"],
  "emotion": ["emotion1", "emotion2"],
  "risk_notes": "short note",
  "recommended_stimulus": "naratif | relasional | spiritual | kontribusi | komunitas | praktis",
  "scoring_reason": {{
    "N": "short reason",
    "S": "short reason",
    "R": "short reason",
    "C": "short reason",
    "E": "short reason",
    "A": "short reason"
  }}
}}
"""


def is_low_variance_default_score(data: dict) -> bool:
    scores = [float(data.get(k, 70)) for k in ["N", "S", "R", "C", "E", "A"]]
    all_70 = all(abs(s - 70) < 0.01 for s in scores)
    very_low_variance = max(scores) - min(scores) <= 2
    return all_70 or very_low_variance


def heuristic_grace_score(raw_story: str, narrative: str) -> dict:
    """
    Lightweight deterministic scoring to avoid flat LLM defaults.
    This is not the final scientific model, but it gives useful variance for early testing.
    """
    text = f"{raw_story} {narrative}".lower()

    N = 55
    S = 45
    R = 45
    C = 40
    E = 40
    A = 50

    # Narrative continuity
    if any(w in text for w in [
        "ingat", "dulu", "masa lalu", "waktu kecil", "pengalaman",
        "cerita", "hidup saya", "kenangan"
    ]):
        N += 15

    if any(w in text for w in [
        "makna", "pelajaran", "berarti", "bersyukur", "menyadari",
        "mengingatkan", "nilai"
    ]):
        N += 15

    # Spiritual grounding
    if any(w in text for w in [
        "tuhan", "doa", "gereja", "iman", "bersyukur", "syukur",
        "berkat", "pengharapan", "damai", "menerima"
    ]):
        S += 25

    # Relational vitality
    if any(w in text for w in [
        "anak", "cucu", "keluarga", "tetangga", "teman", "sahabat",
        "istri", "suami", "orang lain"
    ]):
        R += 25

    if any(w in text for w in [
        "sepi", "sendiri", "belum menelepon", "tidak menelepon",
        "rindu", "dilupakan"
    ]):
        R -= 15

    # Contribution and usefulness
    if any(w in text for w in [
        "mengajar", "nasihat", "menolong", "membantu", "melayani",
        "mendoakan", "warisan", "pesan untuk cucu", "memberi",
        "berguna"
    ]):
        C += 30

    # Community and world engagement
    if any(w in text for w in [
        "teras", "tetangga", "gereja", "komunitas", "pasar", "acara",
        "sekolah", "pelayanan", "persekutuan", "lingkungan"
    ]):
        E += 20

    # Adaptive resilience
    if any(w in text for w in [
        "tetap", "bertahan", "kuat", "sabar", "menerima",
        "berharap", "bangkit", "tidak menyerah"
    ]):
        A += 20

    if any(w in text for w in [
        "sedih", "sakit", "lemah", "takut", "kehilangan", "duka",
        "capek", "cemas"
    ]):
        A -= 10

    def clamp(x):
        return max(0, min(100, round(x, 1)))

    scores = {
        "N": clamp(N),
        "S": clamp(S),
        "R": clamp(R),
        "C": clamp(C),
        "E": clamp(E),
        "A": clamp(A),
    }

    lowest = min(scores, key=scores.get)

    stimulus_map = {
        "N": "naratif",
        "S": "spiritual",
        "R": "relasional",
        "C": "kontribusi",
        "E": "komunitas",
        "A": "praktis",
    }

    values = detect_values(raw_story)

    return {
        "N": scores["N"],
        "S": scores["S"],
        "R": scores["R"],
        "C": scores["C"],
        "E": scores["E"],
        "A": scores["A"],
        "values": values,
        "emotion": [],
        "risk_notes": "Heuristic scoring used because LLM returned low-variance default scores.",
        "recommended_stimulus": stimulus_map[lowest],
        "scoring_reason": {
            "N": "Heuristic estimate based on narrative meaning and memory cues.",
            "S": "Heuristic estimate based on spiritual or gratitude cues.",
            "R": "Heuristic estimate based on relational and loneliness cues.",
            "C": "Heuristic estimate based on usefulness and contribution cues.",
            "E": "Heuristic estimate based on outside-world or community cues.",
            "A": "Heuristic estimate based on resilience and difficulty cues."
        }
    }


def analyze_grace_with_llm(
    raw_story: str,
    narrative: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL
) -> dict:
    """
    Meminta Ollama memberi skor GRACE.
    Jika JSON gagal atau skor terlalu flat, pakai heuristic guardrail.
    """
    prompt = build_grace_tagging_prompt(raw_story, narrative)
    result = call_ollama(prompt, ollama_url=ollama_url, model=model)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1

        if start == -1 or end <= start:
            raise ValueError("No valid JSON object found in LLM response.")

        json_text = result[start:end]
        data = json.loads(json_text)

        for key in ["N", "S", "R", "C", "E", "A"]:
            data[key] = max(0, min(100, float(data.get(key, 70))))

        data.setdefault("values", [])
        data.setdefault("emotion", [])
        data.setdefault("risk_notes", "Tidak ada catatan khusus.")
        data.setdefault("recommended_stimulus", "naratif")
        data.setdefault("scoring_reason", {})

        if is_low_variance_default_score(data):
            corrected = heuristic_grace_score(raw_story, narrative)
            corrected["raw_llm_scoring_response"] = result
            corrected["scoring_source"] = "heuristic_low_variance_guardrail"
            return corrected

        data["raw_llm_scoring_response"] = result
        data["scoring_source"] = "llm"
        return data

    except Exception:
        fallback = heuristic_grace_score(raw_story, narrative)
        fallback["raw_llm_scoring_response"] = result if "result" in locals() else ""
        fallback["scoring_source"] = "heuristic_json_parse_failed"
        fallback["risk_notes"] = (
            "Heuristic scoring used because LLM JSON parsing failed."
        )
        return fallback


# =========================
# Simulated daily story for Digital Twin
# =========================

def build_simulated_daily_story_prompt(
    profile: dict,
    event_name: str,
    grace_before: dict
) -> str:
    return f"""
Anda adalah GRACE Daily Story Creation Engine.

Tugas Anda adalah membuat cerita harian singkat untuk digital twin lansia berdasarkan profil dan kejadian hari ini.

Profil lansia:
Nama: {profile.get("name")}
Usia: {profile.get("age")}
Latar belakang: {profile.get("background")}
Peran hidup: {profile.get("roles")}
Nilai utama: {profile.get("values")}
Kekuatan: {profile.get("strengths")}
Kerentanan: {profile.get("vulnerabilities")}

Kondisi GRACE sebelum kejadian:
{grace_before}

Kejadian hari ini:
{event_name}

Buat keluaran dalam format berikut:

JUDUL:
[judul pendek]

CERITA HARIAN:
[1-2 paragraf dari sudut pandang lansia, hangat, realistis, tidak berlebihan]

MAKNA:
[2-3 butir makna hidup]

HERO JOURNEY TAGS:
Ordinary World:
Trial:
Guide:
Courage:
Insight:
Return:

REKOMENDASI STIMULUS:
[pilih satu: naratif / relasional / spiritual / kontribusi / komunitas / praktis / none]

CATATAN:
[catatan singkat risiko atau peluang]
"""


def generate_simulated_daily_story(
    profile: dict,
    event_name: str,
    grace_before: dict,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL
) -> str:
    prompt = build_simulated_daily_story_prompt(
        profile=profile,
        event_name=event_name,
        grace_before=grace_before
    )
    return call_ollama(prompt, ollama_url=ollama_url, model=model)


# =========================
# Fallback value detection
# =========================

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

    if "senang" in text or "syukur" in text or "bersyukur" in text:
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