import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from story_agent import generate_simple_story, analyze_grace_with_llm
from grace_core import GraceState, calculate_grace_index, get_zone


# =========================
# Load environment
# =========================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://100.112.221.112:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "dolphin-mistral"
)

DATA_PATH = "data/telegram_stories.csv"

os.makedirs("data", exist_ok=True)


# =========================
# Helper functions
# =========================

def save_telegram_story(row: dict):
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(DATA_PATH, index=False)


def parse_story_type(text: str) -> tuple[str, str]:
    """
    Mengambil jenis cerita dari command Telegram.
    """
    text = text.strip()
    story_type = "Hari ini"

    if text.startswith("/hariini"):
        story_type = "Hari ini"
        text = text.replace("/hariini", "", 1).strip()
    elif text.startswith("/masalalu"):
        story_type = "Masa lalu"
        text = text.replace("/masalalu", "", 1).strip()
    elif text.startswith("/masadepan"):
        story_type = "Masa depan"
        text = text.replace("/masadepan", "", 1).strip()

    return story_type, text


def build_reply(narrative: str, grace_score: float, zone: str, grace_data: dict) -> str:
    values = ", ".join(grace_data.get("values", [])) or "-"
    emotion = ", ".join(grace_data.get("emotion", [])) or "-"
    stimulus = grace_data.get("recommended_stimulus", "naratif")
    risk_notes = grace_data.get("risk_notes", "Tidak ada catatan khusus.")

    return f"""
🌿 MyGRACE telah membaca cerita Anda.

{narrative}

📊 GRACE Index
Score: {round(grace_score, 2)}
Zone: {zone}

🏷️ Nilai yang tampak:
{values}

😊 Emosi yang tampak:
{emotion}

💡 Stimulus yang disarankan:
{stimulus}

⚠️ Catatan:
{risk_notes}
""".strip()


async def process_story(update: Update, text: str):
    user = update.effective_user
    story_type, clean_text = parse_story_type(text)

    if not clean_text:
        await update.message.reply_text(
            "Silakan kirim cerita setelah command.\n\n"
            "Contoh:\n"
            "/hariini Hari ini saya duduk di teras dan merasa agak sepi."
        )
        return

    elder_name = user.first_name or "Bapak/Ibu"

    await update.message.reply_text(
        "Terima kasih. GRACE sedang membaca dan memaknai cerita ini..."
    )

    narrative = generate_simple_story(
        raw_story=clean_text,
        story_type=story_type,
        elder_name=elder_name,
        ollama_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
    )

    grace_data = analyze_grace_with_llm(
        raw_story=clean_text,
        narrative=narrative,
        ollama_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
    )

    state = GraceState(
        N=grace_data["N"],
        S=grace_data["S"],
        R=grace_data["R"],
        C=grace_data["C"],
        E=grace_data["E"],
        A=grace_data["A"],
    )

    grace_score = calculate_grace_index(state)
    zone = get_zone(grace_score)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "telegram_user_id": user.id,
        "telegram_username": user.username,
        "telegram_first_name": user.first_name,
        "telegram_last_name": user.last_name,
        "story_type": story_type,
        "raw_story": clean_text,
        "narrative": narrative,
        "N": grace_data["N"],
        "S": grace_data["S"],
        "R": grace_data["R"],
        "C": grace_data["C"],
        "E": grace_data["E"],
        "A": grace_data["A"],
        "GRACE": round(grace_score, 2),
        "zone": zone,
        "values": ", ".join(grace_data.get("values", [])),
        "emotion": ", ".join(grace_data.get("emotion", [])),
        "recommended_stimulus": grace_data.get("recommended_stimulus", "naratif"),
        "risk_notes": grace_data.get("risk_notes", ""),
        "ollama_url": OLLAMA_URL,
        "model": OLLAMA_MODEL,
    }

    save_telegram_story(row)

    reply = build_reply(
        narrative=narrative,
        grace_score=grace_score,
        zone=zone,
        grace_data=grace_data,
    )

    # Plain text dipakai agar aman dari error Markdown akibat tanda baca dari LLM.
    await update.message.reply_text(reply)


# =========================
# Telegram handlers
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = """
Halo, saya MyGRACE Story Bot 🌿

Kirimkan cerita singkat tentang:
- kejadian hari ini,
- kenangan masa lalu,
- atau harapan masa depan.

Contoh:
Hari ini saya duduk di teras. Anak belum menelepon, tetapi tetangga menyapa saya.

Command yang tersedia:
/hariini cerita...
/masalalu cerita...
/masadepan cerita...
/help
""".strip()

    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = """
Cara pakai MyGRACE Story Bot:

1. Kirim cerita langsung sebagai chat biasa.
2. Atau gunakan command:

/hariini Hari ini saya merasa sepi, tetapi cucu mengirim foto sekolahnya.

/masalalu Dulu waktu saya masih mengajar, saya merasa hidup saya berarti.

/masadepan Saya ingin meninggalkan pesan untuk cucu saya agar hidup jujur dan takut akan Tuhan.

GRACE akan membantu:
- merapikan cerita,
- menemukan makna,
- menghitung GRACE Index,
- menyarankan stimulus lanjutan.
""".strip()

    await update.message.reply_text(message)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    await process_story(update, text)


async def handle_command_story(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    await process_story(update, text)


# =========================
# Main
# =========================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN belum diset. "
            "Isi file .env atau set environment variable TELEGRAM_BOT_TOKEN."
        )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(CommandHandler("hariini", handle_command_story))
    app.add_handler(CommandHandler("masalalu", handle_command_story))
    app.add_handler(CommandHandler("masadepan", handle_command_story))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    print("MyGRACE Telegram Bot is running...")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Ollama Model: {OLLAMA_MODEL}")

    app.run_polling()


if __name__ == "__main__":
    main()