import os
import html
import pandas as pd
from datetime import datetime

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
# Config
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://100.112.221.112:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

DATA_PATH = "data/telegram_stories.csv"

os.makedirs("data", exist_ok=True)


# =========================
# Helper
# =========================

def save_telegram_story(row: dict):
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(DATA_PATH, index=False)


def build_reply_markdown_legacy(narrative: str, grace_score: float, zone: str, grace_data: dict) -> str:
    values = ", ".join(grace_data.get("values", []))
    emotion = ", ".join(grace_data.get("emotion", []))
    stimulus = grace_data.get("recommended_stimulus", "naratif")
    risk_notes = grace_data.get("risk_notes", "Tidak ada catatan khusus.")

    return f"""
🌿 *MyGRACE telah membaca cerita Anda.*

{narrative}

📊 *GRACE Index*
Score: *{round(grace_score, 2)}*
Zone: *{zone}*

🏷️ *Nilai yang tampak:*
{values if values else "-"}

😊 *Emosi yang tampak:*
{emotion if emotion else "-"}

💡 *Stimulus yang disarankan:*
{stimulus}

⚠️ *Catatan:*
{risk_notes}
""".strip()


def escape_html(value) -> str:
    return html.escape(str(value), quote=False)


def build_reply(narrative: str, grace_score: float, zone: str, grace_data: dict) -> str:
    values = ", ".join(grace_data.get("values", []))
    emotion = ", ".join(grace_data.get("emotion", []))
    stimulus = grace_data.get("recommended_stimulus", "naratif")
    risk_notes = grace_data.get("risk_notes", "Tidak ada catatan khusus.")

    return f"""
<b>MyGRACE telah membaca cerita Anda.</b>

{escape_html(narrative)}

<b>GRACE Index</b>
Score: <b>{escape_html(round(grace_score, 2))}</b>
Zone: <b>{escape_html(zone)}</b>

<b>Nilai yang tampak:</b>
{escape_html(values if values else "-")}

<b>Emosi yang tampak:</b>
{escape_html(emotion if emotion else "-")}

<b>Stimulus yang disarankan:</b>
{escape_html(stimulus)}

<b>Catatan:</b>
{escape_html(risk_notes)}
""".strip()


# =========================
# Telegram Handlers
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = """
Halo, saya *MyGRACE Story Bot* 🌿

Kirimkan cerita singkat tentang:
- kejadian hari ini,
- kenangan masa lalu,
- atau harapan masa depan.

Contoh:
"Hari ini saya duduk di teras. Anak belum menelepon, tetapi tetangga menyapa saya."

Saya akan membantu merapikan cerita, menemukan makna, dan memberi analisis GRACE sederhana.
""".strip()

    await update.message.reply_text(message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = """
Cara pakai:

1. Kirim cerita langsung sebagai chat biasa.
2. Awali dengan salah satu tag bila mau:
   /hariini cerita...
   /masalalu cerita...
   /masadepan cerita...

Contoh:
/hariini Saya merasa sepi, tetapi cucu mengirim foto sekolahnya.
""".strip()

    await update.message.reply_text(message)


async def handle_story(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text.strip()

    if not text:
        await update.message.reply_text("Silakan kirim cerita terlebih dahulu.")
        return

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

    elder_name = user.first_name or "Bapak/Ibu"

    await update.message.reply_text(
        "Terima kasih. GRACE sedang membaca dan memaknai cerita ini..."
    )

    narrative = generate_simple_story(
        raw_story=text,
        story_type=story_type,
        elder_name=elder_name,
        ollama_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
    )

    grace_data = analyze_grace_with_llm(
        raw_story=text,
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
        "story_type": story_type,
        "raw_story": text,
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

    await update.message.reply_text(reply, parse_mode="HTML")


# =========================
# Main
# =========================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN belum diset. "
            "Set environment variable TELEGRAM_BOT_TOKEN terlebih dahulu."
        )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_story))
    app.add_handler(CommandHandler("hariini", handle_story))
    app.add_handler(CommandHandler("masalalu", handle_story))
    app.add_handler(CommandHandler("masadepan", handle_story))

    print("MyGRACE Telegram Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
