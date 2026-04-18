"""
Wasurenai Telegram Bot
A warm companion for persons with dementia and their caregivers.

Features:
  /start           — Warm welcome
  /help            — List all commands
  /diagnosis       — Guidance after suspected self-diagnosis
  /checkin         — Voice check-in (placeholder for audio anomaly detection)
    /setkeystroke    — Set your patient keystroke ID
  /addmed          — Add a medicine reminder
  /addappt         — Add an appointment reminder
  /list            — List all your reminders
  /delete          — Remove a reminder
  /cancel          — Cancel the current conversation
"""
import logging
import sqlite3
import os
import tempfile
import warnings
import asyncio
import re
from datetime import time, timezone, timedelta

import librosa
import numpy as np
import requests

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")  # set this in your shell
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("WASURENAI_DB_PATH", os.path.join(BASE_DIR, "wasurenai.db"))
AUDIO_ARCHIVE_DIR = os.environ.get("AUDIO_ARCHIVE_DIR", os.path.join(BASE_DIR, "audio_checkins"))
USE_REAL_ANALYSIS = os.environ.get("USE_REAL_ANALYSIS", "0").lower() in {"1", "true", "yes", "on"}
VOICE_ANALYSIS_ENDPOINT = os.environ.get(
    "VOICE_ANALYSIS_ENDPOINT",
    "https://wasurenaibackend.uiutech.xyz/submit",
)
VOICE_ANALYSIS_TIMEOUT_SECONDS = float(os.environ.get("VOICE_ANALYSIS_TIMEOUT_SECONDS", "10"))
WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://wasurenai.uiutech.xyz")

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
# Quiet down the noisy httpx logger
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
GMT_PLUS_8 = timezone(timedelta(hours=8))

# Ignore noisy environment-level deprecation warnings unrelated to bot logic.
warnings.filterwarnings(
    "ignore",
    message=".*TripleDES has been moved.*",
    category=DeprecationWarning,
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables used by reminders and doctor dashboard."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    INTEGER NOT NULL,
            kind       TEXT    NOT NULL,          -- 'medicine' or 'appointment'
            text       TEXT    NOT NULL,
            hour       INTEGER NOT NULL,
            minute     INTEGER NOT NULL
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS checkins (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id       INTEGER NOT NULL,
            source        TEXT    NOT NULL,          -- 'voice_note' or 'audio_file'
            voice_status  TEXT    NOT NULL,          -- 'normal', 'mild_anomaly', 'high_anomaly'
            anomaly_score REAL,
            audio_file_path TEXT,
            created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_checkins_chat_time
        ON checkins (chat_id, created_at DESC)
        """
    )
    _ensure_column_exists(c, "checkins", "audio_file_path", "TEXT")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_checkin_settings (
            chat_id     INTEGER PRIMARY KEY,
            enabled     INTEGER NOT NULL DEFAULT 0,
            hour        INTEGER NOT NULL DEFAULT 20,
            minute      INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            chat_id               INTEGER PRIMARY KEY,
            patient_keystroke_id  TEXT UNIQUE,
            created_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS keystroke_checkins (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_keystroke_id TEXT NOT NULL,
            status               TEXT NOT NULL,
            anomaly_score        REAL,
            created_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_keystroke_checkins_pid_time
        ON keystroke_checkins (patient_keystroke_id, created_at DESC)
        """
    )
    conn.commit()
    conn.close()


def _ensure_column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str, column_def: str) -> None:
    cols = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in cols}
    if column_name in existing:
        return
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def add_reminder(chat_id: int, kind: str, text: str, hour: int, minute: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders (chat_id, kind, text, hour, minute) VALUES (?, ?, ?, ?, ?)",
        (chat_id, kind, text, hour, minute),
    )
    reminder_id = c.lastrowid
    conn.commit()
    conn.close()
    return reminder_id


def get_reminders(chat_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, kind, text, hour, minute FROM reminders WHERE chat_id = ? ORDER BY hour, minute",
        (chat_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_reminders():
    """Used on startup to reschedule every saved reminder."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, chat_id, kind, text, hour, minute FROM reminders")
    rows = c.fetchall()
    conn.close()
    return rows


def delete_reminder(chat_id: int, reminder_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "DELETE FROM reminders WHERE id = ? AND chat_id = ?",
        (reminder_id, chat_id),
    )
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def add_checkin(
    chat_id: int,
    source: str,
    voice_status: str,
    anomaly_score: float | None,
    audio_file_path: str | None,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO checkins (chat_id, source, voice_status, anomaly_score, audio_file_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (chat_id, source, voice_status, anomaly_score, audio_file_path),
    )
    conn.commit()
    conn.close()


def get_all_audio_checkin_settings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT chat_id, enabled, hour, minute
        FROM audio_checkin_settings
        ORDER BY chat_id
        """
    )
    rows = c.fetchall()
    conn.close()
    return rows


def ensure_patient_exists(chat_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO patients (chat_id)
        VALUES (?)
        ON CONFLICT(chat_id) DO NOTHING
        """,
        (chat_id,),
    )
    conn.commit()
    conn.close()


def set_patient_keystroke_id(chat_id: int, patient_keystroke_id: str) -> bool:
    ensure_patient_exists(chat_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        UPDATE patients
        SET patient_keystroke_id = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chat_id = ?
        """,
        (patient_keystroke_id, chat_id),
    )
    changed = c.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def get_patient_keystroke_id(chat_id: int) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute(
        "SELECT patient_keystroke_id FROM patients WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return row[0]


# ---------------------------------------------------------------------------
# Reminder firing
# ---------------------------------------------------------------------------

async def send_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback that runs at the scheduled time and sends the reminder message."""
    job = context.job
    data = job.data
    kind = data["kind"]
    text = data["text"]

    if kind == "medicine":
        message = (
            f"💊  Gentle reminder: it's time for your {text}.\n\n"
            "Take your time, and have some water nearby. "
            "You're doing wonderfully."
        )
    else:
        message = (
            f"📅  Gentle reminder about your appointment:\n\n"
            f"   {text}\n\n"
            "No rush. Take a deep breath, and go when you're ready."
        )

    await context.bot.send_message(chat_id=job.chat_id, text=message)


async def send_audio_checkin_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(
        chat_id=context.job.chat_id,
        text=(
            "Hi there 🌼 Gentle audio check-in reminder.\n\n"
            "When you're ready, type /checkin and send me a short voice note."
        ),
    )


def schedule_reminder(application, reminder_id, chat_id, kind, text, hour, minute):
    """Register a daily job with the JobQueue."""
    application.job_queue.run_daily(
        send_reminder,
        time=time(hour=hour, minute=minute),
        chat_id=chat_id,
        name=f"reminder_{reminder_id}",
        data={"kind": kind, "text": text},
    )


def sync_audio_checkin_jobs(application: Application) -> None:
    """Ensure daily audio check-in reminder jobs match DB settings."""
    settings = get_all_audio_checkin_settings()
    for chat_id, enabled, hour, minute in settings:
        job_name = f"audio_checkin_{chat_id}"
        existing = application.job_queue.get_jobs_by_name(job_name)
        if not enabled:
            for job in existing:
                job.schedule_removal()
            continue

        if existing:
            continue

        application.job_queue.run_daily(
            send_audio_checkin_reminder,
            time=time(hour=hour, minute=minute),
            chat_id=chat_id,
            name=job_name,
            data={"kind": "audio_checkin"},
        )


async def refresh_audio_checkin_jobs(context: ContextTypes.DEFAULT_TYPE) -> None:
    sync_audio_checkin_jobs(context.application)


# ---------------------------------------------------------------------------
# Basic commands
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_patient_exists(update.effective_chat.id)
    name = update.effective_user.first_name or "friend"
    await update.message.reply_text(
        f"Hello {name} 🌸\n\n"
        "I'm Wasurenai, your gentle companion. I'm here to help you remember "
        "what matters and to walk alongside you.\n\n"
        "Here's what I can do:\n"
        "  /diagnosis  — guidance if you're worried about memory\n"
        "  /checkin    — a quick voice check-in with me\n"
        "  /setkeystroke <id> — set your keystroke patient ID\n"
        "  /addmed     — set a medicine reminder\n"
        "  /addappt    — set an appointment reminder\n"
        "  /list       — see your reminders\n"
        "  /delete     — remove a reminder\n"
        "  /help       — show this list again\n\n"
        "Before we begin keystroke tracking:\n"
        f"1) Download our keyboard app from {WEBSITE_URL}\n"
        "2) Get your patient keystroke ID from the keyboard app\n"
        "3) Send it here with /setkeystroke <id>\n\n"
        "Take your time. I'm always here. 💛"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Here's everything I can do:\n\n"
        "  /diagnosis  — guidance if you're worried about memory\n"
        "  /checkin    — a quick voice check-in\n"
        "  /setkeystroke <id> — set your keystroke patient ID\n"
        "  /addmed     — set a medicine reminder\n"
        "  /addappt    — set an appointment reminder\n"
        "  /list       — see your reminders\n"
        "  /delete     — remove a reminder\n"
        "  /cancel     — cancel what we're doing right now\n\n"
        "Don't worry about remembering all of this. "
        "Just type /help whenever you need me. 🌼"
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["awaiting_delete"] = False
    context.user_data["awaiting_voice"] = False
    await update.message.reply_text(
        "That's alright, we'll stop here. 🌷\nType /help whenever you're ready.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# /diagnosis — guidance after self-diagnosis
# ---------------------------------------------------------------------------

async def diagnosis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I'm really glad you reached out. 💛\n\n"
        "If you've been noticing little changes — forgetting words, misplacing "
        "things, feeling a bit lost in familiar places — please know you're not "
        "alone, and there are gentle next steps we can take together:\n\n"
        "1️⃣  *Talk to someone you trust.* A family member or friend can support "
        "you through this.\n\n"
        "2️⃣  *See your doctor soon.* Early conversations lead to better care. "
        "Tell them what you've been noticing. It helps to bring a short list.\n\n"
        "3️⃣  *Keep using me.* I can remind you about the appointment once you "
        "book it. Just type /addappt and I'll help.\n\n"
        "4️⃣  *Be kind to yourself.* This doesn't define you. "
        "One small step at a time is more than enough. 🌱\n\n"
        "Would you like to set a reminder to book that appointment? "
        "Type /addappt whenever you're ready.",
    )


# ---------------------------------------------------------------------------
# /checkin — placeholder for audio anomaly detection
# ---------------------------------------------------------------------------

async def checkin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi there 🌼 Let's do a little check-in.\n\n"
        "Please send me a voice note or an audio file — anything you like! "
        "You could tell me about your day, read a short sentence, "
        "or just say hello.\n\n"
        "I'll listen carefully. 💛"
    )
    context.user_data["awaiting_voice"] = True


async def set_keystroke_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Assign or update the user's patient keystroke ID."""
    if not context.args:
        await update.message.reply_text(
            "Get our keyboard from wasurenai.uiutech.xyz/keyboard\n"
            "Please provide your keystroke ID like this:\n"
            "/setkeystroke [your-id]\n\n"
            "3-64 characters: letters, numbers, dash, underscore."
        )
        return

    keystroke_id = context.args[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", keystroke_id):
        await update.message.reply_text(
            "That ID format isn't valid. Use 3-64 characters with letters, "
            "numbers, dash, or underscore only."
        )
        return

    chat_id = update.effective_chat.id
    try:
        changed = set_patient_keystroke_id(chat_id, keystroke_id)
    except sqlite3.IntegrityError:
        await update.message.reply_text(
            "That keystroke ID is already linked to another patient. "
            "Please use a different ID."
        )
        return

    if changed:
        await update.message.reply_text(
            f"Done 🌷 Your keystroke patient ID is now set to: {keystroke_id}"
        )
    else:
        await update.message.reply_text(
            "I couldn't update your keystroke ID right now. Please try again."
        )


# ---------------------------------------------------------------------------
# /addmed — conversation to add a medicine reminder
# ---------------------------------------------------------------------------

MED_NAME, MED_TIME = range(2)
APPT_TEXT, APPT_TIME = range(2, 4)


async def addmed_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Of course 🌸 Let's set up a medicine reminder.\n\n"
        "What medicine should I remind you about?\n"
        "(You can type something like: *blue blood pressure pill*)\n\n"
        "Type /cancel any time to stop.",
    )
    return MED_NAME


async def addmed_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["med_name"] = update.message.text.strip()
    await update.message.reply_text(
        "Thank you 💛 What time each day should I remind you?\n\n"
        "Please type the time in 24-hour format, like *08:00* for 8 in the "
        "morning, or *20:30* for half past 8 in the evening.",
    )
    return MED_TIME


async def addmed_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    parsed = parse_time(text)
    if parsed is None:
        await update.message.reply_text(
            "Hmm, I didn't quite catch that 🌼\n\n"
            "Could you type the time like *08:00* or *20:30*?\n"
            "Or type /cancel to stop.",
        )
        return MED_TIME

    hour, minute = parsed
    med_name = context.user_data["med_name"]
    chat_id = update.effective_chat.id

    reminder_id = add_reminder(chat_id, "medicine", med_name, hour, minute)
    schedule_reminder(
        context.application, reminder_id, chat_id, "medicine", med_name, hour, minute
    )

    await update.message.reply_text(
        f"All set 🌷\n\n"
        f"I'll gently remind you to take your {med_name} "
        f"every day at {hour:02d}:{minute:02d}.\n\n"
        "You can type /list any time to see your reminders.",
    )
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# /addappt — conversation to add an appointment reminder
# ---------------------------------------------------------------------------

async def addappt_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Wonderful 🌸 Let's add an appointment reminder.\n\n"
        "What's the appointment?\n"
        "(Something like: *Doctor visit with Dr. Tan on Friday*)\n\n"
        "Type /cancel any time to stop.",
    )
    return APPT_TEXT


async def addappt_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["appt_text"] = update.message.text.strip()
    await update.message.reply_text(
        "Thank you 💛 What time of day should I remind you?\n\n"
        "Please type the time in 24-hour format, like *09:00* or *14:30*.",
    )
    return APPT_TIME


async def addappt_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    parsed = parse_time(text)
    if parsed is None:
        await update.message.reply_text(
            "Hmm, I didn't quite catch that 🌼\n\n"
            "Could you type the time like *09:00* or *14:30*?\n"
            "Or type /cancel to stop.",
        )
        return APPT_TIME

    hour, minute = parsed
    appt_text = context.user_data["appt_text"]
    chat_id = update.effective_chat.id

    reminder_id = add_reminder(chat_id, "appointment", appt_text, hour, minute)
    schedule_reminder(
        context.application, reminder_id, chat_id, "appointment", appt_text, hour, minute
    )

    await update.message.reply_text(
        f"All set 🌷\n\n"
        f"I'll gently remind you about {appt_text} "
        f"at {hour:02d}:{minute:02d} each day until you remove it.\n\n"
        "You can type /list any time to see your reminders, "
        "or /delete to remove one.",
    )
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# /list and /delete
# ---------------------------------------------------------------------------

async def list_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = get_reminders(chat_id)

    if not rows:
        await update.message.reply_text(
            "You don't have any reminders yet 🌸\n\n"
            "Type /addmed or /addappt whenever you'd like to set one."
        )
        return

    lines = ["Here are your reminders 💛\n"]
    for rid, kind, text, hour, minute in rows:
        icon = "💊" if kind == "medicine" else "📅"
        lines.append(f"{icon}  #{rid} — {text}  at  {hour:02d}:{minute:02d}")
    lines.append("\nTo remove one, type /delete.")

    await update.message.reply_text("\n".join(lines))


async def delete_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = get_reminders(chat_id)

    if not rows:
        await update.message.reply_text(
            "You have nothing to remove 🌼 You're all clear!"
        )
        return

    lines = ["Which reminder should I remove?\n"]
    for rid, kind, text, hour, minute in rows:
        icon = "💊" if kind == "medicine" else "📅"
        lines.append(f"{icon}  #{rid} — {text}  at  {hour:02d}:{minute:02d}")
    lines.append(
        "\nReply with the reminder ID shown here (for example, 1) "
        "to remove that reminder. Or type /cancel."
    )

    await update.message.reply_text("\n".join(lines))
    context.user_data["awaiting_delete"] = True


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catches free-text messages outside any conversation."""
    if context.user_data.get("awaiting_delete"):
        text = update.message.text.strip()
        try:
            rid = int(text.lstrip("#"))
        except ValueError:
            await update.message.reply_text(
                "Please just type the reminder ID, like 3, "
                "or /cancel to stop.",
            )
            return

        chat_id = update.effective_chat.id
        if delete_reminder(chat_id, rid):
            # Also remove the scheduled job
            for job in context.job_queue.get_jobs_by_name(f"reminder_{rid}"):
                job.schedule_removal()
            await update.message.reply_text(
                f"Done 🌷 Reminder #{rid} has been removed."
            )
        else:
            await update.message.reply_text(
                "I couldn't find that one. Type /list to see your reminders."
            )
        context.user_data["awaiting_delete"] = False
        return

    # Friendly fallback for unknown text
    await update.message.reply_text(
        "I'm here whenever you need me 🌼\n"
        "Type /help to see what I can do."
    )


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Custom audio handler for the /checkin flow (voice note or audio file)."""
    if not context.user_data.get("awaiting_voice"):
        await update.message.reply_text(
            "Thank you for the audio message 🌸 "
            "If you'd like me to run a check-in, please type /checkin first."
        )
        return

    context.user_data["awaiting_voice"] = False

    voice = update.message.voice
    audio = update.message.audio
    if voice is None and audio is None:
        await update.message.reply_text(
            "I couldn't find an audio message there. Please try sending it again."
        )
        return

    await update.message.reply_text("Listening carefully… 🎧")

    archived_audio_path = None
    try:
        audio_file = await (voice or audio).get_file()
        os.makedirs(AUDIO_ARCHIVE_DIR, exist_ok=True)
        suffix = _resolve_audio_suffix(voice, audio)
        filename = _build_audio_archive_filename(update, suffix)
        archived_audio_path = os.path.join(AUDIO_ARCHIVE_DIR, filename)
        await audio_file.download_to_drive(custom_path=archived_audio_path)
        features = await asyncio.to_thread(preprocess_voice_file, archived_audio_path)
    except Exception as exc:
        logger.exception("Failed to preprocess audio message: %s", exc)
        await update.message.reply_text(
            "I had trouble reading that audio message. Please try sending it again."
        )
        return

    chat_id = update.effective_chat.id
    patient_keystroke_id = get_patient_keystroke_id(chat_id)
    voice_status, anomaly_score = await analyze_voice_features(
        features,
        chat_id=chat_id,
        patient_keystroke_id=patient_keystroke_id,
    )

    source = "voice_note" if voice is not None else "audio_file"
    relative_audio_path = (
        os.path.relpath(archived_audio_path, BASE_DIR) if archived_audio_path else None
    )
    add_checkin(chat_id, source, voice_status, anomaly_score, relative_audio_path)

    await update.message.reply_text(
        "All done. Everything sounds lovely today 💛\n\n"
        "Thank you for checking in with me. I'll be here whenever you need me."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time(text: str):
    """Accepts HH:MM, HH.MM or HHMM. Returns (hour, minute) or None."""
    text = text.replace(" ", "").replace(".", ":")
    if ":" not in text and text.isdigit() and len(text) == 4:
        text = text[:2] + ":" + text[2:]
    try:
        parts = text.split(":")
        if len(parts) != 2:
            return None
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return hour, minute
    except ValueError:
        return None


def preprocess_voice_file(file_path: str) -> np.ndarray:
    """Resample audio to 16 kHz and build a log Mel-spectrogram.

    The output uses 80 Mel bins, a 25 ms analysis window, and a 10 ms hop.
    """
    audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    window_length = int(0.025 * 16000)
    hop_length = int(0.010 * 16000)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=80,
        n_fft=window_length,
        win_length=window_length,
        hop_length=hop_length,
        window="hann",
        power=2.0,
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def _resolve_audio_suffix(voice, audio) -> str:
    if voice is not None:
        return ".ogg"

    mime_type = (audio.mime_type or "").lower() if audio is not None else ""
    if "mpeg" in mime_type or "mp3" in mime_type:
        return ".mp3"
    if "wav" in mime_type:
        return ".wav"
    if "m4a" in mime_type or "mp4" in mime_type:
        return ".m4a"
    if "ogg" in mime_type:
        return ".ogg"
    return ".audio"


def _build_audio_archive_filename(update: Update, suffix: str) -> str:
    chat_id = update.effective_chat.id
    message_count = _next_audio_archive_counter(chat_id)
    message_date = update.effective_message.date

    if message_date.tzinfo is None:
        message_date = message_date.replace(tzinfo=timezone.utc)
    timestamp = message_date.astimezone(GMT_PLUS_8).strftime("%Y%m%dT%H%M%S")

    return f"{chat_id}_{message_count}_{timestamp}{suffix}"


def _next_audio_archive_counter(chat_id: int) -> int:
    prefix = f"{chat_id}_"
    highest = 0

    try:
        for entry in os.scandir(AUDIO_ARCHIVE_DIR):
            if not entry.is_file() or not entry.name.startswith(prefix):
                continue

            parts = entry.name.split("_", 2)
            if len(parts) < 3:
                continue

            try:
                count = int(parts[1])
            except ValueError:
                continue

            if count > highest:
                highest = count
    except FileNotFoundError:
        return 1

    return highest + 1


def _submit_mel_to_backend(
    features: np.ndarray,
    chat_id: int,
    #patient_keystroke_id: str | None,
) -> tuple[str, float | None]:
    payload = {
        "mel_spectrogram": features.tolist(),
        "shape": list(features.shape),
        "source": "telegram_checkin",
        "chat_id": chat_id,
        #"patient_keystroke_id": patient_keystroke_id,
    }
    response = requests.post(
        VOICE_ANALYSIS_ENDPOINT,
        json=payload,
        timeout=VOICE_ANALYSIS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()

    if not isinstance(body, dict):
        raise ValueError("Backend response must be a JSON object")

    status = body.get("voice_status") or body.get("status") or "normal"
    if status not in {"normal", "mild_anomaly", "high_anomaly"}:
        status = "normal"

    score = body.get("anomaly_score")
    if score is None:
        score = body.get("score")
    if score is not None:
        score = float(score)

    return status, score


async def analyze_voice_features(
    features: np.ndarray,
    chat_id: int,
    patient_keystroke_id: str | None,
) -> tuple[str, float | None]:
    if not USE_REAL_ANALYSIS:
        return "normal", None

    try:
        return await asyncio.to_thread(
            _submit_mel_to_backend,
            features,
            chat_id,
            patient_keystroke_id,
        )
    except Exception as exc:
        logger.exception("Real voice analysis failed, falling back to normal: %s", exc)
        return "normal", None


# ---------------------------------------------------------------------------
# Startup: rehydrate scheduled jobs from the database
# ---------------------------------------------------------------------------

async def post_init(application: Application) -> None:
    count = 0
    for rid, chat_id, kind, text, hour, minute in get_all_reminders():
        schedule_reminder(application, rid, chat_id, kind, text, hour, minute)
        count += 1
    logger.info("Rehydrated %d reminder(s) from the database.", count)

    sync_audio_checkin_jobs(application)
    application.job_queue.run_repeating(
        refresh_audio_checkin_jobs,
        interval=60,
        first=5,
        name="audio_checkin_sync",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TOKEN:
        raise RuntimeError(
            "Please set the TELEGRAM_BOT_TOKEN environment variable "
            "before running the bot."
        )

    init_db()

    application = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    # Simple commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("diagnosis", diagnosis))
    application.add_handler(CommandHandler("checkin", checkin))
    application.add_handler(CommandHandler("setkeystroke", set_keystroke_id))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(CommandHandler("list", list_reminders))
    application.add_handler(CommandHandler("delete", delete_start))

    # Voice notes and audio uploads are routed through the /checkin handler.
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_message))

    # /addmed conversation
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("addmed", addmed_start)],
        states={
            MED_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, addmed_name)],
            MED_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, addmed_time)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    ))

    # /addappt conversation
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("addappt", addappt_start)],
        states={
            APPT_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, addappt_text)],
            APPT_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, addappt_time)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    ))

    # Free-text fallback (must come last)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Wasurenai is starting up 🌸")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()