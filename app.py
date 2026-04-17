from pathlib import Path
import os
import sqlite3
from datetime import datetime, timezone

from flask import abort, Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
from flask.typing import ResponseReturnValue


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static_site"
DOCTOR_TEMPLATE_DIR = BASE_DIR / "telegram_bot" / "doctor_dashboard"
DB_PATH = Path(os.environ.get("WASURENAI_DB_PATH", str(BASE_DIR / "telegram_bot" / "wasurenai.db")))
BOT_DIR = BASE_DIR / "telegram_bot"

app = Flask(__name__, template_folder=str(DOCTOR_TEMPLATE_DIR))
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-secret-change-me")

# Demo-only doctor credentials. 
DOCTOR_USERNAME = os.environ.get("DOCTOR_USERNAME", "doctor")
DOCTOR_PASSWORD = os.environ.get("DOCTOR_PASSWORD", "wasurenai-demo")
KEYSTROKE_API_KEY = os.environ.get("KEYSTROKE_API_KEY", "")


def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_dashboard_schema() -> None:
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    INTEGER NOT NULL,
            kind       TEXT    NOT NULL,
            text       TEXT    NOT NULL,
            hour       INTEGER NOT NULL,
            minute     INTEGER NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS checkins (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id       INTEGER NOT NULL,
            source        TEXT    NOT NULL,
            voice_status  TEXT    NOT NULL,
            anomaly_score REAL,
            audio_file_path TEXT,
            created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_column_exists(cursor, "checkins", "audio_file_path", "TEXT")
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_checkins_chat_time
        ON checkins (chat_id, created_at DESC)
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            chat_id               INTEGER PRIMARY KEY,
            patient_keystroke_id  TEXT UNIQUE,
            created_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
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
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_keystroke_checkins_pid_time
        ON keystroke_checkins (patient_keystroke_id, created_at DESC)
        """
    )
    cursor.execute(
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
    conn.commit()
    conn.close()


def _ensure_column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str, column_def: str) -> None:
    cols = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in cols}
    if column_name in existing:
        return
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def _is_doctor_authenticated() -> bool:
    return bool(session.get("doctor_authenticated"))


def set_audio_checkin_setting(chat_id: int, enabled: bool) -> None:
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audio_checkin_settings (chat_id, enabled, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chat_id) DO UPDATE SET
            enabled = excluded.enabled,
            updated_at = CURRENT_TIMESTAMP
        """,
        (chat_id, 1 if enabled else 0),
    )
    conn.commit()
    conn.close()


def format_relative_time(raw_timestamp: str | None) -> str:
    if not raw_timestamp:
        return "No check-in yet"
    try:
        parsed = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return raw_timestamp

    delta = datetime.now(timezone.utc) - parsed
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "Just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    days = hours // 24
    return f"{days} day ago" if days == 1 else f"{days} days ago"


def voice_status_meta(status: str) -> tuple[str, str, str]:
    if status == "high_anomaly":
        return "High anomaly", "alert", "Doctor follow-up"
    if status == "mild_anomaly":
        return "Mild anomaly", "warn", "Call caregiver"
    if status == "normal":
        return "Normal", "ok", "Routine review"
    return "No data", "warn", "Collect baseline"


def keystroke_status_meta(status: str) -> tuple[str, str]:
    if status == "high_anomaly":
        return "High anomaly", "alert"
    if status == "mild_anomaly":
        return "Mild anomaly", "warn"
    if status == "normal":
        return "Normal", "ok"
    return "No data", "warn"


def load_dashboard_data() -> dict:
    conn = get_db_conn()
    cursor = conn.cursor()

    patient_ids = [
        row["chat_id"]
        for row in cursor.execute(
            """
            SELECT chat_id FROM reminders
            UNION
            SELECT chat_id FROM checkins
            UNION
            SELECT chat_id FROM patients
            ORDER BY chat_id
            """
        ).fetchall()
    ]

    patient_keystroke_ids = {
        row["chat_id"]: row["patient_keystroke_id"]
        for row in cursor.execute(
            "SELECT chat_id, patient_keystroke_id FROM patients"
        ).fetchall()
    }

    reminder_counts = {
        row["chat_id"]: row["count"]
        for row in cursor.execute(
            "SELECT chat_id, COUNT(*) AS count FROM reminders GROUP BY chat_id"
        ).fetchall()
    }

    audio_checkin_settings = {
        row["chat_id"]: bool(row["enabled"])
        for row in cursor.execute(
            "SELECT chat_id, enabled FROM audio_checkin_settings"
        ).fetchall()
    }

    alert_24h = cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM checkins
        WHERE voice_status IN ('mild_anomaly', 'high_anomaly')
          AND created_at >= datetime('now', '-1 day')
        """
    ).fetchone()["count"]

    keystroke_alert_24h = cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM keystroke_checkins
        WHERE status IN ('mild_anomaly', 'high_anomaly')
          AND created_at >= datetime('now', '-1 day')
        """
    ).fetchone()["count"]

    rows = []
    pending_followups = 0
    for chat_id in patient_ids:
        latest = cursor.execute(
            """
            SELECT voice_status, anomaly_score, created_at
            FROM checkins
            WHERE chat_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (chat_id,),
        ).fetchone()

        latest_status = latest["voice_status"] if latest else "no_data"
        status_label, status_class, next_action = voice_status_meta(latest_status)

        patient_keystroke_id = patient_keystroke_ids.get(chat_id)
        latest_keystroke = None
        if patient_keystroke_id:
            latest_keystroke = cursor.execute(
                """
                SELECT status, created_at
                FROM keystroke_checkins
                WHERE patient_keystroke_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (patient_keystroke_id,),
            ).fetchone()

        keystroke_label, keystroke_class = keystroke_status_meta(
            latest_keystroke["status"] if latest_keystroke else "no_data"
        )

        if status_class == "alert" or keystroke_class == "alert":
            next_action = "Doctor follow-up"
        elif status_class == "warn" or keystroke_class == "warn":
            next_action = "Call caregiver"

        if status_class in {"warn", "alert"} or keystroke_class in {"warn", "alert"}:
            pending_followups += 1

        rows.append(
            {
                "chat_id": chat_id,
                "patient_id": f"P-{chat_id}",
                "last_checkin": format_relative_time(latest["created_at"] if latest else None),
                "voice_status_label": status_label,
                "voice_status_class": status_class,
                "patient_keystroke_id": patient_keystroke_id or "Not set",
                "keystroke_status_label": keystroke_label,
                "keystroke_status_class": keystroke_class,
                "next_action": next_action,
                "reminder_count": reminder_counts.get(chat_id, 0),
                "audio_checkin_enabled": audio_checkin_settings.get(chat_id, False),
            }
        )

    conn.close()
    return {
        "patients_monitored": len(patient_ids),
        "alerts_24h": alert_24h + keystroke_alert_24h,
        "pending_followups": pending_followups,
        "rows": rows,
    }


def _status_to_score(status: str) -> int:
    if status == "high_anomaly":
        return 2
    if status == "mild_anomaly":
        return 1
    return 0


def load_patient_history(chat_id: int, limit: int) -> dict:
    conn = get_db_conn()
    cursor = conn.cursor()

    patient_keystroke_id_row = cursor.execute(
        "SELECT patient_keystroke_id FROM patients WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    patient_keystroke_id = (
        patient_keystroke_id_row["patient_keystroke_id"] if patient_keystroke_id_row else None
    )

    voice_rows = cursor.execute(
        """
        SELECT id, voice_status, anomaly_score, source, created_at, audio_file_path
        FROM checkins
        WHERE chat_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (chat_id, limit),
    ).fetchall()

    keystroke_rows = []
    if patient_keystroke_id:
        keystroke_rows = cursor.execute(
            """
            SELECT status, anomaly_score, created_at
            FROM keystroke_checkins
            WHERE patient_keystroke_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (patient_keystroke_id, limit),
        ).fetchall()

    conn.close()

    voice_rows_ordered = list(reversed(voice_rows))
    keystroke_rows_ordered = list(reversed(keystroke_rows))

    voice_points = [
        {
            "x": row["created_at"],
            "y": _status_to_score(row["voice_status"]),
            "status": row["voice_status"],
            "anomaly_score": row["anomaly_score"],
        }
        for row in voice_rows_ordered
    ]

    keystroke_points = [
        {
            "x": row["created_at"],
            "y": _status_to_score(row["status"]),
            "status": row["status"],
            "anomaly_score": row["anomaly_score"],
        }
        for row in keystroke_rows_ordered
    ]

    voice_events = []
    for row in voice_rows:
        voice_events.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "status": row["voice_status"],
                "source": row["source"],
                "anomaly_score": row["anomaly_score"],
                "has_audio": bool(row["audio_file_path"]),
            }
        )

    return {
        "chat_id": chat_id,
        "patient_id": f"P-{chat_id}",
        "patient_keystroke_id": patient_keystroke_id or "Not set",
        "limit": limit,
        "voice_points": voice_points,
        "keystroke_points": keystroke_points,
        "voice_events": voice_events,
    }


@app.route("/")
def index() -> ResponseReturnValue:
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/doctor", methods=["GET", "POST"])
def doctor_portal() -> ResponseReturnValue:
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == DOCTOR_USERNAME and password == DOCTOR_PASSWORD:
            session["doctor_authenticated"] = True
            return redirect(url_for("doctor_portal"))
        return render_template("doctor_login.html", error="Invalid username or password.")

    if not session.get("doctor_authenticated"):
        return render_template("doctor_login.html", error=None)

    ensure_dashboard_schema()
    dashboard = load_dashboard_data()
    return render_template("doctor_dashboard.html", dashboard=dashboard)


@app.route("/doctor/logout")
def doctor_logout() -> ResponseReturnValue:
    session.pop("doctor_authenticated", None)
    return redirect(url_for("doctor_portal"))


@app.route("/doctor/audio-checkin", methods=["POST"])
def doctor_audio_checkin_toggle() -> ResponseReturnValue:
    if not _is_doctor_authenticated():
        return redirect(url_for("doctor_portal"))

    ensure_dashboard_schema()
    chat_id_raw = request.form.get("chat_id", "")
    enabled_raw = request.form.get("enabled", "0")
    try:
        chat_id = int(chat_id_raw)
    except ValueError:
        return redirect(url_for("doctor_portal"))

    set_audio_checkin_setting(chat_id, enabled_raw == "1")
    return redirect(url_for("doctor_portal"))


@app.route("/doctor/patient/<int:chat_id>")
def doctor_patient_history(chat_id: int) -> ResponseReturnValue:
    if not _is_doctor_authenticated():
        return redirect(url_for("doctor_portal"))

    ensure_dashboard_schema()
    n_raw = request.args.get("n", "25")
    try:
        n = int(n_raw)
    except ValueError:
        n = 25
    if n not in {10, 25, 50, 100}:
        n = 25

    focus = request.args.get("focus", "voice")
    if focus not in {"voice", "keystroke"}:
        focus = "voice"

    history = load_patient_history(chat_id, n)
    return render_template("doctor_patient_history.html", history=history, focus=focus)


@app.route("/doctor/checkin-audio/<int:checkin_id>")
def doctor_checkin_audio(checkin_id: int) -> ResponseReturnValue:
    if not _is_doctor_authenticated():
        return redirect(url_for("doctor_portal"))

    ensure_dashboard_schema()
    conn = get_db_conn()
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT audio_file_path FROM checkins WHERE id = ?",
        (checkin_id,),
    ).fetchone()
    conn.close()

    if not row or not row["audio_file_path"]:
        return abort(404)

    relative_path = Path(row["audio_file_path"])
    full_path = (BOT_DIR / relative_path).resolve()
    if not str(full_path).startswith(str(BOT_DIR.resolve())):
        return abort(403)
    if not full_path.exists() or not full_path.is_file():
        return abort(404)

    return send_file(full_path, as_attachment=False)


@app.route("/api/keystroke-checkin", methods=["POST"])
def submit_keystroke_checkin() -> ResponseReturnValue:
    ensure_dashboard_schema()

    if KEYSTROKE_API_KEY:
        received_key = request.headers.get("X-API-Key", "")
        if received_key != KEYSTROKE_API_KEY:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    patient_keystroke_id = str(payload.get("patient_keystroke_id", "")).strip()
    status = str(payload.get("status", "")).strip()
    anomaly_score = payload.get("anomaly_score")

    if not patient_keystroke_id:
        return jsonify({"ok": False, "error": "patient_keystroke_id required"}), 400
    if status not in {"normal", "mild_anomaly", "high_anomaly"}:
        return jsonify({"ok": False, "error": "invalid status"}), 400

    if anomaly_score is not None:
        try:
            anomaly_score = float(anomaly_score)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid anomaly_score"}), 400

    conn = get_db_conn() 
    cursor = conn.cursor()
    # TODO: POTENTIALLY ATTACKERS CAN WASTE OUR STORAGE
    # BY REGISTERING AND THEN ADDING RECORDS THAT DON'T AFFECT ANYONE? 
    '''
    patient_exists = cursor.execute(
        "SELECT 1 FROM patients WHERE patient_keystroke_id = ?",
        (patient_keystroke_id,),
    ).fetchone()
    if not patient_exists:
        conn.close()
        return jsonify({"ok": False, "error": "unknown patient_keystroke_id"}), 404
    '''

    cursor.execute(
        """
        INSERT INTO keystroke_checkins (patient_keystroke_id, status, anomaly_score)
        VALUES (?, ?, ?)
        """,
        (patient_keystroke_id, status, anomaly_score),
    )
    conn.commit()
    conn.close()

    return jsonify({"ok": True})

@app.route("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)