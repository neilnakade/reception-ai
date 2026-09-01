import json
import sqlite3
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path("reception.db")


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    # -----------------------------------------
    # Appointments
    # -----------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'booked'
        )
    """)

    # -----------------------------------------
    # Conversations
    # -----------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # -----------------------------------------
    # Conversation messages
    # -----------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def create_conversation(conversation_id: str, initial_state: dict):
    connection = get_connection()
    cursor = connection.cursor()

    now = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT OR IGNORE INTO conversations
        (
            conversation_id,
            state_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            conversation_id,
            json.dumps(initial_state),
            now,
            now,
        ),
    )

    connection.commit()
    connection.close()


def get_conversation_state(conversation_id: str) -> dict | None:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT state_json
        FROM conversations
        WHERE conversation_id = ?
        """,
        (conversation_id,),
    )

    row = cursor.fetchone()

    connection.close()

    if not row:
        return None

    return json.loads(row[0])


def save_conversation_state(
    conversation_id: str,
    state: dict,
):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET state_json = ?,
            updated_at = ?
        WHERE conversation_id = ?
        """,
        (
            json.dumps(state),
            datetime.utcnow().isoformat(),
            conversation_id,
        ),
    )

    connection.commit()
    connection.close()


def add_message(
    conversation_id: str,
    role: str,
    content: str,
):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO conversation_messages
        (
            conversation_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content,
            datetime.utcnow().isoformat(),
        ),
    )

    connection.commit()
    connection.close()


def get_messages(conversation_id: str) -> list[dict]:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM conversation_messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        {
            "role": role,
            "content": content,
        }
        for role, content in rows
    ]