import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DATABASE_PATH = Path("reception.db")

DATABASE_URL = os.getenv("DATABASE_URL")


# =========================================================
# POSTGRESQL CONNECTION POOL
# =========================================================

POSTGRES_POOL = None


class PooledConnection:
    """
    Compatibility wrapper around a pooled PostgreSQL connection.

    Existing application code does:

        connection = get_connection()
        ...
        connection.commit()
        connection.close()

    For PostgreSQL, close() returns the connection to the pool
    instead of permanently closing the database connection.
    """

    def __init__(self, pool, connection):
        self._pool = pool
        self._connection = connection
        self._returned = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        if self._returned:
            return

        try:
            # Clear any unfinished transaction before returning
            # the connection to the pool.
            self._connection.rollback()
        except Exception:
            pass

        self._pool.putconn(self._connection)
        self._returned = True

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


def get_postgres_pool():
    """
    Create the PostgreSQL connection pool once and reuse it.
    """

    global POSTGRES_POOL

    if POSTGRES_POOL is None:

        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not configured."
            )

        POSTGRES_POOL = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=5,
            open=False,
            timeout=10,
        )

        POSTGRES_POOL.open(
            wait=True,
            timeout=10,
        )

    return POSTGRES_POOL


# =========================================================
# GET DATABASE CONNECTION
# =========================================================

def get_connection():
    """
    Production:
        PostgreSQL via connection pool

    Local fallback:
        SQLite
    """

    if DATABASE_URL:

        pool = get_postgres_pool()

        connection = pool.getconn()

        return PooledConnection(
            pool,
            connection,
        )

    return sqlite3.connect(
        DATABASE_PATH
    )


# =========================================================
# DATABASE TYPE
# =========================================================

def is_postgres():
    return bool(DATABASE_URL)


# =========================================================
# SQL EXECUTION HELPER
# =========================================================

def execute_query(
    cursor,
    query,
    params=(),
):
    """
    SQLite uses '?'
    PostgreSQL uses '%s'
    """

    if is_postgres():

        query = query.replace(
            "?",
            "%s",
        )

    cursor.execute(
        query,
        params,
    )


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    try:

        if is_postgres():

            # -------------------------------------------------
            # APPOINTMENTS
            # -------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'booked'
                )
                """
            )

            # -------------------------------------------------
            # CONVERSATIONS
            # -------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # -------------------------------------------------
            # CONVERSATION MESSAGES
            # -------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

        else:

            # -------------------------------------------------
            # SQLITE APPOINTMENTS
            # -------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'booked'
                )
                """
            )

            # -------------------------------------------------
            # SQLITE CONVERSATIONS
            # -------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # -------------------------------------------------
            # SQLITE CONVERSATION MESSAGES
            # -------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# CREATE CONVERSATION
# =========================================================

def create_conversation(
    conversation_id: str,
    initial_state: dict,
):

    connection = get_connection()
    cursor = connection.cursor()

    now = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        if is_postgres():

            execute_query(
                cursor,
                """
                INSERT INTO conversations
                (
                    conversation_id,
                    state_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT (conversation_id)
                DO NOTHING
                """,
                (
                    conversation_id,
                    json.dumps(
                        initial_state
                    ),
                    now,
                    now,
                ),
            )

        else:

            execute_query(
                cursor,
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
                    json.dumps(
                        initial_state
                    ),
                    now,
                    now,
                ),
            )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# GET CONVERSATION STATE
# =========================================================

def get_conversation_state(
    conversation_id: str,
) -> dict | None:

    connection = get_connection()
    cursor = connection.cursor()

    try:

        execute_query(
            cursor,
            """
            SELECT state_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (
                conversation_id,
            ),
        )

        row = cursor.fetchone()

    finally:

        connection.close()

    if not row:
        return None

    return json.loads(
        row[0]
    )


# =========================================================
# SAVE CONVERSATION STATE
# =========================================================

def save_conversation_state(
    conversation_id: str,
    state: dict,
):

    connection = get_connection()
    cursor = connection.cursor()

    try:

        execute_query(
            cursor,
            """
            UPDATE conversations
            SET
                state_json = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (
                json.dumps(state),
                datetime.now(
                    timezone.utc
                ).isoformat(),
                conversation_id,
            ),
        )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# ADD MESSAGE
# =========================================================

def add_message(
    conversation_id: str,
    role: str,
    content: str,
):

    connection = get_connection()
    cursor = connection.cursor()

    try:

        execute_query(
            cursor,
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
                datetime.now(
                    timezone.utc
                ).isoformat(),
            ),
        )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# GET MESSAGES
# =========================================================

def get_messages(
    conversation_id: str,
) -> list[dict]:

    connection = get_connection()
    cursor = connection.cursor()

    try:

        execute_query(
            cursor,
            """
            SELECT role, content
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (
                conversation_id,
            ),
        )

        rows = cursor.fetchall()

    finally:

        connection.close()

    return [
        {
            "role": role,
            "content": content,
        }
        for role, content in rows
    ]