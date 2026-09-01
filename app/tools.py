from app.database import get_connection


def check_availability(date: str, time: str) -> str:
    """Check whether an appointment slot is available."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM appointments
        WHERE date = ?
          AND time = ?
          AND status = 'booked'
        """,
        (date, time),
    )

    appointment = cursor.fetchone()

    connection.close()

    if appointment:
        return "BOOKED"

    return "AVAILABLE"


def book_appointment(date: str, time: str, name: str) -> str:
    """Book an appointment if the slot is available."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM appointments
        WHERE date = ?
          AND time = ?
          AND status = 'booked'
        """,
        (date, time),
    )

    existing = cursor.fetchone()

    if existing:
        connection.close()
        return "BOOKED"

    cursor.execute(
        """
        INSERT INTO appointments (name, date, time, status)
        VALUES (?, ?, ?, 'booked')
        """,
        (name, date, time),
    )

    connection.commit()
    connection.close()

    return "BOOKED_SUCCESSFULLY"


def cancel_appointment(date: str, time: str, name: str) -> str:
    """Cancel an existing booked appointment."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM appointments
        WHERE name = ?
          AND date = ?
          AND time = ?
          AND status = 'booked'
        """,
        (name, date, time),
    )

    appointment = cursor.fetchone()

    if not appointment:
        connection.close()
        return "NOT_FOUND"

    appointment_id = appointment[0]

    cursor.execute(
        """
        UPDATE appointments
        SET status = 'cancelled'
        WHERE id = ?
        """,
        (appointment_id,),
    )

    connection.commit()
    connection.close()

    return "CANCELLED"


def reschedule_appointment(
    name: str,
    old_date: str,
    old_time: str,
    new_date: str,
    new_time: str,
) -> str:
    """
    Move an existing appointment to a new date and time.
    """

    connection = get_connection()
    cursor = connection.cursor()

    # 1. Find the customer's existing appointment
    cursor.execute(
        """
        SELECT id
        FROM appointments
        WHERE name = ?
          AND date = ?
          AND time = ?
          AND status = 'booked'
        """,
        (name, old_date, old_time),
    )

    appointment = cursor.fetchone()

    if not appointment:
        connection.close()
        return "OLD_APPOINTMENT_NOT_FOUND"

    # 2. Make sure the new slot is free
    cursor.execute(
        """
        SELECT id
        FROM appointments
        WHERE date = ?
          AND time = ?
          AND status = 'booked'
        """,
        (new_date, new_time),
    )

    new_slot = cursor.fetchone()

    if new_slot:
        connection.close()
        return "NEW_SLOT_BOOKED"

    # 3. Cancel the old appointment
    old_appointment_id = appointment[0]

    cursor.execute(
        """
        UPDATE appointments
        SET status = 'cancelled'
        WHERE id = ?
        """,
        (old_appointment_id,),
    )

    # 4. Create the new appointment
    cursor.execute(
        """
        INSERT INTO appointments (name, date, time, status)
        VALUES (?, ?, ?, 'booked')
        """,
        (name, new_date, new_time),
    )

    connection.commit()
    connection.close()

    return "RESCHEDULED"