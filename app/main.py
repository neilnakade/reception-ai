import re
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from pydantic import BaseModel

from app.business import load_business_config
from app.database import (
    add_message,
    create_conversation,
    get_conversation_state,
    get_messages,
    initialize_database,
    save_conversation_state,
)
from app.llm import get_llm_response
from app.prompts import build_system_prompt
from app.schemas import ChatRequest, ChatResponse
from app.tools import (
    book_appointment,
    cancel_appointment,
    check_availability,
    reschedule_appointment,
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="ReceptionAI",
    description="AI receptionist API",
    version="0.1.0",
)


# =========================================================
# INITIALIZATION
# =========================================================

initialize_database()

BUSINESS_CONFIG = load_business_config()
SYSTEM_PROMPT = build_system_prompt(
    BUSINESS_CONFIG
)

# India timezone
IST = ZoneInfo("Asia/Kolkata")


# =========================================================
# DEFAULT CONVERSATION STATE
# =========================================================

def default_state() -> dict:
    return {
        "name": None,

        # Booking
        "date": None,
        "time": None,

        # Cancellation
        "old_date": None,
        "old_time": None,

        # Rescheduling
        "new_date": None,
        "new_time": None,

        # Current action
        "action": None,

        # Date clarification
        "asked_for_date": False,
    }


# =========================================================
# DATE EXTRACTION
# =========================================================

def extract_date(message: str):
    """
    Extract dates from common Indian and natural-language
    formats.

    Examples:
    - today
    - tomorrow
    - 05/09/2026
    - 05-09-2026
    - 05.09.2026
    - 2026-09-05
    - 5 September
    - 5th September
    - 5 September 2026
    - September 5
    - September 5th
    - 5 Sep
    - Sep 5
    """

    if not message:
        return None

    message_lower = message.lower().strip()

    # Use India time, not the server's timezone.
    today = datetime.now(IST).date()

    # -----------------------------------------------------
    # TODAY
    # -----------------------------------------------------

    if re.search(
        r"\btoday\b",
        message_lower,
    ):
        return today.isoformat()

    # -----------------------------------------------------
    # TOMORROW
    # -----------------------------------------------------

    if re.search(
        r"\btomorrow\b",
        message_lower,
    ):
        return (
            today + timedelta(days=1)
        ).isoformat()

    # -----------------------------------------------------
    # YYYY-MM-DD
    # -----------------------------------------------------

    iso_match = re.search(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        message_lower,
    )

    if iso_match:

        year, month, day = map(
            int,
            iso_match.groups(),
        )

        try:

            return date(
                year,
                month,
                day,
            ).isoformat()

        except ValueError:

            return None

    # -----------------------------------------------------
    # DD/MM/YYYY
    # DD-MM-YYYY
    # DD.MM.YYYY
    #
    # Indian convention
    # -----------------------------------------------------

    indian_match = re.search(
        r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b",
        message_lower,
    )

    if indian_match:

        day, month, year = map(
            int,
            indian_match.groups(),
        )

        try:

            return date(
                year,
                month,
                day,
            ).isoformat()

        except ValueError:

            return None

    # -----------------------------------------------------
    # MONTH NAMES
    # -----------------------------------------------------

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    short_months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    all_months = {
        **months,
        **short_months,
    }

    month_pattern = "|".join(
        all_months.keys()
    )

    # -----------------------------------------------------
    # DAY + MONTH + OPTIONAL YEAR
    # -----------------------------------------------------

    reverse_match = re.search(
        rf"\b"
        rf"(\d{{1,2}})"
        rf"(?:st|nd|rd|th)?"
        rf"\s+"
        rf"({month_pattern})"
        rf"(?:\s*,?\s*(\d{{4}}))?"
        rf"\b",
        message_lower,
    )

    if reverse_match:

        day = int(
            reverse_match.group(1)
        )

        month_name = (
            reverse_match.group(2)
        )

        month = all_months[
            month_name
        ]

        year_text = (
            reverse_match.group(3)
        )

        if year_text:

            year = int(year_text)

            try:

                return date(
                    year,
                    month,
                    day,
                ).isoformat()

            except ValueError:

                return None

        year = today.year

        try:

            candidate = date(
                year,
                month,
                day,
            )

            if candidate < today:

                candidate = date(
                    year + 1,
                    month,
                    day,
                )

            return candidate.isoformat()

        except ValueError:

            return None

    # -----------------------------------------------------
    # MONTH + DAY + OPTIONAL YEAR
    # -----------------------------------------------------

    normal_match = re.search(
        rf"\b"
        rf"({month_pattern})"
        rf"\s+"
        rf"(\d{{1,2}})"
        rf"(?:st|nd|rd|th)?"
        rf"(?:\s*,?\s*(\d{{4}}))?"
        rf"\b",
        message_lower,
    )

    if not normal_match:
        return None

    month_name = (
        normal_match.group(1)
    )

    day = int(
        normal_match.group(2)
    )

    month = all_months[
        month_name
    ]

    year_text = (
        normal_match.group(3)
    )

    if year_text:

        year = int(year_text)

        try:

            return date(
                year,
                month,
                day,
            ).isoformat()

        except ValueError:

            return None

    year = today.year

    try:

        candidate = date(
            year,
            month,
            day,
        )

        if candidate < today:

            candidate = date(
                year + 1,
                month,
                day,
            )

        return candidate.isoformat()

    except ValueError:

        return None


# =========================================================
# CUSTOMER DATE FORMAT
# =========================================================

def format_date_for_customer(
    date_string: str,
) -> str:

    try:

        return datetime.strptime(
            date_string,
            "%Y-%m-%d",
        ).strftime(
            "%d/%m/%Y"
        )

    except (
        ValueError,
        TypeError,
    ):

        return date_string


# =========================================================
# CUSTOMER TIME FORMAT
# =========================================================

def format_time_for_customer(
    time_string: str,
) -> str:

    try:

        return datetime.strptime(
            time_string,
            "%H:%M",
        ).strftime(
            "%I:%M %p"
        ).lstrip("0")

    except (
        ValueError,
        TypeError,
    ):

        return time_string


# =========================================================
# TIME EXTRACTION
# =========================================================

def extract_time(message: str):

    if not message:
        return None

    message_lower = message.lower().strip()

    # -----------------------------------------------------
    # 12-HOUR FORMAT
    # -----------------------------------------------------

    match_12 = re.search(
        r"\b(1[0-2]|0?[1-9])"
        r"(?::([0-5]\d))?"
        r"\s*"
        r"(am|pm)\b",
        message_lower,
    )

    if match_12:

        hour = int(
            match_12.group(1)
        )

        minute = int(
            match_12.group(2) or "00"
        )

        period = match_12.group(3)

        if (
            period == "pm"
            and hour != 12
        ):
            hour += 12

        if (
            period == "am"
            and hour == 12
        ):
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # -----------------------------------------------------
    # NATURAL DAY PERIOD
    # -----------------------------------------------------

    match_period = re.search(
        r"\b(1[0-2]|0?[1-9])"
        r"(?::([0-5]\d))?"
        r"\s+"
        r"(?:in the\s+)?"
        r"(morning|afternoon|evening|night)\b",
        message_lower,
    )

    if match_period:

        hour = int(
            match_period.group(1)
        )

        minute = int(
            match_period.group(2) or "00"
        )

        period = match_period.group(3)

        if (
            period in (
                "afternoon",
                "evening",
            )
            and hour != 12
        ):
            hour += 12

        if (
            period == "morning"
            and hour == 12
        ):
            hour = 0

        if (
            period == "night"
            and hour != 12
        ):
            hour += 12

        return f"{hour:02d}:{minute:02d}"

    # -----------------------------------------------------
    # 24-HOUR FORMAT
    # -----------------------------------------------------

    match_24 = re.search(
        r"\b([01]\d|2[0-3]):([0-5]\d)\b",
        message_lower,
    )

    if match_24:

        hour = int(
            match_24.group(1)
        )

        minute = int(
            match_24.group(2)
        )

        return f"{hour:02d}:{minute:02d}"

    return None


# =========================================================
# NAME EXTRACTION
# =========================================================

def extract_name(message: str):

    if not message:
        return None

    text = re.sub(
        r"\s+",
        " ",
        message.strip(),
    )

    patterns = [
        # My name is Neil
        r"\bmy name is\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s*(?:[,.;]|$|and\b|wants\b|is\b|has\b|needs\b))",

        # Name is Neil
        r"\bname is\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s*(?:[,.;]|$|and\b|wants\b|is\b|has\b|needs\b))",

        # Customer's name is Neil
        r"\bcustomer(?:'s)? name is\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s*(?:[,.;]|$|and\b|wants\b|is\b|has\b|needs\b))",

        # I'm Neil
        r"\bI'm\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s*(?:[,.;]|$|and\b|wants\b|is\b|has\b|needs\b))",

        # I am Neil
        r"\bI am\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s*(?:[,.;]|$|and\b|wants\b|is\b|has\b|needs\b))",

        # This is Neil
        r"\bthis is\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s*(?:[,.;]|$|and\b|wants\b|is\b|has\b|needs\b))",

        # Customer Neil wants...
        r"\bcustomer\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s+(?:wants|is|has|needs|would|requested|is requesting)\b)",

        # Patient Neil wants...
        r"\bpatient\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s+(?:wants|is|has|needs|would|requested|is requesting)\b)",

        # Caller Neil wants...
        r"\bcaller\s+"
        r"([A-Za-z][A-Za-z .'-]{0,49}?)"
        r"(?=\s+(?:wants|is|has|needs|would|requested|is requesting)\b)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            name = (
                match.group(1)
                .strip()
                .rstrip(" .,!?")
            )

            if name:
                return name

    # If the entire value is simply a name,
    # accept it.
    if re.fullmatch(
        r"[A-Za-z][A-Za-z .'-]{1,49}",
        text,
    ):

        return text

    return None


# =========================================================
# INTENT DETECTION FOR WEB CHAT
# =========================================================

def detect_action(message: str):

    message_lower = message.lower()

    # Cancellation
    if any(
        keyword in message_lower
        for keyword in [
            "cancel",
            "cancellation",
            "cancelled",
            "canceled",
        ]
    ):
        return "cancel"

    # Rescheduling
    if any(
        keyword in message_lower
        for keyword in [
            "reschedule",
            "rescheduled",
            "change my appointment",
            "move my appointment",
            "change the appointment",
        ]
    ):
        return "reschedule"

    # Booking
    if any(
        keyword in message_lower
        for keyword in [
            "book",
            "booking",
            "schedule",
            "reserve",
            "appointment",
        ]
    ):
        return "book"

    return None


# =========================================================
# NEW BOOKING DETECTION
# =========================================================

def is_new_booking_request(
    message: str,
) -> bool:

    message_lower = (
        message.lower().strip()
    )

    booking_words = [
        "book",
        "booking",
        "schedule",
        "reserve",
        "appointment",
    ]

    date_value = extract_date(
        message
    )

    time_value = extract_time(
        message
    )

    contains_booking_word = any(
        word in message_lower
        for word in booking_words
    )

    return (
        contains_booking_word
        and date_value is None
        and time_value is None
    )


# =========================================================
# BOOKING HANDLER FOR WEB CHAT
# =========================================================

def handle_booking(
    message: str,
    state: dict,
):

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    if not state.get("date"):

        extracted_date = extract_date(
            message
        )

        if extracted_date:

            state["date"] = (
                extracted_date
            )

            state["asked_for_date"] = (
                False
            )

        else:

            if is_new_booking_request(
                message
            ):

                state[
                    "asked_for_date"
                ] = True

                return (
                    "Sure. What date would "
                    "you like the appointment?"
                )

            if state.get(
                "asked_for_date"
            ):

                return (
                    "I couldn't quite "
                    "understand that date. "
                    "You can say something "
                    "like 5 September, "
                    "05/09/2026, or tomorrow."
                )

            state[
                "asked_for_date"
            ] = True

            return (
                "Sure. What date would "
                "you like the appointment?"
            )

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    if not state.get("time"):

        extracted_time = extract_time(
            message
        )

        if extracted_time:

            state["time"] = (
                extracted_time
            )

        else:

            return (
                "What time would you like "
                "the appointment?"
            )

    # -----------------------------------------------------
    # NAME
    # -----------------------------------------------------

    if not state.get("name"):

        extracted_name = extract_name(
            message
        )

        if extracted_name:

            state["name"] = (
                extracted_name
            )

        else:

            return (
                "May I have your name?"
            )

    # -----------------------------------------------------
    # BOOK
    # -----------------------------------------------------

    if (
        state.get("date")
        and state.get("time")
        and state.get("name")
    ):

        availability = check_availability(
            date=state["date"],
            time=state["time"],
        )

        customer_date = (
            format_date_for_customer(
                state["date"]
            )
        )

        customer_time = (
            format_time_for_customer(
                state["time"]
            )
        )

        if availability == "BOOKED":

            return (
                f"Sorry, {customer_time} "
                f"on {customer_date} is "
                "already booked. Please "
                "choose another time."
            )

        result = book_appointment(
            date=state["date"],
            time=state["time"],
            name=state["name"],
        )

        if result == "BOOKED_SUCCESSFULLY":

            response = (
                f"Your appointment is "
                f"booked for {customer_date} "
                f"at {customer_time}. "
                f"Name: {state['name']}."
            )

            state.clear()
            state.update(
                default_state()
            )

            return response

        if result == "BOOKED":

            return (
                f"Sorry, {customer_time} "
                f"on {customer_date} "
                "was just booked. Please "
                "choose another time."
            )

    return (
        "Please provide the appointment "
        "details."
    )


# =========================================================
# CANCELLATION HANDLER FOR WEB CHAT
# =========================================================

def handle_cancellation(
    message: str,
    state: dict,
):

    if not state.get("date"):

        extracted_date = extract_date(
            message
        )

        if extracted_date:

            state["date"] = (
                extracted_date
            )

        else:

            return (
                "What date is the "
                "appointment you would "
                "like to cancel?"
            )

    if not state.get("time"):

        extracted_time = extract_time(
            message
        )

        if extracted_time:

            state["time"] = (
                extracted_time
            )

        else:

            return (
                "What time is the "
                "appointment you would "
                "like to cancel?"
            )

    if not state.get("name"):

        extracted_name = extract_name(
            message
        )

        if extracted_name:

            state["name"] = (
                extracted_name
            )

        else:

            return (
                "May I have the name "
                "on the appointment?"
            )

    if (
        state.get("date")
        and state.get("time")
        and state.get("name")
    ):

        result = cancel_appointment(
            date=state["date"],
            time=state["time"],
            name=state["name"],
        )

        customer_date = (
            format_date_for_customer(
                state["date"]
            )
        )

        customer_time = (
            format_time_for_customer(
                state["time"]
            )
        )

        if result == "CANCELLED":

            response = (
                f"Your appointment on "
                f"{customer_date} at "
                f"{customer_time} has "
                "been cancelled."
            )

            state.clear()
            state.update(
                default_state()
            )

            return response

        if result == "NOT_FOUND":

            return (
                "I couldn't find a booked "
                "appointment matching "
                "those details."
            )

    return (
        "Please provide the appointment "
        "details."
    )


# =========================================================
# RESCHEDULING HANDLER FOR WEB CHAT
# =========================================================

def handle_rescheduling(
    message: str,
    state: dict,
):

    if not state.get("name"):

        extracted_name = extract_name(
            message
        )

        if extracted_name:

            state["name"] = (
                extracted_name
            )

        else:

            return (
                "May I have the name "
                "on the appointment?"
            )

    if not state.get("old_date"):

        extracted_date = extract_date(
            message
        )

        if extracted_date:

            state["old_date"] = (
                extracted_date
            )

        else:

            return (
                "What is the current date "
                "of your appointment?"
            )

    if not state.get("old_time"):

        extracted_time = extract_time(
            message
        )

        if extracted_time:

            state["old_time"] = (
                extracted_time
            )

        else:

            return (
                "What is the current time "
                "of your appointment?"
            )

    if not state.get("new_date"):

        extracted_date = extract_date(
            message
        )

        if (
            extracted_date
            and extracted_date
            != state.get("old_date")
        ):

            state["new_date"] = (
                extracted_date
            )

        else:

            return (
                "What new date would you "
                "like for the appointment?"
            )

    if not state.get("new_time"):

        extracted_time = extract_time(
            message
        )

        if (
            extracted_time
            and extracted_time
            != state.get("old_time")
        ):

            state["new_time"] = (
                extracted_time
            )

        else:

            return (
                "What new time would you "
                "like for the appointment?"
            )

    if (
        state.get("name")
        and state.get("old_date")
        and state.get("old_time")
        and state.get("new_date")
        and state.get("new_time")
    ):

        result = reschedule_appointment(
            name=state["name"],
            old_date=state["old_date"],
            old_time=state["old_time"],
            new_date=state["new_date"],
            new_time=state["new_time"],
        )

        new_customer_date = (
            format_date_for_customer(
                state["new_date"]
            )
        )

        new_customer_time = (
            format_time_for_customer(
                state["new_time"]
            )
        )

        if result == "RESCHEDULED":

            response = (
                f"Your appointment has "
                f"been rescheduled to "
                f"{new_customer_date} at "
                f"{new_customer_time}."
            )

            state.clear()
            state.update(
                default_state()
            )

            return response

        if (
            result
            == "OLD_APPOINTMENT_NOT_FOUND"
        ):

            return (
                "I couldn't find your "
                "existing appointment "
                "with those details."
            )

        if result == "NEW_SLOT_BOOKED":

            return (
                f"Sorry, "
                f"{new_customer_time} on "
                f"{new_customer_date} "
                "is already booked. "
                "Please choose another time."
            )

    return (
        "Please provide the appointment "
        "details."
    )


# =========================================================
# NORMAL CHAT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "ReceptionAI API is running"
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    conversation_id = (
        request.conversation_id
    )

    state = get_conversation_state(
        conversation_id
    )

    if state is None:

        state = default_state()

        create_conversation(
            conversation_id,
            state,
        )

    history = get_messages(
        conversation_id
    )

    add_message(
        conversation_id,
        "user",
        request.message,
    )

    detected_action = detect_action(
        request.message
    )

    if detected_action:

        if (
            detected_action == "book"
            and is_new_booking_request(
                request.message
            )
            and not state.get("date")
            and not state.get("time")
        ):

            state["name"] = None
            state["date"] = None
            state["time"] = None
            state["old_date"] = None
            state["old_time"] = None
            state["new_date"] = None
            state["new_time"] = None
            state["asked_for_date"] = False

        state["action"] = (
            detected_action
        )

    action = state.get("action")

    if action == "book":

        response = handle_booking(
            request.message,
            state,
        )

    elif action == "cancel":

        response = handle_cancellation(
            request.message,
            state,
        )

    elif action == "reschedule":

        response = handle_rescheduling(
            request.message,
            state,
        )

    else:

        llm_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        for message in history:

            llm_messages.append(message)

        llm_messages.append(
            {
                "role": "user",
                "content": request.message,
            }
        )

        response = get_llm_response(
            llm_messages,
            state,
        )

    save_conversation_state(
        conversation_id,
        state,
    )

    add_message(
        conversation_id,
        "assistant",
        response,
    )

    return ChatResponse(
        response=response
    )


# =========================================================
# VOICE ACTION REQUEST
# =========================================================

class VoiceActionRequest(BaseModel):
    """
    Structured arguments sent by the Sarvam voice agent.

    action:
        check_availability
        book
        cancel
        reschedule

    date/time:
        Existing or requested appointment details.

    new_date/new_time:
        New appointment details for rescheduling.
    """

    action: str
    name: str | None = None
    date: str | None = None
    time: str | None = None
    new_date: str | None = None
    new_time: str | None = None


# =========================================================
# NORMALIZE VOICE ACTION
# =========================================================

def normalize_voice_action(
    action: str,
) -> str | None:

    if not action:
        return None

    cleaned = re.sub(
        r"[\s-]+",
        "_",
        action.strip().lower(),
    )

    aliases = {
        "availability": "check_availability",
        "check_availability": "check_availability",
        "checkavailability": "check_availability",
        "book": "book",
        "booking": "book",
        "schedule": "book",
        "reserve": "book",
        "cancel": "cancel",
        "cancellation": "cancel",
        "canceled": "cancel",
        "cancelled": "cancel",
        "reschedule": "reschedule",
        "rescheduling": "reschedule",
    }

    return aliases.get(cleaned)


# =========================================================
# PARSE STRUCTURED VOICE DATE
# =========================================================

def parse_voice_date(
    value: str | None,
) -> str | None:

    if not value:
        return None

    return extract_date(
        value
    )


# =========================================================
# PARSE STRUCTURED VOICE TIME
# =========================================================

def parse_voice_time(
    value: str | None,
) -> str | None:

    if not value:
        return None

    return extract_time(
        value
    )


# =========================================================
# PARSE STRUCTURED VOICE NAME
# =========================================================

def parse_voice_name(
    value: str | None,
) -> str | None:

    if not value:
        return None

    extracted = extract_name(
        value
    )

    if extracted:
        return extracted

    cleaned = value.strip()

    if cleaned:
        return cleaned

    return None


# =========================================================
# VOICE ACTION ENDPOINT
# =========================================================

@app.post(
    "/voice-action",
    response_model=ChatResponse,
)
def voice_action(
    request: VoiceActionRequest,
):

    total_start = time.perf_counter()

    action = normalize_voice_action(
        request.action
    )

    name = parse_voice_name(
        request.name
    )

    appointment_date = parse_voice_date(
        request.date
    )

    appointment_time = parse_voice_time(
        request.time
    )

    new_date = parse_voice_date(
        request.new_date
    )

    new_time = parse_voice_time(
        request.new_time
    )

    # -----------------------------------------------------
    # DEBUG LOGGING
    # -----------------------------------------------------

    print(
        f"[VOICE] action={action}"
        f" name={name}"
        f" date={appointment_date}"
        f" time={appointment_time}"
        f" new_date={new_date}"
        f" new_time={new_time}"
    )

    # -----------------------------------------------------
    # INVALID ACTION
    # -----------------------------------------------------

    if not action:

        response = (
            "I can help with checking "
            "appointment availability, "
            "booking, cancellation, or "
            "rescheduling. Which would "
            "you like?"
        )

        print(
            f"[PERF] TOTAL voice_action: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return ChatResponse(
            response=response
        )

    # =====================================================
    # CHECK AVAILABILITY
    # =====================================================

    if action == "check_availability":

        if not appointment_date:

            return ChatResponse(
                response=(
                    "What date would you "
                    "like me to check?"
                )
            )

        if not appointment_time:

            return ChatResponse(
                response=(
                    "What time would you "
                    "like me to check?"
                )
            )

        checkpoint = time.perf_counter()

        availability = check_availability(
            date=appointment_date,
            time=appointment_time,
        )

        print(
            f"[PERF] voice check_availability: "
            f"{time.perf_counter() - checkpoint:.3f}s"
        )

        customer_date = (
            format_date_for_customer(
                appointment_date
            )
        )

        customer_time = (
            format_time_for_customer(
                appointment_time
            )
        )

        if availability == "BOOKED":

            response = (
                f"Sorry, {customer_time} "
                f"on {customer_date} "
                "is already booked."
            )

        else:

            response = (
                f"Yes, {customer_time} "
                f"on {customer_date} "
                "is available."
            )

        print(
            f"[PERF] TOTAL voice_action: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return ChatResponse(
            response=response
        )

    # =====================================================
    # BOOK
    # =====================================================

    if action == "book":

        if not appointment_date:

            return ChatResponse(
                response=(
                    "What date would "
                    "you like the appointment?"
                )
            )

        if not appointment_time:

            return ChatResponse(
                response=(
                    "What time would "
                    "you like the appointment?"
                )
            )

        if not name:

            return ChatResponse(
                response=(
                    "May I have your name?"
                )
            )

        # -------------------------------------------------
        # CHECK AVAILABILITY
        # -------------------------------------------------

        checkpoint = time.perf_counter()

        availability = check_availability(
            date=appointment_date,
            time=appointment_time,
        )

        print(
            f"[PERF] voice availability check: "
            f"{time.perf_counter() - checkpoint:.3f}s"
        )

        customer_date = (
            format_date_for_customer(
                appointment_date
            )
        )

        customer_time = (
            format_time_for_customer(
                appointment_time
            )
        )

        if availability == "BOOKED":

            response = (
                f"Sorry, {customer_time} "
                f"on {customer_date} "
                "is already booked. "
                "Please choose another time."
            )

            print(
                f"[PERF] TOTAL voice_action: "
                f"{time.perf_counter() - total_start:.3f}s"
            )

            return ChatResponse(
                response=response
            )

        # -------------------------------------------------
        # BOOK
        # -------------------------------------------------

        checkpoint = time.perf_counter()

        result = book_appointment(
            date=appointment_date,
            time=appointment_time,
            name=name,
        )

        print(
            f"[PERF] voice book_appointment: "
            f"{time.perf_counter() - checkpoint:.3f}s"
        )

        if result == "BOOKED_SUCCESSFULLY":

            response = (
                f"Your appointment is "
                f"booked for {customer_date} "
                f"at {customer_time}. "
                f"Name: {name}."
            )

        elif result == "BOOKED":

            response = (
                f"Sorry, {customer_time} "
                f"on {customer_date} "
                "was just booked. "
                "Please choose another time."
            )

        else:

            response = (
                "I wasn't able to complete "
                "the booking right now. "
                "Please try again."
            )

        print(
            f"[PERF] TOTAL voice_action: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return ChatResponse(
            response=response
        )

    # =====================================================
    # CANCEL
    # =====================================================

    if action == "cancel":

        if not appointment_date:

            return ChatResponse(
                response=(
                    "What date is the "
                    "appointment you want "
                    "to cancel?"
                )
            )

        if not appointment_time:

            return ChatResponse(
                response=(
                    "What time is the "
                    "appointment you want "
                    "to cancel?"
                )
            )

        if not name:

            return ChatResponse(
                response=(
                    "May I have the name "
                    "on the appointment?"
                )
            )

        checkpoint = time.perf_counter()

        result = cancel_appointment(
            date=appointment_date,
            time=appointment_time,
            name=name,
        )

        print(
            f"[PERF] voice cancel_appointment: "
            f"{time.perf_counter() - checkpoint:.3f}s"
        )

        customer_date = (
            format_date_for_customer(
                appointment_date
            )
        )

        customer_time = (
            format_time_for_customer(
                appointment_time
            )
        )

        if result == "CANCELLED":

            response = (
                f"Your appointment on "
                f"{customer_date} at "
                f"{customer_time} has "
                "been cancelled."
            )

        elif result == "NOT_FOUND":

            response = (
                "I couldn't find a booked "
                "appointment matching "
                "those details."
            )

        else:

            response = (
                "I wasn't able to complete "
                "the cancellation right now."
            )

        print(
            f"[PERF] TOTAL voice_action: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return ChatResponse(
            response=response
        )

    # =====================================================
    # RESCHEDULE
    # =====================================================

    if action == "reschedule":

        if not name:

            return ChatResponse(
                response=(
                    "May I have the name "
                    "on the appointment?"
                )
            )

        if not appointment_date:

            return ChatResponse(
                response=(
                    "What is the current "
                    "date of your appointment?"
                )
            )

        if not appointment_time:

            return ChatResponse(
                response=(
                    "What is the current "
                    "time of your appointment?"
                )
            )

        if not new_date:

            return ChatResponse(
                response=(
                    "What new date would "
                    "you like?"
                )
            )

        if not new_time:

            return ChatResponse(
                response=(
                    "What new time would "
                    "you like?"
                )
            )

        checkpoint = time.perf_counter()

        result = reschedule_appointment(
            name=name,
            old_date=appointment_date,
            old_time=appointment_time,
            new_date=new_date,
            new_time=new_time,
        )

        print(
            f"[PERF] voice reschedule_appointment: "
            f"{time.perf_counter() - checkpoint:.3f}s"
        )

        new_customer_date = (
            format_date_for_customer(
                new_date
            )
        )

        new_customer_time = (
            format_time_for_customer(
                new_time
            )
        )

        if result == "RESCHEDULED":

            response = (
                f"Your appointment has "
                f"been rescheduled to "
                f"{new_customer_date} "
                f"at {new_customer_time}."
            )

        elif (
            result
            == "OLD_APPOINTMENT_NOT_FOUND"
        ):

            response = (
                "I couldn't find your "
                "existing appointment "
                "with those details."
            )

        elif result == "NEW_SLOT_BOOKED":

            response = (
                f"Sorry, {new_customer_time} "
                f"on {new_customer_date} "
                "is already booked. "
                "Please choose another time."
            )

        else:

            response = (
                "I wasn't able to complete "
                "the rescheduling right now."
            )

        print(
            f"[PERF] TOTAL voice_action: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return ChatResponse(
            response=response
        )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    print(
        f"[PERF] TOTAL voice_action: "
        f"{time.perf_counter() - total_start:.3f}s"
    )

    return ChatResponse(
        response=(
            "I wasn't able to process "
            "that appointment request."
        )
    )