from datetime import datetime, timedelta
import re

from fastapi import FastAPI

from app.llm import get_llm_response
from app.prompts import build_system_prompt
from app.schemas import ChatRequest, ChatResponse

from app.database import (
    initialize_database,
    create_conversation,
    get_conversation_state,
    save_conversation_state,
    add_message,
    get_messages,
)

from app.tools import (
    check_availability,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
)

from app.business import load_business_config


# ==========================================================
# INITIALIZATION
# ==========================================================

initialize_database()

BUSINESS_CONFIG = load_business_config()

SYSTEM_PROMPT = build_system_prompt(
    BUSINESS_CONFIG
)


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="ReceptionAI",
    description="AI receptionist API",
    version="0.1.0",
)


# ==========================================================
# DEFAULT APPOINTMENT STATE
# ==========================================================

def default_state() -> dict:
    """
    Create a fresh appointment state for a conversation.
    """

    return {
        "name": None,
        "date": None,
        "time": None,

        # Rescheduling fields
        "old_date": None,
        "old_time": None,
        "new_date": None,
        "new_time": None,

        # Current action
        "action": None,
    }


# ==========================================================
# DATE EXTRACTION
# ==========================================================

def extract_date(message: str) -> str | None:
    """
    Convert common natural-language date expressions
    into YYYY-MM-DD format.

    Supported examples:

    today
    tomorrow
    2026-09-02
    September 2
    September 2nd
    Sep 2
    Sep 2nd
    """

    text = message.lower().strip()

    today = datetime.now().date()

    # ------------------------------------------------------
    # Relative dates
    # ------------------------------------------------------

    if "tomorrow" in text:
        return str(
            today + timedelta(days=1)
        )

    if "today" in text:
        return str(today)

    # ------------------------------------------------------
    # ISO format: YYYY-MM-DD
    # ------------------------------------------------------

    match = re.search(
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        text,
    )

    if match:
        year, month, day = match.groups()

        try:
            return str(
                datetime(
                    int(year),
                    int(month),
                    int(day),
                ).date()
            )

        except ValueError:
            return None

    # ------------------------------------------------------
    # Month-name formats
    #
    # September 2
    # September 2nd
    # Sep 2
    # Sep 2nd
    # ------------------------------------------------------

    month_pattern = (
        r"\b("
        r"january|february|march|april|may|june|"
        r"july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
        r")\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?\b"
    )

    match = re.search(
        month_pattern,
        text,
    )

    if match:
        month_name = match.group(1)
        day = int(match.group(2))

        month_numbers = {
            "january": 1,
            "jan": 1,

            "february": 2,
            "feb": 2,

            "march": 3,
            "mar": 3,

            "april": 4,
            "apr": 4,

            "may": 5,

            "june": 6,
            "jun": 6,

            "july": 7,
            "jul": 7,

            "august": 8,
            "aug": 8,

            "september": 9,
            "sep": 9,
            "sept": 9,

            "october": 10,
            "oct": 10,

            "november": 11,
            "nov": 11,

            "december": 12,
            "dec": 12,
        }

        month = month_numbers[month_name]

        year = today.year

        try:
            candidate = datetime(
                year,
                month,
                day,
            ).date()

            # If the date has already passed this year,
            # use next year's occurrence.
            if candidate < today:
                candidate = datetime(
                    year + 1,
                    month,
                    day,
                ).date()

            return str(candidate)

        except ValueError:
            return None

    return None


# ==========================================================
# TIME EXTRACTION
# ==========================================================

def extract_time(message: str) -> str | None:
    """
    Convert common time formats into HH:MM 24-hour format.

    Examples:

    4 PM
    4:00 PM
    4pm
    4:00pm
    16:00
    """

    text = message.lower().strip()

    # ------------------------------------------------------
    # 12-hour format
    # ------------------------------------------------------

    match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        text,
    )

    if match:
        hour = int(match.group(1))
        minute = int(
            match.group(2) or 0
        )
        period = match.group(3)

        if hour < 1 or hour > 12:
            return None

        if period == "pm" and hour != 12:
            hour += 12

        if period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # ------------------------------------------------------
    # 24-hour format
    # ------------------------------------------------------

    match = re.search(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
        text,
    )

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        return f"{hour:02d}:{minute:02d}"

    return None


# ==========================================================
# INTENT DETECTION
# ==========================================================

def looks_like_cancel_request(
    message: str,
) -> bool:
    """
    Detect cancellation requests.
    """

    text = message.lower()

    cancel_words = [
        "cancel",
        "cancellation",
        "cancelled",
        "canceled",
    ]

    return any(
        word in text
        for word in cancel_words
    )


def looks_like_reschedule_request(
    message: str,
) -> bool:
    """
    Detect rescheduling requests.
    """

    text = message.lower()

    reschedule_phrases = [
        "reschedule",
        "rescheduled",
        "change my appointment",
        "move my appointment",
        "change the appointment",
    ]

    return any(
        phrase in text
        for phrase in reschedule_phrases
    )


def looks_like_booking_request(
    message: str,
) -> bool:
    """
    Detect booking requests.
    """

    if looks_like_cancel_request(
        message
    ):
        return False

    if looks_like_reschedule_request(
        message
    ):
        return False

    text = message.lower()

    booking_words = [
        "book",
        "booking",
        "schedule",
        "reserve",
        "appointment",
    ]

    return any(
        word in text
        for word in booking_words
    )


# ==========================================================
# BOOKING FLOW
# ==========================================================

def handle_booking(
    message: str,
    state: dict,
) -> str:
    """
    Collect booking details and create
    an appointment once all required
    information is available.
    """

    text = message.strip()

    # ------------------------------------------------------
    # Date
    # ------------------------------------------------------

    date = extract_date(text)

    if date:
        state["date"] = date

    # ------------------------------------------------------
    # Time
    # ------------------------------------------------------

    time = extract_time(text)

    if time:
        state["time"] = time

    # ------------------------------------------------------
    # Name
    # ------------------------------------------------------

    if (
        state["date"]
        and state["time"]
        and not state["name"]
        and not date
        and not time
    ):
        state["name"] = text

    # ------------------------------------------------------
    # Missing information
    # ------------------------------------------------------

    if not state["date"]:
        return (
            "Sure. What date would you like "
            "the appointment?"
        )

    if not state["time"]:
        return (
            "What time would you like "
            "the appointment?"
        )

    if not state["name"]:
        return (
            "Sure. May I have your name?"
        )

    # ------------------------------------------------------
    # Check availability
    # ------------------------------------------------------

    availability = check_availability(
        date=state["date"],
        time=state["time"],
    )

    if availability == "BOOKED":

        return (
            f"Sorry, {state['time']} on "
            f"{state['date']} is already booked. "
            "Please choose another time."
        )

    # ------------------------------------------------------
    # Create appointment
    # ------------------------------------------------------

    result = book_appointment(
        date=state["date"],
        time=state["time"],
        name=state["name"],
    )

    if result == "BOOKED_SUCCESSFULLY":

        response = (
            "Your appointment has been successfully booked.\n"
            f"Name: {state['name']}\n"
            f"Date: {state['date']}\n"
            f"Time: {state['time']}"
        )

        # Clear completed booking state
        state.clear()
        state.update(
            default_state()
        )

        return response

    return (
        "I couldn't complete the booking."
    )


# ==========================================================
# CANCELLATION FLOW
# ==========================================================

def handle_cancellation(
    message: str,
    state: dict,
) -> str:
    """
    Collect appointment information and
    cancel an existing appointment.
    """

    text = message.strip()

    date = extract_date(text)

    if date:
        state["date"] = date

    time = extract_time(text)

    if time:
        state["time"] = time

    # ------------------------------------------------------
    # Name
    # ------------------------------------------------------

    if (
        state["date"]
        and state["time"]
        and not state["name"]
        and not date
        and not time
    ):
        state["name"] = text

    # ------------------------------------------------------
    # Missing information
    # ------------------------------------------------------

    if not state["date"]:
        return (
            "Sure. What date is the appointment "
            "you want to cancel?"
        )

    if not state["time"]:
        return (
            "What time is that appointment?"
        )

    if not state["name"]:
        return (
            "May I have the name on the appointment?"
        )

    # ------------------------------------------------------
    # Cancel
    # ------------------------------------------------------

    result = cancel_appointment(
        date=state["date"],
        time=state["time"],
        name=state["name"],
    )

    if result == "CANCELLED":

        response = (
            "Your appointment has been successfully "
            "cancelled.\n"
            f"Name: {state['name']}\n"
            f"Date: {state['date']}\n"
            f"Time: {state['time']}"
        )

        state.clear()
        state.update(
            default_state()
        )

        return response

    return (
        "I couldn't find a booked appointment "
        "matching those details. Please check "
        "the name, date, and time."
    )


# ==========================================================
# RESCHEDULING FLOW
# ==========================================================

def handle_rescheduling(
    message: str,
    state: dict,
) -> str:
    """
    Collect old and new appointment details
    and reschedule the appointment.
    """

    text = message.strip()

    date = extract_date(text)
    time = extract_time(text)

    # ------------------------------------------------------
    # Existing appointment date
    # ------------------------------------------------------

    if date:

        if not state["old_date"]:
            state["old_date"] = date

        else:
            state["new_date"] = date

    # ------------------------------------------------------
    # Existing appointment time
    # ------------------------------------------------------

    if time:

        if not state["old_time"]:
            state["old_time"] = time

        else:
            state["new_time"] = time

    # ------------------------------------------------------
    # Name
    # ------------------------------------------------------

    if (
        state["old_date"]
        and state["old_time"]
        and not state["name"]
        and not date
        and not time
    ):
        state["name"] = text

    # ------------------------------------------------------
    # Missing information
    # ------------------------------------------------------

    if not state["old_date"]:
        return (
            "What date is your current appointment?"
        )

    if not state["old_time"]:
        return (
            "What time is your current appointment?"
        )

    if not state["name"]:
        return (
            "May I have the name on the appointment?"
        )

    if not state["new_date"]:
        return (
            "What new date would you like?"
        )

    if not state["new_time"]:
        return (
            "What new time would you like?"
        )

    # ------------------------------------------------------
    # Reschedule
    # ------------------------------------------------------

    result = reschedule_appointment(
        name=state["name"],
        old_date=state["old_date"],
        old_time=state["old_time"],
        new_date=state["new_date"],
        new_time=state["new_time"],
    )

    if result == "OLD_APPOINTMENT_NOT_FOUND":
        return (
            "I couldn't find your existing appointment "
            "with those details."
        )

    if result == "NEW_SLOT_BOOKED":
        return (
            f"Sorry, {state['new_time']} on "
            f"{state['new_date']} is already booked. "
            "Please choose another time."
        )

    if result == "RESCHEDULED":

        response = (
            "Your appointment has been successfully "
            "rescheduled.\n"
            f"Name: {state['name']}\n"
            f"New date: {state['new_date']}\n"
            f"New time: {state['new_time']}"
        )

        state.clear()
        state.update(
            default_state()
        )

        return response

    return (
        "I couldn't complete the rescheduling."
    )


# ==========================================================
# API ROUTES
# ==========================================================

@app.get("/")
def root():
    """
    Basic health-check endpoint.
    """

    return {
        "message": "ReceptionAI API is running"
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):
    """
    Main ReceptionAI chat endpoint.
    """

    conversation_id = request.conversation_id

    # ------------------------------------------------------
    # Load conversation state
    # ------------------------------------------------------

    state = get_conversation_state(
        conversation_id
    )

    # ------------------------------------------------------
    # Create conversation if needed
    # ------------------------------------------------------

    if state is None:

        state = default_state()

        create_conversation(
            conversation_id=conversation_id,
            initial_state=state,
        )

    # ------------------------------------------------------
    # Load previous messages
    # ------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(
        get_messages(conversation_id)
    )

    user_message = request.message

    # ------------------------------------------------------
    # Save incoming message
    # ------------------------------------------------------

    add_message(
        conversation_id,
        "user",
        user_message,
    )

    # ------------------------------------------------------
    # Detect requested action
    # ------------------------------------------------------

    if looks_like_reschedule_request(
        user_message
    ):

        state["action"] = "reschedule"

    elif looks_like_cancel_request(
        user_message
    ):

        state["action"] = "cancel"

    elif looks_like_booking_request(
        user_message
    ):

        state["action"] = "book"

    # ------------------------------------------------------
    # Execute business workflow
    # ------------------------------------------------------

    if state["action"] == "book":

        response = handle_booking(
            message=user_message,
            state=state,
        )

    elif state["action"] == "cancel":

        response = handle_cancellation(
            message=user_message,
            state=state,
        )

    elif state["action"] == "reschedule":

        response = handle_rescheduling(
            message=user_message,
            state=state,
        )

    else:

        response = get_llm_response(
            messages=messages + [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            appointment_state=state,
        )

    # ------------------------------------------------------
    # Save assistant response
    # ------------------------------------------------------

    add_message(
        conversation_id,
        "assistant",
        response,
    )

    # ------------------------------------------------------
    # Save updated state
    # ------------------------------------------------------

    save_conversation_state(
        conversation_id,
        state,
    )

    return ChatResponse(
        response=response
    )