import re
from datetime import date, datetime, timedelta

from fastapi import FastAPI

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


app = FastAPI(
    title="ReceptionAI",
    description="AI receptionist API",
    version="0.1.0",
)


# ---------------------------------------------------------
# DATABASE + BUSINESS CONFIGURATION
# ---------------------------------------------------------

initialize_database()

BUSINESS_CONFIG = load_business_config()
SYSTEM_PROMPT = build_system_prompt(BUSINESS_CONFIG)


# ---------------------------------------------------------
# DEFAULT CONVERSATION STATE
# ---------------------------------------------------------

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

        "action": None,

        # Used to give a proper message when
        # a customer provides an invalid date.
        "asked_for_date": False,
    }


# ---------------------------------------------------------
# DATE EXTRACTION
# ---------------------------------------------------------

def extract_date(message: str):
    """
    Extract a date from common Indian and natural-language formats.

    Supported examples:

    today
    tomorrow

    05/09/2026
    5/9/2026
    05-09-2026
    5-9-2026

    2026-09-05

    5 September
    5th September
    05 September

    September 5
    September 5th

    5 Sep
    Sep 5
    """

    message_lower = message.lower().strip()

    today = datetime.now().date()

    # -----------------------------------------------------
    # TODAY
    # -----------------------------------------------------

    if re.search(r"\btoday\b", message_lower):
        return today.isoformat()

    # -----------------------------------------------------
    # TOMORROW
    # -----------------------------------------------------

    if re.search(r"\btomorrow\b", message_lower):
        return (today + timedelta(days=1)).isoformat()

    # -----------------------------------------------------
    # YYYY-MM-DD
    #
    # Example:
    # 2026-09-05
    # -----------------------------------------------------

    iso_match = re.search(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        message_lower,
    )

    if iso_match:
        year, month, day = map(int, iso_match.groups())

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
    #
    # India-friendly format.
    #
    # Examples:
    # 05/09/2026
    # 5/9/2026
    # 05-09-2026
    # -----------------------------------------------------

    indian_match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
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
    # DAY + MONTH
    #
    # Examples:
    # 5 September
    # 5th September
    # 05 September
    # 5 Sep
    # -----------------------------------------------------

    reverse_match = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})\b",
        message_lower,
    )

    if reverse_match:

        day = int(
            reverse_match.group(1)
        )

        month_name = reverse_match.group(2)

        month = all_months[month_name]

    else:

        # -------------------------------------------------
        # MONTH + DAY
        #
        # Examples:
        # September 5
        # September 5th
        # Sep 5
        # -------------------------------------------------

        normal_match = re.search(
            rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
            message_lower,
        )

        if not normal_match:
            return None

        month_name = normal_match.group(1)

        day = int(
            normal_match.group(2)
        )

        month = all_months[month_name]

    # -----------------------------------------------------
    # YEAR HANDLING
    #
    # If customer doesn't specify the year:
    # use current year.
    #
    # If that date has already passed:
    # use next year.
    # -----------------------------------------------------

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


# ---------------------------------------------------------
# CUSTOMER-FACING DATE FORMAT
# ---------------------------------------------------------

def format_date_for_customer(date_string: str) -> str:
    """
    Convert internal YYYY-MM-DD format
    into Indian DD/MM/YYYY format.
    """

    try:

        return datetime.strptime(
            date_string,
            "%Y-%m-%d",
        ).strftime("%d/%m/%Y")

    except (ValueError, TypeError):

        return date_string


# ---------------------------------------------------------
# TIME EXTRACTION
# ---------------------------------------------------------

def extract_time(message: str):
    """
    Extract appointment time.

    Supports examples:

    4 PM
    4:00 PM
    04 PM

    16:00
    09:30
    """

    message_lower = message.lower().strip()

    # -----------------------------------------------------
    # 12-HOUR FORMAT
    #
    # Examples:
    # 4 PM
    # 4:00 PM
    # 04 PM
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

        if period == "pm" and hour != 12:
            hour += 12

        if period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # -----------------------------------------------------
    # 24-HOUR FORMAT
    #
    # Examples:
    # 16:00
    # 09:30
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


# ---------------------------------------------------------
# INTENT DETECTION
# ---------------------------------------------------------

def detect_action(message: str):

    message_lower = message.lower()

    # Cancellation has priority.
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

    # Rescheduling has priority over normal booking.
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

    # Booking keywords.
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


# ---------------------------------------------------------
# BOOKING HANDLER
# ---------------------------------------------------------

def handle_booking(
    message: str,
    state: dict,
):

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    if not state.get("date"):

        extracted_date = extract_date(message)

        if extracted_date:

            state["date"] = extracted_date
            state["asked_for_date"] = False

        else:

            if state.get("asked_for_date"):

                return (
                    "I couldn't quite understand that date. "
                    "You can say something like 5 September, "
                    "05/09/2026, or tomorrow."
                )

            state["asked_for_date"] = True

            return (
                "Sure. What date would you like "
                "the appointment?"
            )

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    if not state.get("time"):

        extracted_time = extract_time(message)

        if extracted_time:

            state["time"] = extracted_time

        else:

            return (
                "What time would you like "
                "the appointment?"
            )

    # -----------------------------------------------------
    # NAME
    # -----------------------------------------------------

    if not state.get("name"):

        cleaned_message = message.strip()

        extracted_time = extract_time(
            cleaned_message
        )

        extracted_date = extract_date(
            cleaned_message
        )

        # Don't accidentally save a date/time
        # as the person's name.
        if (
            cleaned_message
            and not extracted_time
            and not extracted_date
        ):

            state["name"] = cleaned_message

        else:

            return "May I have your name?"

    # -----------------------------------------------------
    # FINAL BOOKING
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

        customer_date = format_date_for_customer(
            state["date"]
        )

        if availability == "BOOKED":

            return (
                f"Sorry, {state['time']} on "
                f"{customer_date} is already booked. "
                "Please choose another time."
            )

        result = book_appointment(
            date=state["date"],
            time=state["time"],
            name=state["name"],
        )

        if result == "BOOKED_SUCCESSFULLY":

            response = (
                f"Your appointment is booked for "
                f"{customer_date} at "
                f"{state['time']}. "
                f"Name: {state['name']}."
            )

            # Reset booking state after success.
            state.clear()
            state.update(default_state())

            return response

        if result == "BOOKED":

            return (
                f"Sorry, {state['time']} on "
                f"{customer_date} was just booked. "
                "Please choose another time."
            )

    return "Please provide the appointment details."


# ---------------------------------------------------------
# CANCELLATION HANDLER
# ---------------------------------------------------------

def handle_cancellation(
    message: str,
    state: dict,
):

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    if not state.get("date"):

        extracted_date = extract_date(message)

        if extracted_date:
            state["date"] = extracted_date

        else:
            return (
                "What date is the appointment "
                "you would like to cancel?"
            )

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    if not state.get("time"):

        extracted_time = extract_time(message)

        if extracted_time:
            state["time"] = extracted_time

        else:
            return (
                "What time is the appointment "
                "you would like to cancel?"
            )

    # -----------------------------------------------------
    # NAME
    # -----------------------------------------------------

    if not state.get("name"):

        cleaned_message = message.strip()

        extracted_time = extract_time(
            cleaned_message
        )

        extracted_date = extract_date(
            cleaned_message
        )

        if (
            cleaned_message
            and not extracted_time
            and not extracted_date
        ):

            state["name"] = cleaned_message

        else:

            return (
                "May I have the name "
                "on the appointment?"
            )

    # -----------------------------------------------------
    # CANCEL
    # -----------------------------------------------------

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

        customer_date = format_date_for_customer(
            state["date"]
        )

        if result == "CANCELLED":

            response = (
                f"Your appointment on "
                f"{customer_date} at "
                f"{state['time']} has been cancelled."
            )

            state.clear()
            state.update(default_state())

            return response

        if result == "NOT_FOUND":

            return (
                "I couldn't find a booked appointment "
                "matching those details."
            )

    return "Please provide the appointment details."


# ---------------------------------------------------------
# RESCHEDULE HANDLER
# ---------------------------------------------------------

def handle_rescheduling(
    message: str,
    state: dict,
):

    # -----------------------------------------------------
    # CUSTOMER NAME
    # -----------------------------------------------------

    if not state.get("name"):

        cleaned_message = message.strip()

        extracted_time = extract_time(
            cleaned_message
        )

        extracted_date = extract_date(
            cleaned_message
        )

        if (
            cleaned_message
            and not extracted_time
            and not extracted_date
        ):
            state["name"] = cleaned_message

        else:
            return "May I have the name on the appointment?"

    # -----------------------------------------------------
    # OLD DATE
    # -----------------------------------------------------

    if not state.get("old_date"):

        extracted_date = extract_date(message)

        if extracted_date:
            state["old_date"] = extracted_date

        else:
            return (
                "What is the current date of "
                "your appointment?"
            )

    # -----------------------------------------------------
    # OLD TIME
    # -----------------------------------------------------

    if not state.get("old_time"):

        extracted_time = extract_time(message)

        if extracted_time:
            state["old_time"] = extracted_time

        else:
            return (
                "What is the current time of "
                "your appointment?"
            )

    # -----------------------------------------------------
    # NEW DATE
    # -----------------------------------------------------

    if not state.get("new_date"):

        extracted_date = extract_date(message)

        # Don't accidentally use the old date again.
        if (
            extracted_date
            and extracted_date != state.get("old_date")
        ):
            state["new_date"] = extracted_date

        else:
            return (
                "What new date would you like "
                "for the appointment?"
            )

    # -----------------------------------------------------
    # NEW TIME
    # -----------------------------------------------------

    if not state.get("new_time"):

        extracted_time = extract_time(message)

        if (
            extracted_time
            and extracted_time != state.get("old_time")
        ):

            state["new_time"] = extracted_time

        else:

            return (
                "What new time would you like "
                "for the appointment?"
            )

    # -----------------------------------------------------
    # FINAL RESCHEDULE
    # -----------------------------------------------------

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

        new_customer_date = format_date_for_customer(
            state["new_date"]
        )

        if result == "RESCHEDULED":

            response = (
                f"Your appointment has been rescheduled "
                f"to {new_customer_date} at "
                f"{state['new_time']}."
            )

            state.clear()
            state.update(default_state())

            return response

        if result == "OLD_APPOINTMENT_NOT_FOUND":

            return (
                "I couldn't find your existing appointment "
                "with those details."
            )

        if result == "NEW_SLOT_BOOKED":

            return (
                f"Sorry, {state['new_time']} on "
                f"{new_customer_date} is already booked. "
                "Please choose another time."
            )

    return "Please provide the appointment details."


# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "ReceptionAI API is running"
    }


# ---------------------------------------------------------
# CHAT ENDPOINT
# ---------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    conversation_id = request.conversation_id

    # -----------------------------------------------------
    # LOAD OR CREATE CONVERSATION
    # -----------------------------------------------------

    state = get_conversation_state(
        conversation_id
    )

    if state is None:

        state = default_state()

        create_conversation(
            conversation_id,
            state,
        )

    # -----------------------------------------------------
    # LOAD PREVIOUS MESSAGES
    # -----------------------------------------------------

    history = get_messages(
        conversation_id
    )

    # Add current user message.
    add_message(
        conversation_id,
        "user",
        request.message,
    )

    # -----------------------------------------------------
    # DETECT ACTION
    # -----------------------------------------------------

    detected_action = detect_action(
        request.message
    )

    # Only update action if a new actionable
    # intent was detected.
    if detected_action:

        state["action"] = detected_action

    action = state.get("action")

    # -----------------------------------------------------
    # HANDLE BOOKING
    # -----------------------------------------------------

    if action == "book":

        response = handle_booking(
            request.message,
            state,
        )

    # -----------------------------------------------------
    # HANDLE CANCELLATION
    # -----------------------------------------------------

    elif action == "cancel":

        response = handle_cancellation(
            request.message,
            state,
        )

    # -----------------------------------------------------
    # HANDLE RESCHEDULING
    # -----------------------------------------------------

    elif action == "reschedule":

        response = handle_rescheduling(
            request.message,
            state,
        )

    # -----------------------------------------------------
    # NORMAL AI RESPONSE
    # -----------------------------------------------------

    else:

        llm_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # Include previous conversation history.
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

    # -----------------------------------------------------
    # SAVE STATE + RESPONSE
    # -----------------------------------------------------

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