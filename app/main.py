import json
import re
from datetime import date, datetime, timedelta

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

        # Tracks whether date was already requested
        "asked_for_date": False,
    }


# =========================================================
# DATE EXTRACTION
# =========================================================

def extract_date(message: str):
    """
    Extract dates from common Indian and natural-language
    formats.

    Supported examples:

    today
    tomorrow

    05/09/2026
    5/9/2026
    05-09-2026
    5-9-2026
    05.09.2026

    2026-09-05

    5 September
    5th September
    05 September
    5 September 2026
    5th September 2026

    September 5
    September 5th
    September 5, 2026

    5 Sep
    Sep 5
    """

    message_lower = message.lower().strip()

    today = datetime.now().date()

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
    # Indian date convention
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
    #
    # Examples:
    # 5 September
    # 5th September
    # 5 September 2026
    # 5th September 2026
    # 5 Sep
    # -----------------------------------------------------

    reverse_match = re.search(
        rf"\b"
        rf"(\d{{1,2}})"
        rf"(?:st|nd|rd|th)?"
        rf"\s+"
        rf"({month_pattern})"
        rf"(?:"
        rf"\s*,?\s*(\d{{4}})"
        rf")?"
        rf"\b",
        message_lower,
    )

    if reverse_match:

        day = int(
            reverse_match.group(1)
        )

        month_name = reverse_match.group(2)

        month = all_months[
            month_name
        ]

        year_text = reverse_match.group(3)

        if year_text:

            year = int(year_text)

        else:

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

    else:

        # -------------------------------------------------
        # MONTH + DAY + OPTIONAL YEAR
        #
        # Examples:
        # September 5
        # September 5th
        # September 5, 2026
        # Sep 5
        # -------------------------------------------------

        normal_match = re.search(
            rf"\b"
            rf"({month_pattern})"
            rf"\s+"
            rf"(\d{{1,2}})"
            rf"(?:st|nd|rd|th)?"
            rf"(?:"
            rf"\s*,?\s*(\d{{4}})"
            rf")?"
            rf"\b",
            message_lower,
        )

        if not normal_match:

            return None

        month_name = normal_match.group(1)

        day = int(
            normal_match.group(2)
        )

        month = all_months[
            month_name
        ]

        year_text = normal_match.group(3)

        if year_text:

            year = int(year_text)

        else:

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
    # DATE WITH EXPLICIT YEAR
    # -----------------------------------------------------

    try:

        return date(
            year,
            month,
            day,
        ).isoformat()

    except ValueError:

        return None


# =========================================================
# CUSTOMER-FACING DATE FORMAT
# =========================================================

def format_date_for_customer(
    date_string: str,
) -> str:
    """
    Internal:
        2026-09-05

    Customer:
        05/09/2026
    """

    try:

        return datetime.strptime(
            date_string,
            "%Y-%m-%d",
        ).strftime("%d/%m/%Y")

    except (
        ValueError,
        TypeError,
    ):

        return date_string


# =========================================================
# CUSTOMER-FACING TIME FORMAT
# =========================================================

def format_time_for_customer(
    time_string: str,
) -> str:
    """
    Internal:
        16:00

    Customer:
        4:00 PM
    """

    try:

        return datetime.strptime(
            time_string,
            "%H:%M",
        ).strftime("%I:%M %p").lstrip("0")

    except (
        ValueError,
        TypeError,
    ):

        return time_string


# =========================================================
# TIME EXTRACTION
# =========================================================

def extract_time(message: str):
    """
    Supports:

    4 PM
    4:00 PM
    04 PM

    16:00
    09:30

    4 in the afternoon
    9 in the morning
    """

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
    #
    # Examples:
    # 4 in the afternoon
    # 9 in the morning
    # 7 in the evening
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
            period in [
                "afternoon",
                "evening",
            ]
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

        if (
            period == "night"
            and hour == 24
        ):
            hour = 0

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
# INTENT DETECTION
# =========================================================

def detect_action(message: str):

    message_lower = message.lower()

    # -----------------------------------------------------
    # CANCELLATION
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RESCHEDULING
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # BOOKING
    # -----------------------------------------------------

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
# BOOKING HANDLER
# =========================================================

def handle_booking(
    message: str,
    state: dict,
):

    # =====================================================
    # DATE
    # =====================================================

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

    # =====================================================
    # TIME
    # =====================================================

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

    # =====================================================
    # NAME
    # =====================================================

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
            and not is_new_booking_request(
                cleaned_message
            )
        ):

            state["name"] = (
                cleaned_message
            )

        else:

            return (
                "May I have your name?"
            )

    # =====================================================
    # COMPLETE BOOKING
    # =====================================================

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

        # -------------------------------------------------
        # SLOT ALREADY BOOKED
        # -------------------------------------------------

        if availability == "BOOKED":

            return (
                f"Sorry, {customer_time} "
                f"on {customer_date} is "
                "already booked. Please "
                "choose another time."
            )

        # -------------------------------------------------
        # BOOK APPOINTMENT
        # -------------------------------------------------

        result = book_appointment(
            date=state["date"],
            time=state["time"],
            name=state["name"],
        )

        if result == (
            "BOOKED_SUCCESSFULLY"
        ):

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

        # -------------------------------------------------
        # SLOT BOOKED BETWEEN CHECK
        # AND INSERT
        # -------------------------------------------------

        if result == "BOOKED":

            return (
                f"Sorry, {customer_time} "
                f"on {customer_date} was "
                "just booked. Please choose "
                "another time."
            )

    return (
        "Please provide the appointment "
        "details."
    )


# =========================================================
# CANCELLATION HANDLER
# =========================================================

def handle_cancellation(
    message: str,
    state: dict,
):

    # =====================================================
    # DATE
    # =====================================================

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

    # =====================================================
    # TIME
    # =====================================================

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

    # =====================================================
    # NAME
    # =====================================================

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

            state["name"] = (
                cleaned_message
            )

        else:

            return (
                "May I have the name on "
                "the appointment?"
            )

    # =====================================================
    # CANCEL
    # =====================================================

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
# RESCHEDULING HANDLER
# =========================================================

def handle_rescheduling(
    message: str,
    state: dict,
):

    # =====================================================
    # NAME
    # =====================================================

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

            state["name"] = (
                cleaned_message
            )

        else:

            return (
                "May I have the name on "
                "the appointment?"
            )

    # =====================================================
    # OLD DATE
    # =====================================================

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

    # =====================================================
    # OLD TIME
    # =====================================================

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

    # =====================================================
    # NEW DATE
    # =====================================================

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

    # =====================================================
    # NEW TIME
    # =====================================================

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

    # =====================================================
    # RESCHEDULE
    # =====================================================

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

        if result == (
            "OLD_APPOINTMENT_NOT_FOUND"
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
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "ReceptionAI API is running"
    }


# =========================================================
# NORMAL CHAT ENDPOINT
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    conversation_id = (
        request.conversation_id
    )

    # =====================================================
    # LOAD OR CREATE CONVERSATION
    # =====================================================

    state = get_conversation_state(
        conversation_id
    )

    if state is None:

        state = default_state()

        create_conversation(
            conversation_id,
            state,
        )

    # =====================================================
    # LOAD HISTORY
    # =====================================================

    history = get_messages(
        conversation_id
    )

    # Save current user message.
    add_message(
        conversation_id,
        "user",
        request.message,
    )

    # =====================================================
    # DETECT INTENT
    # =====================================================

    detected_action = detect_action(
        request.message
    )

    if detected_action:

        # -------------------------------------------------
        # Fresh booking request after an incomplete
        # previous booking conversation.
        # -------------------------------------------------

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

    # =====================================================
    # BOOKING
    # =====================================================

    if action == "book":

        response = handle_booking(
            request.message,
            state,
        )

    # =====================================================
    # CANCELLATION
    # =====================================================

    elif action == "cancel":

        response = handle_cancellation(
            request.message,
            state,
        )

    # =====================================================
    # RESCHEDULING
    # =====================================================

    elif action == "reschedule":

        response = handle_rescheduling(
            request.message,
            state,
        )

    # =====================================================
    # NORMAL LLM RESPONSE
    # =====================================================

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

    # =====================================================
    # SAVE STATE + RESPONSE
    # =====================================================

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
# VOICE CHAT REQUEST
# =========================================================

class VoiceChatRequest(BaseModel):
    transcript: str
    interaction_id: str


# =========================================================
# EXTRACT LATEST USER UTTERANCE
# =========================================================

def extract_latest_user_utterance(
    transcript: str,
):
    """
    Sarvam sends the full call transcript through
    the Call Transcript variable.

    This function extracts the latest caller/user turn
    so that ReceptionAI processes only the new request.

    Supported transcript labels include:

    User:
    Caller:
    Customer:
    """

    transcript = transcript.strip()

    if not transcript:
        return None

    # -----------------------------------------------------
    # Try JSON transcript first.
    #
    # Supports structures containing:
    #
    # [
    #   {
    #       "role": "user",
    #       "en_text": "..."
    #   }
    # ]
    # -----------------------------------------------------

    try:

        parsed = json.loads(
            transcript
        )

        if isinstance(
            parsed,
            list,
        ):

            user_messages = []

            for turn in parsed:

                if not isinstance(
                    turn,
                    dict,
                ):
                    continue

                role = str(
                    turn.get(
                        "role",
                        ""
                    )
                ).lower()

                if role == "user":

                    text = (
                        turn.get("en_text")
                        or turn.get("text")
                        or turn.get("content")
                    )

                    if text:

                        user_messages.append(
                            str(text).strip()
                        )

            if user_messages:

                return user_messages[-1]

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        pass

    # -----------------------------------------------------
    # Normal text transcript.
    #
    # Example:
    #
    # Agent: Hello...
    # User: I want an appointment
    # Agent: What date?
    # User: Tomorrow
    # -----------------------------------------------------

    pattern = re.compile(
        r"(?:^|\n)\s*"
        r"(user|caller|customer)"
        r"\s*[:\-]\s*"
        r"(.+?)(?=\n\s*"
        r"(?:user|caller|customer|agent|assistant|receptionai)"
        r"\s*[:\-]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(
        transcript
    )

    if matches:

        return matches[-1][1].strip()

    # -----------------------------------------------------
    # If the transcript has no speaker labels and contains
    # only one line, it may simply be the current utterance.
    # -----------------------------------------------------

    lines = [
        line.strip()
        for line in transcript.splitlines()
        if line.strip()
    ]

    if len(lines) == 1:

        return lines[0]

    # -----------------------------------------------------
    # Multiple unlabeled lines cannot safely tell us which
    # line belongs to the caller.
    # -----------------------------------------------------

    return None


# =========================================================
# VOICE CHAT ENDPOINT
# =========================================================

@app.post(
    "/voice-chat",
    response_model=ChatResponse,
)
def voice_chat(
    request: VoiceChatRequest,
):

    # =====================================================
    # EXTRACT LATEST CALLER MESSAGE
    # =====================================================

    latest_user_message = (
        extract_latest_user_utterance(
            request.transcript
        )
    )

    if not latest_user_message:

        return ChatResponse(
            response=(
                "I didn't catch that. "
                "Could you please repeat?"
            )
        )

    # =====================================================
    # USE SARVAM INTERACTION ID AS OUR
    # CONVERSATION ID
    # =====================================================

    conversation_id = (
        request.interaction_id
    )

    # =====================================================
    # LOAD OR CREATE CONVERSATION
    # =====================================================

    state = get_conversation_state(
        conversation_id
    )

    if state is None:

        state = default_state()

        create_conversation(
            conversation_id,
            state,
        )

    # =====================================================
    # LOAD HISTORY
    # =====================================================

    history = get_messages(
        conversation_id
    )

    # =====================================================
    # SAVE ONLY THE NEW USER TURN
    #
    # We deliberately do NOT save the entire Sarvam
    # transcript because the previous turns are already
    # stored in our conversation history.
    # =====================================================

    add_message(
        conversation_id,
        "user",
        latest_user_message,
    )

    # =====================================================
    # DETECT INTENT
    # =====================================================

    detected_action = detect_action(
        latest_user_message
    )

    if detected_action:

        # -------------------------------------------------
        # Fresh booking request.
        # -------------------------------------------------

        if (
            detected_action == "book"
            and is_new_booking_request(
                latest_user_message
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

    # =====================================================
    # BOOKING
    # =====================================================

    if action == "book":

        response = handle_booking(
            latest_user_message,
            state,
        )

    # =====================================================
    # CANCELLATION
    # =====================================================

    elif action == "cancel":

        response = handle_cancellation(
            latest_user_message,
            state,
        )

    # =====================================================
    # RESCHEDULING
    # =====================================================

    elif action == "reschedule":

        response = handle_rescheduling(
            latest_user_message,
            state,
        )

    # =====================================================
    # NORMAL LLM RESPONSE
    # =====================================================

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
                "content": latest_user_message,
            }
        )

        response = get_llm_response(
            llm_messages,
            state,
        )

    # =====================================================
    # SAVE STATE + RESPONSE
    # =====================================================

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