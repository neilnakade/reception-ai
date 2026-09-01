from groq import Groq
import json

from app.config import GROQ_API_KEY, MODEL_NAME
from app.tools import check_availability, book_appointment


client = Groq(api_key=GROQ_API_KEY)


tools = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check whether a specific appointment date and time "
                "is available. Use this only when the user is asking "
                "about availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format."
                    },
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM 24-hour format."
                    }
                },
                "required": ["date", "time"]
            }
        }
    }
]


def get_llm_response(
    messages: list[dict],
    appointment_state: dict
) -> str:

    # Give the LLM the current appointment state.
    state_message = {
        "role": "system",
        "content": (
            "CURRENT APPOINTMENT STATE:\n"
            f"Name: {appointment_state.get('name')}\n"
            f"Date: {appointment_state.get('date')}\n"
            f"Time: {appointment_state.get('time')}\n\n"
            "IMPORTANT:\n"
            "Never invent or change a date, time, or name that is "
            "already present in CURRENT APPOINTMENT STATE."
        )
    }

    llm_messages = messages + [state_message]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=llm_messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )

    message = response.choices[0].message

    if not message.tool_calls:
        return message.content

    tool_call = message.tool_calls[0]

    if tool_call.function.name == "check_availability":

        arguments = json.loads(tool_call.function.arguments)

        # Prefer the values stored by Python.
        date = appointment_state.get("date") or arguments["date"]
        time = appointment_state.get("time") or arguments["time"]

        result = check_availability(
            date=date,
            time=time,
        )

        if result == "AVAILABLE":
            return f"Yes, {time} on {date} is available."

        return f"Sorry, {time} on {date} is already booked."

    return "I'm sorry, I couldn't complete that request."