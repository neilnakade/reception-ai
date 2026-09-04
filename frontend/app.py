import os
import uuid

import requests
import streamlit as st


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/chat",
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="ReceptionAI",
    page_icon="🦷",
    layout="centered",
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🦷 ReceptionAI")
st.caption("AI receptionist for Smile Dental Clinic")


# ---------------------------------------------------------
# NEW CONVERSATION
# ---------------------------------------------------------

if st.button(
    "🆕 New Conversation",
    use_container_width=True,
):

    st.session_state.conversation_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []

    st.rerun()


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

user_message = st.chat_input(
    "How can I help you?"
)


if user_message:

    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    # -----------------------------------------------------
    # SEND TO FASTAPI
    # -----------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            json={
                "message": user_message,
                "conversation_id": (
                    st.session_state.conversation_id
                ),
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        assistant_message = data["response"]

    except requests.exceptions.ConnectionError:

        assistant_message = (
            "I couldn't connect to the ReceptionAI server. "
            "Please make sure the FastAPI server is running."
        )

    except requests.exceptions.Timeout:

        assistant_message = (
            "The request took too long. "
            "Please try again."
        )

    except requests.exceptions.HTTPError as error:

        assistant_message = (
            f"The server returned an error: {error}"
        )

    except (KeyError, ValueError):

        assistant_message = (
            "The server returned an unexpected response."
        )

    except requests.exceptions.RequestException as error:

        assistant_message = (
            f"Something went wrong: {error}"
        )

    # -----------------------------------------------------
    # SHOW ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_message)