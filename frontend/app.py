import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(
    page_title="ReceptionAI",
    page_icon="🦷",
    layout="centered",
)


st.title("🦷 ReceptionAI")
st.caption("AI receptionist for Smile Dental Clinic")


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "web-user-001"

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

user_message = st.chat_input("How can I help you?")


if user_message:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    # Send request to FastAPI
    try:

        response = requests.post(
            API_URL,
            json={
                "message": user_message,
                "conversation_id": st.session_state.conversation_id,
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
            "The request took too long. Please try again."
        )

    except requests.exceptions.RequestException:

        assistant_message = (
            "Something went wrong while contacting the receptionist."
        )

    # Show assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_message)