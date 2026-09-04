# 🦷 ReceptionAI

AI-powered receptionist for appointment-based businesses.

ReceptionAI can understand customer requests, answer business questions,
check appointment availability, book appointments, cancel appointments,
reschedule appointments, and maintain conversation state.

The project is currently configured for a dental clinic and can be adapted
for other appointment-based businesses such as salons, clinics, gyms,
law offices, and service businesses.

---

## 🚀 Live Demo

### Frontend
https://reception-ai-neil.streamlit.app/

### Backend API
https://reception-ai-b706.onrender.com

### API Documentation

https://reception-ai-b706.onrender.com/docs


---

## ✨ Features

- AI-powered customer conversations
- Business information and service queries
- Appointment availability checking
- Appointment booking
- Appointment cancellation
- Appointment rescheduling
- Persistent conversation state
- Conversation IDs
- PostgreSQL database
- Streamlit customer interface
- FastAPI backend
- Groq-powered LLM
- Configurable business information
- Natural-language date and time handling
- Indian-friendly date formats such as `05/09/2026`
- Fallback handling for unsupported date formats

---

## 🧠 Architecture

```text
Customer
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
ReceptionAI Logic
   ↓
Groq LLM
   ↓
Business Tools
   ↓
Supabase PostgreSQL



🛠️ Tech Stack
Backend
Python
FastAPI
Uvicorn
Pydantic
AI
Groq API
LLM-based conversational interface
Tool-based business actions
Database
PostgreSQL
Supabase
Frontend
Streamlit
Development
Git
GitHub
Virtual environment
📁 Project Structure
reception-ai/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── llm.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── tools.py
│   ├── database.py
│   └── business.py
│
├── frontend/
│   └── app.py
│
├── business.json
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Local Setup
1. Clone the repository
git clone https://github.com/neilnakade/reception-ai.git
cd reception-ai
2. Create a virtual environment
python -m venv .venv
3. Activate the environment

Windows PowerShell:

.venv\Scripts\Activate
4. Install dependencies
pip install -r requirements.txt
5. Create .env

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
MODEL_NAME=openai/gpt-oss-20b
DATABASE_URL=your_postgresql_connection_string

Never commit .env to GitHub.

▶️ Run Locally
Start FastAPI
uvicorn app.main:app --reload

Backend:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs
Start Streamlit

Open a second terminal:

streamlit run frontend/app.py

Frontend:

http://localhost:8501
💬 Example Conversation
Customer:
I want to book an appointment.

ReceptionAI:
Sure. What date would you like the appointment?

Customer:
5 September

ReceptionAI:
What time would you like the appointment?

Customer:
4 PM

ReceptionAI:
May I have your name?

Customer:
Neil

ReceptionAI:
Your appointment is booked for 05/09/2026 at 16:00.
Name: Neil.

ReceptionAI accepts common date formats such as:

5 September
5th September
05/09/2026
05-09-2026
September 5
Sep 5
tomorrow
today

Internally, dates are stored in an unambiguous database format.

🔄 Appointment Operations
Availability

The system checks the database before confirming an appointment slot.

Booking

A booking is created only after the selected slot is confirmed as available.

Cancellation

The system searches for the matching booked appointment and changes
its status to cancelled.

Rescheduling

The existing appointment is cancelled and a new appointment is created
at the requested available date and time.

💾 Persistent Conversations

Each chat session receives a unique conversation ID.

Conversation state and messages are stored in PostgreSQL so the backend
can maintain context across requests.

Starting a new conversation creates a new conversation ID without deleting
previous conversations.

🏢 Business Configuration

Business information is separated from the application logic in:

business.json

This allows the same ReceptionAI system to be adapted for different
businesses without rewriting the core application.

🔐 Environment Variables

The application uses environment variables for sensitive configuration:

GROQ_API_KEY
MODEL_NAME
DATABASE_URL

API keys and database credentials should never be committed to the
repository.

🌐 Deployment

The current architecture uses:

Streamlit
    ↓
Render
    ↓
FastAPI
    ↓
Supabase PostgreSQL
    ↓
Groq

The backend and frontend are deployed separately.

🎯 Current MVP

ReceptionAI currently demonstrates:

Conversational customer interaction
Business information retrieval
Appointment scheduling
Availability checking
Cancellation
Rescheduling
Persistent conversations
Production-style API architecture
Cloud PostgreSQL storage
🚧 Future Improvements
Voice receptionist
Phone-call integration
Customer phone/email identification
Authentication
Admin dashboard
Real calendar integration
Multiple businesses / multi-tenant architecture
Analytics
Automated reminders
👨‍💻 Author

Neil Nakade

GitHub:
https://github.com/neilnakade
