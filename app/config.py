import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

if not MODEL_NAME:
    raise ValueError("MODEL_NAME is not set in the .env file.")