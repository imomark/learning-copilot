import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

settings = Settings()