from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# 🔥 load with override
load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not found")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # 🔥 use latest
    google_api_key=api_key
)

result = model.invoke("What is the capital of India?")
print(result.content)