import os
from dotenv import load_dotenv

load_dotenv()

print("From Python:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))