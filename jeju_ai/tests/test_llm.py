import os


from dotenv import load_dotenv
from google import genai


load_dotenv()

MODEL = os.getenv("GEMINI_MODEL")

client = genai.Client(api_key = os.getenv("GEMINI_KEY"))
response  = client.models.generate_content(model = MODEL,contents = "안녕")
print(response.text)