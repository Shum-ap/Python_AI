from dotenv import load_dotenv
from transformers import AutoTokenizer
from google import genai
import os

# 1. Загружаем ключ ТОЛЬКО из .env
if not load_dotenv():
    raise FileNotFoundError("Файл .env не найден!")

api_key = os.getenv("GEMINI_API_KEY", None)
if not api_key:
    raise ValueError("GEMINI_API_KEY отсутствует в файле .env")

# 2. Токенизация текста
text = "i love to programm      and learn stuff"

# Словарная токенизация
result1 = text.split()
print("Word-based:", result1)

# Символьная токенизация
result2 = list(" ".join(text.split()))
print("Char-based:", result2)

# Токенизация через transformers
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
result3 = tokenizer.tokenize(text)
print("Transformers tokenizer:", result3)

# 3. Запрос к Gemini API
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)

print("\nОтвет от Gemini:")
print(response.text)
