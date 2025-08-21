import os
import requests
from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Убедись, что OPENAI_API_KEY корректно задан в переменных окружения
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Установи переменную окружения OPENAI_API_KEY!")

# Инициализация модели
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)


def get_text_from_url(url: str) -> str:
    """Получает текст со страницы по URL"""
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")

    # Извлекаем текст из всех <p>
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text() for p in paragraphs)
    return text


def summarize_text(text: str) -> str:
    """Делает суммаризацию текста с помощью ChatOpenAI"""
    messages = [
        HumanMessage(content=f"Сделай краткое и понятное резюме этого текста:\n{text}")
    ]
    # Используем invoke() вместо устаревшего __call__
    result = llm.invoke(messages)
    return result.content


def summarize_url(url: str) -> str:
    """Суммаризация текста со страницы URL"""
    text = get_text_from_url(url)
    # Ограничиваем длинный текст для модели (первые 2000 символов)
    summary = summarize_text(text[:2000])
    return summary


if __name__ == "__main__":
    url = "https://example.com"  # Замени на нужный URL
    print("--- Суммаризация URL ---")
    print(summarize_url(url))


# Получение текста с веб-страницы
def get_text_from_url(url: str) -> str:
    response = requests.get(url)
    response.encoding = 'utf-8'  # чтобы правильно декодировать символы
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

# Суммаризация текста
def summarize_text(text: str) -> str:
    messages = prompt.format_messages(text=text)
    # Используем __call__, а не predict_messages
    result = llm(messages)
    return result.content

# Суммаризация URL
def summarize_url(url: str) -> str:
    text = get_text_from_url(url)
    if not text.strip():
        return "Текст на странице не найден."
    # Ограничим длину текста для модели
    summary = summarize_text(text[:2000])
    return summary

# Пример
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    print("--- Суммаризация URL ---")
    print(summarize_url(url))
