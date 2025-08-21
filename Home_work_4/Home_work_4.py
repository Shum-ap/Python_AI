import time
import threading
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sentence_transformers import SentenceTransformer, util

# ==============================
# 🔹 Настройка API
# ==============================
# Положи свой ключ в отдельный файл api_key.txt и считай его
with open("../api_key.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()


# ==============================
# 🔹 Rate Limit
# ==============================
MAX_REQUESTS_PER_MIN = 200
INTERVAL = 60 / MAX_REQUESTS_PER_MIN
_last_call_time = 0.0
_lock = threading.Lock()


def rate_limiter():
    """Простейший локальный rate limiter."""
    global _last_call_time
    with _lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)
        _last_call_time = time.time()


# ==============================
# 🔹 Запрос с retry и timeout
# ==============================
@retry(
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
def request_with_retry(url: str, params: dict = None, timeout: int = 10):
    rate_limiter()
    try:
        # Преобразуем ключ в ASCII на всякий случай
        api_key_ascii = str(API_KEY).encode("ascii", errors="ignore").decode("ascii")
        headers = {"Authorization": f"Bearer {api_key_ascii}"}

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                print(f"⚠️ Rate limit, жду {retry_after} сек...")
                time.sleep(retry_after)
                raise httpx.RequestError("Rate limited")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"Ошибка запроса: {e}, повтор...")
        raise


# ==============================
# 🔹 Работа с эмбеддингами
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")


def compare_texts(text1: str, text2: str) -> float:
    """Сравнить два текста по смыслу (косинусное сходство)."""
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return similarity.item()


def semantic_search(query: str, corpus: list, top_k: int = 3):
    """Поиск похожих текстов в коллекции."""
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_emb = model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_emb, corpus_embeddings)[0]
    top_results = similarities.topk(k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append((corpus[idx], float(score)))
    return results


# ==============================
# 🔹 Пример использования
# ==============================
if __name__ == "__main__":
    print("=== Тест API-запроса с retry и timeout ===")
    url = "https://httpbin.org/get"
    data = request_with_retry(url, params={"q": "test"})
    print(data)

    print("\n=== Сравнение текстов ===")
    s = compare_texts("Я люблю программирование.", "Кодинг – это моё хобби.")
    print(f"Сходство: {s:.4f}")

    print("\n=== Поиск похожих текстов ===")
    corpus = [
        "Кошки любят молоко.",
        "Собаки верные друзья человека.",
        "Программирование на Python очень популярно.",
        "Искусственный интеллект меняет мир.",
        "Котята милые и игривые."
    ]
    query = "Питание кошек"
    results = semantic_search(query, corpus, top_k=3)
    for text, score in results:
        print(f"{text} (сходство {score:.4f})")
