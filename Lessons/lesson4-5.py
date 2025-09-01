# from transformers import AutoTokenizer
from google import genai
import time
# from google.genai import types
# import faiss
# from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
# from requests import ReadTimeout
import os

def get_response(prompt, seconds_to_sleep=2):
    time.sleep(seconds_to_sleep)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response

text = 'write a function to find factorial'
response = get_response(text)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

timeout_seconds = 1
client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_response(prompt, seconds_to_sleep=2):
    time.sleep(seconds_to_sleep)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return response
    except ReadTimeout:
        print(f'The time of the response is out')
    except Exception as e:
        print(f'Exception was raised: {e}')


Teacher
18
Teacher
18
10: 11


def get_embedding(text):
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text)

        return np.array(response.embeddings[0].values)
    except Exception as e:
        print(f'Exception was raised: {e}')


reviews = ['погода была хорошая',
           'кофе был отличный',
           'стулья были сломаны']

embeddings_array = np.array([get_embedding(review) for review in reviews])
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)


def semantic_search(query, index, texts, k=2):
    query_embedding = get_embedding(query).reshape(1, -1)
    D, I = index.search(query_embedding, k)
    results = [texts[i] for i in I[0]]
    return results


query = 'Мне нравится латте'
result_search = semantic_search(query, index, reviews, k=1)
result_search
# from transformers import AutoTokenizer
from google import genai
import time
from google.genai import types
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from requests import ReadTimeout
import os
import numpy as np