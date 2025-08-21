from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import requests
from bs4 import BeautifulSoup

# 1. Получаем текст с веб-страницы
url = "https://example.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
text = soup.get_text()

# 2. Создаем шаблон промпта для суммаризации
template = """
Сделай краткое, но информативное резюме следующего текста:

{text}
"""

prompt = PromptTemplate(template=template, input_variables=["text"])

# 3. Создаем цепочку с LLM
llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Запускаем суммаризацию
summary = chain.run(text=text)
print(summary)
