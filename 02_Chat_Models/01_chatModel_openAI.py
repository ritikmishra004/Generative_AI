from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4',temperature=0,max_completion_tokens=20)

result = model.invoke("suggest me 5 indian male names")

print(result.content)