from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


prompt1 = PromptTemplate(
    template="generate a small report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate a 2 pointer summary of the following text \n {text}",
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model |parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()