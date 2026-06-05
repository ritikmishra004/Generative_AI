from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('/Users/ritikmishra/Desktop/Gen_AI/09_Document_loader/seminar_ritik copy.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
    separator='\n'
)

result = splitter.split_documents(docs)

print(result[1].page_content)