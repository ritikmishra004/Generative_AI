from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

# 🔥 HF API embedding
doc_embeddings = client.feature_extraction(documents)
query_embedding = client.feature_extraction(query)

# 👉 convert to numpy
doc_embeddings = np.array(doc_embeddings)
query_embedding = np.array(query_embedding)

# 🔥 similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query:", query)
print("Most relevant document:", documents[index])
print("Similarity score:", score)