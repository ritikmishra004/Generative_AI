from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# 🔥 embedding API call
embeddings = client.feature_extraction(documents)

print(embeddings)


# from huggingface_hub import InferenceClient
# from dotenv import load_dotenv
# import os
# import numpy as np

# load_dotenv()

# client = InferenceClient(
#     model="sentence-transformers/all-mpnet-base-v2",
#     token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# documents = [
#     "Delhi is the capital of India",
#     "Kolkata is the capital of West Bengal",
#     "Paris is the capital of France"
# ]

# # 🔥 API embedding call
# embeddings = client.feature_extraction(documents)

# # 👉 convert to numpy (important)
# embeddings = np.array(embeddings)

# # ✅ total documents
# print("Number of vectors:", len(embeddings))

# # ✅ dimension of each vector
# print("Dimension of each vector:", len(embeddings[0]))

# # ✅ first 10 values of each vector (clean view)
# for i, vec in enumerate(embeddings):
#     print(f"\nVector {i+1} (first 10 values):")
#     print(vec[:10])