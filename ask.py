from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# 1. Load embedding model
# -----------------------------
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# -----------------------------
# 2. Connect to ChromaDB
# -----------------------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function
)


# -----------------------------
# 3. Load local LLM
# -----------------------------
model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# -----------------------------
# 4. Ask the user
# -----------------------------
question = input("Ask a question: ")


# -----------------------------
# 5. Retrieve relevant chunks
# -----------------------------
results = vectorstore.similarity_search(
    question,
    k=3
)
print("\nRetrieved Context:\n")

for i, result in enumerate(results, 1):
    print(f"--- Chunk {i} ---")
    print(result.page_content)
    print()


# -----------------------------
# 6. Combine retrieved context
# -----------------------------
context = "\n\n".join(
    result.page_content for result in results
)


# -----------------------------
# 7. Create prompt
# -----------------------------
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# 8. Generate answer
# -----------------------------
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True
)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100
)


# -----------------------------
# 9. Display final answer
# -----------------------------
answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nAnswer:")
print(answer)