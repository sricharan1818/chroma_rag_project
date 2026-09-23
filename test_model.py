from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully.")

inputs = tokenizer(
    "What is the capital of India?",
    return_tensors="pt"
)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Answer:", answer)