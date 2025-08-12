# %%
from models import GetHuggingFaceModel
# %%
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
model = GetHuggingFaceModel(model_id=MODEL_ID)
model.system_prompt = "You are a helpful and knowledgeable assistant. Answer the user's questions correctly and concisely."
# %%
question = "What is the capital of France?"
response = model.ask_question(question)
print(f"Question: {question}")
print(f"response: {response}")
# %%



