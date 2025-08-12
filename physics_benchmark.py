# %%
from models import GetHuggingFaceModel
from datasets import load_dataset
# %%
# Load the data
dataset = load_dataset("UGPhysics/ugphysics", "QuantumMechanics", split="en")
# %%
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
model = GetHuggingFaceModel(model_id=MODEL_ID)
model.system_prompt = "You are a helpful and knowledgeable assistant. Answer the user's questions correctly and concisely."
# %%
