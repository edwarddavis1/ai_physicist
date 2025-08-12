# %%
from models import GetHuggingFaceModel
from datasets import load_dataset
# %%
# Load the data
subject = "QuantumMechanics"
dataset = load_dataset("UGPhysics/ugphysics", subject, split="en")
print(f"Number of questions on {subject}: {len(dataset)}")
# %%
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
model = GetHuggingFaceModel(model_id=MODEL_ID)
model.system_prompt = "You are a helpful and knowledgeable assistant. Answer the user's questions correctly and concisely."
# %%
# This allows me to read each Q&A a little easier

from latex_renderer import LaTeXRenderer

renderer = LaTeXRenderer()

idx = 0
question = dataset['problem'][idx]
answer = dataset['solution'][idx]

_ = renderer.display_problem(question, answer, idx)
# %%
# Model response

response = model.ask_question(question, max_length=2048)

# Display the model response with formatting
_ = renderer.display_model_response(response, MODEL_ID)

# Or display everything together (question, correct answer, and model response)
# _ = renderer.display_problem_with_model_response(question, answer, response, idx, MODEL_ID)