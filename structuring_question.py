# %%
from datasets import load_dataset
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import os
from langchain_core.prompts import ChatPromptTemplate

# %%
# Load the data
# subject = "QuantumMechanics"
# dataset_name = "UGPhysics/ugphysics"
# dataset = load_dataset(dataset_name, subject, split="en")

dataset_name = "TIGER-Lab/MMLU-Pro"
subject = "physics"
dataset = load_dataset(dataset_name, split="test") 
dataset = dataset.filter(lambda x: x['category'] == subject)


print(f"Number of questions on {subject}: {len(dataset)}")

# %%

# Get a list of all unique src 
unique_src = list(set([item["src"] for item in dataset]))

print(f"There are {len(unique_src)} unique sources of questions in the dataset.")


subject = unique_src[-1] # More mathsy.


dataset_for_subject = dataset.filter(lambda x: x['src'] == subject)

print(f"Number of questions on {subject}: {len(dataset_for_subject)}")

# %%
# Look at some example questions

idx = 1
question = dataset_for_subject['question'][idx]
options = dataset_for_subject['options'][idx]
answer_index = dataset_for_subject['answer_index'][idx]

formatted_question = f"{question}\n\nOptions:\n"
for i, option in enumerate(options):
    formatted_question += f"{i}. {option}\n"

print(formatted_question)


