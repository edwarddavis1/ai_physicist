# %%
from datasets import load_dataset
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import os
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json
from tqdm import tqdm

# %%
# Load the data
subject = "QuantumMechanics"
dataset_name = "UGPhysics/ugphysics"
dataset = load_dataset(dataset_name, subject, split="en")

results_dir = "results/UGPhysics_benchmarking"

print(f"Number of questions on {subject}: {len(dataset)}")

# %%

# Get a list of all unique src 
question_type = list(set([item["level"] for item in dataset]))

print(f"There are {len(question_type)} unique types of questions in the dataset.")


question_type = "Laws Application"

dataset_for_question_type = dataset.filter(lambda x: x['level'] == question_type)

print(f"Number of questions on {question_type}: {len(dataset_for_question_type)}")

# %%
# Look at some example questions

idx = 0
problem = dataset_for_question_type['problem'][idx]
solution = dataset_for_question_type['solution'][idx]

print(problem)
print(solution)



# %%
# MODEL_ID = "Qwen/Qwen3-235B-A22B"
MODEL_ID = "openai/gpt-oss-20B"
# MODEL_ID = "Qwen/Qwen3-32B"
# MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Minimal HuggingFace model setup (extracted from GetHuggingFaceModel class)
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
if api_token is None:
    print("Warning: No HuggingFace API token provided. Some models may not work.")

print(f"Initializing HuggingFace Inference client for model: {MODEL_ID}")

# Initialize the inference client
client = InferenceClient(model=MODEL_ID, token=api_token)

# Initialize tokenizer for message formatting
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    has_tokenizer = True
    print(f"Tokenizer loaded for message formatting")
except Exception as e:
    print(f"Could not load tokenizer for {MODEL_ID}: {e}")
    print("Will use simple string formatting for messages")
    has_tokenizer = False

# %%
# # Compute a rough estimate of the average number of input tokens per question and the total number of tokens across all questions 

# total_tokens = 0
# for item in dataset:
#     question = item['question']
#     options = item['options']
#     formatted_question = f"{question}\n\nOptions:\n"
#     for i, option in enumerate(options):
#         formatted_question += f"{i}. {option}\n"
#     total_tokens += len(tokenizer(formatted_question)["input_ids"])

# average_tokens = total_tokens / len(dataset) if dataset else 0
# print(f"Average number of input tokens per question: {average_tokens}")
# print(f"Total number of tokens across all questions: {total_tokens}")


# %%
idx = 0
problem = dataset_for_question_type['problem'][idx]
solution = dataset_for_question_type['solution'][idx]

PROMPT_TEMPLATE = """Please reason the following problem step by step, and put your final answer within \\boxed{{}}.

Problem:
{problem}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(problem=problem)

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=512,
)

thinking = response.choices[0].message.content or ""
print(thinking)

print(f"\nModel answer:\n\n{solution}")
# %%