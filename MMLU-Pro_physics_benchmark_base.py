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
# subject = "QuantumMechanics"
# dataset_name = "UGPhysics/ugphysics"
# dataset = load_dataset(dataset_name, subject, split="en")

dataset_name = "TIGER-Lab/MMLU-Pro"
subject = "physics"
dataset = load_dataset(dataset_name, split="test") 
dataset = dataset.filter(lambda x: x['category'] == subject)

results_dir = "results/MMLU-Pro_benchmarking"

print(f"Number of questions on {subject}: {len(dataset)}")

# %%

# Get a list of all unique src 
unique_src = list(set([item["src"] for item in dataset]))

print(f"There are {len(unique_src)} unique sources of questions in the dataset.")


subject = "ori_mmlu-astronomy"
# subject = "ori_mmlu-conceptual_physics" 


dataset_for_subject = dataset.filter(lambda x: x['src'] == subject)

print(f"Number of questions on {subject}: {len(dataset_for_subject)}")

# %%
# Look at some example questions

idx = 0
question = dataset_for_subject['question'][idx]
options = dataset_for_subject['options'][idx]
answer_index = dataset_for_subject['answer_index'][idx]

formatted_question = f"{question}\n\nOptions:\n"
for i, option in enumerate(options):
    formatted_question += f"{i}. {option}\n"

print(formatted_question)



# %%
# MODEL_ID = "Qwen/Qwen3-235B-A22B"
MODEL_ID = "openai/gpt-oss-20B"
# MODEL_ID = "Qwen/Qwen3-32B"
# MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

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
answers_by_question = []

for idx in tqdm(range(10)):

    question_id = dataset_for_subject['question_id'][idx]
    question = dataset_for_subject['question'][idx]
    options = dataset_for_subject['options'][idx]
    answer_index = dataset_for_subject['answer_index'][idx]

    formatted_question = f"{question}\n\nOptions:\n"
    for i, option in enumerate(options):
        formatted_question += f"{i}. {option}\n"


    PROMPT_TEMPLATE = """
    You are a helpful and knowledgeable assistant, specialising in Physics. Below is a multiple-choice question. Provide a chain of thought as you solve the problem, stating any relevant principles, concepts or equations. After this thinking, state the correct answer to the question based on the index of the correct option. 

    It is essential that the your final answer index is formatted as "Answer: <index>", with no additional text or punctuation, and that your total response is less than 512 characters.

    Question:
    {formatted_question}

    """

    # Use this prompt if struggling to get an answer out in 512 tokens

    # PROMPT_TEMPLATE = """
    # You are a helpful and knowledgeable assistant, specialising in Physics. Below is a multiple-choice question. Provide only the index of your selected answer - no other surrounding words or punctuation.

    # Question:
    # {formatted_question}

    # Answer:
    # """


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(formatted_question=formatted_question)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )

    thinking = response.choices[0].message.content or ""

    # NOTE: crude extraction of answer index
    ai_answer = thinking.split("Answer:")[-1].strip()

    # Validate and convert the answer index to an integer
    try:
        ai_answer = int(ai_answer)
    except ValueError:
        print("Could not extract answer index")
        continue

    correct = answer_index == ai_answer

    answers_by_question.append({
        # Question data
        "question_id": question_id,
        "question": question,
        "options": options,
        "category": dataset_for_subject['category'][idx],
        "src": dataset_for_subject['src'][idx],
        "answer_index": answer_index,

        # Response data
        "model_id": MODEL_ID,
        "thinking": thinking,
        "ai_answer": ai_answer,
        "correct": correct
    })

    print(f"Processed question {idx + 1}/{10}, AI Answer: {ai_answer}, Correct Answer: {answer_index}, Correct: {correct}")

# Save based on model and clock (to avoid overwriting)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{results_dir}/base_model/answers_by_question_{MODEL_ID.replace('/', '_')}_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(answers_by_question, f, indent=4)

# %%
# Performance Analysis

correct_answers = sum(1 for answer in answers_by_question if answer["correct"])
total_questions = len(answers_by_question)
accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0

print(f"Performance Analysis: {MODEL_ID}")
print(f"Total Questions: {total_questions}")
print(f"Correct Answers: {correct_answers}")
print(f"Accuracy: {accuracy:.2f}%")

# %%
# Average number of characters in thinking 
average_thinking_length = sum(len(answer["thinking"]) for answer in answers_by_question) / total_questions if total_questions > 0 else 0
print(f"Average Thinking Length: {average_thinking_length:.2f} characters")

# %%
