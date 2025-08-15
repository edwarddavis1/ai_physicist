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
dataset_name = "xw27/scibench"
dataset = load_dataset(dataset_name, split="train") 


subject_sources = list(set([item["source"] for item in dataset]))

subject = "quan"
dataset_for_subject = dataset.filter(lambda x: x['source'] == subject)

print(f"There are {len(dataset_for_subject)} examples for subject: {subject}")
# %%
