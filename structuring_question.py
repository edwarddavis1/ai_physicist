# %%
from datasets import load_dataset
import os
from datetime import datetime
import json
from tqdm import tqdm
import langextract as lx
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re

# %%
# Load the data
dataset_name = "TIGER-Lab/MMLU-Pro"
subject = "physics"
dataset = load_dataset(dataset_name, split="test") 
dataset = dataset.filter(lambda x: x['category'] == subject)

results_dir = "results/MMLU-Pro_question_extraction"
os.makedirs(results_dir, exist_ok=True)

print(f"Number of questions on {subject}: {len(dataset)}")

# %%
# Get a list of all unique src 
unique_src = list(set([item["src"] for item in dataset]))
print(f"There are {len(unique_src)} unique sources of questions in the dataset.")

# Filter to specific subject for analysis
subject_filter = "ori_mmlu-astronomy"
# subject_filter = "ori_mmlu-conceptual_physics" 
# subject_filter = "stemez-Mechanics"

dataset_for_subject = dataset.filter(lambda x: x['src'] == subject_filter)
print(f"Number of questions on {subject_filter}: {len(dataset_for_subject)}")

# %%
# Look at some example questions to understand structure
idx = 0
question = dataset_for_subject['question'][idx]
options = dataset_for_subject['options'][idx]
answer_index = dataset_for_subject['answer_index'][idx]

formatted_question = f"{question}\n\nOptions:\n"
for i, option in enumerate(options):
    formatted_question += f"{i}. {option}\n"

print("Example question:")
print(formatted_question)
print(f"Correct answer index: {answer_index}")
print("\n" + "="*80 + "\n")

# %%
# Define Pydantic models for structured extraction

class PhysicsVariable(BaseModel):
    """Represents a physics variable mentioned in the question"""
    symbol: str = Field(description="The mathematical symbol or variable name")
    name: str = Field(description="The full name or description of the variable")
    value: Optional[str] = Field(description="Numerical value if given, with units")
    unit: Optional[str] = Field(description="Unit of measurement if applicable")

class PhysicsFormula(BaseModel):
    """Represents a physics formula or equation relevant to the question"""
    name: str = Field(description="Name of the formula or principle")
    equation: str = Field(description="Mathematical representation of the formula")
    topic: str = Field(description="Physics topic or area (e.g., 'Mechanics', 'Thermodynamics')")

class PhysicsConcept(BaseModel):
    """Represents a physics concept mentioned in the question"""
    name: str = Field(description="Name of the physics concept")
    description: str = Field(description="Brief description of the concept")
    topic: str = Field(description="Physics topic or area")

class QuestionStructure(BaseModel):
    """Structured representation of a physics question"""
    question_type: str = Field(description="Type of question (e.g., 'calculation', 'conceptual', 'application')")
    physics_topic: str = Field(description="Main physics topic addressed")
    sub_topics: List[str] = Field(description="Specific sub-topics or areas within the main topic")
    
    # Core content
    problem_statement: str = Field(description="The main problem being asked")
    given_information: List[str] = Field(description="Information provided in the question")
    what_to_find: str = Field(description="What the question is asking to determine")
    
    # Physics-specific elements
    variables: List[PhysicsVariable] = Field(description="Variables mentioned in the question")
    relevant_formulas: List[PhysicsFormula] = Field(description="Formulas that might be relevant")
    concepts: List[PhysicsConcept] = Field(description="Physics concepts involved")
    
    # Question metadata
    difficulty_level: str = Field(description="Estimated difficulty (basic, intermediate, advanced)")
    requires_calculation: bool = Field(description="Whether the question requires numerical calculation")
    requires_diagram: bool = Field(description="Whether understanding requires visualization/diagrams")

# %%
EXTRACTION_PROMPT = """
You are an expert physics educator analyzing physics questions. Extract structured information from the given physics question text.

Focus on identifying:
1. The type and nature of the question
2. Physics topics and concepts involved
3. Variables, values, and units mentioned
4. Relevant formulas or principles
5. What information is given vs. what needs to be found

Be thorough but precise in your analysis. If information is not explicitly stated, indicate this appropriately.

Question to analyze:
{question_text}
"""

# %%
# Process a single question with LangExtract
extracted_question = None
extraction_error = None

# Select a specific question to analyze (can be changed)
idx = 1

question_id = dataset_for_subject['question_id'][idx]
question = dataset_for_subject['question'][idx]
options = dataset_for_subject['options'][idx]
answer_index = dataset_for_subject['answer_index'][idx]

# Combine question and options for full context
full_question_text = f"{question}\n\nOptions:\n"
for i, option in enumerate(options):
    full_question_text += f"{i}. {option}\n"

print("Question to be analyzed:")
print("="*60)
print(full_question_text)
print("="*60)

# Format prompt for extraction
prompt = EXTRACTION_PROMPT.format(question_text=full_question_text)

examples = [
    lx.data.ExampleData(
        text="Determine the root-mean-square (rms) values of displacement, velocity, and acceleration for a damped forced harmonic oscillator operating at steady state.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task_objective",
                extraction_text="Determine the root-mean-square (rms) values",
                attributes={"calculation_type": "Root-Mean-Square"}
            ),
            lx.data.Extraction(
                extraction_class="physical_system",
                extraction_text="a damped forced harmonic oscillator",
                attributes={
                    "name": "Damped Forced Harmonic Oscillator",
                    "key_concepts": ["damping", "external_force", "oscillation", "resonance"]
                }
            ),
            lx.data.Extraction(
                extraction_class="operating_condition",
                extraction_text="operating at steady state",
                attributes={"state": "Steady State"}
            ),
            lx.data.Extraction(
                extraction_class="quantity_to_calculate",
                extraction_text="displacement",
                attributes={"symbol": "x_rms"}
            ),
            lx.data.Extraction(
                extraction_class="quantity_to_calculate",
                extraction_text="velocity",
                attributes={"symbol": "v_rms"}
            ),
            lx.data.Extraction(
                extraction_class="quantity_to_calculate",
                extraction_text="acceleration",
                attributes={"symbol": "a_rms"}
            ),
        ]
    )
]

LANGEXTRACT_API_KEY = os.getenv("LANGEXTRACT_API_KEY")

structured_question = lx.extract(
    text_or_documents=full_question_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    api_key=LANGEXTRACT_API_KEY
)


# lx.io.save_annotated_documents([structured_question], output_name="results/MMLU-Pro_question_extraction/extraction_results.jsonl", output_dir=".")

# # Generate the visualization from the file
# html_content = lx.visualize("results/MMLU-Pro_question_extraction/extraction_results.jsonl")
# with open("visualization.html", "w") as f:
#     if hasattr(html_content, 'data'):
#         f.write(html_content.data)  # For Jupyter/Colab
#     else:
#         f.write(html_content)