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
# subject_filter = "ori_mmlu-astronomy"
# subject_filter = "ori_mmlu-conceptual_physics" 
subject_filter = "stemez-Mechanics"

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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

structured_question = lx.extract(
    text_or_documents=full_question_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

# Combine original data with extracted structure
extracted_question = {
    # Original question data
    "question_id": question_id,
    "original_question": question,
    "options": options,
    "correct_answer_index": answer_index,
    "category": dataset_for_subject['category'][idx],
    "src": dataset_for_subject['src'][idx],
    
    # Extracted structured data
    "structured_analysis": structured_question.dict(),
    
    # Metadata
    "extraction_timestamp": datetime.now().isoformat(),
    "full_question_text": full_question_text
}

print("Successfully extracted structured information!")
    
# except Exception as e:
#     extraction_error = {
#         "question_id": dataset_for_subject['question_id'][idx],
#         "question_index": idx,
#         "error": str(e),
#         "question_preview": question[:100] + "..." if len(question) > 100 else question
#     }
#     print(f"Failed to extract structure: {e}")
#     extracted_question = None

# %%
# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{results_dir}/structured_question_{subject_filter.replace('/', '_')}_{timestamp}.json"
error_file = f"{results_dir}/extraction_error_{subject_filter.replace('/', '_')}_{timestamp}.json"

# Save successful extraction or error
if extracted_question:
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(extracted_question, f, indent=4, ensure_ascii=False)
    print(f"Extraction complete! Results saved to: {output_file}")
elif extraction_error:
    with open(error_file, "w", encoding='utf-8') as f:
        json.dump(extraction_error, f, indent=4, ensure_ascii=False)
    print(f"Extraction failed. Error details saved to: {error_file}")
else:
    print("No extraction attempted.")

# %%
# Analysis and Statistics for Single Question

def analyze_single_question(extracted_question):
    """Analyze the extracted structured data for a single question"""
    
    if not extracted_question:
        print("No successfully extracted question to analyze.")
        return
    
    print("="*80)
    print("SINGLE QUESTION EXTRACTION ANALYSIS")
    print("="*80)
    
    analysis = extracted_question["structured_analysis"]
    
    print(f"\nQuestion Details:")
    print(f"  Question ID: {extracted_question['question_id']}")
    print(f"  Category: {extracted_question['category']}")
    print(f"  Source: {extracted_question['src']}")
    print(f"  Correct Answer Index: {extracted_question['correct_answer_index']}")
    
    print(f"\nStructured Analysis:")
    print(f"  Question Type: {analysis['question_type']}")
    print(f"  Physics Topic: {analysis['physics_topic']}")
    print(f"  Sub-topics: {', '.join(analysis['sub_topics']) if analysis['sub_topics'] else 'None identified'}")
    print(f"  Difficulty Level: {analysis['difficulty_level']}")
    print(f"  Requires Calculation: {analysis['requires_calculation']}")
    print(f"  Requires Diagram: {analysis['requires_diagram']}")
    
    print(f"\nProblem Structure:")
    print(f"  Problem Statement: {analysis['problem_statement']}")
    print(f"  What to Find: {analysis['what_to_find']}")
    
    if analysis['given_information']:
        print(f"\nGiven Information:")
        for i, info in enumerate(analysis['given_information'], 1):
            print(f"    {i}. {info}")
    
    if analysis['variables']:
        print(f"\nVariables Identified ({len(analysis['variables'])}):")
        for var in analysis['variables']:
            unit_str = f" ({var['unit']})" if var.get('unit') else ""
            value_str = f" = {var['value']}" if var.get('value') else ""
            print(f"    • {var['symbol']}: {var['name']}{value_str}{unit_str}")
    
    if analysis['relevant_formulas']:
        print(f"\nRelevant Formulas ({len(analysis['relevant_formulas'])}):")
        for formula in analysis['relevant_formulas']:
            print(f"    • {formula['name']}: {formula['equation']} (Topic: {formula['topic']})")
    
    if analysis['concepts']:
        print(f"\nPhysics Concepts ({len(analysis['concepts'])}):")
        for concept in analysis['concepts']:
            print(f"    • {concept['name']}: {concept['description']} (Topic: {concept['topic']})")
    
    print("="*80)

# Run analysis on the extracted question
if extracted_question:
    analyze_single_question(extracted_question)
else:
    print("No question was successfully extracted for analysis.")

# %%
# Display detailed example of extracted structure
if extracted_question:
    print("\n" + "="*80)
    print("DETAILED EXTRACTION EXAMPLE")
    print("="*80)
    
    print(f"Original Question:")
    print(f"{extracted_question['original_question']}")
    
    print(f"\nOptions:")
    for i, option in enumerate(extracted_question['options']):
        marker = "✓" if i == extracted_question['correct_answer_index'] else " "
        print(f"  {i}. {option} {marker}")
    
    analysis = extracted_question['structured_analysis']
    
    print(f"\nExtracted Structure:")
    print(f"  Type: {analysis['question_type']}")
    print(f"  Topic: {analysis['physics_topic']}")
    print(f"  Difficulty: {analysis['difficulty_level']}")
    print(f"  Calculation Required: {analysis['requires_calculation']}")
    
    print(f"\n  Problem Statement: {analysis['problem_statement']}")
    print(f"  What to Find: {analysis['what_to_find']}")
    
    if analysis['variables']:
        print(f"\n  Variables ({len(analysis['variables'])}):")
        for var in analysis['variables']:
            print(f"    - {var['symbol']}: {var['name']}")
    
    if analysis['concepts']:
        print(f"\n  Key Concepts ({len(analysis['concepts'])}):")
        for concept in analysis['concepts']:
            print(f"    - {concept['name']}")
    
    print("="*80)

# %%
# Summary
print(f"\nExtraction Summary:")
if extracted_question:
    print("✓ Successfully extracted structured information from the question")
    print(f"  - Saved to: {output_file}")
    print(f"  - Extraction timestamp: {extracted_question['extraction_timestamp']}")
elif extraction_error:
    print("✗ Failed to extract structured information")
    print(f"  - Error details saved to: {error_file}")
    print(f"  - Error: {extraction_error['error']}")
else:
    print("? No extraction was attempted")

print(f"\nThis script demonstrates how LangExtract can be used to convert")
print(f"free-text physics questions into structured, analyzable data.")

# %%
