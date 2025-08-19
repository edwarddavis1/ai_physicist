## Problem Statement

**Central Language Model for AI Physicist**

**Background:** At FirstPrinciples we are developing an AI Physicist, an automated AI system that will assist in the day-to-day tasks of a human physicist to tackle any or all aspects of the scientific process, including generating a properly formulated research hypothesis and executing the testing of a hypothesis. An AI Physicist can be thought of as a "brain with hands", where the brain is a central language model specialized in physics, or certain subdomains, and the hands are external tools able to perform tasks that the language model would not be good at, e.g. mathematical reasoning. The specialized AI Physicist should demonstrate increased performance in said subdomains or tasks compared to generic models and be able to discover novel insights about physics.

**Task:** For this assignment, we would like to focus on the central language model. How would you go about specializing a model in the domain of physics? Feel free to go into any direction you see fit, for example post-training, fine tuning, reinforcement learning, tool usage, etc.

Outline your approach and submit your code in the format of your choice, e.g. Jupyter Notebook, GitHub repo, Python scripts. Use whichever dataset, model, training procedure, or benchmarking you see fit, and feel free to include comments, documentation, data, visualization as necessary. We acknowledge that this is an open ended question that cannot be solved in a week, we are interested in understanding your thought process but not in a production level tool. There are no right or wrong answers.

# Exploration

## Relevant Literature

There were a few papers I remember from ICLR that came from Jure Leskovec's group:

-   [Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems](https://arxiv.org/pdf/2307.08423)
-   [Biomni: A General-Purpose Biomedical AI Agent](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1.full.pdf)

-   From the above I remember the team talking about how they approach the problem of hypothesis generation - which could be of significant use here.
-   Beyond hypothesis generation, the mechanism by which they specialise the knowledge of science should be reviewed here.

Otherwise there are a fair few other highly-notable papers in this space:

-   [The ai scientist: Towards fully automated open-ended scientific discovery](https://arxiv.org/pdf/2408.06292?) (2024, 372 citations)
-   [The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search](https://arxiv.org/pdf/2504.08066?) (2025, 32 citations) "the first instance of a fully AI-generated paper successfully navigating a peer review (accepted at ICLR workshop)"
-   [Towards an AI co-scientist](https://arxiv.org/pdf/2502.18864) (2025, 90 citations) Google Deepmind et. al.

## Key Questions / Deliverables

-   Take a pre-trained LLM and improve it's ability to answer physics questions (by any means).
-   What is the most effective way to evaluate its physics performance?

## Takeaways from Literature Review

[Full Literature Review](../literature_review.md)

## Relevant Benchmarks and Evalutations

-   [UGPhysics](https://huggingface.co/datasets/UGPhysics/ugphysics)

## High-level Plan / Project Scope

### Early Project Aim - Pipeline to Improve Inherent Performance

Fundamentally we want to take a pre-trained model and improve it's performance on physics-related questions - e.g. those posed in the _UGPhysics_ dataset.

Therefore, seen as I'm very compute-limited, I should focus on very small and open-source models. If I can come up with a pipeline that will improve physics performance for small models, perhaps this pipeline would also improve physics performance for larger base models.

### Later Aim - Enhance with tools + RAG

Once we have a model that has more knowledge, I should turn my attention to grounding it's knowledge base with RAG, and explore useful MCP servers to aid it's ability to answer the questions.

## Notable Small Base Models

-   gpt-oss-20b (activates 3.6B params at inference time)
-   Qwen/Qwen2.5-0.5B-Instruct
-   meta-llama/Llama-3.2-1B-Instruct
-   google/gemma-3-4b-it
-   google/gemma-3-270m-it (just came out!)

Models around this scale are possible of running locally on my CPU without slowing me down too much. E.g. a simple question to the 0.5B param Qwen model takes just over a second to get a response.

## Is the UGPhysics dataset a good benchmark?

While this is a decently large dataset containing questions and answers, these questions involve mathematical reasoning, which language models (especially small ones) are no good at. In these situations the ideal behaviour would be to call some tool to help reason the question.

## A simpler benchmark - MMLU-Pro Physics Questions

These appear to be more like high school/sixth-form-level questions.

-   Average num tokens per q: 171
-   Total num tokens: 200k

## High Level Plan: Structure the question, RAG, math tooling.

First, get some idea of how well base models do on this dataset.

Let us initially assume that most of these (now high-school-level) questions can be solved by first selecting the correct formula, based on the variables, then plugging the values into the formula to compute the answer. Then consider the following pipeline:

1. Use LangExtract to convert the free text question into a JSON containing the known and unknown physics variables by name, value and unit.
2. Based on the information from the structured question, select a formula from a formula sheet (or knowledge base) which contains the physics variables. I.e. have an embedding of each equation based on the physics variables it contains, this will then be found based on a query including those same variables.
3. Once the RAG pipeline identifies an equation, for calculation, use an external tool (Wolfram Alpha API?)

### Required formatting for the formula sheet

-   Name of all variables included (excluding constants)
-   SI units
-   Standard symbol for each variable ({"Energy":"E", "Potential Difference":"V"})
-   Latex formatting of equation (so it can write it in the working)
-   Equation name (for reference)
-   Physics topic

_Example: Newton's Second Law_

```
{
    "formula_name": "Newton's Second Law",
    "equation_latex": "F = ma",
    "topic": "Mechanics",
    "variables": [
      {
        "symbol": "F",
        "name": "Force",
        "unit": "Newtons (N)"
      },
      {
        "symbol": "m",
        "name": "Mass",
        "unit": "Kilograms (kg)"
      },
      {
        "symbol": "a",
        "name": "Acceleration",
        "unit": "m/s^2"
      }
    ]
}
```

## Initial Base Model Performance on MMLU-Pro Physics Questions

_Note that if an answer could not be extracted from any of the 10 questions, those question were omitted from testing_

| Dataset                     | Model              | Total Questions | Correct Answers | Accuracy |
| --------------------------- | ------------------ | --------------- | --------------- | -------- |
| ori_mmlu-astronomy          | openai/gpt-oss-20B | 9               | 5               | 55.56%   |
| stemez-Mechanics            | openai/gpt-oss-20B | 6               | 6               | 100.00%  |
| stemez-Optics               | openai/gpt-oss-20B | 3               | 2               | 66.67%   |
| stemez-Physics              | openai/gpt-oss-20B | 8               | 7               | 87.50%   |
| scibench-thermo             | openai/gpt-oss-20B | 6               | 6               | 100.00%  |
| ori_mmlu-college_physics    | openai/gpt-oss-20B | 5               | 4               | 80.00%   |
| ori_mmlu-conceptual_physics | openai/gpt-oss-20B | 10              | 9               | 90.00%   |

### Initial Results Discussion: gpt-oss-20B

Contrary to what I would have thought, it performed better at the more mathsy topics (when formatting allowed to extract an answer) in comparison to the astronomy. I expected the mathsy topics to be worse as LLMs are famously bad at arithmetic computation.

Note that it might well be my stringent token limit of 512 which is the reason why I can't get an answer from the model. So currently I'm not going to focus on why extraction seems to fail fairly often.

Overall I'm surprised at how well the gpt-oss-20B model did. **Almost suspicious**. Could it be the case that gpt-oss was just benchmaxxed to this dataset?

### Other models

-   "Qwen/Qwen3-4B-Thinking-2507": Takes a crazy amount of output tokens to get to an answer - even when explicitly told to return only an answer index. E.g. it won't provide an answer even with max_tokens set to 2040.

#### "Qwen/Qwen3-32B":

| Dataset                     | Model          | Total Questions | Correct Answers | Accuracy |
| --------------------------- | -------------- | --------------- | --------------- | -------- |
| ori_mmlu-conceptual_physics | Qwen/Qwen3-32B | 9               | 8               | 88.89%   |
| ori_mmlu-astronomy          | Qwen/Qwen3-32B | 8               | 7               | 87.50%   |

Qwen does even better than gpt-oss - maybe they're all better at Physics than I first thought...

Is the benchmark too easy?

I could go back to the UGPhysics benchmark, but the answers are very unstructured in those datasets, making evaluation difficult. Due to time constraints, it would be much easier working with this dataset.

# Attempts at Increasing Answer Quality

Going forward I'm going to focus on the astronomy questions from the MMLU-Pro benchmark, as (in the case of oss-20B) this represents the largest margin for improvement (~55% base performance on the first 10).

## Zero-shot Chain-of-thought (astronomy questions)

Let's start with a really basic change, add "Let's think step by step" to the questioning prompt.

| Model       | Method | Accuracy | Average Thinking Length (chars) | Effect                  |
| ----------- | ------ | -------- | ------------------------------- | ----------------------- |
| gpt-oss-20b | Base   | 55.56%   | 229.56                          | -                       |
| gpt-oss-20b | ZS CoT | 55.56%   | 289.44                          | None, more output chars |
| Qwen3-32B   | Base   | 80.00%   | 11.00                           | -                       |
| Qwen3-32B   | ZS CoT | 100.00%  | 67.14                           | Positive effect         |

_How the addition of a zero-shot chain-of-thought prompt changes the output quality of base models_

So, depending on the model, this represents a simple way of improving reasoning capabilities (not specific to Physics).

## Adding a knowledge base: RAG from Textbook

[This](https://openstax.org/details/books/astronomy-2e) is an open-source astronomy textbook. Let's allow the models to retrieve information from this textbook and see if it improves answer quality.

1. Chunk the PDF and embed each chunk.
2. Perform a semantic search based on the question and multiple choice options (as a normal string input).
3. The semantically similar chunks are then included in the system prompt, as well as the question and options.

_Result_

| Dataset            | Model              | Method | Total Questions | Correct Answers | Accuracy | Avg Thinking Length (chars) |
| ------------------ | ------------------ | ------ | --------------- | --------------- | -------- | --------------------------- |
| ori_mmlu-astronomy | openai/gpt-oss-20B | Base   | 9               | 5               | 55.56%   | 229.56                      |
| ori_mmlu-astronomy | openai/gpt-oss-20B | RAG    | 10              | 8               | 80.00%   | 119.00                      |
| ori_mmlu-astronomy | Qwen/Qwen3-32B     | Base   | 8               | 7               | 87.50%   | -                           |
| ori_mmlu-astronomy | Qwen/Qwen3-32B     | RAG    | 10              | 8               | 80.00%   | 175.20                      |

Clearly much better responses from oss-20B. While the accuracy has dropped for Qwen, this is due to more questions being processed. So adding the RAG step has actually helped with answer formatting (or it's reduced the output characters to within the limit I imposed). In either case, RAG is beneficial.

### RAG improvements: LangExtract?
