# [2025-8-12 Tue]

## Problem Statement

**Central Language Model for AI Physicist**

**Background:** At FirstPrinciples we are developing an AI Physicist, an automated AI system that will assist in the day-to-day tasks of a human physicist to tackle any or all aspects of the scientific process, including generating a properly formulated research hypothesis and executing the testing of a hypothesis. An AI Physicist can be thought of as a "brain with hands", where the brain is a central language model specialized in physics, or certain subdomains, and the hands are external tools able to perform tasks that the language model would not be good at, e.g. mathematical reasoning. The specialized AI Physicist should demonstrate increased performance in said subdomains or tasks compared to generic models and be able to discover novel insights about physics.

**Task:** For this assignment, we would like to focus on the central language model. How would you go about specializing a model in the domain of physics? Feel free to go into any direction you see fit, for example post-training, fine tuning, reinforcement learning, tool usage, etc.

Outline your approach and submit your code in the format of your choice, e.g. Jupyter Notebook, GitHub repo, Python scripts. Use whichever dataset, model, training procedure, or benchmarking you see fit, and feel free to include comments, documentation, data, visualization as necessary. We acknowledge that this is an open ended question that cannot be solved in a week, we are interested in understanding your thought process but not in a production level tool. There are no right or wrong answers.

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

# [2025-8-15 Fri]

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

## Initial Performance on MMLU-Pro Physics Questions

Based on the first ten mechanics questions:

Performance Analysis: openai/gpt-oss-20B
Total Questions: 9
Correct Answers: 5
Accuracy: 55.56%
