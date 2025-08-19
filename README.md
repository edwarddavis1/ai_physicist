# A Physics-specialist AI Model

For full details of thought process see the [log](progress_log/log.md)

## Final Runs

Running the gpt-oss-20B model on 80 astronomy questions from the MMLU-Pro benchmark.

| Model              | Method    | Total Questions | Correct Answers | Accuracy | Avg Thinking Length (chars) |
| ------------------ | --------- | --------------- | --------------- | -------- | --------------------------- |
| openai/gpt-oss-20B | Base      | 63              | 53              | 84.13%   | 222.02                      |
| openai/gpt-oss-20B | CoT       | 75              | 59              | 78.67%   | 244.12                      |
| openai/gpt-oss-20B | RAG       | 80              | 65              | 81.25%   | 117.50                      |
| openai/gpt-oss-20B | RAG + CoT | 79              | 63              | 79.75%   | 123.13                      |

_CoT increased the number of questions that could be answered, and RAG allowed all questions to be answered._

_Having both CoT and RAG seemed to be worse than RAG alone, but I'd take this with a pinch of salt because I have not tuned the RAG parameters due to time constraints._

