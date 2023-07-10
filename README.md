# LLM4Academic

LLM4Academic is a repository for everything I want to know about large language models (LLMs). There are two parts in this repository: (1) Theory: reading list, survey, curated sources; (2) Practice: insightful experiment (demo, framework, etc.) implemented by myself.

> *"**In theory, theory and practice are the same. In practice, they are not**."*

**Table of Contents**

- [Theory](#theory)

  - [Reading List](#reading-list)
  - [Dataset Collections](#dataset-collections)
  - [Open Source LLM](#open-source-llms)
  - [Evaluation Benchmarks](#evaluation-benchmarks)
- [Practice](#Practice)

  - [API](#api)
  - [Instruction Tuning](#instruction-tuning)

## Theory

### Reading List

Reading list and related notes for LLM, see [Reading List](Theory/ReadingList.md) for details.

> - Instruction Tuning
> - In Context Learning
> - Chain of Thought
> - Reference (survey, lists, and etc.)

### Courses

- **Johns Hopkins University**: CS 601.x71 NLP: Self-supervised Models [Spring 2023](https://self-supervised.cs.jhu.edu/sp2023/)/[Fall 2022](https://self-supervised.cs.jhu.edu/fa2022/)
- **Stanford University**: CS25: Transformers United V2 [Fall 2021](https://web.stanford.edu/class/cs25/prev_years/2021_fall/)/[Winter 2023](https://web.stanford.edu/class/cs25/)
- **Stanford University**: CS 324 - Advances in Foundation Models [Winter 2022](https://stanford-cs324.github.io/winter2022/)/[Winter 2023](https://stanford-cs324.github.io/winter2023/)
- **Princeton University**: COS 597G: Understanding Large Language Models [Fall 2022](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)

### Dataset Collections

Datasets for Pretrain/Finetune/Instruction-tune LLMs, see [Datasets](Theory/Dataset.md) for details.

> - Pretraining Corpora
> - Instruction

### Open Source LLMs

Collection of various open-source LLMs, see [Open Souce LLMs](Theory/OpenSourceLLM.md) for details.

> - Pretrained Model
> - Multitask Supervised Finetuned Model
> - Instruction Finetuned Model
>   - English
>   - Chinese
>   - Multilingual
> - Human Feedback Finetuned Model
> - Domain Finetuned Model
> - Open Source Projects
>   - reproduce/framework
>   - accelerate
>   - evaluation
>   - deployment/demo
> - Reference

### Evaluation Benchmarks

Collection of automatic evaluation benchmarks, see [Evaluation Benchmarks](Theory/EvaluationBenchmarks.md) for details.

> - English
>   - Comprehensive
>   - Knowledge
>   - Reason
>     - Hard Mathematical, Theorem
>   - Code
>   - Personalization
> - Chinese
>   - Comprehensive
>   - Safety
> - Multilingual

## Practice

### API

LLM API demos, see [API](Practice/API/README.md) for details.

> - Claude
> - ChatGPT

### Instruction Tuning

1. Instruction Construct: Construct Instruction by mixture or self-instruct
2. Fine Tuning: Instruction Tuning on 4 LLM with multilingual instructions

see [Instruction Tuning](Practice/Instruction_Tuning/READEME.md) for details.

> - Experiments
>   - Datasets
>     - Collection
>     - Bootstrap
>   - Model Cards
>   - Usage
> - Results
