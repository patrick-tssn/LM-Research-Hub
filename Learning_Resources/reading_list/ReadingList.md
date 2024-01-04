# LLM Reading

Table of Contents

- [Key Findings](#key-findings)

*arranged in alphabetical order*

- [Code Learning](#code-learning)
- [Efficiency](#efficiency)
- [Human Alignment](#human-alignment)
- [Instruction Tuning](#instruction-tuning)
- [Interpretability](#interpretability)
- [In Context Learning](#in-context-learning)
- [Mixture of Experts (MoE)](#mixture-of-experts-moe)
- [Reasoning](#reasoning)
  - [Abstract Reasoning](#abstract-reasoning)
  - [Chain of Thought](#chain-of-thought)
  - [Symbolic Reasoning](#symbolic-reasoning)

## Key Findings

| Title                                                                 | Pub          | Preprint                                                                                                                                                                                                                                                                                               | Supplementary                      |
| --------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| Are Emergent Abilities of Large Language Models a Mirage?             | Neurips 2023 | [2304.15004](https://arxiv.org/abs/2304.15004) <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F29c7f009df21d0112c48dec254ff80cc45fac3af%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)            | mirage emergent ability, Stanford  |
| Self-Instruct: Aligning LM with Self Generated Instructions           | ACL 2023     | [2212.10560](https://arxiv.org/abs/2212.10560)        <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe65b346d442e9962a4276dc1c1af2956d9d5f1eb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)   | self-instruct, UW                  |
| Emergent Abilities of Large Language Models                           | TMLR 2022    | [2206.07682](https://arxiv.org/abs/2206.07682)  <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdac3a172b504f4e33c029655e9befb3386e5f63a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)         | emergent ability,  Google         |
| Training language models to follow instructions with human feedback   |              | [2203.02155](https://arxiv.org/abs/2203.02155)  <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd766bffc357127e0dc86dd69561d5aeb520d6f4c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)           | InstructGPT, OpenAI                |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models |              | [2201.11903](https://arxiv.org/abs/2201.11903)   <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1b6e810ce0afd0dd093f789d2b2742d047e316d5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | CoT, Google                        |
| Finetuned Language Models Are Zero-Shot Learners                      | ICLR 2022    | [2109.01652](https://arxiv.org/abs/2109.01652)   <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fff0b2681d7b05e16c46dfb71d980cc2f605907cd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | Flan, instruction finetune, Google |
| Language Models are Few-Shot Learners                                 |              | [2005.14165](https://arxiv.org/abs/2005.14165)    <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6b85b63579a916f705a8e10a49bd8d849d91b1fc%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)         | GPT3, OpenAI                       |
| Scaling Laws for Neural Language Models                               |              | [2001.08361](https://arxiv.org/abs/2001.08361) <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe6c561d02500b2596a230b341a8eb8b921ca5bf2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | scaling law, OpenAI                |
| Attention Is All You Need                               |              | [1706.03762](https://arxiv.org/abs/1706.03762) <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | Transformer, Google                |

## Code Learning

Papers about code generation, see [Code](Code.md) for details.

> Papers

Related Collections

- [Awesome-Code-LLM](https://github.com/huybery/Awesome-Code-LLM) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/huybery/Awesome-Code-LLM?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/huybery/Awesome-Code-LLM.svg?style=social&label=Star), An awesome and curated list of best code-LLM for research.

## Efficiency

Papers about Efficiency/Compression/Extrapolation of neural network/language model, see [Efficiency](Efficiency.md) for details.

> - Papers
>   - Adapt
>   - Architecture
>   - Pruning
>   - Quantization
> - Blogs
> - Benchmarks
> - Projects

Related Collections (efficiency)

- [Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/horseee/Awesome-Efficient-LLM?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/horseee/Awesome-Efficient-LLM.svg?style=social&label=Star), A curated list for Efficient Large Language Models
- [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/DefTruth/Awesome-LLM-Inference?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social&label=Star), A small Collection for Awesome LLM Inference [Papers|Blogs|Docs] with codes, contains TensorRT-LLM, streaming-llm, SmoothQuant, WINT8/4, Continuous Batching, FlashAttention, PagedAttention etc.
- [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/HuangOwen/Awesome-LLM-Compression?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/HuangOwen/Awesome-LLM-Compression.svg?style=social&label=Star), Awesome LLM compression research papers and tools to accelerate the LLM training and inference

Related Collections (extensibility)
- [Awesome-Long-Context](https://github.com/showlab/Awesome-Long-Context) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/showlab/Awesome-Long-Context?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/showlab/Awesome-Long-Context.svg?style=social&label=Star), A curated list of resources about long-context in large-language models and video understanding.

- [Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling), ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Xnhyacinth/Awesome-LLM-Long-Context-Modeling?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/Xnhyacinth/Awesome-LLM-Long-Context-Modeling.svg?style=social&label=Star) This repo includes papers and blogs about Efficient Transformers, Length Extrapolation, Long Term Memory, Retrieval Augmented Generation(RAG), and Evaluation for Long Context Modeling.

## Human Alignment

Papers about human alignment, see [Human Alignment](Humanalignment.md) for details

> Papers

Related Collections

- [AlignLLMHumanSurvey](https://github.com/GaryYufei/AlignLLMHumanSurvey) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/GaryYufei/AlignLLMHumanSurvey?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/GaryYufei/AlignLLMHumanSurvey.svg?style=social&label=Star), A collection of papers and resources about aligning large language models (LLMs) with human.

- [awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/EdinburghNLP/awesome-hallucination-detection?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/EdinburghNLP/awesome-hallucination-detection.svg?style=social&label=Star), List of papers on hallucination detection in LLMs.

## Instruction Tuning

Papers/blogs about Instruction Tuning, see [Instruction Tuning](InstructionTuning.md) for details.

> - Papers
> - Blogs

Related Collections

- [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/SinclairCoder/Instruction-Tuning-Papers?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/SinclairCoder/Instruction-Tuning-Papers.svg?style=social&label=Star), A trend starts from Natrural-Instruction (ACL 2022), FLAN (ICLR 2022) and T0 (ICLR 2022).

## Interpretability

Papers about interpretability of language model, see [Interpretability](Interpretability.md) for details

Related Collections

- [awesome-llm-interpretability](https://github.com/JShollaj/awesome-llm-interpretability) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/JShollaj/awesome-llm-interpretability?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/JShollaj/awesome-llm-interpretability.svg?style=social&label=Star), A curated list of amazingly awesome tools, papers, articles, and communities focused on Large Language Model (LLM) Interpretability.

## In Context Learning

Papers about In Context Learning, see [In Context Learning](ICL.md) for details.

> - Papers

Related Collections

- [ICL_PaperList](https://github.com/dqxiu/ICL_PaperList) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/dqxiu/ICL_PaperList?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/dqxiu/ICL_PaperList.svg?style=social&label=Star), Paper List for In-context Learning
- [LMaaS-Papers](https://github.com/txsun1997/LMaaS-Papers) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/txsun1997/LMaaS-Papers?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/txsun1997/LMaaS-Papers.svg?style=social&label=Star), Awesome papers on Language-Model-as-a-Service (LMaaS)

## Mixture of Experts (MoE)

Papers about Mixture of Experts (MoE), see [Mixture of Experts](MoE.md) for details.

> - Papers
> - Projects
> - Blogs

- [Awesome-Mixture-of-Experts-Papers](https://github.com/codecaution/Awesome-Mixture-of-Experts-Papers) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/codecaution/Awesome-Mixture-of-Experts-Papers?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/codecaution/Awesome-Mixture-of-Experts-Papers.svg?style=social&label=Star), A curated reading list of research in Mixture-of-Experts(MoE).

## Reasoning

### Chain of Thought

Papers about Chain of Thought, see [Chain of Thought](reasoning/CoT.md) for details.

> - Papers

- [Chain-of-ThoughtsPapers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Timothyxxx/Chain-of-ThoughtsPapers?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/Timothyxxx/Chain-of-ThoughtsPapers.svg?style=social&label=Star), A trend starts from "Chain of Thought Prompting Elicits Reasoning in Large Language Models"
- [Reasoning in Large Language Models](https://github.com/atfortes/LLM-Reasoning-Papers) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/atfortes/LLM-Reasoning-Papers?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/atfortes/LLM-Reasoning-Papers.svg?style=social&label=Star), Collection of papers and resources on how to unlock the reasoning ability of Large Language Models.

### Abstract Reasoning

Short survey about Abstract Reasoning, see [Abstract Reasoning](reasoning/AR.md) for details.

> - Datasets
> - Learning Methods
> - Models
> - Representative works

### Symbolic Reasoning

Papers about symbolic reasoning, see [Symbolic](reasoning/Symbolic.md) for details

> - Papers

Related Collections

- [Awesome-Reasoning-Foundation-Models](https://github.com/reasoning-survey/Awesome-Reasoning-Foundation-Models) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/reasoning-survey/Awesome-Reasoning-Foundation-Models?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/reasoning-survey/Awesome-Reasoning-Foundation-Models.svg?style=social&label=Star), A curated list of awesome large AI models, or foundation models, for reasoning
