# LLM Reading

Table of Contents

- [Key Findings](#key-findings)

*arranged in alphabetical order*

- [Architecture](#architecture)
- [Instruction Tuning](#instruction-tuning)
- [In Context Learning](#in-context-learning)
- [Mixture of Experts (MoE)](#mixture-of-experts-moe)
- [Reasoning](#reasoning)
  - [Abstract Reasoning](#abstract-reasoning)
  - [Chain of Thought](#chain-of-thought)

## Key Findings

| Title                                                                 | Pub       | Preprint                                                                                                                                                                                                                                                                                               | Supplementary                      |
| --------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| RWKV: Reinventing RNNs for the Transformer Era                        |           | [2305.13048](https://arxiv.org/abs/2305.13048)          <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F026b3396a63ed5772329708b7580d633bb86bec9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) | RWKV, Blink                        |
| Self-Instruct: Aligning LM with Self Generated Instructions           | ACL 2023  | [2212.10560](https://arxiv.org/abs/2212.10560)        <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe65b346d442e9962a4276dc1c1af2956d9d5f1eb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)   | self-instruct, UW                  |
| Emergent Abilities of Large Language Models                           | TMLR 2022 | [2206.07682](https://arxiv.org/abs/2206.07682)  <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdac3a172b504f4e33c029655e9befb3386e5f63a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)         | emergent ability,  Google         |
| Training language models to follow instructions with human feedback   |           | [2203.02155](https://arxiv.org/abs/2203.02155)  <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd766bffc357127e0dc86dd69561d5aeb520d6f4c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)           | InstructGPT, OpenAI                |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models |           | [2201.11903](https://arxiv.org/abs/2201.11903)   <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1b6e810ce0afd0dd093f789d2b2742d047e316d5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | CoT, Google                        |
| Finetuned Language Models Are Zero-Shot Learners                      | ICLR 2022 | [2109.01652](https://arxiv.org/abs/2109.01652)   <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fff0b2681d7b05e16c46dfb71d980cc2f605907cd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | Flan, instruction finetune, Google |
| Language Models are Few-Shot Learners                                 |           | [2005.14165](https://arxiv.org/abs/2005.14165)    <br /> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6b85b63579a916f705a8e10a49bd8d849d91b1fc%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)         | GPT3, OpenAI                       |
| Scaling Laws for Neural Language Models                               |           | [2001.08361](https://arxiv.org/abs/2001.08361) <br />   ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe6c561d02500b2596a230b341a8eb8b921ca5bf2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          | scaling law, OpenAI                |

## Architecture

Papers about Architecturn/Framework of neural network/language model beyond Transformer, see [Architecture](Architecture.md) for details.

> - Papers

## Efficiency

Papers about Efficiency/Compression of neural network/language model, see [Efficiency](Efficiency.md) for details.

> - Papers

Related Collections

- [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/DefTruth/Awesome-LLM-Inference?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social&label=Star), A small Collection for Awesome LLM Inference [Papers|Blogs|Docs] with codes, contains TensorRT-LLM, streaming-llm, SmoothQuant, WINT8/4, Continuous Batching, FlashAttention, PagedAttention etc.
- [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/HuangOwen/Awesome-LLM-Compression?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/HuangOwen/Awesome-LLM-Compression.svg?style=social&label=Star), Awesome LLM compression research papers and tools to accelerate the LLM training and inference

## Extensibility

Papers concentrate on the scalability and extensibility of LMs, i.e. longer context

> - Projects

Related Collections

- [Awesome-Long-Context](https://github.com/showlab/Awesome-Long-Context) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/showlab/Awesome-Long-Context?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/showlab/Awesome-Long-Context.svg?style=social&label=Star), A curated list of resources about long-context in large-language models and video understanding.


## Instruction Tuning

Papers/blogs about Instruction Tuning, see [Instruction Tuning](InstructionTuning.md) for details.

> - Papers
> - Blogs

- [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers) ![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/SinclairCoder/Instruction-Tuning-Papers?style=flat)![Dynamic JSON Badge](https://img.shields.io/github/stars/SinclairCoder/Instruction-Tuning-Papers.svg?style=social&label=Star), A trend starts from Natrural-Instruction (ACL 2022), FLAN (ICLR 2022) and T0 (ICLR 2022).

Related Collections

## In Context Learning

Papers about In Context Learning, see [In Context Learning](ICL.md) for details.

> - Papers
> - Reference

## Mixture of Experts (MoE)

Papers about Mixture of Experts (MoE), see [Mixture of Experts](MoE.md) for details.

> - Papers
> - Reference

## Reasoning

### Chain of Thought

Papers about Chain of Thought, see [Chain of Thought](reasoning/CoT.md) for details.

> - Papers
> - Reference

### Abstract Reasoning

Short survey about Abstract Reasoning, see [Abstract Reasoning](reasoning/AR.md) for details.

> - Datasets
> - Learning Methods
> - Models
> - Representative works
