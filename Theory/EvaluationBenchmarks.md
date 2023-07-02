# Evaluation Benchmarks

Table of Contents

- [English](#english)
  - [Comprehensive](#comprehensive)
  - [Knowledge](#knowledge)
  - [Reason](#reason)
    - [Hard Mathematical, Theorem](#hard-mathematical-theorem)
  - [Code](#code)
  - [Personalization](#personalization)
- [Chinese](#chinese)
  - [Comprehensive](#comprehensive-1)
  - [Safety](#safety)
- [Multilingual]()

## English

### Comprehensive

- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub) | [paper](https://arxiv.org/abs/2305.17306) | [blog](https://yaofu.notion.site/Towards-Complex-Reasoning-the-Polaris-of-Large-Language-Models-c2b4a51355b44764975f88e6a42d4e75)

  > COT prompt, complex reasoning, *by Edinburgh, 2023*
  >
- [Open-LLM-Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/tree/main) | [leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

  > For open-source LLMs, *by Huggingface, 2023*
  >
- [Chatbo-Arena](https://github.com/lm-sys/FastChat#evaluation) | [leaderboard](https://chat.lmsys.org/?arena) | [paper](https://arxiv.org/abs/2306.05685)  | [blog](https://lmsys.org/blog/2023-05-03-arena/)

  > Elo rating system, *by LMSYS, 2023*
  >
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) | [leaderboard](https://tatsu-lab.github.io/alpaca_eval/)

  > Instruction-following tasks, *by Stanford, 2023*
  >
- [HELM](https://github.com/stanford-crfm/helm) | [paper](https://arxiv.org/abs/2211.09110) | [website](https://crfm.stanford.edu/helm/latest/)

  > *by Stanford, 2022*
  >
- [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness)

  > 200+ tasks, *by EleutherAI, 2021*
  >

### Knowledge

- [MMLU](https://github.com/hendrycks/test) | [paper (ICLR 2021)](https://arxiv.org/abs/2009.03300)
  > 15k problems under 57 subjects, high school and college knowledge, *by UCB, 2020*
  >

### Reason

- [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) | [paper](https://arxiv.org/abs/2210.09261)

  > 6.5k problems within 23 subsets, symbolic and text reasoning, *by Google, 2022*
  >
- [GSM8K](https://github.com/openai/grade-school-math) | [paper](https://arxiv.org/abs/2110.14168) | [blog](https://openai.com/blog/grade-school-math/)

  > a dataset of 8.5K high quality linguistically diverse grade school math word problems, *by OpenAI, 2021*
  >

#### Hard Mathematical, Theorem

- [MATH](https://github.com/hendrycks/math) | [paper (NeurIPS 2021)](https://arxiv.org/abs/2103.03874)
  > Hard, 12k problems within 7 categories, very hard math and natural science, *by UCB, 2021*
  >
- [TheoremQA](https://github.com/wenhuchen/TheoremQA) | [paper](https://arxiv.org/abs/2305.12524)
  > Hard, 800 QA pairs covering 350+ theorems spanning across Math, EE&CS, Physics and Finance, *by UWaterloo, 2023*
  >

### Code

- [HumanEval](https://github.com/openai/human-eval) | [paper](https://arxiv.org/abs/2107.03374)

  > *by OpenAI, 2021*
  >
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) | [paper](https://arxiv.org/abs/2108.07732)

  > *by Google, 2021*
  >

### Personalization

- [LaMP](https://github.com/LaMP-Benchmark/LaMP) | [leaderboard](https://lamp-benchmark.github.io/leaderboard) | [paper](https://arxiv.org/abs/2304.11406)
  > *by UMASS, 2023*
  >

### Factuality

- [SummEdits](https://github.com/salesforce/factualNLG) | [paper](https://arxiv.org/abs/2305.14540)
  > 6.3k factual consistency reasoning problems within 10 domains, *by Salesforce, 2023*
  >

## Chinese

### Comprehensive

- [SuperCLUE](https://github.com/CLUEbenchmark/SuperCLUE) | [leaderboard](https://www.cluebenchmarks.com/superclue.html)
  > *by CLUE, 2023*
  >

### Knowledge

- [C-Eval](https://github.com/SJTU-LIT/ceval) | [leaderboard](https://cevalbenchmark.com/static/leaderboard.html) | [paper](https://arxiv.org/abs/2305.08322)
  > Exams, Multi-choice, *by SJTU, 2023*
  >

### Safety

- [Safety-Prompts](https://github.com/thu-coai/Safety-Prompts) | [leaderboard](http://115.182.62.166:18000/public) | [paper](https://arxiv.org/abs/2304.10436) | [blog](https://cevalbenchmark.com/index.html#home)
  > *by THU, 2023*
  >

## Multilingual

- [AGIEval](https://github.com/microsoft/AGIEval) | [paper](https://arxiv.org/abs/2304.06364)
  > Chinese&English, high-standard admission and qualification exams, *by Microsoft, 2023*
  >

## Reference

- [Instruction Tuning阶段性总结](https://yaofu.notion.site/2023-06-Instruction-Tuning-935b48e5f26448e6868320b9327374a1), by 符尧 (Yao Fu), 2023.06
