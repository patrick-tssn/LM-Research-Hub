# LLM4Academic

LLM4Academic is a repository about everything I would like to know about large language models (LLMs). There are two parts in this repository: (1) Practice: insightful experiment, demo, framework; (2) Theory: reading list, survey, curated sources

> *"Talk is cheap, show me the code."*

**Table of Contents**

- [Theory](#theory)

  - [Datasets](#datasets)
  - [Open Source LLM](#open-source-llms)
    - [English Community](#english-community)
    - [Chinese Community](#chinese-community)
  - [Evaluation](#evaluation)
  - [Learning Materials](#tutorial-course-blog-talk-curated-list)
- [Practice](#Practice)

  - [API](#api)
  - [Instruction Tuning](#instruction-tuning)

## Theory

### Datasets

1. Commonly Used

| Data                                                                                     | Language                                                                            | Source                                      | prompt                                                                   | Size | Link                                                                                                       |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ | ---- | ---------------------------------------------------------------------------------------------------------- |
| [OIG](https://laion.ai/blog/oig-dataset/)                                                   |                                                                                     |                                             |                                                                          | 43M  | [OIG](https://huggingface.co/datasets/laion/OIG)                                                              |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                                      | English                                                                             | text-davinci-003                            | [prompt](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) | 52k  | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                             |
| [Belle](https://github.com/LianjiaTech/BELLE)                                               | Chinese                                                                             | text-davinci-003                            | [prompt](https://github.com/LianjiaTech/BELLE/blob/main/prompt_cn.txt)      | 543k | [Belle (0.5M)](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)                            |
| [Luotuo](https://github.com/LC1332/Chinese-alpaca-lora)                                     | Chinese                                                                             | text-davinci-003+ChatGPT (Translate Alpaca) | -                                                                        | 52k  | [Alpaca_Chinese](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json) |
| [Guannaco](https://guanaco-model.github.io/)                                                | English<br />Simplified Chinese<br />Traditional Chinese<br />Japanese<br />Deutsch |                                             |                                                                          | 534k | [Guannaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)                                     |
| [Firefly](https://github.com/yangjianxin1/Firefly)                                          | Chinese                                                                             |                                             |                                                                          | 1.1M | [firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)                             |
| [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | English, Chinese                                                                    | GPT4                                        |                                                                          |      | [data-gpt4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release)                          |
| [gpt4all](https://github.com/nomic-ai/gpt4all)                                              | English                                                                             | GPT4                                        |                                                                          |      | [gpt4all_data](https://github.com/nomic-ai/gpt4all#Reproducibility)                                           |
| [Dolly](https://github.com/databrickslabs/dolly/tree/master/data)                           | English                                                                             | Human (Databricks employee)                 |                                                                          | 15k  | [databricks-dolly-15k](https://github.com/databrickslabs/dolly/tree/master/data)                              |
| [COIG](agithub.com/BAAI-Zlab/COIG)                                                          | Chinese                                                                             | Dialogue, Code, Exam, Human, Translation    |                                                                          |      | [COIG](agithub.com/BAAI-Zlab/COIG)                                                                            |

2. COT

| Data                                               | Language | Source                                       | Template                                                                             | Statistic        |
| -------------------------------------------------- | -------- | -------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------- |
| [Alpaca-Cot](https://github.com/PhoebusSi/Alpaca-CoT) | English  | [Flan](https://github.com/google-research/FLAN) | [Flan-template](https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py) | 9 datasets (75k) |

3. Code

| Data             | Source | Size                                                               | Link                                                                        |
| ---------------- | ------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Instruct-to-Code |        | 700k                                                               | [Instruct-to-Code](https://huggingface.co/datasets/Graverman/Instruct-to-Code) |
| BigCode          |        | [bigcode-dataset](https://github.com/bigcode-project/bigcode-dataset) |                                                                             |

4. Dialogue

| Data                           | Language | Source  | Size     | Link                                                            |
| ------------------------------ | -------- | ------- | -------- | --------------------------------------------------------------- |
| [ShareGPT](https://sharegpt.com/) |          | ChatGPT | 52K, 90K | [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) |


5. Human Preference

| Data                                          | Language | Source | Size | Link                                                                                                                                                                                          |
| --------------------------------------------- | -------- | ------ | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [hh-rlhf](https://github.com/anthropics/hh-rlhf) | English  | Human  |      | [human preference data](https://github.com/anthropics/hh-rlhf#Human-preference-data-about-helpfulness-and-harmlessness), [red teaming data](https://github.com/anthropics/hh-rlhf#Red-teaming-data) |

### Evaluation

- [HELM](https://github.com/stanford-crfm/helm) | comprehensive
- [AGIEval](https://github.com/microsoft/AGIEval) | comprehensive
- [C-Eval](https://github.com/SJTU-LIT/ceval) | comprehensive, Chinese
- [SuperCLUE](https://github.com/CLUEbenchmark/SuperCLUE) | comprehensive, Chinese
- [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness) | safety
- [Safety-Prompts](https://github.com/thu-coai/Safety-Prompts) | safety
- [LaMP](https://lamp-benchmark.github.io/) | personalization

### Open Source LLMs

#### English Community

| Open Source Repository                                                                                  | Base Language Model                                | Language         | Accelerate                                                                                          | Tuning                                                     |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)                                            | LLaMA7B                                            | English          | [fsdp](https://huggingface.co/docs/accelerate/usage_guides/fsdp)                                       | Instruction Tuning                                         |
| [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter)                                             | LlaMA                                              | English          |                                                                                                     | adapter                                                    |
| [alpaca-lora](https://github.com/tloen/alpaca-lora)                                                        | LLaMA7B                                            | English          | [peft](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Instruction Tuning                                         |
| [Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)                                                      | LLaMA7B                                            | Chinese, English | [peft](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Instruction Tuning                                         |
| [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) *(Lightning)*                                    | LLaMA                                              | English          | [LoRA](https://github.com/microsoft/LoRA)                                                              | Instruction Tuning                                         |
| [Llama-X](https://github.com/AetherCortex/Llama-X)                                                         | LLaMA                                              |                  |                                                                                                     | Instruction Tuning, RLHF (possible)), etc.                 |
| [Dromedary](https://github.com/IBM/Dromedary)                                                              | LLaMA                                              | English          |                                                                                                     | Instruction Tuning (safety)                                |
| [PandaLM](https://github.com/WeOpenML/PandaLM)                                                             | LlaMA, BLOOM                                       | English          |                                                                                                     | Instruction Tuning, evaluation                             |
| [Open-Llama](https://github.com/s-JoL/Open-Llama)                                                          | Transformers-Llama                                 |                  |                                                                                                     | Pretraining, Tuning (possible), RLHF (possible)            |
| [OpenAlpaca](https://github.com/yxuansu/OpenAlpaca)                                                        | OpenLLaMA                                          | English          |                                                                                                     | Instruction Tuning                                         |
| [Dolly](https://github.com/databrickslabs/dolly), v2                                                       | GPT-J, EleutherAI pythia (v2)                      | English          |                                                                                                     | Insturction Tuning                                         |
| [Koala (EasyLM)](https://bair.berkeley.edu/blog/2023/04/03/koala/)                                         | LLaMA, GPT-J, OPT, RoBERTa                         |                  | JAX/Flax                                                                                            | Instruction Tuning                                         |
| [Vicuna (FastChat)](https://github.com/lm-sys/FastChat)                                                    | LLaMA13B                                           | English          | [fsdp](https://huggingface.co/docs/accelerate/usage_guides/fsdp)                                       | Instruction Tuning                                         |
| [LMFlow](https://github.com/OptimalScale/LMFlow)                                                           | LLaMA, GPT2, GPT-Neo, Galactica                    |                  | [deepspeed](https://www.deepspeed.ai/docs/config-json/)                                                | Instruction Tuning, Inference, Alignment Tuning (possible) |
| [Coati](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)                               | LLaMA7B                                            | English          |                                                                                                     | Instruction Tuning, RLHF                                   |
| [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)  | OPT, BLOOM, GPT-NEOX, GPTJ, GPT-NEO, GPT2, CODEGEN |                  |                                                                                                     | Instruction Tuning, RLHF                                   |
| [StableLM](https://github.com/Stability-AI/StableLM)                                                       | Transformer                                        | English          |                                                                                                     | Pretraining, Fine-tuning                                   |
| [LLM Foundary](https://github.com/mosaicml/llm-foundry)                                                    | Transformer-GPT                                    | English          |                                                                                                     | Pretraining, Fine-tuning                                   |

#### Chinese Community

| Open Source Repository                                               | Base Language Model | Language              | Accelerate                                                               | Tuning                   |
| -------------------------------------------------------------------- | ------------------- | --------------------- | ------------------------------------------------------------------------ | ------------------------ |
| [骆驼 Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM) | LLaMA7B             | Chinese               | [peft](https://github.com/huggingface/peft)                                 | Instruction Tuning       |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)   | LLaMA               | Chinese               |                                                                          | Instruction Tuning       |
| [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)              | LLaMA               | Chinese               | LoRA                                                                     | Instruction Tuning       |
| [白泽 Baize](https://github.com/project-baize/baize-chatbot)            | LLaMA               | Chinese               | [LoRA](https://github.com/microsoft/LoRA)                                   | Instruction Tuning       |
| [流萤 Firefly](https://github.com/yangjianxin1/Firefly)                 | BLOOM               | Chinese               | [LLMPruner](https://github.com/yangjianxin1/LLMPruner)                      | Instruction Tuning       |
| [BELLE](https://github.com/LianjiaTech/BELLE)                           | BLOOMZ-7B1-mt       | Chinese               |                                                                          | Instruction Tuning       |
| [TigerBot](https://github.com/TigerResearch/TigerBot)                   | BLOOM               | Chinese, English      | GPTQ                                                                     | Pretrain, Tuning         |
| [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)                       | GLM6B               | Chinese, English      | [p-tuning](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md) | Instruciton Tuning, RLHF |
| [InstructGLM](https://github.com/yanqiangmiffy/InstructGLM)             | ChatGLM-6B          | Chienese              | [deepspeed](https://www.deepspeed.ai/docs/config-json/)                     | Instruction Tuning       |
| [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)                | LLaMA, Bloomz       | Chinese, Multilingual |                                                                          | Instruction Tuning       |
| [MOSS](https://github.com/OpenLMLab/MOSS)                               | Transformer         | Chinese, English      |                                                                          | Pretraining, Fine-Tuning |

#### Related Repositories

- [LLM-Zoo](https://github.com/DAMO-NLP-SG/LLM-Zoo)
- [FindTheChatGPTer](https://github.com/chenking2020/FindTheChatGPTer)

### Tutorial, Course, Blog, Talk, Curated List

- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
- [Awesome-Colorful-LLM](https://github.com/patrick-tssn/Awesome-Colorful-LLM)
- [Awesome-ALM](https://github.com/pbhu1024/awesome-augmented-language-model#action-and-plan)

## Practice

### API

LLM API demos, see [API](API/README.md) for details.

### Instruction Tuning

Instruction Tuning on 4 LLM with multilingual instructions, see [Instruction Tuning](Instruction_Tuning/READEME.md) for details.
