# Datasets

Commonly Used Datasets for Pretraining, Finetuning, Instruction Tuning

Table of Contents

- [Pretraining Corpora](#pretraining-corpora)
- [Instruction](#instruction)

## Pretraining corpora

- [MNBVC](https://github.com/esbatmop/MNBVC), MNBVC(Massive Never-ending BT Vast Chinese corpus)超大规模中文语料集。对标chatGPT训练的40T数据。MNBVC数据集不但包括主流文化，也包括各个小众文化甚至火星文的数据。MNBVC数据集包括新闻、作文、小说、书籍、杂志、论文、台词、帖子、wiki、古诗、歌词、商品介绍、笑话、糗事、聊天记录等一切形式的纯文本中文数据。

## Instruction

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

6. Social interactions

| Data                                           | Language | Source            | Size | Link                                                   |
| ---------------------------------------------- | -------- | ----------------- | ---- | ------------------------------------------------------ |
| [KokoMind](https://github.com/CHATS-lab/KokoMind) | English  | GPT4, Movie, ToMi | 770  | [KokoMind](https://github.com/CHATS-lab/KokoMind#dataset) |

6. Specific Domain

| Data                                                        | Language | Source | Size  | Link                                                        |
| ----------------------------------------------------------- | -------- | ------ | ----- | ----------------------------------------------------------- |
| [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions) | English  |        | 2052K | [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions) |

## Reference

- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts), This repo includes ChatGPT prompt curation to use ChatGPT better
- [🧠ChatGPT 中文调教指南](https://github.com/PlexPt/awesome-chatgpt-prompts-zh), ChatGPT 中文调教指南。各种场景使用指南。学习怎么让它听你的话
