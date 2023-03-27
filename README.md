# LLM4Academic

Find the proper FOUNDATION model for research (empirical study, including instruct tuning, RLHF, accelerate, etc,.)

## Datasets

### Instructs

1. Commonly Used

| Data     | Language                                                                            | Source                                     | prompt                                                                   | Size | Link                                                                                                       |
| -------- | ----------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ | ---- | ---------------------------------------------------------------------------------------------------------- |
| Alpaca   | English                                                                             | text-davinci-003                           | [prompt](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) | 52k  | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                             |
| Belle    | Chinese                                                                             | text-davinci-003                           | [prompt](https://github.com/LianjiaTech/BELLE/blob/main/prompt_cn.txt)      | 543k | [Belle (0.5M)](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)                            |
| Luotuo   | Chinese                                                                             | text-davinci-003+ChatGPT (Tranlate Alpaca) | -                                                                        | 52k  | [Alpaca_Chinese](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json) |
| Guannaco | English<br />Simplified Chinese<br />Traditional Chinese<br />Japanese<br />Deutsch |                                            |                                                                          | 534k | [Guannaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)                                     |

2. COT

| Data                                               | Language | Source                                       | Template                                                                             | Statistic        |
| -------------------------------------------------- | -------- | -------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------- |
| [Alpaca-Cot](https://github.com/PhoebusSi/Alpaca-CoT) | English  | [Flan](https://github.com/google-research/FLAN) | [Flan-template](https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py) | 9 datasets (75k) |

## Baselines

| Model     | Params | Pretrianed                                                                                                                                   | Language | Affiliation | Foundation | tuning                             |
| --------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----------- | ---------- | ---------------------------------- |
| Galactica | 6.7b   | Self-construct                                                                                                                               |          | Meta        | Galactica  |                                    |
| Llama     | 7b     | Self-construct                                                                                                                               |          | Meta        | Llama      |                                    |
| Bloomz    | 7.1b   | [BigscienceCorpus (1.5T)](https://huggingface.co/spaces/bigscience/BigScienceCorpus) + [xP3mt](https://huggingface.co/datasets/bigscience/xP3mt) |          | Bigscience  | Bloom      | +finetune                          |
| Flan-T5   | 11b    | [C4 (750G)](https://www.tensorflow.org/datasets/catalog/c) + Multitask Datasets                                                                |          | Google      | T5         | +finetune<br />+instruct           |
| GLM       | 10b    | [Pile (825G)](https://pile.eleuther.ai/)                                                                                                        |          | THUDM       | GLM        |                                    |
| GLM-CN    | 10b    | WuDaoCorpora (3T/200G)                                                                                                                       | Chinese  | THUDM       | GLM        |                                    |
| ChatGLM   | 6b     | Self-construct (1T)                                                                                                                          |          | THUDM       | GLM        | +fintune<br />+instruct<br />+rlhf |

## Benchmark

| Benchmarks | Tuning          | Language | Result                                                  |
| ---------- | --------------- | -------- | ------------------------------------------------------- |
| Z-Bench    | Instruct Tuning | Chinese  | [Basic](evaluations/zbench_basic.csv) `<space>` Advanced |
