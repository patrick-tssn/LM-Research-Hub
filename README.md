# LLM4Academic

Find the proper FOUNDATION model for research (empirical study)

## Datasets

### Instructs

| Data     | Language                                                                            | Source                                     | prompt                                                                   | Size | Link                                                                                                       |
| -------- | ----------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ | ---- | ---------------------------------------------------------------------------------------------------------- |
| Alpaca   | English                                                                             | text-davinci-003                           | [prompt](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) | 52k  | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                             |
| Belle    | Chinese                                                                             | text-davinci-003                           | [prompt](https://github.com/LianjiaTech/BELLE/blob/main/prompt_cn.txt)      | 543k | [Belle (0.5M)](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)                            |
| Luotuo   | Chinese                                                                             | text-davinci-003+ChatGPT (Tranlate Alpaca) | -                                                                        | 52k  | [Alpaca_Chinese](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json) |
| Guannaco | English<br />Simplified Chinese<br />Traditional Chinese<br />Japanese<br />Deutsch |                                            |                                                                          | 534k | [Guannaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)                                     |

## Baselines

| Model     | Params | Pretrianed                                                                                                                                   | English | Chinese | Affiliation | Foundation | tuning                             |
| --------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | ----------- | ---------- | ---------------------------------- |
| Galactica | 6.7b   | Self-construct                                                                                                                               |         |         | Meta        | Galactica  |                                    |
| Llama     | 7b     | Self-construct                                                                                                                               |         |         | Meta        | Llama      |                                    |
| Bloomz    | 7.1b   | [BigscienceCorpus (1.5T)](https://huggingface.co/spaces/bigscience/BigScienceCorpus) + [xP3mt](https://huggingface.co/datasets/bigscience/xP3mt) |         |         | Bigscience  | Bloom      | +finetune                          |
| Flan-T5   | 11b    | [C4 (750G)](https://www.tensorflow.org/datasets/catalog/c) + multitask                                                                         |         |         | Google      | T5         | +instruct                          |
| GLM       | 10b    | [Pile (825G)](https://pile.eleuther.ai/)                                                                                                        |         |         | THUDM       | GLM        |                                    |
| GLM-CN    | 10b    | WuDaoCorpora (3T/200G)                                                                                                                       |         |         | THUDM       | GLM        |                                    |
| ChatGLM   | 6b     | 1T                                                                                                                                           |         |         | THUDM       | GLM        | +fintune<br />+instruct<br />+rlhf |

## Benchmark

| Benchmarks | Tuning          | Language | Result                                                  |
| ---------- | --------------- | -------- | ------------------------------------------------------- |
| Z-Bench    | Instruct Tuning | Chinese  | [Basic](evaluations/zbench_basic.csv) <space>\|<space> Advanced |
