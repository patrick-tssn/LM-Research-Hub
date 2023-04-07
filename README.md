# LLM4Academic

Empirical study of common foundation models, which aims to find the proper FOUNDATION model for research (instruct tuning, RLHF, accelerate, etc,.)

**Table of Contents**

- [Instruction Tuning](#instruction-tuning)
  - [Datasets](#datasets)
  - [Experiments](#experiments)
    - [Model Cards](#model-cards)
    - [Usage](#usage)
  - [Results](#results)
- [RLHF](#rlhf)
- [Accelerate](#accelerate)
- [Multimodal](#accelerate)
- [Augmumented](#augumented)
- [Reference](#reference)
  - [Open Source Repository](#open-source-repositories)
  - [Learning Materials](#tutorial-course-blog-talk-curated-list)

## Instruction tuning

### Datasets

1. Commonly Used

| Data                                                 | Language                                                                            | Source                                     | prompt                                                                   | Size | Link                                                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ | ---- | ---------------------------------------------------------------------------------------------------------- |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)  | English                                                                             | text-davinci-003                           | [prompt](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) | 52k  | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                             |
| [Belle](https://github.com/LianjiaTech/BELLE)           | Chinese                                                                             | text-davinci-003                           | [prompt](https://github.com/LianjiaTech/BELLE/blob/main/prompt_cn.txt)      | 543k | [Belle (0.5M)](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)                            |
| [Luotuo](https://github.com/LC1332/Chinese-alpaca-lora) | Chinese                                                                             | text-davinci-003+ChatGPT (Tranlate Alpaca) | -                                                                        | 52k  | [Alpaca_Chinese](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json) |
| [Guannaco](https://guanaco-model.github.io/)            | English<br />Simplified Chinese<br />Traditional Chinese<br />Japanese<br />Deutsch |                                            |                                                                          | 534k | [Guannaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)                                     |

2. COT

| Data                                               | Language | Source                                       | Template                                                                             | Statistic        |
| -------------------------------------------------- | -------- | -------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------- |
| [Alpaca-Cot](https://github.com/PhoebusSi/Alpaca-CoT) | English  | [Flan](https://github.com/google-research/FLAN) | [Flan-template](https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py) | 9 datasets (75k) |

### Experiments

#### Model Cards

| Model     | Params | Pretrianed                                                                                                                                   | Language | Affiliation | Foundation | tuning                             |
| --------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----------- | ---------- | ---------------------------------- |
| Galactica | 6.7b   | Self-construct                                                                                                                               |          | Meta        | Galactica  |                                    |
| Llama     | 7b     | Self-construct                                                                                                                               |          | Meta        | Llama      |                                    |
| Bloomz    | 7.1b   | [BigscienceCorpus (1.5T)](https://huggingface.co/spaces/bigscience/BigScienceCorpus) + [xP3mt](https://huggingface.co/datasets/bigscience/xP3mt) |          | Bigscience  | Bloom      | +finetune                          |
| Flan-T5   | 11b    | [C4 (750G)](https://www.tensorflow.org/datasets/catalog/c) + Multitask Datasets                                                                |          | Google      | T5         | +finetune<br />+instruct           |
| GLM       | 10b    | [Pile (825G)](https://pile.eleuther.ai/)                                                                                                        |          | THUDM       | GLM        |                                    |
| GLM-CN    | 10b    | WuDaoCorpora (3T/200G)                                                                                                                       | Chinese  | THUDM       | GLM        |                                    |
| ChatGLM   | 6b     | Self-construct (1T)                                                                                                                          |          | THUDM       | GLM        | +fintune<br />+instruct<br />+rlhf |

#### Usage

##### Traning

The code is heavily based on [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), and we use the A100 (80G) for training

- Llama

  ```
  torchrun --nproc_per_node=4 --master_port=10017 train.py \
      --model_name_or_path /prev_trained_models/llama-7b-hf \
      --data_path ../data/instructs/alpaca_mix.json \
      --bf16 True \
      --output_dir output/llama-mix \
      --num_train_epochs 3 \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap" \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True
  ```
- Bloomz

  ```
  torchrun --nproc_per_node=4 --master_port=10013 train.py \
      --model_name_or_path /prev_trained_models/bloomz-7b1-mt \
      --data_path ./instructs/alpaca_mix.json \
      --bf16 True \
      --output_dir output/bloomz-mix \
      --num_train_epochs 3 \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap" \
      --fsdp_transformer_layer_cls_to_wrap 'BloomBlock' \
      --tf32 True
  ```
- Galactica

  ```
  torchrun --nproc_per_node=4 --master_port=10014 train.py \
      --model_name_or_path /prev_trained_models/galactica-6b \
      --data_path ./instructs/alpaca_mix.json \
      --bf16 True \
      --output_dir output/galactica-mix \
      --num_train_epochs 3 \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap" \
      --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
      --tf32 True
  ```
- Flan-T5

  ```
  torchrun --nproc_per_node=4 --master_port=10015 train.py \
      --model_name_or_path /prev_trained_models/flan-t5-xxl \
      --data_path ./instructs/alpaca_mix.json \
      --bf16 True \
      --output_dir output/flant5-mix \
      --num_train_epochs 3 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 32 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap" \
      --fsdp_transformer_layer_cls_to_wrap 'T5Block' \
      --tf32 True
  ```

##### Evaluating

```
python evaluate.py
```

### Results

| Benchmarks                                   | Data                                     | Language                            | Result                                                                                                                                                                   |
| -------------------------------------------- | ---------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Z-Bench](https://github.com/zhenbench/z-bench) | Alpaca, Alpaca_CN, Belle(0.5M), Guannaco | Chinese, English, Japanese, Deutsch | [Basic](evaluations/z-bench/chinese_eval_basic_new.csv), [Advanced](evaluations/z-bench/chinese_eval_advanced_new.csv), [Domain](evaluations/z-bench/chinese_eval_domain_new.csv) |

## RLHF

## Accelerate

## Multimodal

## Augumented

## Evaluation

- [HELM](https://github.com/stanford-crfm/helm)

## Reference

### Open Source Repositories

| Open Source Repository                                                    | Base Language Model             | Language         | Accelerate                                                                                          | Tuning                                                     |
| ------------------------------------------------------------------------- | ------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)              | LLaMA7B                         | English          | [fsdp](https://huggingface.co/docs/accelerate/usage_guides/fsdp)                                       | Instruction Tuning                                         |
| [alpaca-lora](https://github.com/tloen/alpaca-lora)                          | LLaMA7B                         | English          | [peft](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Instruction Tuning                                         |
| [Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM)           | LLaMA7B                         | Chinese          | [peft](https://github.com/huggingface/peft)                                                            | Instruction Tuning                                         |
| [Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)                        | LLaMA7B                         | Chinese, English | [peft](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Instruction Tuning                                         |
| [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) *(Lightning)*      | LLaMA                           | English          | [LoRA](https://github.com/microsoft/LoRA)                                                              | Instruction Tuning                                         |
| [Llama-X](https://github.com/AetherCortex/Llama-X) (*Recommend*)           | LLaMA                           |                  |                                                                                                     |                                                            |
| [Coati](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) | LLaMA7B                         | English          |                                                                                                     | Instruction Tuning, RLHF                                   |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)        | LLaMA                           |                  |                                                                                                     | Instruction Tuning                                         |
| [Vicuna](https://github.com/lm-sys/FastChat)                                 | LLaMA13B                        | English          | [fsdp](https://huggingface.co/docs/accelerate/usage_guides/fsdp)                                       | Instruction Tuning                                         |
| [Baize](https://github.com/project-baize/baize-chatbot)                      | LLaMA                           | Chinese          | [LoRA](https://github.com/microsoft/LoRA)                                                              | Instruction Tuning                                         |
| [Dolly](https://github.com/databrickslabs/dolly)                             | GPT-J                           | English          |                                                                                                     | Insturction Tuning                                         |
| [BELLE](https://github.com/LianjiaTech/BELLE)                                | BLOOMZ-7B1-mt                   | Chinese          |                                                                                                     | Instruction Tuning                                         |
| [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)                            | GLM6B                           | Chinese, English | [p-tuning](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md)                            | Instruciton Tuning, RLHF                                   |
| [LMFlow](https://github.com/OptimalScale/LMFlow)                             | LLaMA, GPT2, GPT-Neo, Galactica |                  | [deepspeed](https://www.deepspeed.ai/docs/config-json/)                                                | Instruction Tuning, Inference, Alignment Tuning (possible) |

### Tutorial, Course, Blog, Talk, Curated List

- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
- [Awesome-Colorful-LLM](https://github.com/patrick-tssn/Awesome-Colorful-LLM)
- [Awesome-ALM](https://github.com/pbhu1024/awesome-augmented-language-model#action-and-plan)
