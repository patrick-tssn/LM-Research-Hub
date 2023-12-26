# Instruction tuning

Table of Contents

- [Experiments](#experiments)
  - [Datasets](#datasets-1)
    - [Collection](#collected-instructions)
    - [Bootstrap](#bootsrap-instructions)
  - [Model Cards](#model-cards)
  - [Usage](#usage)
- [Results](#results)

## Experiments

### Datasets

#### Collected Instructions

We collect 1M **Multilingual** instruction datasets for experiments, including:

- Chinese: [Belle](https://github.com/LianjiaTech/BELLE), [Alpaca_Chinese](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json), [Guannaco](https://guanaco-model.github.io/)
- English: [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Guannaco](https://guanaco-model.github.io/)
- Deutsch: [Guannaco](https://guanaco-model.github.io/)

#### Bootsrap Instructions

to construct instruction manually, we follow [self-instruct](https://github.com/yizhongw/self-instruct), and [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

```
python bootstrap_instruction.py generate_instruction_following_data\
	--output_dir ./ \
	--num_instructions_to_generate 10 \
	--model_name="text-davinci-003"
```

### Model Cards

We select 4 open-source LLM for experiments, including:

| Model     | Params | Pretrianed                                                                                                                                   | Language | Affiliation | Foundation | tuning                   |
| --------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----------- | ---------- | ------------------------ |
| Galactica | 6.7b   | Self-construct                                                                                                                               |          | Meta        | Galactica  |                          |
| LLaMA     | 7b     | Self-construct                                                                                                                               |          | Meta        | Llama      |                          |
| Bloomz    | 7.1b   | [BigscienceCorpus (1.5T)](https://huggingface.co/spaces/bigscience/BigScienceCorpus) + [xP3mt](https://huggingface.co/datasets/bigscience/xP3mt) |          | Bigscience  | Bloom      | +finetune                |
| Flan-T5   | 11b    | [C4 (750G)](https://www.tensorflow.org/datasets/catalog/c) + Multitask Datasets                                                                |          | Google      | T5         | +finetune<br />+instruct |

### Usage

**LOG:**

- [X] We adopt instruction tuning on 10b-level LLM (Llama-7b, galactica-6.7b, bloomz-7b1-mt, flant5-11b) with open source instructions (stanford_alpaca, chinese_alpaca, belle0.5m, guanaco), and evaluate these models on Z-Bench.

```
cd instruction
```

#### Traning

*NOTE: The code is heavily based on [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), and we use the A100 (80G) for training*

- <details><summary>Llama</summary>

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

  </details>
- <details><summary>Bloomz</summary>

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

  </details>
- <details><summary>Galactica</summary>

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

  </details>
- <details><summary>Flan-T5</summary>

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

  </details>

#### Evaluating

```
python evaluate.py
```

## Results

| Benchmarks                                   | Data                                     | Language                            | Result                                                                                                                                                                   |
| -------------------------------------------- | ---------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Z-Bench](https://github.com/zhenbench/z-bench) | Alpaca, Alpaca_CN, Belle(0.5M), Guannaco | Chinese, English, Japanese, Deutsch | [Basic](evaluations/z-bench/chinese_eval_basic_new.csv), [Advanced](evaluations/z-bench/chinese_eval_advanced_new.csv), [Domain](evaluations/z-bench/chinese_eval_domain_new.csv) |
