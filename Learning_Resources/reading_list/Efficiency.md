# Efficiency

## Table of Contents

- [Papers](#papers)
    - [Adapt](#adapt)
    - [Architecture](#architecture)
    - [Pruning](#pruning)
    - [Quantization](#quantization)
- [Blogs](#blogs)

## Papers

### Adapt

| Title                                                                          | Pub       | Preprint                                    | Supplementary                                                         |
| ------------------------------------------------------------------------------ | --------- | ------------------------------------------- | --------------------------------------------------------------------- |
| LoRA: Low-Rank Adaptation of Large Language Models                             | ICLR 2022 | [2106.09685](https://arxiv.org/abs/2106.09685) | LoRA, Microsoft                                                       |



### Architecture


|Title                                                                          | Pub       | Preprint                                    | Supplementary                                                         |
| ------------------------------------------------------------------------------ | --------- | ------------------------------------------- | --------------------------------------------------------------------- |
| RWKV: Reinventing RNNs for the Transformer Era                                                |           | [2305.13048](https://arxiv.org/abs/2305.13048) | linear, Blink                              |
| Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder |           | [2304.04052](https://arxiv.org/abs/2304.04052) | encoder-decoder or decoder-only, Cambridge |
| Hyena Hierarchy: Towards Larger Convolutional Language Models                                 | ICML 2023 | [2302.10866](https://arxiv.org/abs/2302.10866) | CNN,Stanford                               |
| Transformer Quality in Linear Time                                 | ICML 2022 | [2202.10447](https://arxiv.org/abs/2202.10447) | GAU,Google, [Su's blog](https://spaces.ac.cn/archives/8934)                               |
| MoEfication: Transformer Feed-forward Layers are Mixtures of Experts           | ACL 2022  | [2110.01786](https://arxiv.org/abs/2110.01786) | [MoEfication](https://github.com/thunlp/MoEfication), THU                |
| Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth           | ICML 2021  | [2103.03404](https://arxiv.org/abs/2103.03404) | Problems within low rank matrix, Google
| Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention           | AAAI 2021  | [2102.03902](https://arxiv.org/abs/2102.03902) | linear attention, UWM                |
| Rethinking Attention with Performers           |   | [2009.14794](https://arxiv.org/abs/2009.14794) | linear attention, Google                |
| Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention                                 | ICML 2020 | [2006.16236](https://arxiv.org/abs/2006.16236) | Linear Attention, EPFL                               |
| Linformer: Self-Attention with Linear Complexity
 Transformer                                 |  | [2006.04768](https://arxiv.org/abs/2006.04768) | Subsampling (pooling), Meta                               |
 | Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing                                 | Neurips 2020  | [2006.03236](https://arxiv.org/abs/2006.03236) | Subsampling,  Google                              |
 | Longformer: The Long-Document Transformer                                 |  | [2004.05150](https://arxiv.org/abs/2004.05150) | Sparse Attention, AI2                               |
 | PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination                                | ICML 2020  | [2001.08950](https://arxiv.org/abs/2001.08950) | Subsampling, IBM                               |
| Reformer: The Efficient Transformer                                 | ICLR 2020 | [2001.04451](https://arxiv.org/abs/2001.04451) | Sparse Attention, Google                               |
| Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection
Transformers                                 |  | [1912.11637](https://arxiv.org/abs/1912.11637) | Sparse Attention, PKU                               |
| Generating Long Sequences with Sparse Transformers                                 |  | [1904.10509](https://arxiv.org/abs/1904.10509) | Sparse Attention, OpenAI                               |
| Efficient Attention: Attention with Linear Compexities                                 | WACV 2021 | [1812.01243](https://arxiv.org/abs/1812.01243) | Linear Attention, Sensetime                               |


### Pruning


| Title                                                                          | Pub       | Preprint                                    | Supplementary                                                         |
| ------------------------------------------------------------------------------ | --------- | ------------------------------------------- | --------------------------------------------------------------------- |
| Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning |           | [2310.06694](https://arxiv.org/abs/2310.06694) | [LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing), Princeton |


### Quantization


| Title                                                                          | Pub       | Preprint                                    | Supplementary                                                         |
| ------------------------------------------------------------------------------ | --------- | ------------------------------------------- | --------------------------------------------------------------------- |
| LLM-QAT: Data-Free Quantization Aware Training for Large Language Models       |           | [2305.17888](https://arxiv.org/abs/2305.17888) | Meta                                                                  |

## Blogs

- 线性Attention的探索：Attention必须有个Softmax吗？[Jianlin Su's blog](https://spaces.ac.cn/archives/7546)
- 为节约而生：从标准Attention到稀疏Attention, [Jianlin Su's blog](https://spaces.ac.cn/archives/6853)


## Projects

- (2023-10) [MemGPT](https://github.com/cpacker/MemGPT), Memory-GPT (or MemGPT in short) is a system that intelligently manages different memory tiers in LLMs in order to effectively provide extended context within the LLM's limited context window
- (2023-08) [Long-Context](https://github.com/abacusai/Long-Context), This repository contains code and tooling for the Abacus.AI LLM Context Expansion project. Also included are evaluation scripts and benchmark tasks that evaluate a model’s information retrieval capabilities with context expansion. We also include key experimental results and instructions for reproducing and building on them
- (2023-08) [LongBench](https://github.com/THUDM/LongBench), LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding



