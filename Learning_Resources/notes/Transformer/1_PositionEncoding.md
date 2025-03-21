# Position Encoding

**Comments**: Positional encoding is crucial in Transformer architectures because the original model lacks an inherent sense of sequence order.

## Table of Contents

- [Absolute Position Embedding `Learnable` `Absolute`](#1-absolute-position-embedding)
- [Sinusoidal Position Encoding `Fixed` `Absolute`](#2-sinusoidal-position-encoding)
- [Relative Position Encoding `Learnable` `Relative`](#3-relative-position-encoding)
- [Rotary Position Embedding `Fixed` `Relative`](#4-rotary-position-embedding)
  - [NTK](#41-ntk)
  - [Frequency/theta summary](#42-frequenciestheta-summary)
  - [N-d RoPE](#43-n-d-rope)
  - [MRoPE](#44-mrope)
  - [VideoRoPE](#45-videorope)

## 1. Absolute Position Embedding

**Implementation**

```python
# in the embedding part
import torch
from torch import nn
position_ids = torch.arange(INPUT_LENGTH).unsqueeze(0)
position_embedding = nn.Embedding(MAX_POSITION, HIDDEN_SIZE)
position_embedding = position_embedding(position_id)
# Then add into word embedding
```

**References**

> - Devlin, Jacob et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” North American Chapter of the Association for Computational Linguistics (2019).
> - Liu, Yinhan et al. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” ArXiv abs/1907.11692 (2019): n. pag.

## 2. Sinusoidal Position Encoding

**Brief Introduction:**
$t$ is the position step, $d$ is the hidden size of the model, $i$ is the dimension of the hidden size

$$
\begin{cases}
PE(t, 2i) &= sin(\frac{t}{10000^{2i/d}}) \\
PE(t, 2i+1) &= cos(\frac{t}{10000^{2i/d}})
\end{cases}
$$

according to the Trigonometric Identities `sin(a±b)=sin(a)cos(b)±cos(a)sin(b)`,`cos(a+b)=cos(a)cos(b)-sin(a)sin(b)`, `cos(a-b)=cos(a)cos(b)+sin(a)sin(b)`

$$
\begin{cases}
PE(t + k, 2i) &= PE(t, 2i) \times PE(k, 2i+1) + PE(t, 2i+1) \times PE(k, 2i)  \\
PE(t + k, 2i+1) &= PE(t, 2i+1) \times PE(k, 2i+1) - PE(t, 2i) \times PE(k, 2i)
\end{cases}
$$

Property1: $PE_{t+k}$ can be represented as linear function of $PE_{t}$, Therefore, $PE_{t}^TPE_{t+k}$ is dependent on $k$, which indicates this kind of position encoding methods have the potentials of applying relative position

let $\theta_i = \frac{1}{10000^{2i/d}}$

$$
PE_{t} = 
\begin{bmatrix}
sin(\theta_0t)\\
cos(\theta_0t)\\
\cdots\\
sin(\theta_{\frac{d}{2}-1}t) \\
cos(\theta_{\frac{d}{2}-1}t)
\end{bmatrix}
$$

Property2:

$$
\begin{align}
PE_{t}^TPE_{t+k} &= \sum_{i=0}^{\frac{d}{2}-1}[sin(\theta_it)sin(\theta_i(t+k)) + cos(\theta_it)cos(\theta_i(t+k))] \\
& = \sum_{i=0}^{\frac{d}{2}-1}cos(\theta_i(t-(t+k))) \\
& = \sum_{i=0}^{\frac{d}{2}-1}cos(\theta_ik)\\
& = PE_{t'}^TPE_{t'+k} \\
& = PE_{t}^TPE_{t-k}
\end{align}
$$

This means that sinusoidal position embeddings are unaware of direction

**Implementation**

```python
import numpy as np
import torch
position_embedding = torch.zeros(MAX_POSITION, HIDDEN_SIZE).unsqueeze(0)
position_enc = np.array([[pos / np.power(10000, 2*(j//2)/HIDDEN_SIZE) for j in range(HIDDEN_SIZE)] for pos in range(MAX_POSITION)])
position_embedding[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
position_embedding[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
# inputs + PE
```

**References**

> - Vaswani, Ashish et al. “Attention is All you Need.” Neural Information Processing Systems (2017).
> - Yan, Hang et al. “TENER: Adapting Transformer Encoder for Named Entity Recognition.” ArXiv abs/1911.04474 (2019): n. pag.
> - [The Annotated Transformer
>   ](https://nlp.seas.harvard.edu/2018/04/03/attention.html), HarvardNLP's blog

## 3. Relative Position Encoding

**Comments**: The core of self-attention is dot-product

**Brief Introduction**

- Original Self-Attention

$W^Q, W^K, W^V$ are parameter matrices, the attention score is calculated as $e_{ij}=\frac{(x_iW^Q)(x_jW^K)^T}{\sqrt{d}}$, where $d$ is the hidden size of single head, $x_i$ is the embedding $i^{th}$ token, here is a row vector. So the attention weight is calculated as $\alpha_{ij}=\frac{exp(e_{ij})}{ \sum\limits_{k=1}^{n} exp(e_{ij})}$. The output is $z_i= \sum\limits_{j=1}^{n} \alpha_{ij}(x_jW^V)$.

- Fuse Relative Position Information into Self-Attention

the relative position is actually pair-wise relationship between input elements, represented by vectors $a_{ij}^K, a_{ij}^V$. We first add it into attention score $e_{ij}=\frac{(x_iW^Q)(x_jW^K + a_{ij}^K)^T}{\sqrt{d}}$, then add it into output $z_i= \sum\limits_{j=1}^{n} \alpha_{ij}(x_jW^V + a_{ij}^V)$. In practice, there will be clip operation.

- Transformations

$$
\begin{align}
q_ik_j^T &= ((x_i+p_i)W^Q)((x_j+p_j)W^K)^T \\
&= x_iW^Q{W^K}^Tx_j^T + x_iW^Q{W^K}^Tp_j^T+p_iW^Q{W^K}^Tx_j^T+p_iW^Q{W^K}^Tp_j^T\\
&\approx x_iW^Q{W^K}^Tx_j^T + x_iW^Q{W^K}^TR_{i-j}^T+uW^Q{W^K}^Tx_j^T+vW^Q{W^K}^TR_{i-j}^T\\
&\approx x_iW^Q{W^K}^Tx_j^T + x_iW^Q{W^{K,R}}^TR_{i-j}^T+u{W^K}^Tx_j^T+v{W^{K,R}}^TR_{i-j}^T \quad (XLNet)\\
&\approx x_iW^Q{W^K}^Tx_j^T + x_iW^Q{W^K}^TR_{i,j}^T+R_{j,i}{W^K}^Tx_j^T \quad (DeBERTa)\\
&\approx x_iW^Q{W^K}^Tx_j^T + \beta_{i,j} \quad (T5)\\
\end{align}
$$

**Implementation 1**

```python
# in the attention part (clipping)
import torch
from torch import nn
position_ids_l = torch.arange(QUERY_LENGTH).view(-1, 1)
position_ids_r = torch.arange(KEY_LENGTH).view(-1, 1)
distance = position_ids_l - position_ids_r
distance_embedding = nn.Embedding(2*MAX_POSITION-1, HEAD_SIZE)
position_embedding = distance_embedding(distance + MAX_POSITION - 1) # lrd
relative_position_scores = torch.matmul(QUERY, position_embedding) # bhld @ lrd -> bhlr
attention_scores = attention_scores + relative_position_scores
```

**Implementation 2**

```python
# in attention part (sinusoidal)
import torch
from torch import nn
position_ids = torch.arange(INPUT_LENGTH-1, -1, -1.0)
position_embedding = 1 / (10000 ** (torch.arange(0.0, HIDDEN_SIZE, 2.0) / HIDDEN_SIZE))
position_embedding = torch.outer(position_embedding, position_ids).unsqueeze(1) # l1d
r = nn.Linear(HIDDEN_SIZE, HEAD * HEAD_SIZE, bias=False)
position_embedding = r(position_embedding)
position_embedding = position_embedding.view(INPUT_LENGTH, HEAD, HEAD_SIZE)
relative_position_score = torch.einsum("ibnd,jbnd->bnij", QUERY, position_embedding)
attention_scores = attention_scores + relative_position_score
```

**References**

> - Shaw, Peter et al. “Self-Attention with Relative Position Representations.” North American Chapter of the Association for Computational Linguistics (2018).
> - Dai, Zihang et al. “Transformer-XL: Attentive Language Models beyond a Fixed-Length Context.” ArXiv abs/1901.02860 (2019): n. pag.
> - Yang, Zhilin et al. “XLNet: Generalized Autoregressive Pretraining for Language Understanding.” Neural Information Processing Systems (2019).
> - Raffel, Colin et al. “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.” J. Mach. Learn. Res. 21 (2019): 140:1-140:67.
> - He, Pengcheng et al. “DeBERTa: Decoding-enhanced BERT with Disentangled Attention.” ArXiv abs/2006.03654 (2020): n. pag.
> - [让研究人员绞尽脑汁的Transformer位置编码](https://spaces.ac.cn/archives/8130), Jianlin Su's blog

## 4. Rotary Position Embedding

**Brief Introduction**
The primary objective of RoPE (Rotary Position Embedding) is to identify an operation that enables the inner product to incorporate relative positional information effectively. i.e. find a solution of the equation $< f(q, m), f(k, n)>=g(q,k,m-n)$
Intuitively, we introduce complex number, let arbitrary $\theta \in (0, \frac{\pi}{2N}]$

$$
\begin{align}
RoPE(x, m) &= xe^{im\theta}\\
< RoPE(q_j,m), RoPE(k_j,n)> &= < q_je^{im\theta}, k_je^{in\theta}>\\
 &=(q_je^{im\theta})(k_je^{in\theta})^* \\
 &=q_jk_je^{i(m-n)\theta}\\
 &=RoPE(q_jk_j, m-n) 
\end{align}
$$

For detailed derivation, please refer to the [original paper](https://arxiv.org/pdf/2104.09864.pdf).

**Implementation**
In a two-dimensional context, a complex number can be represented in the form of a matrix, which geometrically corresponds to a rotation vector

$$
f(q, m) = qe^{im\theta}=
\begin{pmatrix}
\cos m\theta & -\sin m\theta\\ 
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
q_0 \\ 
q_1
\end{pmatrix}
$$

the rotary matrix could be a combination of several 2D rotary matrix

$$
\begin{pmatrix} 
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ 
\end{pmatrix} 
\begin{pmatrix}
q_0 \\ 
q_1 \\ 
q_2 \\ 
q_3 \\ 
\vdots \\ 
q_{d-2} \\ 
q_{d-1}
\end{pmatrix}
$$

visualize the implementation

<!-- ![image](assets/rope.png) -->

<center>
  <img src="../assets/rope.png">
  <figcaption>implementation of RoPE</figcaption>
</center>

compute trick: convert matrix multiplication to element-wise multiplication (the origin of rotate_half())

$$
\begin{pmatrix}
q'_0 \\ 
q'_1
\end{pmatrix}=
\begin{pmatrix}
\cos m\theta & -\sin m\theta\\ 
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
q_0 \\ 
q_1
\end{pmatrix}
$$

$$
q'_0 = q_0 \cdot \cos m\theta + (-q_1) \cdot \sin m\theta \\
q'_1 = q_1 \cdot \cos m\theta + q_0 \cdot \sin m\theta
$$

$$
\begin{pmatrix}
q'_0 \\ 
q'_1
\end{pmatrix}=
\cos m\theta \cdot 
\begin{pmatrix}
q_0 \\ 
q_1
\end{pmatrix}
+
\sin m\theta \cdot 
\begin{pmatrix}
-q_1 \\ 
q_0
\end{pmatrix}
$$


```python
# in the embedding part (apply rotary position to QUERY and KEY)
import numpy as np
import torch
from torch import nn
position_ids = torch.arange(0, KEY_LENGTH)
position_embedding = nn.Embedding(MAX_POSITION, HEAD_SIZE)
position_embedding.weight.requires_grad = False
position_enc = np.array([[pos/np.power(10000, 2*(j//2)/dim) for j in range(dim)] for pos in range(n_pos)]) # 1, dim
position_embedding.weight[:, :dim//2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
position_embedding.weight[:, dim//2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
sinusoidal_pos = position_embedding(position_ids) # L, dim
sin, cos = sinusoidal_pos.chunk(2, dim=-1) # L, dim/2
sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos) # L, dim
cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos) # L, dim
rotate_half_QUERY = torch.stack([-QUERY[..., 1::2], QUERY[..., ::2]], dim=-1).reshape_as(QUERY) # B, N, L, dim
QUERY = QUERY * cos_pos + rotate_hals_QUERY * sin_pos
rotate_half_KEY = torch.stack([-KEY[..., 1::2], KEY[..., ::2]], dim=-1).reshape_as(KEY)
KEY = KEY * cos_pos + rotate_half_KEY * sin_pos
```

**References**

> - Su, Jianlin et al. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” ArXiv abs/2104.09864 (2021): n. pag.
> - [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265), Jianlin Su's blog
> - [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/), Eleuther's blog
> - [Positional Encodings I. Main Approaches](https://medium.com/mantisnlp/positional-encodings-i-main-approaches-bd1199d6770d), Medium



### 4.1 NTK

NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation. See [blog](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

Common methods for context length extension:

- Extrapolation: direct
- Interpolation: [SuperHOT](https://kaiokendev.github.io/til#extending-context-to-8k)/[PI](https://arxiv.org/abs/2306.15595): position n --> position n/k
- NTK

NTK derivation (simple version):

sinusoidal position at n

$$
[cos(\frac{n}{\beta^0}), sin(\frac{n}{\beta^0}), \ldots, cos(\frac{n}{\beta^{d/2-1}}), sin(\frac{n}{\beta^{d/2-1}})]
$$

where $\beta$ is $base^\frac{2}{d}$, NTK want to combine Extrapolation for high frequency with Interpolation in low frequency, specifically, they introduce $k$ to $\beta$, then making the value equal to the interpolation format

$$
\frac{n}{(\lambda\beta)^{d/2-1}} = \frac{n/k}{\beta^{d/2-1}}
$$

we get $\lambda=k^{2/(d-2)}$

Another interpretation is $\beta$ base, see reference[1] for details

**Reference**

> [1] [Transformer升级之路：10、RoPE是一种β进制编码](https://kexue.fm/archives/9675) Jianlin Su's blog

### 4.2 Frequencies/theta summary

```python
# default: d->0 theta->1 high_freq; d->dim theta->0 low_freq
1.0 / (base ** torch.arange(0, dim, 2) / dim)
# linear scaling: factor = sl/mp
1.0 / (base ** torch.arange(0, dim, 2) / dim) / factor
# ntk: d->0 (sl/mp)^-d/dim->1 high_freq_extrapolation; d->dim (sl/mp)^d/dim->(sl/mp)^-1 low_freq_interpolation
base = base * (seq_len / max_position_embeddings) ** (dim / (dim - 2))
1.0 / (base ** torch.arange(0, dim, 2) / dim) # base * (sl/mp)
# dynamic ntk
base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2)) # base * (factor * (sl/ml - 1) + 1)
1.0 / (base ** torch.arange(0, dim, 2) / dim)
# ntk by parts / yarn: wave length: 2pi/base**(-2d/dim); L/(2pi*base**(-2d/dim))<1 low_freq_extrapolation
extrapolation = 1.0 / (base ** torch.arange(0, dim, 2) / dim)
interpolation = 1.0 / (base ** torch.arange(0, dim, 2) / dim) / factor
low = dim * math.log(max_pos / （32 * 2 * math.pi）) / (2 * math.log(base)) # dim * log(L/32 * 2pi) / 2log(b)
high = dim * math.log(max_pos / (1 * 2 * math.pi)) / (2 * math.log(base)) # dim * log(L /* 2pi) / 2log(b)
extrapolation_factor = 1 - torch.clamp((torch.arange(dim) - low) / (high - low) , 0, 1)
interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor
attention_factor = 0.1 * math.log(factor) + 1.0 # difference between yarn and ntk by parts
```

**Reference**

> [从ROPE到Yarn, 一条通用公式速通长文本大模型中的位置编码
](https://zhuanlan.zhihu.com/p/15311461897) note: there is an error in section 6.2


### 4.3 N-d RoPE

2d RoPE

$$
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y}=\left(
\begin{array}{cc:cc}
\cos x\theta & -\sin x\theta & 0 & 0 \\
\sin x\theta & \cos x\theta & 0 & 0 \\
\hdashline
0 & 0 & \cos y\theta & -\sin y\theta \\
0 & 0 & \sin y\theta & \cos y\theta \\
\end{array}\right)\end{equation}
$$

we get

`Relative`: 

```math
\boldsymbol{\mathcal{R}}_{x_1,y_1}^{\top}\boldsymbol{\mathcal{R}}_{x_2,y_2} = \boldsymbol{\mathcal{R}}_{x_2-x_1,y_2-y_1}
```

`Reversible (lossless)`: Given $\boldsymbol{\mathcal{R}}_{x,y}$, we could obtain $x, y$

comparison of RoPE-1D and RoPE-2D

```math
\scriptsize{\begin{array}{c}\begin{array}{c}\text{RoPE-1D}\\ (\boldsymbol{\mathcal{R}}_n)\end{array}= \begin{pmatrix} 
\cos n\theta_0 & -\sin n\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
\sin n\theta_0 & \cos n\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \cos n\theta_1 & -\sin n\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \sin n\theta_1 & \cos n\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos n\theta_{d/2-2} & -\sin n\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & \sin n\theta_{d/2-2} & \cos n\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos n\theta_{d/2-1} & -\sin n\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin n\theta_{d/2-1} & \cos n\theta_{d/2-1} \\ 
\end{pmatrix} \\[16pt] 
\begin{array}{c}\text{RoPE-2D}\\ (\boldsymbol{\mathcal{R}}_{x,y})\end{array}= \begin{pmatrix} 
\cos x\theta_0 & -\sin x\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
\sin x\theta_0 & \cos x\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \cos y\theta_1 & -\sin y\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
0 & 0 & \sin y\theta_1 & \cos y\theta_1 & \cdots & 0 & 0 & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos x\theta_{d/2-2} & -\sin x\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & \sin x\theta_{d/2-2} & \cos x\theta_{d/2-2} & 0 & 0 \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos y\theta_{d/2-1} & -\sin y\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin y\theta_{d/2-1} & \cos y\theta_{d/2-1} \\ 
\end{pmatrix}\end{array}}
```

we can natually derive N-d RoPE

**Reference**

> - [1] [Transformer升级之路：4、二维位置的旋转式位置编码](https://kexue.fm/archives/8397) Jianlin Su's blog
> - [2] [恒等式 det(exp(A)) = exp(Tr(A)) 赏析](https://kexue.fm/archives/6377) Jianlin Su's blog
> - [3] [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) Meta

### 4.4 MRoPE


```python
# in the embedding part (apply rotary position to QUERY and KEY)
# we comment code in origianl 1-d rope for comparison
import numpy as np
import torch
from torch import nn
# position_ids = torch.arange(0, KEY_LENGTH) # defaul position id
# Examples:
#     input_ids: [T T T T T], here T is for text.
#     temporal position_ids: [0, 1, 2, 3, 4]
#     height position_ids: [0, 1, 2, 3, 4]
#     width position_ids: [0, 1, 2, 3, 4]
position_ids = torch.arange(0, KEY_LENGTH).expand(3, 1, KEY_LENGTH) # position for temporal, height, weight of text: (3, BS, L)
# For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
# and 1D rotary position embedding for text part.
# Examples:
#     Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
#     input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
#     vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
#     vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
#     vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
#     text temporal position_ids: [3, 4, 5, 6, 7]
#     text height position_ids: [3, 4, 5, 6, 7]
#     text width position_ids: [3, 4, 5, 6, 7]
#     Here we calculate the text start position_ids as the max vision position_ids plus 1.
position_ids = get_rope_index() # the input-output is as above
position_embedding = nn.Embedding(MAX_POSITION, HEAD_SIZE)
position_embedding.weight.requires_grad = False
position_enc = np.array([[pos/np.power(10000, 2*(j//2)/dim) for j in range(dim)] for pos in range(n_pos)]) # here we use the default rope for simplicity
position_embedding.weight[:, :dim//2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
position_embedding.weight[:, dim//2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
sinusoidal_pos = position_embedding(position_ids) # 3, BSZ, KEY_LENGTH, HEAD_SIZE
sin, cos = sinusoidal_pos.chunk(2, dim=-1)
sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos) # 3, BSZ, KEY_LENGTH, HEAD_SIZE
cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
############apply mrope here############
mrope_section = mrope_section * 2 # temporal, height, weight e.g. in Qwen2-VL [16, 24, 24] -> [16, 24, 24, 16, 24, 24] -> head_size = 128
sin_pos = torch.cat([m[i%3] for i, m in enumerate(sin_pos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim) # unsqueeze for broadcast on head
cos_pos = torch.cat([m[i%3] for i, m in enumerate(cos_pos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim) # BSZ, KEY_LENGTH, HEAD_SIZE: [[t1,t2,t3,t4,t5,t6],[h1,h2,h3,h4,h5,h6],[w1,w2,w3,w4,w5,w6]] -> [t1,t2,h3,h4,w5,w6]
#########################################
rotate_half_QUERY = torch.stack([-QUERY[..., 1::2], QUERY[..., ::2]], dim=-1).reshape_as(QUERY)
QUERY = QUERY * cos_pos + rotate_hals_QUERY * sin_pos
rotate_half_KEY = torch.stack([-KEY[..., 1::2], KEY[..., ::2]], dim=-1).reshape_as(KEY)
KEY = KEY * cos_pos + rotate_half_KEY * sin_pos
```

<center>
  <img src="../assets/mrope.png">
  <figcaption>implementation of MRoPE</figcaption>
</center>

### 4.5 VideoRoPE

three modification on MRoPE:
1. low-frequency temporal allocation: high dimension for temporal position
2. spatial symmetry: diagonal layout 
3. temporal index scaling: adjustable temporal spacing

```python
############apply videorope here############
# For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
# and 1D rotary position embedding for text part.
# Examples:
#     Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
#     input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
#     vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
#     vision height position_ids: [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3] # h_index = t_index + h_index
#     vision width position_ids: [0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 2, 3] # w_index = t_index + w_index
#     text temporal position_ids: [3, 4, 5, 6, 7]
#     text height position_ids: [3, 4, 5, 6, 7]
#     text width position_ids: [3, 4, 5, 6, 7]
#     Here we calculate the text start position_ids as the max vision position_ids plus 1.
position_ids = get_t_scale_rope_index() # MODIFICATION 2 + 3: adjustable temporal spacing, diagonal layout
# t_index = t_index * scale_factor
# h_index = t_index + h_index
# w_index = t_index + w_index
mrope_section = [mrope_section[0], mrope_section[1]+mrope_section[2]]
mrope_section = mrope_section * 2 # temporal, height, weight e.g. in VideoRoPE [16, 48] -> [16, 48, 16, 48] -> head_size = 128
mrope_section = mrope_section[::-1] # MODIFICATION 1: low-frequency temporal allocation (interpolation) [48, 16, 48, 16]
result_cos, result_sin = [], []
index=0
for i, section in enumerate(mrope_section):
  if i%2 == 0:
    for j in range(section): # MODIFICATION 2: diagonal layout
      row = 1 if j % 2 == 0 else 2
      result_sin.append(sin_pos[row, ..., index:index+1])
      result_cos.append(cos_pos[row, ..., index:index+1])
  else:
    result_sin.append(sin_pos[0, ..., index:index+section])
    result_cos.append(cos_pos[0, ..., index:index+section])
sin_pos = torch.cat([m[i%3] for i, m in enumerate(sin_pos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim) # unsqueeze for broadcast on head
cos_pos = torch.cat([m[i%3] for i, m in enumerate(cos_pos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim) # BSZ, KEY_LENGTH, HEAD_SIZE: [[h1,h2,h3,h4,h5,h6],[w1,w2,w3,w4,w5,w6],[t1,t2,t3,t4,t5,t6]] -> [h1,w2,h3,w4,t5,t6]
#############################################
```

**Reference**

> - [1] [Transformer升级之路：17、多模态位置编码的简单思考](https://spaces.ac.cn/archives/10040) Jianlin Su's blog
> - [1] [Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) Alibaba
> - [2] [“闭门造车”之多模态思路浅谈（三）：位置编码](https://kexue.fm/archives/10352) Jianlin Su's blog
> - [3] [VideoRoPE: What Makes for Good Video Rotary Position Embedding?](https://arxiv.org/abs/2502.05173) Shanghai AI Lab
