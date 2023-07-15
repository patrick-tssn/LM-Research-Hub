# Constrained Generation method for LM

Table of Content

- [Common method](#common-method)
- [Trie + Beam search](#trie--beam-search)

## Common method

- [ ] generation method: sample/greedy search/beam search + constrained vocabulary

## Trie + Beam search

paper [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904 "https://arxiv.org/abs/2010.00904") (ICLR 2021) proposed the constrained sample method (i.e, trie + beam search)

see `generate.ipynb` for details

reference:  [GENRE](https://github.com/facebookresearch/GENRE/blob/main/genre)

Key algorithm

*NOTE: algorithm will fail while all allowd tokens have score of `inf`*

```python
# from transformers.generation.logits_process
class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent)] = 0

        return scores + mask
```
