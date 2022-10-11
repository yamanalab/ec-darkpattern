import torch
from transformers import PreTrainedTokenizer


def tensor_to_text(tensor: torch.Tensor, tokenizer: PreTrainedTokenizer) -> str:
    """
    Convert tensor to text.
    """
    return tokenizer.decode(tensor)


def text_to_tensor(
    text: str, tokenizer: PreTrainedTokenizer, max_length: int
) -> torch.Tensor:
    """
    Convert text to tensor.
    """
    return torch.Tensor(
        tokenizer.encode(text, max_length=max_length, pad_to_max_length=True)
    ).to(torch.long)
