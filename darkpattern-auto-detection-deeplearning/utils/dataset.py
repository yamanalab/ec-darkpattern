from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset


class DarkpatternDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        text_to_tensor: Callable[[str], torch.Tensor],
    ) -> None:
        self.texts: List[str] = texts
        self.labels: torch.Tensor = torch.tensor(labels)
        self.text_to_tensor = text_to_tensor

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [text_tensor, label_tensor]
        text_encoded = self.text_to_tensor(str(self.texts[index]))
        return text_encoded, self.labels[index]

    def __len__(self) -> int:
        return len(self.texts)
