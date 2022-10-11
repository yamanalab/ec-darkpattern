import torch
from torch import nn
from transformers import BertModel


class DarkpatternClassifierBert(nn.Module):
    def __init__(
        self,
        pretrained: str = "bert-base-uncased",
        dropout_rate: float = 0.1,
        output_layer: nn.Linear = nn.Linear(in_features=768, out_features=2),
    ):
        super(DarkpatternClassifierBert, self).__init__()
        self.__pretrained = pretrained
        self.bert = BertModel.from_pretrained(self.__pretrained)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = output_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)[0]  # [batch_size,seq_len,768]
        x = x[:, 0, :]  # [batch_size,1,768]
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
