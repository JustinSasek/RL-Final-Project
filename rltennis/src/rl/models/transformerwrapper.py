from dataclasses import field

import torch
from torch import nn

from .transformer import Transformer
from .util import ModelInterface, nn_dataclass


@nn_dataclass
class SingleVectorWrapper(ModelInterface):
    transformer: Transformer = field(default_factory=Transformer)
    input_size: int = 1
    output_size: int = 1
    dropout = 0.1

    def __post_init__(self):
        self.fc_in = nn.Linear(self.input_size, self.transformer.d_model)
        self.transformer = self.transformer
        self.act_out = nn.LeakyReLU()
        self.dropout_out = nn.Dropout(self.dropout)
        self.fc_out = nn.Linear(self.transformer.d_model, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.transformer(x)
        x = self.act_out(x)
        x = self.dropout_out(x)
        x = self.fc_out(x)
        return x
