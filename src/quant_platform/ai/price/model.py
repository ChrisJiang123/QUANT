"""Optional PyTorch Price Encoder with independent 14d/28d regression heads."""
import math

import torch
from torch import nn


class PriceEncoder(nn.Module):
    """Produces a price representation; reserved for later SIDGTN fusion."""

    def __init__(self, config, features=5):
        super().__init__()
        self.projection = nn.Linear(features, config.d_model)
        self.asset_embedding = nn.Embedding(2, config.d_model)
        position = torch.arange(config.window, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model))
        encoding = torch.zeros(config.window, config.d_model)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position*divisor), torch.cos(position*divisor)
        self.register_buffer("position", encoding)
        layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.heads,
                    dim_feedforward=config.feedforward, dropout=config.dropout, batch_first=True,
                    activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, prices, asset):
        # All 90 bars precede decision time. Bidirectional attention inside this
        # already observed window is allowed; no future window enters the encoder.
        embedded = self.projection(prices) + self.position[None] + self.asset_embedding(asset)[:, None]
        return self.norm(self.transformer(embedded)[:, -1])


class PriceReturnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.price_encoder = PriceEncoder(config)
        self.return_14d = nn.Linear(config.d_model, 1)
        self.return_28d = nn.Linear(config.d_model, 1)
        # TransformerEncoder clones layers. Explicitly initialize them separately.
        for name, parameter in self.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("weight"):
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(self, prices, asset):
        representation = self.price_encoder(prices, asset)
        return torch.cat((self.return_14d(representation), self.return_28d(representation)), dim=1)
