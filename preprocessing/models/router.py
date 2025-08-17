from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F

from models.owkin.utils.mlp import MLP


class MILRouter(nn.Module):
    def __init__(self, model: nn.Module, drop_p: float = 0.1, temperature: float = 1.0):
        super(MILRouter, self).__init__()
        self.drop = nn.Dropout(drop_p)
        self.gate = model
        self.temperature = temperature
        self.initialize_weights()

    def forward(self, x, mask=None):
        """
        Expects input of shape:
            x: B, N, D
            mask: B, N, 1 (optional)
        Where:
          B: Batch size
          N: Number of patch features
          D: The sum of all expert feature dimensions

        Example:
        We have an MoE that works with chief (D: 768), uni (D: 1024), and resnet50 (D: 2048) features.
        The expected input shape would be:
          B, N, 3840

        The output will be a probability distribution over the number of experts.
        The actual processing of the combined features is done by the `self.model` module.
        Examples of potential models are:
          - ABMIL,
          - DSMILWrap
          - HIPTMILWrap
          - MeanPool
          - TransMIL
        """
        x = self.drop(x)
        logits = self.gate(x, mask)
        probabilities = F.softmax(logits / self.temperature, dim=1)
        return probabilities, logits

    def initialize_weights(self):
        """
        Init weights of the model with Xavier initialization
        and biases to zero.
        """
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Router(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        temperature: float = 1.0,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[nn.Module] = nn.Sigmoid(),
    ):
        super(Router, self).__init__()
        self.gate = MLP(
            in_features=in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.temperature = temperature
        self.initialize_weights()

    def forward(self, x):
        """
        Expects input of shape:
          B, D
        Where:
          B: Batch size
          D: The sum of all expert feature dimensions

        Example:
        We have an MoE that works with chief (D: 768), uni (D: 1024), and resnet50 (D: 2048) features.
        The expected input shape would be:
          B, 3840

        The output will be a probability distribution over the number of experts.
        """
        # B, D = x.shape
        logits = self.gate(x)
        probabilities = F.softmax(logits / self.temperature, dim=1)
        return probabilities, logits

    def initialize_weights(self):
        """
        Init weights of the model with Xavier initialization
        and biases to zero.
        """
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
