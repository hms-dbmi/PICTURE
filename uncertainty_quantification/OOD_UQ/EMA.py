import torch
from torch import nn
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
from typing import Literal
# for type hint
from torch import Tensor
class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float, starts_at: int =0, updates_every:int=1, ens_mode:Literal['avg','exp']='avg'):
        super().__init__()
        self.decay = decay
        self.model = model
        self.ens_mode = ens_mode
        self.starts_at = starts_at
        self.updates_every = updates_every
        self.shadow = deepcopy(self.model)
        for param in self.shadow.parameters():
            param.detach_()
        self.count=0
        self.ens_count=0.
    @torch.no_grad()
    def update(self):
        if self.count < self.starts_at:
            self.count = self.count + 1
            return
        elif self.count % self.updates_every !=0 :
            self.count = self.count + 1
            return
        self.count = self.count + 1
        self.ens_count = self.ens_count + 1
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return
        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())
        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()
        ## determining decay
        if self.ens_count == 1:
            decay=0
        elif self.ens_mode == 'avg':
            decay = 1. - 1/self.ens_count
        else:
            decay = self.decay
        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - decay) * (shadow_params[name] - param))
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())
        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()
        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)
    def forward(self, *args) -> Tensor:
        if self.training:
            return self.model(*args)
        else:
            return self.shadow(*args)
