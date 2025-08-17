import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2



class VITAttentionGradRollout:
    """ Performs gradient rollout on the model given an input image and a category index.
    """
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_full_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, batch, category_index, quantile = 0):
        self.model.zero_grad()
        self.attentions = []
        self.attention_gradients = []
        output = self.model.step(batch)[2]
        category_mask = torch.zeros(output.size()).to(batch["image"].device)
        category_mask[:, category_index] = 1
        loss = (output).sum()
        loss.backward()

        return self.grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio, quantile)
    
    def visualize_attention(self, batch, category_index, quantile = 0):
        heatmap = self(batch, category_index, quantile)
        heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = torch.nn.functional.interpolate(heatmap, size=(224,224), mode='nearest')
        heatmap = heatmap.squeeze(0).squeeze(0).numpy()
        return heatmap
        
    def grad_rollout(self, attentions, gradients, discard_ratio, quantile = 0):
        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            for attention, grad in zip(attentions, gradients):                
                weights = grad
                attention_heads_fused = (attention*weights).mean(axis=1)
                attention_heads_fused[attention_heads_fused < 0] = 0

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                #indices = indices[indices != 0]
                flat[0, indices] = 0
                
                attention_heads_fused = flat.reshape(attention_heads_fused.shape)

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1,keepdim=True)
                result = torch.matmul(a, result)
        
        # and the image patches
        mask = result[0, 0 , 1 :]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        
        # If we want to use a quantile for threshold on mask
        if quantile > 0:
            quantile = np.quantile(mask, quantile)
            # If mask value is greater than quantile, we replace it by quantile with np where
            mask = np.where(mask >= quantile, mask.min(), mask)
        
        return mask    