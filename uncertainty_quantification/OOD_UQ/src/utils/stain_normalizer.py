import torchstain
import torch
from typing import Callable

def stain_normalizer(to_fit: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """

    Args:
        to_fit (torch.Tensor): _description_

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: _description_
    """

    torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    torch_normalizer.fit(to_fit*255)

    def stain_normalizer_(image: torch.Tensor):
        if image.std() < 5e-2:
            return image
        return (torch_normalizer.normalize(I=image*255, stains=True)[0].mT/255).permute(1,0,2)

    return stain_normalizer_