from src.models.slides_module import SlidesModule, ViT8
from torch.optim import AdamW
import torch
import torchvision.transforms as T
import torch
import numpy as np
import gradio as gr


if __name__ == '__main__':
    def to_heatmap(attentions, grid_size: int = 6):
        """Converts a list of attention maps to a single heatmap, by averaging the attention maps.
        The returned heatmap is of shape (grid_size, grid_size).

        Args:
            attentions: A list of attention maps.
            grid_size (int, optional): Desired grid size. Defaults to 6.

        Returns:
            Tensor: Heatmap of shape (grid_size, grid_size).
        """
        heatmap = T.Resize((grid_size,grid_size))(torch.Tensor(attentions.mean(axis=(1)).reshape((attentions.shape[0],1,60,60))))
        return heatmap

    def to_coordinates(heatmap, kth: int = 10):
        """Converts a heatmap to a list of coordinates, by selecting the kth highest values.
    
        
        Args:
            heatmap (Tensor): Heatmap of shape (grid_size, grid_size).
            kth (int, optional): Number of coordinates to return. Defaults to 10.
            
        Returns:
            List: List of the kth coordinates.
        """
        
        grid_size = heatmap.shape[-1]
        idxs = np.argsort(heatmap.T.ravel())[::-1][:kth]

        rows, cols = idxs//grid_size, idxs%grid_size
        return list(zip(rows, cols))


    def greet(image, model_type, grid_size : int, kth : int):
        """Function to be used by gradio. It takes an image as input, and returns the coordinates of the kth highest values in the heatmap.

        Args:
            image (np.array): Image of shape (H, W, 3). 
            model_type (str): Type of model to use. Can be either 'brain', 'cellphone', 'brain-blurred', 'whole-slide'.
            grid_size (int): Desired grid size.
            kth (int): Number of coordinates to return.

        Returns:
            Heatmap_high (np.array): Average attention map of the model.
            Heatmap (np.array): Heatmap of the model.
            Coordinates (list): List of the kth coordinates.
        """
        
        model = ViT8(type=model_type).to('cuda:0')
        inp = T.ToTensor()(image).unsqueeze(0)
        inp = T.Resize((224,224))(inp)
        
        
        attentions =  model.visualize_attention(inp.to('cuda:0'), grid=False)
        
        heatmap_high = T.Resize((200,200), interpolation=T.InterpolationMode.NEAREST)(attentions).squeeze().mean(0).numpy()
        heatmap_high = (heatmap_high)/heatmap_high.max()
        
        heatmap = to_heatmap(attentions, grid_size)
        coordinates = to_coordinates(heatmap.numpy(), kth=kth)
        heatmap = T.Resize((200,200), interpolation=T.InterpolationMode.NEAREST)(heatmap)/heatmap.max()
        heatmap = heatmap.numpy().squeeze()
        
        return heatmap_high, heatmap, coordinates

    demo = gr.Interface(fn=greet, inputs=[gr.Image(value="data/download.png"), gr.Radio(choices=["brain", "cellphone", "brain-blurred", "whole-slide"], value = "cellphone"), gr.Slider(minimum= 2, maximum = 60, value=16,step = 1), gr.Slider(minimum= 1, maximum = 36, value=5,step = 1)], outputs=["image","image", "text"])

    _, _, url = demo.launch(share=True, height = 700)   
    
    demo.launch()