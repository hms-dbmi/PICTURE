import argparse
import enum
import os
from typing import Optional

import requests
import timm
import torch
import torch.nn as nn
import torchvision
from huggingface_hub import hf_hub_download, login
from timm.models.vision_transformer import VisionTransformer
from torch.hub import load_state_dict_from_url
from transformers import CLIPModel, SwinModel, ViTModel
# from models import gigapath 
# from models import gigapath
from models.gigapath.slide_encoder import create_model as create_gigapath_WSI
from models.CHIEF.CHIEF import CHIEF

class ModelType(enum.Enum):
    CTRANS = 1
    LUNIT = 2
    RESNET50 = 3
    UNI = 4
    SWIN224 = 5
    PHIKON = 6
    CHIEF = 7
    PLIP = 8
    GIGAPATH = 9
    CIGAR = 10
    VIRCHOW = 11
    VIRCHOW2 = 12
    GIGAPATH_WSI = 13
    CHIEF_WSI = 14
    NONE = None

    def __str__(self):
        return self.name


def parse_model_type(models_str):
    models = models_str.upper().split(",")
    try:
        return [ModelType[model] for model in models]
    except KeyError as e:
        raise argparse.ArgumentTypeError(f"Invalid model name: {e}")


def get_model(args, model: str) -> nn.Module:
    m_type = ModelType[model.upper()]
    if m_type == ModelType.CTRANS:
        model = get_ctrans()
    elif m_type == ModelType.LUNIT:
        model = get_lunit()
    elif m_type == ModelType.RESNET50:
        model = get_resnet50()
    elif m_type == ModelType.UNI:
        model = get_uni(args.hf_token)
    elif m_type == ModelType.SWIN224:
        model = get_swin_224()
    elif m_type == ModelType.PHIKON:
        model = get_phikon()
    elif m_type == ModelType.CHIEF:
        model = get_chief()
    elif m_type == ModelType.PLIP:
        model = get_plip()
    elif m_type == ModelType.GIGAPATH:
        model = get_gigapath(args.hf_token)
    elif m_type == ModelType.CIGAR:
        model = get_cigar()
    elif m_type == ModelType.VIRCHOW:
        model = get_virchow(args.hf_token, v=1)
    elif m_type == ModelType.VIRCHOW2:
        model = get_virchow(args.hf_token, v=2)
    elif m_type == ModelType.GIGAPATH_WSI:
        model = get_gigapath_WSI(args.hf_token)
    elif m_type == ModelType.CHIEF_WSI:
        model = get_chief_wsi()
    else:
        raise Exception("Invalid model type")
    return model


def get_resnet50():
    model = timm.create_model("resnet50", pretrained=True)
    model.fc = nn.Identity()
    return model


def get_phikon():
    # =============== Owkin Phikon ===============
    # https://github.com/owkin/HistoSSLscaling
    # =============== Owkin Phikon ===============
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = ViTModel.from_pretrained(
                "owkin/phikon", add_pooling_layer=True
            )

        def forward(self, x):
            x = {"pixel_values": x}
            features = self.model(**x).last_hidden_state[:, 0, :]
            return features

    return Model()


def get_lunit():
    # =============== Lunit Dino ===============
    # https://github.com/lunit-io/benchmark-ssl-pathology
    # =============== Lunit Dino ===============
    url_prefix = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    key = "DINO_p16"
    pretrained_url = f"{url_prefix}/{model_zoo_registry.get(key)}"
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0
    )
    state_dict = load_state_dict_from_url(pretrained_url, progress=False)
    model.load_state_dict(state_dict)
    return model


def get_uni(hf_token: Optional[str] = None):
    """
    =============== UNI ===============
    https://huggingface.co/MahmoodLab/UNI
    Warning: this model requires an access request to the model owner.
    =============== UNI ===============
    """
    if hf_token:
        login(token=hf_token)

    model_dir = os.path.expanduser("~/.models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = hf_hub_download(
        "MahmoodLab/UNI",
        filename="pytorch_model.bin",
        cache_dir=model_dir,
        force_download=False,
    )
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    return model


def get_swin_224():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = SwinModel.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224"
            )

        def forward(self, x):
            return self.model(x).pooler_output

    model = Model()
    return model


def get_ctrans():
    # =============== CTransPath ===============
    # requires the ctranspath.pth file to be located
    # at CTRANS_PTH environment variable
    # =============== CTransPath ===============
    model_pth = os.environ.get("CTRANS_PTH", "ctranspath.pth")

    import oldtimm
    import torch.nn as nn
    from oldtimm.models.layers.helpers import to_2tuple

    class ConvStem(nn.Module):

        def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
        ):
            super().__init__()

            assert patch_size == 4
            assert embed_dim % 8 == 0

            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            self.img_size = img_size
            self.patch_size = patch_size
            self.grid_size = (
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.flatten = flatten

            stem = []
            input_dim, output_dim = 3, embed_dim // 8
            for l in range(2):
                stem.append(
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
                stem.append(nn.BatchNorm2d(output_dim))
                stem.append(nn.ReLU(inplace=True))
                input_dim = output_dim
                output_dim *= 2
            stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
            self.proj = nn.Sequential(*stem)

            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        def forward(self, x):
            B, C, H, W = x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)
            return x

    model = oldtimm.create_model(
        "swin_tiny_patch4_window7_224", embed_layer=ConvStem, pretrained=False
    )
    model.head = nn.Identity()
    td = torch.load(model_pth)
    model.load_state_dict(td["model"], strict=True)
    model = model.eval()
    return model


def get_chief():
    # =============== Chief ===============
    # requires the chief.pth file to be located
    # at CHIEF_PTH environment variable
    # =============== Chief ===============
    model_pth = os.environ.get("CHIEF_PTH", "chief.pth")

    import oldtimm
    import torch.nn as nn
    from oldtimm.models.layers.helpers import to_2tuple

    class ConvStem(nn.Module):
        def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
        ):
            super().__init__()

            assert patch_size == 4
            assert embed_dim % 8 == 0

            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            self.img_size = img_size
            self.patch_size = patch_size
            self.grid_size = (
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.flatten = flatten

            stem = []
            input_dim, output_dim = 3, embed_dim // 8
            for l in range(2):
                stem.append(
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
                stem.append(nn.BatchNorm2d(output_dim))
                stem.append(nn.ReLU(inplace=True))
                input_dim = output_dim
                output_dim *= 2
            stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
            self.proj = nn.Sequential(*stem)

            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        def forward(self, x):
            B, C, H, W = x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)
            return x

    model = oldtimm.create_model(
        "swin_tiny_patch4_window7_224", embed_layer=ConvStem, pretrained=False
    )
    model.head = nn.Identity()
    td = torch.load(model_pth)
    model.load_state_dict(td["model"], strict=True)
    model = model.eval()
    return model

def get_chief_wsi():
    model_weight_path = '/n/data2/hms/dbmi/kyu/lab/NCKU/CHIEF_model_weight/CHIEF_pretraining.pth'
        
    model = CHIEF(size_arg="small", dropout=True, n_classes=2)

    # td = torch.load(r'./model_weight/CHIEF_pretraining.pth')
    td = torch.load(model_weight_path)
    model.load_state_dict(td, strict=True)
    model.eval()
    return model


def get_gigapath(hf_token: Optional[str] = None):
    """
    =============== Gigapath ===============
    Requires a huggingface token to download the model

    See here:
        https://github.com/prov-gigapath/prov-gigapath
    =============== Gigapath ===============
    """
    if hf_token:
        login(token=hf_token)
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model.eval()
    return model

def get_gigapath_WSI(hf_token: Optional[str] = None):
    # slide_encoder = create_gigapath_WSI("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
    slide_encoder = create_gigapath_WSI("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=True)
    slide_encoder.eval()

    return slide_encoder



def get_plip():
    """
    =============== PLIP ===============
    Pathology Language and Image Pre-Training (PLIP)
    We only utilize the vision backbone here.

    See here:
        https://github.com/PathologyFoundation/plip
    =============== PLIP ===============
    """

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = CLIPModel.from_pretrained("vinid/plip").vision_model

        def forward(self, x):
            return self.model(x).pooler_output

    return Model()


def get_cigar(ckpt_path: str = "~/.models"):
    """
    =============== CIGAR ===============
    See here:
        https://github.com/ozanciga/self-supervised-histopathology
    =============== CIGAR ===============
    """
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_url = "https://github.com/ozanciga/self-supervised-histopathology/releases/download/nativetenpercent/pytorchnative_tenpercent_resnet18.ckpt"
    ckpt_path = os.path.join(ckpt_path, "pytorchnative_tenpercent_resnet18.ckpt")
    download_checkpoint(ckpt_url, ckpt_path)
    model = torchvision.models.__dict__["resnet18"](weights=None)

    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print("No weight could be loaded..")
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(
            key
        )
    model = load_model_weights(model, state_dict)
    model.fc = nn.Identity()
    return model


def get_virchow(hf_token: Optional[str] = None, v=1):
    """
    =============== VIRCHOV ===============
    A foundation model for clinical-grade computational pathology
    and rare cancers detection

    Model Parameters: 632M

    See here:
        https://huggingface.co/paige-ai/Virchow
        https://huggingface.co/paige-ai/Virchow2
        https://www.nature.com/articles/s41591-024-03141-0
    =============== VIRCHOV ===============
    """
    from timm.layers import SwiGLUPacked

    if hf_token:
        login(token=hf_token)

    class Model(nn.Module):
        def __init__(self, model_name="hf-hub:paige-ai/Virchow"):
            super(Model, self).__init__()
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )

        def forward(self, x):
            # x: [B, 3, 224, 224]
            output = self.model(x)  # size: B x 257 x 1280
            class_token = output[:, 0]  # size: B x 1280
            patch_tokens = output[:, 1:]  # size: B x 256 x 1280
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: B x 2560
            return embedding
    if v == 2:
        return Model(model_name=f"hf-hub:paige-ai/Virchow{v}")
    else:
        return Model()

def download_checkpoint(ckpt_url, ckpt_path):
    if not os.path.exists(ckpt_path):
        print("Downloading checkpoint...")
        response = requests.get(ckpt_url, allow_redirects=True)
        with open(ckpt_path, "wb") as f:
            f.write(response.content)
    else:
        print("Checkpoint exists.")
