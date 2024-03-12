from __future__ import annotations
from typing import Union, List
import os
import glob
from pathlib import Path
import numpy as np
# import torch
# import torch.nn as nn

# from comfy_extras.chainner_models import model_loading
# from comfy import model_management
# import comfy.utils
from comfy.model_management import get_torch_device
import folder_paths
folder_paths.folder_names_and_paths["stylegan"] = ([os.path.join(folder_paths.models_dir, "stylegan")], [".pkl"])

from pathlib import Path
import sys

current_path = Path(__file__).parent
tu_path = current_path / 'torch_utils'
dnnlib_path = current_path / 'dnnlib'
sys.path.append(tu_path)
sys.path.append(dnnlib_path)
print(sys.path)

import pickle
import torch
# import torch.nn as nn
import torch_utils
import dnnlib

class StyleGANModelLoader:
    # def __init__(self):

    @classmethod
    def model_list(cls):
        # model_dir = Path(__file__).parent / "models"
        # print(model_dir)
        # files = glob.glob(str(model_dir / "*.pkl"))
        # return [os.path.basename(file) for file in sorted(files, key=lambda file: (os.stat(file).st_mtime, file), reverse=True)]
        return folder_paths.get_filename_list("stylegan")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (s.model_list(), ),
                             }}
    RETURN_TYPES = ("STYLEGAN",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"
    
    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("stylegan", model_name)
        # sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        with open(model_path, 'rb') as f:
            G = pickle.load(f)['G_ema']

        device = get_torch_device()
        # if device is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        out = G.eval().to(device)

        return (out, )

class StyleGANGenerator:
    def __init__(self):
        self.G = None
        self.device = get_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STYLEGAN", ),
                "seed": ("INT", {"default": 0})
            },
            "optional": {
                "psi": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"

    CATEGORY = "stylegan"
    
    def generate_image(self, model, seed, psi):
        self.G = model
        w = self.get_w_from_seed(seed, psi)
        return self.w_to_img(w)[0]

    def w_to_img(self, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const') -> np.ndarray:
        """
        Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
        returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
        """
        assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
        if len(dlatents.shape) == 2:
            dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
        try:
            synth_image = self.G.synthesis(dlatents, noise_mode=noise_mode)
        except:
            synth_image = self.G.synthesis(dlatents, noise_mode=noise_mode, force_fp32=True)
        
        synth_image = (synth_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return synth_image.cpu().numpy()

    def random_z_dim(self, seed) -> np.ndarray:
        z = np.random.RandomState(seed).randn(1, self.G.z_dim) 
        # if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        # z = torch.tensor(z).float().cpu().numpy() # convert to float32 for mac
        z = torch.tensor(z, dtype=torch.float32).cpu().numpy()

        return z

    def get_w_from_seed(self, seed: int, psi: float) -> torch.Tensor:
        """Get the dlatent from a random seed, using the truncation trick (this could be optional)"""
        z = self.random_z_dim(seed)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        w = w_avg + (w - w_avg) * psi

        return w

    def get_w_from_mean_z(self, psi: float) -> torch.Tensor:
        """Get the dlatent from the mean z space"""
        w = self.G.mapping(torch.zeros((1, self.G.z_dim)).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        w = w_avg + (w - w_avg) * psi

        return w

    def get_w_from_mean_w(self, seed: int, psi: float) -> torch.Tensor:
        """Get the dlatent of the mean w space"""
        w = self.G.mapping.w_avg.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1).to(self.device)
        return w

NODE_CLASS_MAPPINGS = {
    "StyleGAN ModelLoader": StyleGANModelLoader,
    "StyleGAN Generator": StyleGANGenerator,
}

