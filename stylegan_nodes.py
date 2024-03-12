import os
from comfy_extras.chainner_models import model_loading
from comfy import model_management
import torch
import comfy.utils
import folder_paths

from comfy.model_management import get_torch_device
import pickle # TODO convert to safetensors

folder_paths.folder_names_and_paths["stylegan"] = ([os.path.join(folder_paths.models_dir, "stylegan")], [".pkl"])


class StyleGANModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("stylegan"), ),
                             }}
    RETURN_TYPES = ("STYLEGAN",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"
    
    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("stylegan", model_name)
        # sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        with open(path, 'rb') as f:
            G = pickle.load(model_path)['G_ema']

        device = get_torch_device()
        # if device is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        out = G.eval().to(device)

        return (out, )

NODE_CLASS_MAPPINGS = {
    "StyleGANModelLoader": StyleGANModelLoader,
}

