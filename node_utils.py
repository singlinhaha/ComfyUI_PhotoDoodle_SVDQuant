import torch
import numpy as np
from PIL import Image


def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2tensor(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry