import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

def image_to_tensor(pil_image, device):
    from torchvision import transforms
    transform = transforms.ToTensor()
    return transform(pil_image).unsqueeze(0).to(device)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = torch.clamp(tensor, 0, 1)
    np_array = tensor.permute(1, 2, 0).numpy()
    np_array = (np_array * 255).astype(np.uint8)
    return Image.fromarray(np_array)

def safe_bicubic_upscale(tensor, scale_factor=2):
    return F.interpolate(tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False)

def calculate_psnr(img1, img2):
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    img1_np = np.array(img1).astype(np.float64)
    img2_np = np.array(img2).astype(np.float64)
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))
