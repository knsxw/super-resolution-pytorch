import os
from PIL import Image

import torch
from models.working_srcnn import WorkingSRCNN
from models.working_espcn import WorkingESPCN
from models.minimal_sr import MinimalSR
from utils.image_utils import image_to_tensor, tensor_to_image, safe_bicubic_upscale, calculate_psnr

class ReliableSuperResolution:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.models = {
            'Minimal_SR': MinimalSR(),
            'Working_SRCNN': WorkingSRCNN(),
            'Working_ESPCN': WorkingESPCN(scale_factor=2)
        }
        
        for _, model in self.models.items():
            model.to(self.device)
            model.eval()
    
    def load_image_safely(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert('RGB')
    
    def process_with_safety(self, model_name, input_tensor):
        model = self.models[model_name]
        with torch.no_grad():
            if 'ESPCN' in model_name:
                output = model(input_tensor)
            else:
                upscaled = safe_bicubic_upscale(input_tensor, 2)
                output = model(upscaled)
                if 'SRCNN' in model_name:
                    output += upscaled
            return torch.clamp(output, 0, 1)
    
    def process_image_safely(self, image_path, scale_factor=2):
        original_img = self.load_image_safely(image_path)
        lr_img = original_img.resize((original_img.width//scale_factor, original_img.height//scale_factor), Image.Resampling.BICUBIC)
        lr_tensor = image_to_tensor(lr_img, self.device)
        
        output_dir = "muaythai_super_resolution_results"
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        psnr_values = {}
        
        # Bicubic baseline
        bicubic_tensor = safe_bicubic_upscale(lr_tensor, scale_factor)
        bicubic_img = tensor_to_image(bicubic_tensor).resize(original_img.size, Image.Resampling.LANCZOS)
        results['Bicubic'] = bicubic_img
        psnr_values['Bicubic'] = calculate_psnr(original_img, bicubic_img)
        
        for model_name in self.models.keys():
            output_tensor = self.process_with_safety(model_name, lr_tensor)
            output_img = tensor_to_image(output_tensor)
            if output_img.size != original_img.size:
                output_img = output_img.resize(original_img.size, Image.Resampling.LANCZOS)
            results[model_name] = output_img
            psnr_values[model_name] = calculate_psnr(original_img, output_img)
            output_img.save(os.path.join(output_dir, f"{model_name}_muaythai.png"))
        
        return results, psnr_values
