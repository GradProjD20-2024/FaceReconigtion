import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

def transform_image(image: Image.Image, size: tuple = (112, 112)) -> torch.Tensor:
    """
    Transform a PIL Image for face recognition model input.
    
    Args:
        image (PIL.Image): Input image
        size (tuple): Target size for the image (default: 112x112 for most face recognition models)
    
    Returns:
        torch.Tensor: Transformed image tensor
    """
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(size),  # Resize to model's expected input size
        transforms.ToTensor(),    # Convert to tensor (0-1 range)
        transforms.Normalize(     # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)

# Optional: Additional utility functions that might be useful

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor back to PIL Image (useful for debugging).
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        PIL.Image: Converted image
    """
    # Undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    
    # Convert to PIL Image
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()
    
    return transforms.ToPILImage()(tensor)

def preprocess_bytes_image(image_bytes: bytes, size: tuple = (112, 112)) -> torch.Tensor:
    """
    Preprocess image from bytes format.
    
    Args:
        image_bytes (bytes): Image in bytes format
        size (tuple): Target size for the image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transformation
    return transform_image(image, size)

def center_crop_face(image: Image.Image, margin: float = 0.2) -> Image.Image:
    """
    Center crop the face image with a margin.
    
    Args:
        image (PIL.Image): Input image
        margin (float): Margin to add around the face (percentage of image size)
        
    Returns:
        PIL.Image: Cropped image
    """
    width, height = image.size
    
    # Calculate crop dimensions with margin
    crop_size = min(width, height)
    margin_pixels = int(crop_size * margin)
    
    # Calculate crop coordinates
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # Add margin
    left = max(0, left - margin_pixels)
    top = max(0, top - margin_pixels)
    right = min(width, right + margin_pixels)
    bottom = min(height, bottom + margin_pixels)
    
    return image.crop((left, top, right, bottom))