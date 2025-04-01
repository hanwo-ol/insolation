# modules/sobel.py
import torch
import torch.nn.functional as F

def sobel_edges(image_tensor):
    """Calculates Sobel edges for a PyTorch image tensor.

    Args:
        image_tensor: A PyTorch tensor representing a grayscale image (shape: [C, H, W]).

    Returns:
        A PyTorch tensor representing the gradient magnitude (shape: [H, W]).
    """
    # Define Sobel kernels as PyTorch tensors
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Ensure the image tensor has the correct shape [1, 1, H, W] for conv2d
    if image_tensor.ndim == 3:  # [C, H, W]
        image_tensor = image_tensor.unsqueeze(0)  #[1, C, H, W]
    if image_tensor.shape[1] != 1:
        image_tensor = image_tensor[:, :1, :, :]  # Keep only the first channel


    # Apply convolution with Sobel kernels
    grad_x = F.conv2d(image_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(image_tensor, sobel_y, padding=1)

    # Calculate gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    return gradient_magnitude.squeeze() # Remove batch and channel dimensions: [H, W]