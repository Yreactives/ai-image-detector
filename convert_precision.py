import torch

# Load local model file
checkpoint = torch.load("model.pth", map_location="cpu")

# Extract state dict if contained inside a dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Save ONLY the raw state dict (removes optimizer/epoch overhead)
torch.save({"model_state_dict": state_dict}, "model_optimized.pth")