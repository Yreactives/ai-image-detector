import os
import io
import gc
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from model import AI_Image_Detector

# --- MEMORY OPTIMIZATION FOR RENDER (512MB RAM LIMIT) ---
# Restrict PyTorch to single-thread execution to prevent memory/CPU spikes
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI(title="AI Image Detector")

# Enable CORS for cross-origin requests from Streamlit Community Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

size = (128, 128)
transform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

device = torch.device("cpu")
MODEL_PATH = "model_optimized.pth"

# Download model weights from Hugging Face if not available locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from Hugging Face Hub...")
    MODEL_PATH = hf_hub_download(
        repo_id="Yreactives/ai-image-detector-weights",
        filename="model_optimized.pth"
    )

# Load state dict without retaining autograd memory allocations
with torch.no_grad():
    checkpoint = torch.load(MODEL_PATH, map_location=device, mmap=True)
    model = AI_Image_Detector(size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

# Clear raw checkpoint dictionary from RAM after state_dict extraction
del checkpoint
gc.collect()


@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Image Detector API is running. Go to /docs for documentation."}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor_image = transform(pil_image).unsqueeze(0).to(device)

        # Run light-weight inference mode (disables autograd graph entirely)
        with torch.inference_mode():
            output = model(tensor_image)
            probabilities = F.softmax(output, dim=1)[0]

        fake_score = probabilities[0].item()
        real_score = probabilities[1].item()
        label = "AI Generated" if fake_score > real_score else "Real Image"
        confidence = max(fake_score, real_score) * 100

        # Free image Tensors from memory immediately
        del image_bytes, pil_image, tensor_image, output, probabilities
        gc.collect()

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "scores": {
                "ai_generated": round(fake_score, 4),
                "real_image": round(real_score, 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")