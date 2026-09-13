from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import io
from model import AI_Image_Detector

app = FastAPI(title="AI Image Detector")
size = (128,128)
transform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("model.pth", map_location=device)
model = AI_Image_Detector(size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Image Detector API is running. Go to /docs for documentation."}
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    imageBytes = await image.read()
    image = Image.open(io.BytesIO(imageBytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]

    fake_score = probabilities[0].item()
    real_score = probabilities[1].item()
    label = "AI Generated" if fake_score > real_score else "Real Image"
    confidence = max(fake_score, real_score) * 100  # Updated to percentage scale

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "scores": {
            "ai_generated": round(fake_score, 4),
            "real_image": round(real_score, 4)
        }
    }