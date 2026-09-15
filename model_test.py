import torch

from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import AI_Image_Detector

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size = (128, 128)
# Transform (must match training)
transform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Load model
checkpoint = torch.load("model.pth", map_location=device)

model = AI_Image_Detector(size).to(device)
#model = torchvision.models.resnet50(pretrained=True).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return pred_class, confidence


# Folder containing images
image_folder = Path("testimage")

# Supported extensions
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total = 0
ai_count = 0
human_count = 0

print("-" * 70)

for image_path in image_folder.iterdir():

    if image_path.suffix.lower() not in extensions:
        continue

    pred_class, confidence = predict_image(str(image_path))

    prediction = "AI" if pred_class == 0 else "Human"

    if pred_class == 0:
        ai_count += 1
    else:
        human_count += 1

    total += 1

    print(
        f"{image_path.name:<30} "
        f"{prediction:<10} "
        f"{confidence * 100:.2f}%"
    )

print("-" * 70)
print(f"Total Images : {total}")
print(f"AI Images    : {ai_count}")
print(f"Human Images : {human_count}")