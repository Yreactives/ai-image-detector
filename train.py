import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AI_Image_Detector
from PIL import Image
import os
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA').convert('RGB')
if __name__ == '__main__':
    custom_size = (128, 128)
    transform = transforms.Compose([
        transforms.Resize(size=custom_size),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        #transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),

    ])
    transform_test = transforms.Compose([

        transforms.Resize(size=custom_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    batch_size = 64
    lr = 1e-4
    epochs = 1000
    dataset_source = "dataset/manual_data"
    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_source, "train"), transform=transform, loader=pil_loader)
    test_dataset = datasets.ImageFolder(root=os.path.join(dataset_source, "val"), transform=transform_test, loader=pil_loader)
    print("Class mapping:", train_dataset.class_to_idx)
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AI_Image_Detector(size=custom_size).to(device)
    #model = torchvision.models.resnet50(pretrained=True).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience = 0
    max_patience = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        print(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}%"
        )

        # ===== EVALUATION =====
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        acc = 100 * correct / total

        print(f"Test Loss: {test_loss:.4f} | Accuracy: {acc:.2f}%\n")

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            patience = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, "best_model_pixiv.pth")

            print(f"✅ Saved best model (val loss = {best_val_loss:.4f})")

        else:
            patience += 1
            if patience == max_patience:
                break
