import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


MODEL_PATH = "C:/Users/patri/OneDrive/Desktop/JobHunt2025/cv_natural_scenes/models/resnet18_intel_scenes.pt"
IMG_SIZE = 224


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(num_classes: int, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["class_names"]


def preprocess_image(image_path: str):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)
    return x


def predict(image_path: str, top_k: int = 3):
    device = get_device()

    # Load checkpoint to get class names
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    model, class_names = load_model(num_classes, device)
    x = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = probs.argsort()[-top_k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top_idx]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    preds = predict(img_path, top_k=1)
    print("Image:", img_path)
    print("Top predictions:")
    for label, p in preds:
        print(f"  {label}: {p:.4f}")
