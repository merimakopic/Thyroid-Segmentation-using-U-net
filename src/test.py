import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from model import UNet 

MODEL_PATH = "best_model.pth"
IMAGE_PATH = "data/test/400_1.jpg"
OUTPUT_PATH = "outputs/predictions/400_1_version2_prediction.png"
IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)
    mask = output.squeeze().cpu().numpy()
    print("Mask stats:", mask.min(), mask.max(), mask.mean())


gray_mask = (mask * 255).astype(np.uint8)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
gray_mask = cv2.normalize(gray_mask, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite(OUTPUT_PATH, gray_mask)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image.resize(IMAGE_SIZE))
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")
plt.tight_layout()
plt.show()

print(f"Mask saved to: {OUTPUT_PATH}")
