# link Google Drive
from google.colab import drive
drive.mount('/content/drive')

# check the  running environment
import torch
print(torch.__version__)
print(torch.cuda.is_available())

# prepare the dataset
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class BSEPatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".tif")
        ])

        self.patches = []
        self._prepare_patches()

    def _prepare_patches(self):
        for img_name in self.image_files:
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(
                self.mask_dir,
                img_name.replace(".tif", "_mask.tif")
            )

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            h, w = image.shape
            
            for y in range(0, h - self.patch_size + 1, self.patch_size):
                for x in range(0, w - self.patch_size + 1, self.patch_size):
                    self.patches.append((
                        img_path,
                        mask_path,
                        x, y
                    ))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.patches[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = image[y:y+self.patch_size, x:x+self.patch_size]
        mask = mask[y:y+self.patch_size, x:x+self.patch_size]


        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(mask).long()

        # safety check
        assert mask.max()<=3


        return image, mask

# import the google drive
dataset = BSEPatchDataset(
    image_dir="/content/drive/MyDrive/BSE_CNN/dataset/train/img",
    mask_dir="/content/drive/MyDrive/BSE_CNN/dataset/train/mask",
    patch_size=256
)

print("Total patches:", len(dataset))

# define loader
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

# install models
!pip install segmentation-models-pytorch --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

#
num_classes = 4

# 统计整个 dataset 的像素数量
class_counts = torch.zeros(num_classes)
for img, mask in dataset:
    for c in range(num_classes):
        class_counts[c] += torch.sum(mask == c)

total_pixels = class_counts.sum()
class_weights = total_pixels / (num_classes * class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Class weights:", class_weights)

# Unet mmodel define
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,      # BSE 灰度图
    classes=num_classes
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train epoch
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # shape [B, num_classes, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# validate  & visulization
model.eval()
with torch.no_grad():
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    outputs = model(images)
    preds = outputs.argmax(dim=1)

    # 可视化第一张 patch
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(images[0,0].cpu(), cmap='gray')
    axs[0].set_title('Input BSE Patch')
    axs[1].imshow(preds[0].cpu(), cmap='jet', vmin=0, vmax=num_classes-1)
    axs[1].set_title('Predicted Mask')
    axs[2].imshow(masks[0].cpu(), cmap='jet', vmin=0, vmax=num_classes-1)
    axs[2].set_title('Ground Truth Mask')
    plt.show()

# each pixel distribution
for idx in range(images.size(0)):
    print(f"--- Patch {idx} ---")
    total_pix = preds[idx].numel()
    for c in range(num_classes):
        c_pix = (preds[idx]==c).sum().item()
        print(f"Class {c}: {c_pix} pixels, {c_pix/total_pix*100:.2f}%")
    print("Ground truth distribution:")
    for c in range(num_classes):
        c_pix = (masks[idx]==c).sum().item()
        print(f"Class {c}: {c_pix} pixels, {c_pix/total_pix*100:.2f}%")


# Test part
import os
import cv2
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1️⃣ 测试图像路径
test_image_dir = "/content/drive/MyDrive/BSE_CNN/dataset/test/img/"
test_mask_dir = "/content/drive/MyDrive/BSE_CNN/dataset/test/mask/"  

test_image_files = sorted([
    f for f in os.listdir(test_image_dir)
    if f.endswith(".tif")
])

#
def predict_bse(model, image_path, patch_size=256):
    """
    对单张 BSE 图像做 patch-based 推理
    返回完整预测 mask
    """
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    # 用滑动窗口预测
    pred_mask = torch.zeros((h, w), dtype=torch.long)

    stride = patch_size  # 可修改 stride
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
                patch_tensor = patch_tensor.to(device)
                output = model(patch_tensor)
                pred = output.argmax(dim=1).squeeze(0).cpu()
                pred_mask[y:y+patch_size, x:x+patch_size] = pred

    return pred_mask

#
def visualize_prediction(image_path, pred_mask, mask_path=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fig, axs = plt.subplots(1,3 if mask_path else 2, figsize=(12,4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input BSE Image')

    axs[1].imshow(pred_mask, cmap='jet', vmin=0, vmax=3)
    axs[1].set_title('Predicted Mask')

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        axs[2].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axs[2].set_title('Ground Truth Mask')

    plt.show()

#
for img_file in test_image_files:
    img_path = os.path.join(test_image_dir, img_file)
    mask_path = os.path.join(test_mask_dir, img_file.replace(".tif","_mask.tif"))
    pred_mask = predict_bse(model, img_path, patch_size=256)

    if os.path.exists(mask_path):
        visualize_prediction(img_path, pred_mask, mask_path)
    else:
        visualize_prediction(img_path, pred_mask)
end
