import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from utils import OxfordPetsDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(3, 1).to(device)
model.load_state_dict(torch.load('best.pth'))  # if saved

dataset = OxfordPetsDataset(
    './data/OxfordPets/images', './data/OxfordPets/masks')
sample_img, sample_mask = dataset[0]

model.eval()
with torch.no_grad():
    pred = model(sample_img.unsqueeze(0).to(device))
    pred = (pred.squeeze().cpu().numpy() > 0.5).astype(int)

plt.subplot(1, 3, 1)
plt.imshow(sample_img.permute(1, 2, 0))
plt.title("Image")
plt.subplot(1, 3, 2)
plt.imshow(sample_mask)
plt.title("Ground Truth")
plt.subplot(1, 3, 3)
plt.imshow(pred)
plt.title("Prediction")
plt.show()
