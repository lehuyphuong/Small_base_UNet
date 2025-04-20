import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.unet import UNet
from utils import get_loaders, compute_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(3, 1).to(device)

train_loader = get_loaders('./data/OxfordPets/images',
                           './data/OxfordPets/masks', batch_size=2)
test_loader = get_loaders('./data/OxfordPets/images',
                          './data/OxfordPets/masks', batch_size=2)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {'loss': [], 'acc': [], 'iou': [], 'dice': []}

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), targets.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f}")
    history['loss'].append(total_loss / len(train_loader))

    model.eval()
    total_acc, total_iou, total_dice = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc, iou, dice = compute_metrics(
                outputs.squeeze(1), targets.float())
            total_acc += acc
            total_iou += iou
            total_dice += dice

    epoch_acc = total_acc / len(test_loader)
    epoch_iou = total_iou / len(test_loader)
    epoch_dice = total_dice / len(test_loader)

    history['acc'].append(epoch_acc)
    history['iou'].append(epoch_iou)
    history['dice'].append(epoch_dice)

    print(f"\nValidation Metrics after Epoch {epoch+1}:")
    print(f"Pixel Accuracy: {total_acc / len(test_loader):.4f}")
    print(f"Mean IoU: {total_iou / len(test_loader):.4f}")
    print(f"Mean Dice: {total_dice / len(test_loader):.4f}\n")

torch.save(model.state_dict(), 'best.pth')

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.plot(history['loss'], label='Loss')
plt.title('Loss')
plt.subplot(1, 4, 2)
plt.plot(history['acc'], label='Pixel Acc')
plt.title('Pixel Accuracy')
plt.subplot(1, 4, 3)
plt.plot(history['iou'], label='IoU')
plt.title('Mean IoU')
plt.subplot(1, 4, 4)
plt.plot(history['dice'], label='Dice')
plt.title('Mean Dice')
plt.tight_layout()
plt.show()
