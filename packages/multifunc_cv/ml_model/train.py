import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cnn_model import FaceRecogCNN, ContrastiveLoss
from dataset import FaceDataset
import os

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()
        output1 = model(img1)
        output2 = model(img2)

        loss = criterion(output1, output2, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f'Checkpoint saved in {filepath}')

def main():
    os.makedirs('checkpoints', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceRecogCNN().to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = FaceDataset(root_dir='../data/train/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 20

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}: ')

        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f'Average Loss: {avg_loss:.4f}')

    save_checkpoint(model, optimizer, num_epochs, avg_loss, 'checkpoints/final_model.pth')
    print('\nTraining complete')

if __name__ == '__main__':
    main()