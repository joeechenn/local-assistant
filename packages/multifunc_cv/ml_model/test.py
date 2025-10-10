import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from cnn_model import FaceRecogCNN
from dataset import FaceDataset
from torchvision import transforms


class TestDataset(Dataset):
    """Wrapper that limits number of pairs for testing"""

    def __init__(self, base_dataset, num_pairs):
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        return self.base_dataset[idx]


def load_model(checkpoint_path):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceRecogCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device


def calculate_accuracy(model, dataloader, device, threshold=1.5):
    """Test model and calculate accuracy"""
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            output1 = model(img1)
            output2 = model(img2)

            distances = F.pairwise_distance(output1, output2)
            predictions = (distances >= threshold).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100


def main():
    # Load model
    model, device = load_model('checkpoints/final_model.pth')

    # Setup transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create base dataset
    base_dataset = FaceDataset(root_dir='../data/', transform=transform)

    # Wrap with test dataset (no lambda!)
    test_dataset = TestDataset(base_dataset, num_pairs=500)

    # Create dataloader with num_workers=0
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Calculate accuracy
    threshold = 1.5
    accuracy = calculate_accuracy(model, test_loader, device, threshold)

    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()