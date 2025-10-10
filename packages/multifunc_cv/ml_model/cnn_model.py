import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceRecogCNN(nn.Module):
    def __init__(self):
        super(FaceRecogCNN, self).__init__()

        # 224 x 224 x 1
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 112 x 112 x 64
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #56 x 56 x 128
        self.conv_layer3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #28 x 28 x 256
        self.conv_layer4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #14 x 14 x 512
        self.fc5 = nn.Linear(14 * 14 * 512, 512)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(512, 128)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.relu2(out)
        out = self.max_pool2(out)

        out = self.conv_layer3(out)
        out = self.relu3(out)
        out = self.max_pool3(out)

        out = self.conv_layer4(out)
        out = self.relu4(out)
        out = self.max_pool4(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc5(out)
        out = self.relu5(out)
        out = self.dropout5(out)

        out = self.fc6(out)
        return out


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)

        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss

