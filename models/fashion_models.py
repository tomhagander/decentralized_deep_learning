import torch.nn as nn
import torch.nn.functional as F

class CNNFashion(nn.Module):
    def __init__(self, num_classes):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #in-ch, out-ch, kernel_size, 28x28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.output = nn.Linear(64, num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #28x28 -> 26x26 -> 13x13
        x = self.pool(F.relu(self.conv2(x))) #13x13 -> 11x11 -> 5x5
        #x = self.pool(F.relu(self.conv3(x))) #5x5 -> 3x3 -> 1x1
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = self.activation(x)
        return x