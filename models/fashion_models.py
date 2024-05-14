import torch.nn as nn
import torch.nn.functional as F

class fashion_CNN(nn.Module):
    def __init__(self, nbr_classes):
        super(fashion_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #in-ch, out-ch, kernel_size, 28x28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.output = nn.Linear(64, nbr_classes)
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
    

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 20)
        
        # Dropout module with a 0.2 drop probability 
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)    
        # Set the activation functions
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
    
        return x