import torch.nn as nn


class AI_Image_Detector(nn.Module):
    def __init__(self, size):
        self.filter = 64
        self.hidden = 128
        self.size = size
        super(AI_Image_Detector, self).__init__()

        self.conv1 = nn.Conv2d(3, self.filter, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.filter)

        self.conv2 = nn.Conv2d(self.filter, self.filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.filter)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.size = tuple(x//2 for x in self.size)

        self.conv3 = nn.Conv2d(self.filter, self.filter*2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.filter*2)
        self.size = tuple(x//2 for x in self.size)

        self.conv4 = nn.Conv2d(self.filter*2, self.filter*2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.filter*2)
        self.size = tuple(x//2 for x in self.size)

        self.conv5 = nn.Conv2d(self.filter*2, self.filter*4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(self.filter*4)
        self.size = tuple(x//2 for x in self.size)

        self.conv6 = nn.Conv2d(self.filter*4, self.filter*4, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(self.filter*4)
        self.size = tuple(x//2 for x in self.size)

        self.conv7 = nn.Conv2d(self.filter*4, self.filter*8, kernel_size=4, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(self.filter*8)
        self.size = tuple(x-4+1 for x in self.size)

        self.fc1 = nn.Linear(self.filter* 8 * self.size[0] * self.size[1], self.hidden)
        self.fc2 = nn.Linear(self.hidden, 2)
        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(self.relu(out))
        out = self.pool(self.relu(self.bn3(self.conv3(out))))
        out = self.pool(self.relu(self.bn4(self.conv4(out))))
        out = self.pool(self.relu(self.bn5(self.conv5(out))))
        out = self.pool(self.relu(self.bn6(self.conv6(out))))
        out = self.relu(self.bn7(self.conv7(out)))
        out = out.view(x.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


