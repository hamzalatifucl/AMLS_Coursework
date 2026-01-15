import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2, conv1_filters=32, conv2_filters=64, conv3_filters=128, fc_size=256):
        super(CNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_filters)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.4)  # Balanced - enough to prevent overfitting but allow learning
        self.dropout_fc = nn.Dropout(0.5)    # Balanced - prevents overfitting without blocking learning
        
        self.fc1 = nn.Linear(conv3_filters * 3 * 3, fc_size)
        self.fc2 = nn.Linear(fc_size, 128)   # Restored capacity for better learning
        self.fc3 = nn.Linear(128, num_classes)  # Fixed: fc3 should expect 128 (output of fc2)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)  # Add dropout after first conv block
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)  # Add dropout after second conv block
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)  # Add dropout after third conv block too
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(self.relu(self.fc1(x)))
        x = self.dropout_fc(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# hyperparameters - Increased regularization to reduce overfitting
DEFAULT_HYPERPARAMETERS = {
    'num_classes': 2,
    'conv1_filters': 32,
    'conv2_filters': 64,
    'conv3_filters': 128,
    'fc_size': 256,
    'learning_rate': 0.0008,     # Balanced - fast enough to learn but slow enough to prevent memorization
    'weight_decay': 0.0005,      # Balanced L2 regularization
    'num_epochs': 100,
    'batch_size': 64,
    'patience': 12               # Allows enough training for convergence
}

