import numpy as np
import torch
from torch.utils.data import Dataset


class BreastMNISTDataset(Dataset):
    def __init__(self, medmnist_dataset, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for idx in range(len(medmnist_dataset)):
            image, label = medmnist_dataset[idx]
            if hasattr(image, 'numpy'):
                image = image.numpy()
            elif not isinstance(image, np.ndarray):
                image = np.array(image)
            image = image.astype(np.float32) / 255.0
            label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
            
            image_2d = image[:, :, 0] if len(image.shape) == 3 else image
            self.images.append(image_2d)
            self.labels.append(label_value)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.LongTensor([self.labels[idx]])[0]
        return image, label

