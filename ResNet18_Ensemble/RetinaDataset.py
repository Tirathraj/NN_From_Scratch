import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

"""
Subclass torch Dataset
Create Custom Dataset
implement len, getitem

Contains Transform functions as well: example:
 - Convert to PIL Image
 - Resize
 - Horizontal Flip
 - Rotation
 - Colour Jitter
 - Normalize

"""

class RetinaDataset(Dataset):
    
    def __init__(self, images, labels=None, transform=None):
        #Np arr of shape (n x height x width x channel)
        self.images = images
        
        #0-4 for 5 classes
        self.labels = labels

        #torch vision transform pipeline
        self.transform = transform
        #Typical Pipeline: np - PIL - resize - augment - tensor - normalize

    #Length of dataset
    def __len__(self):
        return len(self.images)

    #Retrieve and image by idx
    #Normally np arr
    
    def __getitem__(self, idx):
        img = self.images[idx]

        #Apply transform
        
        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

#Horizontally Flip
#Rotate Randomly by 10 0r -10 degrees
#ColorJitter: increase/decrease contrast/brightness to mirror real-life conditions
def get_train_transforms():

    #Mean and std dev of ImageNet Dataset
    #Official Normalization values
    
    R_mean, R_std = 0.485, 0.229
    G_mean, G_std = 0.456, 0.224
    B_mean, B_std = 0.406, 0.225  
    
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([R_mean, G_mean, B_mean],
                             [R_std, G_std, B_std]),
    ])

#We want validation transform to be as they are
#Therefore we just resize validation transforms...no rotations or filps

def get_val_transforms():

    #Mean and std dev of ImageNet Dataset
    #Official Normalization values
    
    R_mean, R_std = 0.485, 0.229
    G_mean, G_std = 0.456, 0.224
    B_mean, B_std = 0.406, 0.225  
    
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([R_mean, G_mean, B_mean],
                             [R_std, G_std, B_std]),
    ])
    
#Get inverse frequency weights: used for sampling
def weighted_sampling(y_train):
    class_counts = np.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

#Used in error loss function
def get_class_weights(y_train, device):
    class_counts = np.bincount(y_train)
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)
