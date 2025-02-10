import torch
import numpy as np 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

## Dataset
class CustomDataset(Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def __len__(self):
        return len(self.X)
    
    def __getlabels__(self):
        return self.labels
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), self.labels[idx],idx
