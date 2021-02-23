from torch.utils.data import Dataset
import torch

class ReverbDataset(Dataset):
    """
    Reverberation dataset
    """

    def __init__(self, X, y):
        """
        X: (# examples, 1, 128, 340) tensor containing reverberant spectrograms
        y: (# examples, 1, 128, 340) tensor containing target spectrograms
        """

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]