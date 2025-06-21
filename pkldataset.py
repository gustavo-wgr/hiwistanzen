import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import scipy.ndimage

class_to_onehot = {
    'R05': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R10': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R15': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'R20': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'R25': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'R30': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'R35': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'R40': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'R45': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'R50': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

class PKLDataset(Dataset):
    def __init__(self, folder_path, transform_type=None):
        """
        Args:
            folder_path (str): Path to the folder of .pkl files.
            transform_type (str, optional): Transformation to apply. If None, the raw time series is returned.
        """
        self.folder_path = folder_path
        self.transform_type = transform_type
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        self.files.sort()  # Optional: sort for consistent ordering

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get filename and full path
        filename = self.files[idx]
        file_path = os.path.join(self.folder_path, filename)

        # Extract class label (e.g. 'R05' from 'R05_I123.pkl')
        class_label = filename.split('_')[0]
        label_vector = class_to_onehot[class_label]
        label_tensor = torch.tensor(label_vector, dtype=torch.float32)

        # Load the data (assumed to be a 1D or 2D NumPy array)
        with open(file_path, 'rb') as f:
            array_data = pickle.load(f)
        data_tensor = torch.tensor(array_data, dtype=torch.float32)

        return data_tensor, label_tensor

    @staticmethod
    def split_dataset(folder_path, transform_type=None, train_ratio=0.8, random_seed=42):
        """
        Splits the dataset into training and validation subsets with an 80/20 split.

        Args:
            folder_path (str): Path to the folder of .pkl files.
            transform_type (str, optional): Transformation to apply.
            train_ratio (float): Proportion of data to use for training (default 0.8).
            random_seed (int): Seed for reproducibility (default 42).

        Returns:
            train_dataset (Dataset): Training subset.
            val_dataset (Dataset): Validation subset.
        """
        full_dataset = PKLDataset(folder_path, transform_type)
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        return train_dataset, val_dataset

class NoisyPKLDataset(PKLDataset):
    def __init__(self, folder_path, scale_std=0.01, smooth_sigma=1.0, augment_repeats=1, transform_type=None):
        """
        Args:
            folder_path (str): Path to the folder of .pkl files.
            scale_std (float): Standard deviation for the global magnitude scaling factor.
            smooth_sigma (float): Standard deviation for the Gaussian smoothing kernel.
            augment_repeats (int): Number of times to generate augmented versions of each sample
                                   (increasing dataset size by this factor).
            transform_type (str, optional): Transformation to apply. If None, raw time series is returned.
        """
        super().__init__(folder_path, transform_type)
        self.scale_std = scale_std
        self.smooth_sigma = smooth_sigma
        self.augment_repeats = augment_repeats

    def __len__(self):
        """
        Make the dataset 'augment_repeats' times larger by returning multiple
        augmented versions of each original sample.
        """
        return super().__len__() * self.augment_repeats

    def global_magnitude_scaling(self, data_tensor):
        """
        Apply global magnitude scaling to the data_tensor.
        """
        scale_factor = np.random.normal(loc=1.0, scale=self.scale_std)  # Sample a global scale factor near 1
        return data_tensor * scale_factor  # Apply scaling

    def gaussian_smoothing(self, data_tensor):
        """
        Apply Gaussian smoothing to the data_tensor.
        """
        smoothed_np = scipy.ndimage.gaussian_filter(data_tensor.numpy(), sigma=self.smooth_sigma)
        return torch.tensor(smoothed_np, dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Get the base sample and apply global magnitude scaling and Gaussian smoothing.
        """
        # Map the new index to an original dataset index
        original_idx = idx % super().__len__()  # cycle through original data

        # Get the original data and label
        data_tensor, label_tensor = super().__getitem__(original_idx)

        # Apply global magnitude scaling
        scaled_data = self.global_magnitude_scaling(data_tensor)
        
        # Apply Gaussian smoothing
        augmented_data = self.gaussian_smoothing(scaled_data)

        return augmented_data, label_tensor

