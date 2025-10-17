"""
Base class for Vision-Language Datasets.
All dataset implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseVLDataset(Dataset, ABC):
    """Abstract base class for Vision-Language Datasets."""
    
    def __init__(self, split='test', start_index=0, num_samples=None):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split ('train', 'test', 'val', etc.)
            start_index: Starting index for data samples
            num_samples: Number of samples to use (None = all)
        """
        self.split = split
        self.start_index = start_index
        self.num_samples = num_samples
        self.data = []
        
    @abstractmethod
    def load_dataset(self):
        """
        Load the dataset from source.
        Should populate self.data with processed samples.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with keys:
                - 'image': PIL Image
                - 'question': str
                - 'answer': str
                - 'messages': List of message dicts for chat template
        """
        pass
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def get_subset(self, start_idx, end_idx):
        """
        Get a subset of the dataset.
        
        Args:
            start_idx: Start index
            end_idx: End index
            
        Returns:
            List of data samples
        """
        return self.data[start_idx:end_idx]
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"split='{self.split}', "
                f"num_samples={len(self.data)})")
