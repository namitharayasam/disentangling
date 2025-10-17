"""
GQA Dataset loader for PID analysis.
"""

from datasets import load_dataset
from .base_dataset import BaseVLDataset


class GQADataset(BaseVLDataset):
    """GQA (Visual Question Answering) Dataset."""
    
    def __init__(self, split='testdev_balanced', start_index=0, num_samples=None):
        """
        Initialize GQA dataset.
        
        Args:
            split: Dataset split (e.g., 'testdev_balanced')
            start_index: Starting index for samples
            num_samples: Number of samples to load
        """
        super().__init__(split, start_index, num_samples)
        self.load_dataset()
    
    def load_dataset(self):
        """Load GQA dataset from HuggingFace."""
        print(f"Loading GQA dataset (split: {self.split})...")
        
        # Load images and instructions
        images_dataset = load_dataset("lmms-lab/GQA", f"{self.split}_images")
        instructions_dataset = load_dataset("lmms-lab/GQA", f"{self.split}_instructions")
        
        images_data = images_dataset[self.split.replace('_balanced', '')]
        instructions_data = instructions_dataset[self.split.replace('_balanced', '')]
        
        # Create image lookup dictionary
        images_by_id = {item['id']: item['image'] for item in images_data}
        
        # Combine images and instructions
        combined_data = []
        for item in instructions_data:
            image_id = item['imageId']
            if image_id in images_by_id:
                combined_data.append({
                    'image
