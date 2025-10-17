"""
Base class for Vision-Language Models.
All model implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
import torch


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""
    
    def __init__(self, model_name, device='cuda', quantization=True):
        """
        Initialize the model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('cuda' or 'cpu')
            quantization: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.quantization = quantization and self.device == 'cuda'
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def get_text_embeddings(self, inputs):
        """
        Extract text embeddings from the model.
        
        Args:
            inputs: Processed inputs from the processor
            
        Returns:
            torch.Tensor: Text embeddings [seq_len, hidden_dim]
        """
        pass
    
    @abstractmethod
    def get_vision_embeddings(self, inputs):
        """
        Extract vision embeddings from the model.
        
        Args:
            inputs: Processed inputs from the processor
            
        Returns:
            torch.Tensor: Vision embeddings [num_patches, hidden_dim]
        """
        pass
    
    @abstractmethod
    def get_output_hidden_states(self, inputs):
        """
        Get output hidden states from the model.
        
        Args:
            inputs: Processed inputs from the processor
            
        Returns:
            torch.Tensor: Output hidden states [seq_len, hidden_dim]
        """
        pass
    
    @abstractmethod
    def process_inputs(self, image, question):
        """
        Process image and question into model inputs.
        
        Args:
            image: PIL Image
            question: Question string
            
        Returns:
            Processed inputs ready for the model
        """
        pass
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"
