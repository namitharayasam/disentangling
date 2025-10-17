"""
SmolVLM model implementation for PID analysis.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from .base_model import BaseVLM


class SmolVLM(BaseVLM):
    """SmolVLM model for vision-language tasks."""
    
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct", 
                 device='cuda', quantization=True):
        super().__init__(model_name, device, quantization)
        self.text_start_idx = 1067  # Token index where pure text starts
        
    def load_model(self):
        """Load SmolVLM model and processor."""
        print(f"Loading {self.model_name}...")
        print(f"Device: {self.device}")
        print(f"Quantization: {self.quantization}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load model with or without quantization
        if self.quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16,
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded successfully!")
        return self
    
    def process_inputs(self, image, question):
        """
        Process image and question for SmolVLM.
        
        Args:
            image: PIL Image
            question: Question string
            
        Returns:
            Dictionary with processed inputs
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}"}
                ]
            }
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    def get_text_embeddings(self, inputs):
        """
        Extract text embeddings (after image tokens).
        
        Args:
            inputs: Processed inputs dictionary
            
        Returns:
            torch.Tensor: Text embeddings [seq_len, hidden_dim]
        """
        input_ids = inputs['input_ids']
        
        # Slice to get only text tokens (after image tokens)
        input_ids_text = input_ids[:, self.text_start_idx:]
        
        if input_ids_text.shape[1] == 0:
            raise ValueError("No text tokens found after slicing")
        
        # Get embeddings
        text_embs = self.model.get_input_embeddings()(input_ids_text)
        return text_embs.squeeze(0)
    
    def get_vision_embeddings(self, inputs):
        """
        Extract vision embeddings from image encoder.
        
        Args:
            inputs: Processed inputs dictionary
            
        Returns:
            torch.Tensor: Vision embeddings [num_patches, hidden_dim]
        """
        pixel_values = inputs['pixel_values']
        pixel_attention_mask = inputs.get('pixel_attention_mask')
        
        # Get image features
        vision_embs = self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask
        )
        
        # Handle 3D tensor [batch, num_patches, hidden_dim] or [layers, patches, dim]
        if len(vision_embs.shape) == 3:
            vision_embs = vision_embs.mean(dim=0)
        
        return vision_embs
    
    def get_output_hidden_states(self, inputs):
        """
        Get output hidden states from forward pass.
        
        Args:
            inputs: Processed inputs dictionary
            
        Returns:
            torch.Tensor: Output hidden states [seq_len, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get last hidden state
        last_hidden = outputs.hidden_states[-1].squeeze(0)
        return last_hidden
