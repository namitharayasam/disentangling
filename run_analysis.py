"""
Main script to run PID analysis on Vision-Language Models.
"""

import argparse
import os
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Import utilities
from utils import (
    cluster_embeddings,
    clustering,
    convert_data_to_distribution,
    get_measure
)

# Import models and datasets
from models import SmolVLM  # Add other models as you implement them
from data import GQADataset  # Add other datasets as you implement them


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    """Initialize model based on config."""
    model_name = config['model']['name'].lower()
    model_checkpoint = config['model']['checkpoint']
    device = config['model'].get('device', 'cuda')
    quantization = config['model'].get('quantization', True)
    
    if model_name == 'smolvlm':
        model = SmolVLM(
            model_name=model_checkpoint,
            device=device,
            quantization=quantization
        )
    # Add more models here as you implement them
    # elif model_name == 'paligemma':
    #     model = PaliGemma(...)
    # elif model_name == 'llava':
    #     model = LLaVA(...)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_model()
    model.eval()
    return model


def get_dataset(config):
    """Initialize dataset based on config."""
    dataset_name = config['dataset']['name'].lower()
    split = config['dataset'].get('split', 'test')
    start_index = config['dataset'].get('start_index', 0)
    num_samples = config['dataset'].get('num_samples', None)
    
    if dataset_name == 'gqa':
        dataset = GQADataset(
            split=split,
            start_index=start_index,
            num_samples=num_samples
        )
    # Add more datasets here as you implement them
    # elif dataset_name == 'coco':
    #     dataset = COCODataset(...)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def analyze_sample(model, sample, config):
    """
    Run PID analysis on a single sample.
    
    Args:
        model: VLM model instance
        sample: Data sample dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary with PID profile or None if processing fails
    """
    try:
        image = sample['image']
        question = sample['question']
        
        # Process inputs
        inputs = model.process_inputs(image, question)
        
        # Extract embeddings
        text_embs = model.get_text_embeddings(inputs)
        vision_embs = model.get_vision_embeddings(inputs)
        output_hidden = model.get_output_hidden_states(inputs)
        
        # Handle empty text embeddings
        if text_embs.shape[0] == 0:
            print(f"WARNING: No text tokens found. Skipping sample.")
            return None
        
        # Get analysis parameters
        n_clusters = config['analysis'].get('n_clusters', 10)
        use_pca = config['analysis'].get('use_pca', True)
        max_ipfp_iters = config['analysis'].get('max_ipfp_iters', 100)
        
        # Align dimensions by clustering
        text_len = text_embs.shape[0]
        image_len = vision_embs.shape[0]
        
        if text_len < image_len:
            # Cluster vision and output to match text length
            vision_embs = cluster_embeddings(vision_embs, text_len)
            output_hidden = cluster_embeddings(output_hidden, text_len)
        else:
            # Cluster text and output to match image length
            text_embs = cluster_embeddings(text_embs, image_len)
            output_hidden = cluster_embeddings(output_hidden, image_len)
        
        # Apply PCA and K-means clustering
        kmeans_im, _ = clustering(vision_embs, pca=use_pca, n_clusters=n_clusters, n_components=text_len)
        kmeans_txt, _ = clustering(text_embs, pca=use_pca, n_clusters=n_clusters, n_components=text_len)
        kmeans_out, _ = clustering(output_hidden, pca=use_pca, n_clusters=n_clusters, n_components=text_len)
        
        # Reshape for distribution conversion
        kmeans_im = kmeans_im.reshape(-1, 1)
        kmeans_txt = kmeans_txt.reshape(-1, 1)
        kmeans_out = kmeans_out.reshape(-1, 1)
        
        # Convert to probability distribution
        P, _ = convert_data_to_distribution(kmeans_im, kmeans_txt, kmeans_out)
        
        # Compute PID measures
        profile = get_measure(P, name='ipfp', max_iters=max_ipfp_iters)
        
        return {
            'redundancy': float(profile.get('redundancy', 0.0)),
            'unique_text': float(profile.get('unique1', 0.0)),
            'unique_image': float(profile.get('unique2', 0.0)),
            'synergy': float(profile.get('synergy', 0.0)),
        }
        
    except Exception as e:
        print(f"ERROR processing sample: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run PID analysis on VLMs')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize model and dataset
    print("\n" + "="*50)
    print("Initializing model...")
    print("="*50)
    model = get_model(config)
    
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    dataset = get_dataset(config)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=lambda x: x
    )
    
    # Run analysis
    print("\n" + "="*50)
    print("Running PID analysis...")
    print("="*50)
    
    all_profiles = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            sample = batch[0]
            
            profile = analyze_sample(model, sample, config)
            
            if profile is not None:
                all_profiles.append(profile)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"\nSample {i}: "
                          f"Unique_text={profile['unique_text']:.4f}, "
                          f"Unique_image={profile['unique_image']:.4f}")
    
    # Save results
    print("\n" + "="*50)
    print("Saving results...")
    print("="*50)
    
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to DataFrame
    df_profiles = pd.DataFrame(all_profiles)
    
    # Ensure all columns exist
    for col in ['redundancy', 'unique_text', 'unique_image', 'synergy']:
        if col not in df_profiles.columns:
            df_profiles[col] = np.nan
    
    # Generate filename
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    start_idx = config['dataset'].get('start_index', 0)
    save_prefix = config['output'].get('save_prefix', f'{model_name}_{dataset_name}')
    
    output_file = os.path.join(save_dir, f'{save_prefix}_{start_idx}.csv')
    df_profiles.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nMean values:")
    print(df_profiles.mean())
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)


if __name__ == '__main__':
    main()
