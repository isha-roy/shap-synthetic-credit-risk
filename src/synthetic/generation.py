import os
import random
import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

def set_seed(seed):
    """
    Sets global seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_sdv_metadata(df, dataset_name, numerical_features, categorical_features, target_column):
    """
    Creates and validates SDV SingleTableMetadata for the dataset.
    Explicitly forces types according to configurations and removes primary keys.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Reset primary key to None
    metadata.primary_key = None
    
    # Explicitly set sdtypes based on dataset config to avoid false positive key/PII detections
    for col in df.columns:
        if col == target_column:
            metadata.update_column(col, sdtype='categorical')
        elif col in numerical_features:
            metadata.update_column(col, sdtype='numerical')
        elif col in categorical_features:
            metadata.update_column(col, sdtype='categorical')
            
    metadata.validate()
    return metadata

def train_generator(train_df, metadata, generator_type, config_params, seed):
    """
    Instantiates and fits an SDV synthesizer on train_df using the provided parameters.
    """
    set_seed(seed)
    params = config_params.copy()
    # Explicitly run on CPU to avoid CUDA DLL/loading errors on host
    params["cuda"] = False
    
    if generator_type == "ctgan":
        # Ensure only CTGAN valid parameters are passed
        ctgan_keys = [
            'epochs', 'batch_size', 'generator_dim', 'discriminator_dim', 
            'generator_lr', 'generator_decay', 'discriminator_lr', 
            'discriminator_decay', 'discriminator_steps', 'pac', 
            'enforce_min_max_values', 'enforce_rounding', 'cuda', 'verbose'
        ]
        ctgan_params = {k: v for k, v in params.items() if k in ctgan_keys}
        # Convert dimensions to tuples as list types can sometimes cause issues in downstream layers
        if 'generator_dim' in ctgan_params and isinstance(ctgan_params['generator_dim'], list):
            ctgan_params['generator_dim'] = tuple(ctgan_params['generator_dim'])
        if 'discriminator_dim' in ctgan_params and isinstance(ctgan_params['discriminator_dim'], list):
            ctgan_params['discriminator_dim'] = tuple(ctgan_params['discriminator_dim'])
            
        synthesizer = CTGANSynthesizer(metadata, **ctgan_params)
        
    elif generator_type == "tvae":
        # Ensure only TVAE valid parameters are passed
        tvae_keys = [
            'epochs', 'batch_size', 'compress_dims', 'decompress_dims', 
            'embedding_dim', 'l2scale', 'loss_factor', 
            'enforce_min_max_values', 'enforce_rounding', 'cuda'
        ]
        tvae_params = {k: v for k, v in params.items() if k in tvae_keys}
        # Convert dimensions to tuples
        if 'compress_dims' in tvae_params and isinstance(tvae_params['compress_dims'], list):
            tvae_params['compress_dims'] = tuple(tvae_params['compress_dims'])
        if 'decompress_dims' in tvae_params and isinstance(tvae_params['decompress_dims'], list):
            tvae_params['decompress_dims'] = tuple(tvae_params['decompress_dims'])
            
        synthesizer = TVAESynthesizer(metadata, **tvae_params)
        
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
        
    print(f"Fitting {generator_type.upper()} model for seed {seed}...")
    synthesizer.fit(train_df)
    return synthesizer

def generate_synthetic_data(synthesizer, num_rows):
    """
    Samples num_rows of synthetic data from the trained synthesizer.
    """
    return synthesizer.sample(num_rows=num_rows)
