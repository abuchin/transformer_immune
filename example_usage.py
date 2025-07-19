#!/usr/bin/env python3
"""
Example usage of the Transformer Immune Cells pipeline.
This script demonstrates how to use individual components of the pipeline.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import *
from src.utils.data_processing import load_adata, preprocess_adata
from src.utils.scvi_utils import run_full_scvi_pipeline
from src.utils.tokenization import initialize_tokenizer, tokenize_dataset
from src.utils.model_utils import setup_device, load_pretrained_model


def example_data_preprocessing():
    """
    Example of data preprocessing workflow.
    """
    print("=== Data Preprocessing Example ===")
    
    try:
        # Load data
        print("Loading data...")
        adata = load_adata(RAW_DATA_PATH)
        print(f"Loaded data with shape: {adata.shape}")
        
        # Preprocess data
        print("Preprocessing data...")
        adata = preprocess_adata(adata)
        print("Data preprocessing completed")
        
        # Run scVI pipeline
        print("Running scVI pipeline...")
        adata_scvi = run_full_scvi_pipeline(adata)
        print("scVI pipeline completed")
        
        return adata, adata_scvi
        
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None, None


def example_tokenization(adata):
    """
    Example of tokenization workflow.
    """
    print("\n=== Tokenization Example ===")
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = initialize_tokenizer()
        
        # Create gene tokens
        print("Creating gene tokens...")
        genes = adata.var_names.to_list()
        gene_tokens = [tokenizer.gene_token_dict.get(g, 0) for g in genes]
        print(f"Created tokens for {len(genes)} genes")
        
        # Tokenize dataset
        print("Tokenizing dataset...")
        tokenized_dataset = tokenize_dataset(
            adata, 
            gene_tokens, 
            max_len=MAX_SEQUENCE_LENGTH,
            label_key='subject.ageGroup',
            n_jobs=N_JOBS
        )
        print(f"Tokenized {len(tokenized_dataset)} cells")
        
        return tokenized_dataset
        
    except Exception as e:
        print(f"Error in tokenization: {e}")
        return None


def example_model_setup():
    """
    Example of model setup.
    """
    print("\n=== Model Setup Example ===")
    
    try:
        # Setup device
        print("Setting up device...")
        device = setup_device()
        print(f"Using device: {device}")
        
        # Load pre-trained model
        print("Loading pre-trained model...")
        model = load_pretrained_model(GENEFORMER_MODEL_NAME, NUM_LABELS)
        print("Model loaded successfully")
        
        return model
        
    except Exception as e:
        print(f"Error in model setup: {e}")
        return None


def main():
    """
    Main example function.
    """
    print("Transformer Immune Cells - Example Usage")
    print("=" * 50)
    
    # Check if data path exists
    if not Path(RAW_DATA_PATH).exists():
        print(f"Warning: Data file not found at {RAW_DATA_PATH}")
        print("Please update the RAW_DATA_PATH in src/config.py")
        print("Skipping data preprocessing example...")
        return
    
    # Run examples
    adata, adata_scvi = example_data_preprocessing()
    
    if adata is not None:
        tokenized_dataset = example_tokenization(adata)
        model = example_model_setup()
        
        print("\n=== Summary ===")
        print("All examples completed successfully!")
        print(f"Data shape: {adata.shape}")
        if tokenized_dataset:
            print(f"Tokenized dataset size: {len(tokenized_dataset)}")
        if model:
            print("Model loaded successfully")
    else:
        print("Examples could not be completed due to missing data.")


if __name__ == "__main__":
    main() 