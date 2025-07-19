"""
Tokenization utilities for gene expression data.
"""
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from datasets import Dataset as HFDataset
import logging
from typing import List, Dict, Any, Optional
from geneformer.tokenizer import tokenize_cell, TranscriptomeTokenizer

logger = logging.getLogger(__name__)


def initialize_tokenizer() -> TranscriptomeTokenizer:
    """
    Initialize the Geneformer tokenizer.
    
    Returns:
        Initialized TranscriptomeTokenizer
    """
    tk = TranscriptomeTokenizer()
    logger.info("Initialized Geneformer tokenizer")
    return tk


def create_gene_tokens(genes: List[str], gene_token_dict: Dict[str, int]) -> np.ndarray:
    """
    Create gene tokens from gene names using the tokenizer dictionary.
    
    Args:
        genes: List of gene names
        gene_token_dict: Dictionary mapping gene names to token indices
        
    Returns:
        Array of gene tokens
    """
    gene_tokens = np.array([gene_token_dict.get(g, 0) for g in genes], dtype=int)
    logger.info(f"Created gene tokens for {len(genes)} genes")
    return gene_tokens


def process_cell_tokenization(gene_vector: np.ndarray, gene_tokens: np.ndarray, 
                            max_len: int, labels_array: Optional[np.ndarray] = None, 
                            cell_idx: int = 0) -> Dict[str, Any]:
    """
    Process a single cell for tokenization.
    
    Args:
        gene_vector: Gene expression vector for the cell
        gene_tokens: Array of gene tokens
        max_len: Maximum sequence length
        labels_array: Optional array of labels
        cell_idx: Cell index for label lookup
        
    Returns:
        Dictionary with tokenized cell data
    """
    # Get ranked token ids
    ranked_tokens = tokenize_cell(gene_vector, gene_tokens)
    
    # Pad or truncate
    input_ids = list(ranked_tokens[:max_len])
    input_ids += [0] * (max_len - len(input_ids))
    
    attention_mask = [1 if id != 0 else 0 for id in input_ids]
    
    rec = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if labels_array is not None:
        rec["labels"] = int(labels_array[cell_idx])
    
    return rec


def tokenize_dataset(adata, gene_tokens: np.ndarray, max_len: int = 512, 
                    label_key: Optional[str] = None, n_jobs: int = 4) -> HFDataset:
    """
    Tokenize the entire dataset in parallel.
    
    Args:
        adata: AnnData object
        gene_tokens: Array of gene tokens
        max_len: Maximum sequence length
        label_key: Optional key for labels in adata.obs
        n_jobs: Number of parallel jobs
        
    Returns:
        HuggingFace Dataset with tokenized data
    """
    # Prepare labels if available
    labels_array = None
    if label_key and label_key in adata.obs.columns:
        labels_array = adata.obs[label_key].astype("category").cat.codes.values
        logger.info(f"Using labels from column: {label_key}")
    
    # Define the per-cell processing function
    def process_cell(idx):
        gene_vector = adata.X[idx].toarray().flatten() if hasattr(adata.X[idx], "toarray") else adata.X[idx]
        return process_cell_tokenization(gene_vector, gene_tokens, max_len, labels_array, idx)
    
    # Parallel execution
    records = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(i) for i in tqdm(range(adata.n_obs), desc="Tokenizing cells")
    )
    
    # Build HuggingFace dataset
    hf_ds = HFDataset.from_list(records)
    
    logger.info(f"Tokenized dataset with {len(records)} cells")
    return hf_ds


def split_dataset(dataset: HFDataset, test_size: float = 0.2, 
                 seed: int = 42, small_fraction: Optional[float] = None) -> Dict[str, HFDataset]:
    """
    Split dataset into train and test sets, optionally creating smaller versions.
    
    Args:
        dataset: HuggingFace Dataset
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        small_fraction: Optional fraction to create smaller datasets for debugging
        
    Returns:
        Dictionary with train and test datasets
    """
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    train_ds = split_dataset["train"]
    test_ds = split_dataset["test"]
    
    result = {
        "train": train_ds,
        "test": test_ds
    }
    
    # Create smaller datasets for debugging if requested
    if small_fraction:
        small_train_ds = train_ds.shuffle(seed=seed).select(range(int(small_fraction * len(train_ds))))
        small_test_ds = test_ds.shuffle(seed=seed).select(range(int(small_fraction * len(test_ds))))
        
        result["small_train"] = small_train_ds
        result["small_test"] = small_test_ds
        
        logger.info(f"Created smaller datasets: {len(small_train_ds)} train, {len(small_test_ds)} test")
    
    logger.info(f"Split dataset: {len(train_ds)} train, {len(test_ds)} test")
    return result


def save_tokenized_dataset(dataset: HFDataset, save_path: str) -> None:
    """
    Save tokenized dataset to disk.
    
    Args:
        dataset: HuggingFace Dataset to save
        save_path: Path to save the dataset
    """
    dataset.save_to_disk(save_path)
    logger.info(f"Saved tokenized dataset to {save_path}")


def load_tokenized_dataset(load_path: str) -> HFDataset:
    """
    Load tokenized dataset from disk.
    
    Args:
        load_path: Path to load the dataset from
        
    Returns:
        HuggingFace Dataset
    """
    from datasets import load_from_disk
    dataset = load_from_disk(load_path)
    logger.info(f"Loaded tokenized dataset from {load_path}")
    return dataset


def clear_cuda_memory():
    """
    Clear CUDA memory and garbage collect.
    """
    import gc
    
    # Clear Python variables
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA memory")
    else:
        logger.info("CUDA not available, skipped memory clearing") 