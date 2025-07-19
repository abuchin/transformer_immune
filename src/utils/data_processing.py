"""
Data processing utilities for single-cell RNA-seq data.
"""
import scanpy as sc
import pandas as pd
import numpy as np
import mygene
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def load_adata(file_path: str) -> sc.AnnData:
    """
    Load AnnData object from file.
    
    Args:
        file_path: Path to the .h5ad file
        
    Returns:
        AnnData object
    """
    try:
        adata = sc.read_h5ad(file_path)
        logger.info(f"Successfully loaded AnnData with shape {adata.shape}")
        return adata
    except Exception as e:
        logger.error(f"Failed to load AnnData from {file_path}: {e}")
        raise


def preprocess_adata(adata: sc.AnnData, target_sum: float = 1e6) -> sc.AnnData:
    """
    Preprocess AnnData object with normalization and log transformation.
    
    Args:
        adata: AnnData object
        target_sum: Target sum for normalization
        
    Returns:
        Preprocessed AnnData object
    """
    # Save raw data counts
    adata.X = adata.raw.X
    
    # Keep original counts in another layer
    adata.layers['counts'] = adata.raw.X
    
    # Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    logger.info("Preprocessing completed: normalization and log transformation")
    return adata


def map_genes_to_ensembl(gene_symbols: List[str], species: str = "human") -> Dict[str, str]:
    """
    Map gene symbols to Ensembl IDs using MyGene.info.
    
    Args:
        gene_symbols: List of gene symbols
        species: Species name (default: "human")
        
    Returns:
        Dictionary mapping gene symbols to Ensembl IDs
    """
    mg = mygene.MyGeneInfo()
    
    # Query for Ensembl IDs
    gene_info = mg.querymany(
        gene_symbols,
        scopes='symbol',
        fields='ensembl.gene',
        species=species,
        as_dataframe=True
    )
    
    # Create a clean mapping
    ensembl_map = gene_info.reset_index().dropna(subset=['ensembl.gene'])
    id_map = pd.Series(
        ensembl_map['ensembl.gene'].values, 
        index=ensembl_map['query']
    ).to_dict()
    
    logger.info(f"Mapped {len(id_map)} genes to Ensembl IDs")
    return id_map


def convert_to_ensembl_ids(adata: sc.AnnData, species: str = "human") -> sc.AnnData:
    """
    Convert gene symbols to Ensembl IDs in AnnData object.
    
    Args:
        adata: AnnData object
        species: Species name
        
    Returns:
        AnnData object with Ensembl IDs as gene names
    """
    # Keep original gene names
    adata.var['gene_name_HGCN'] = adata.var.index.values
    
    # Map to Ensembl IDs
    gene_symbols = adata.var.index.tolist()
    id_map = map_genes_to_ensembl(gene_symbols, species)
    
    # Add Ensembl ID column
    adata.var['ensembl_id'] = adata.var.index.map(id_map)
    
    # Convert to Ensembl IDs
    adata.var['gene_name'] = adata.var['ensembl_id']
    adata.var_names = adata.var['ensembl_id']
    
    # Remove genes without Ensembl IDs
    valid_var_mask = pd.notnull(adata.var_names)
    adata = adata[:, valid_var_mask]
    
    logger.info(f"Converted to Ensembl IDs. New shape: {adata.shape}")
    return adata


def select_highly_variable_genes(adata: sc.AnnData, n_top_genes: int = 2000) -> sc.AnnData:
    """
    Select highly variable genes from AnnData object.
    
    Args:
        adata: AnnData object
        n_top_genes: Number of top highly variable genes to select
        
    Returns:
        AnnData object with only highly variable genes
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=True)
    adata = adata[:, adata.var.highly_variable].copy()
    
    logger.info(f"Selected {n_top_genes} highly variable genes. New shape: {adata.shape}")
    return adata


def save_preprocessed_data(adata: sc.AnnData, genes: List[str], gene_tokens: np.ndarray, 
                          save_dir: Path) -> None:
    """
    Save preprocessed data to disk.
    
    Args:
        adata: Preprocessed AnnData object
        genes: List of gene names
        gene_tokens: Array of gene tokens
        save_dir: Directory to save the data
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save AnnData
    adata_path = save_dir / "adata_nk_hvg.h5ad"
    adata.write_h5ad(adata_path)
    
    # Save genes and tokens
    genes_path = save_dir / "genes.npy"
    gene_tokens_path = save_dir / "gene_tokens.npy"
    
    np.save(genes_path, genes)
    np.save(gene_tokens_path, gene_tokens)
    
    logger.info(f"Saved preprocessed data to {save_dir}")


def load_preprocessed_data(load_dir: Path) -> Tuple[sc.AnnData, List[str], np.ndarray]:
    """
    Load preprocessed data from disk.
    
    Args:
        load_dir: Directory containing the preprocessed data
        
    Returns:
        Tuple of (AnnData, genes, gene_tokens)
    """
    # Load AnnData
    adata_path = load_dir / "adata_nk_hvg.h5ad"
    adata = sc.read_h5ad(adata_path)
    
    # Load genes and tokens
    genes_path = load_dir / "genes.npy"
    gene_tokens_path = load_dir / "gene_tokens.npy"
    
    genes = np.load(genes_path, allow_pickle=True).tolist()
    gene_tokens = np.load(gene_tokens_path)
    
    logger.info(f"Loaded preprocessed data from {load_dir}")
    return adata, genes, gene_tokens 