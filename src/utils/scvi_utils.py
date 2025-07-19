"""
scVI utilities for single-cell data analysis.
"""
import scanpy as sc
import scvi
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_scvi_data(adata: sc.AnnData, batch_key: str = 'subject.ageGroup', 
                   layer: str = "counts", n_top_genes: int = 2000) -> sc.AnnData:
    """
    Setup AnnData for scVI training.
    
    Args:
        adata: AnnData object
        batch_key: Column name for batch information
        layer: Layer to use for scVI
        n_top_genes: Number of highly variable genes to select
        
    Returns:
        Prepared AnnData object for scVI
    """
    adata_scvi = adata.copy()
    
    # Setup scVI
    scvi.model.SCVI.setup_anndata(adata_scvi, layer=layer, batch_key=batch_key)
    
    # Normalize
    sc.pp.normalize_total(adata_scvi, target_sum=1e6)
    sc.pp.log1p(adata_scvi)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata_scvi, n_top_genes=n_top_genes)
    
    logger.info(f"Setup scVI data with shape {adata_scvi.shape}")
    return adata_scvi


def train_scvi_model(adata: sc.AnnData, max_epochs: int = 10) -> scvi.model.SCVI:
    """
    Train scVI model.
    
    Args:
        adata: Prepared AnnData object
        max_epochs: Maximum number of training epochs
        
    Returns:
        Trained scVI model
    """
    model = scvi.model.SCVI(adata)
    model.train(max_epochs=max_epochs)
    
    logger.info(f"Trained scVI model for {max_epochs} epochs")
    return model


def extract_latent_representation(adata: sc.AnnData, model: scvi.model.SCVI) -> sc.AnnData:
    """
    Extract latent representation from trained scVI model.
    
    Args:
        adata: AnnData object
        model: Trained scVI model
        
    Returns:
        AnnData object with latent representation added
    """
    adata.obsm["X_scVI"] = model.get_latent_representation()
    
    logger.info("Extracted latent representation from scVI model")
    return adata


def run_umap_on_latent(adata: sc.AnnData, use_rep: str = "X_scVI") -> sc.AnnData:
    """
    Run UMAP on latent representation.
    
    Args:
        adata: AnnData object with latent representation
        use_rep: Key for representation to use
        
    Returns:
        AnnData object with UMAP coordinates
    """
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    
    logger.info("Computed UMAP on latent representation")
    return adata


def plot_umap(adata: sc.AnnData, color: list, save_path: Optional[str] = None):
    """
    Plot UMAP with specified color variables.
    
    Args:
        adata: AnnData object with UMAP coordinates
        color: List of variables to color by
        save_path: Optional path to save the plot
    """
    sc.pl.umap(adata, color=color, save=save_path)
    logger.info(f"Plotted UMAP colored by {color}")


def run_full_scvi_pipeline(adata: sc.AnnData, batch_key: str = 'subject.ageGroup',
                          max_epochs: int = 10, n_top_genes: int = 2000) -> sc.AnnData:
    """
    Run complete scVI pipeline from setup to UMAP.
    
    Args:
        adata: Input AnnData object
        batch_key: Column name for batch information
        max_epochs: Maximum number of training epochs
        n_top_genes: Number of highly variable genes
        
    Returns:
        AnnData object with scVI latent representation and UMAP
    """
    # Setup data
    adata_scvi = setup_scvi_data(adata, batch_key=batch_key, n_top_genes=n_top_genes)
    
    # Train model
    model = train_scvi_model(adata_scvi, max_epochs=max_epochs)
    
    # Extract latent representation
    adata_scvi = extract_latent_representation(adata_scvi, model)
    
    # Run UMAP
    adata_scvi = run_umap_on_latent(adata_scvi)
    
    logger.info("Completed full scVI pipeline")
    return adata_scvi 