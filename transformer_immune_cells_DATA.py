#!/usr/bin/env python3
"""
Transformer Immune Cells - Data Preparation Pipeline
===================================================

This script handles the complete data preparation pipeline for immune cell analysis:
1. Loading and preprocessing single-cell RNA-seq data
2. Gene symbol to Ensembl ID mapping
3. Highly variable gene selection
4. scVI latent space analysis
5. Tokenization for transformer models
6. Dataset splitting and saving

Usage:
    python transformer_immune_cells_DATA.py [options]

Options:
    --input-path PATH     Path to input .h5ad file
    --output-dir PATH     Output directory for processed data
    --n-genes INT         Number of highly variable genes (default: 2000)
    --max-len INT         Maximum sequence length (default: 512)
    --test-split FLOAT    Test split ratio (default: 0.2)
    --small-fraction FLOAT Fraction for small datasets (default: 0.25)
    --n-jobs INT          Number of parallel jobs (default: 4)
    --species STR         Species for gene mapping (default: human)
    --batch-key STR       Batch key for scVI (default: subject.ageGroup)
    --label-key STR       Label key for classification (default: subject.ageGroup)
    --scvi-epochs INT     Number of scVI training epochs (default: 10)
    --verbose             Enable verbose logging
    --help                Show this help message
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import mygene
from joblib import Parallel, delayed
from tqdm import tqdm
from datasets import Dataset as HFDataset
import json

# Import utility functions
from src.utils.data_processing import (
    load_adata, preprocess_adata, convert_to_ensembl_ids, 
    select_highly_variable_genes, save_preprocessed_data
)
from src.utils.scvi_utils import run_full_scvi_pipeline
from src.utils.tokenization import (
    initialize_tokenizer, create_gene_tokens, tokenize_dataset,
    split_dataset, save_tokenized_dataset
)


class DataPreparationPipeline:
    """
    Complete data preparation pipeline for transformer immune cell analysis.
    """
    
    def __init__(self, config):
        """
        Initialize the data preparation pipeline.
        
        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "preprocessed").mkdir(exist_ok=True)
        (self.output_dir / "tokenized").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized data preparation pipeline")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "logs" / "data_preparation.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the raw data.
        
        Returns:
            Preprocessed AnnData object
        """
        self.logger.info("Step 1: Loading and preprocessing data")
        
        # Load raw data
        self.logger.info(f"Loading data from: {self.config['input_path']}")
        adata = load_adata(self.config['input_path'])
        
        # Preprocess data
        self.logger.info("Preprocessing data (normalization, log transformation)")
        adata = preprocess_adata(adata, target_sum=self.config.get('target_sum', 1e6))
        
        # Convert to Ensembl IDs
        self.logger.info("Converting gene symbols to Ensembl IDs")
        adata = convert_to_ensembl_ids(adata, species=self.config['species'])
        
        # Select highly variable genes
        self.logger.info(f"Selecting {self.config['n_genes']} highly variable genes")
        adata = select_highly_variable_genes(adata, n_top_genes=self.config['n_genes'])
        
        self.logger.info(f"Preprocessing completed. Final shape: {adata.shape}")
        return adata
    
    def run_scvi_analysis(self, adata):
        """
        Run scVI analysis for latent space representation.
        
        Args:
            adata: Preprocessed AnnData object
            
        Returns:
            AnnData object with scVI latent representation
        """
        self.logger.info("Step 2: Running scVI analysis")
        
        try:
            adata_scvi = run_full_scvi_pipeline(
                adata,
                batch_key=self.config['batch_key'],
                max_epochs=self.config['scvi_epochs'],
                n_top_genes=self.config['n_genes']
            )
            
            # Save scVI results
            scvi_path = self.output_dir / "preprocessed" / "adata_scvi.h5ad"
            adata_scvi.write_h5ad(scvi_path)
            self.logger.info(f"Saved scVI results to: {scvi_path}")
            
            return adata_scvi
            
        except Exception as e:
            self.logger.warning(f"scVI analysis failed: {e}")
            self.logger.info("Continuing without scVI analysis...")
            return adata
    
    def prepare_tokenization(self, adata):
        """
        Prepare data for tokenization.
        
        Args:
            adata: Preprocessed AnnData object
            
        Returns:
            Tuple of (genes, gene_tokens, tokenizer)
        """
        self.logger.info("Step 3: Preparing tokenization")
        
        # Initialize tokenizer
        self.logger.info("Initializing Geneformer tokenizer")
        tokenizer = initialize_tokenizer()
        
        # Create gene tokens
        self.logger.info("Creating gene tokens")
        genes = adata.var_names.to_list()
        gene_tokens = create_gene_tokens(genes, tokenizer.gene_token_dict)
        
        self.logger.info(f"Created tokens for {len(genes)} genes")
        return genes, gene_tokens, tokenizer
    
    def tokenize_and_split_data(self, adata, gene_tokens):
        """
        Tokenize data and split into train/test sets.
        
        Args:
            adata: Preprocessed AnnData object
            gene_tokens: Array of gene tokens
            
        Returns:
            Dictionary with train/test datasets
        """
        self.logger.info("Step 4: Tokenizing and splitting data")
        
        # Tokenize dataset
        self.logger.info("Tokenizing dataset")
        tokenized_dataset = tokenize_dataset(
            adata,
            gene_tokens,
            max_len=self.config['max_len'],
            label_key=self.config['label_key'],
            n_jobs=self.config['n_jobs']
        )
        
        # Split dataset
        self.logger.info("Splitting dataset")
        datasets = split_dataset(
            tokenized_dataset,
            test_size=self.config['test_split'],
            seed=self.config.get('random_seed', 42),
            small_fraction=self.config.get('small_fraction', 0.25)
        )
        
        return datasets, tokenized_dataset
    
    def save_processed_data(self, adata, genes, gene_tokens, datasets, tokenized_dataset):
        """
        Save all processed data to disk.
        
        Args:
            adata: Preprocessed AnnData object
            genes: List of gene names
            gene_tokens: Array of gene tokens
            datasets: Dictionary with train/test datasets
            tokenized_dataset: Full tokenized dataset
        """
        self.logger.info("Step 5: Saving processed data")
        
        # Save preprocessed AnnData
        adata_path = self.output_dir / "preprocessed" / "adata_processed.h5ad"
        adata.write_h5ad(adata_path)
        self.logger.info(f"Saved preprocessed AnnData to: {adata_path}")
        
        # Save genes and tokens
        genes_path = self.output_dir / "preprocessed" / "genes.npy"
        tokens_path = self.output_dir / "preprocessed" / "gene_tokens.npy"
        np.save(genes_path, genes)
        np.save(tokens_path, gene_tokens)
        self.logger.info(f"Saved genes and tokens to: {genes_path}, {tokens_path}")
        
        # Save tokenizer dictionary
        tokenizer_dict_path = self.output_dir / "preprocessed" / "gene_token_dict.json"
        with open(tokenizer_dict_path, 'w') as f:
            json.dump(tokenizer.gene_token_dict, f)
        self.logger.info(f"Saved tokenizer dictionary to: {tokenizer_dict_path}")
        
        # Save tokenized datasets
        tokenized_dir = self.output_dir / "tokenized"
        
        # Save full dataset
        full_dataset_path = tokenized_dir / "full_dataset"
        save_tokenized_dataset(tokenized_dataset, str(full_dataset_path))
        
        # Save individual splits
        for split_name, dataset in datasets.items():
            split_path = tokenized_dir / split_name
            save_tokenized_dataset(dataset, str(split_path))
            self.logger.info(f"Saved {split_name} dataset ({len(dataset)} samples)")
        
        # Save dataset info
        dataset_info = {
            'total_samples': len(tokenized_dataset),
            'train_samples': len(datasets['train']),
            'test_samples': len(datasets['test']),
            'n_genes': len(genes),
            'max_sequence_length': self.config['max_len'],
            'test_split_ratio': self.config['test_split']
        }
        
        if 'small_train' in datasets:
            dataset_info.update({
                'small_train_samples': len(datasets['small_train']),
                'small_test_samples': len(datasets['small_test'])
            })
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f"Saved dataset info to: {info_path}")
    
    def generate_summary_report(self, adata, datasets, processing_time):
        """
        Generate a summary report of the data preparation process.
        
        Args:
            adata: Preprocessed AnnData object
            datasets: Dictionary with train/test datasets
            processing_time: Total processing time
        """
        self.logger.info("Generating summary report")
        
        report = {
            "data_preparation_summary": {
                "input_file": self.config['input_path'],
                "output_directory": str(self.output_dir),
                "processing_time_seconds": processing_time,
                "processing_time_formatted": f"{processing_time//60:.0f}m {processing_time%60:.1f}s"
            },
            "data_statistics": {
                "original_shape": adata.shape,
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "n_highly_variable_genes": self.config['n_genes']
            },
            "dataset_splits": {
                "train_samples": len(datasets['train']),
                "test_samples": len(datasets['test']),
                "test_split_ratio": self.config['test_split']
            },
            "configuration": {
                "max_sequence_length": self.config['max_len'],
                "species": self.config['species'],
                "batch_key": self.config['batch_key'],
                "label_key": self.config['label_key'],
                "n_jobs": self.config['n_jobs']
            }
        }
        
        if 'small_train' in datasets:
            report["dataset_splits"].update({
                "small_train_samples": len(datasets['small_train']),
                "small_test_samples": len(datasets['small_test'])
            })
        
        # Save report
        report_path = self.output_dir / "data_preparation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)
        print(f"Input file: {self.config['input_path']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Processing time: {processing_time//60:.0f}m {processing_time%60:.1f}s")
        print(f"\nData statistics:")
        print(f"  Original shape: {adata.shape}")
        print(f"  Number of cells: {adata.n_obs:,}")
        print(f"  Number of genes: {adata.n_vars:,}")
        print(f"  Highly variable genes: {self.config['n_genes']:,}")
        print(f"\nDataset splits:")
        print(f"  Train samples: {len(datasets['train']):,}")
        print(f"  Test samples: {len(datasets['test']):,}")
        if 'small_train' in datasets:
            print(f"  Small train samples: {len(datasets['small_train']):,}")
            print(f"  Small test samples: {len(datasets['small_test']):,}")
        print("="*60)
        
        self.logger.info(f"Summary report saved to: {report_path}")
    
    def run(self):
        """
        Run the complete data preparation pipeline.
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting data preparation pipeline")
            
            # Step 1: Load and preprocess data
            adata = self.load_and_preprocess_data()
            
            # Step 2: Run scVI analysis
            adata_scvi = self.run_scvi_analysis(adata)
            
            # Step 3: Prepare tokenization
            genes, gene_tokens, tokenizer = self.prepare_tokenization(adata)
            
            # Step 4: Tokenize and split data
            datasets, tokenized_dataset = self.tokenize_and_split_data(adata, gene_tokens)
            
            # Step 5: Save processed data
            self.save_processed_data(adata, genes, gene_tokens, datasets, tokenized_dataset)
            
            # Generate summary report
            processing_time = time.time() - start_time
            self.generate_summary_report(adata, datasets, processing_time)
            
            self.logger.info("Data preparation pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Data preparation pipeline failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer Immune Cells - Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transformer_immune_cells_DATA.py --input-path data/raw_data.h5ad --output-dir processed_data
  python transformer_immune_cells_DATA.py --input-path data/raw_data.h5ad --output-dir processed_data --n-genes 3000 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to input .h5ad file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--n-genes',
        type=int,
        default=2000,
        help='Number of highly variable genes (default: 2000)'
    )
    
    parser.add_argument(
        '--max-len',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test split ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--small-fraction',
        type=float,
        default=0.25,
        help='Fraction for small datasets (default: 0.25)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='Number of parallel jobs (default: 4)'
    )
    
    parser.add_argument(
        '--species',
        type=str,
        default='human',
        help='Species for gene mapping (default: human)'
    )
    
    parser.add_argument(
        '--batch-key',
        type=str,
        default='subject.ageGroup',
        help='Batch key for scVI (default: subject.ageGroup)'
    )
    
    parser.add_argument(
        '--label-key',
        type=str,
        default='subject.ageGroup',
        help='Label key for classification (default: subject.ageGroup)'
    )
    
    parser.add_argument(
        '--scvi-epochs',
        type=int,
        default=10,
        help='Number of scVI training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--target-sum',
        type=float,
        default=1e6,
        help='Target sum for normalization (default: 1e6)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create configuration dictionary
    config = {
        'input_path': args.input_path,
        'output_dir': args.output_dir,
        'n_genes': args.n_genes,
        'max_len': args.max_len,
        'test_split': args.test_split,
        'small_fraction': args.small_fraction,
        'n_jobs': args.n_jobs,
        'species': args.species,
        'batch_key': args.batch_key,
        'label_key': args.label_key,
        'scvi_epochs': args.scvi_epochs,
        'target_sum': args.target_sum,
        'random_seed': args.random_seed,
        'verbose': args.verbose
    }
    
    # Run pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

