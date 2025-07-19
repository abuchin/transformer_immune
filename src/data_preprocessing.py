"""
Main data preprocessing script for the Transformer Immune Cells project.
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_processing import (
    load_adata, preprocess_adata, convert_to_ensembl_ids, 
    select_highly_variable_genes, save_preprocessed_data
)
from utils.scvi_utils import run_full_scvi_pipeline
from utils.tokenization import (
    initialize_tokenizer, create_gene_tokens, tokenize_dataset,
    split_dataset, save_tokenized_dataset
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'data_preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Main data preprocessing pipeline.
    """
    logger.info("Starting data preprocessing pipeline")
    
    try:
        # Step 1: Load raw data
        logger.info("Loading raw data...")
        adata = load_adata(RAW_DATA_PATH)
        
        # Step 2: Preprocess data
        logger.info("Preprocessing data...")
        adata = preprocess_adata(adata, target_sum=TARGET_SUM)
        
        # Step 3: Convert to Ensembl IDs
        logger.info("Converting gene symbols to Ensembl IDs...")
        adata = convert_to_ensembl_ids(adata, species=SPECIES)
        
        # Step 4: Select highly variable genes
        logger.info("Selecting highly variable genes...")
        adata = select_highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
        
        # Step 5: Run scVI pipeline (optional)
        logger.info("Running scVI pipeline...")
        adata_scvi = run_full_scvi_pipeline(
            adata, 
            batch_key='subject.ageGroup',
            max_epochs=10,
            n_top_genes=N_TOP_GENES
        )
        
        # Step 6: Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = initialize_tokenizer()
        
        # Step 7: Create gene tokens
        logger.info("Creating gene tokens...")
        genes = adata.var_names.to_list()
        gene_tokens = create_gene_tokens(genes, tokenizer.gene_token_dict)
        
        # Step 8: Save preprocessed data
        logger.info("Saving preprocessed data...")
        save_preprocessed_data(adata, genes, gene_tokens, DATA_DIR)
        
        # Step 9: Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = tokenize_dataset(
            adata, 
            gene_tokens, 
            max_len=MAX_SEQUENCE_LENGTH,
            label_key='subject.ageGroup',
            n_jobs=N_JOBS
        )
        
        # Step 10: Split dataset
        logger.info("Splitting dataset...")
        datasets = split_dataset(
            tokenized_dataset,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            seed=RANDOM_SEED,
            small_fraction=0.25  # Create smaller datasets for debugging
        )
        
        # Step 11: Save tokenized datasets
        logger.info("Saving tokenized datasets...")
        tokenized_data_path = DATA_DIR / "tokenized_datasets"
        save_tokenized_dataset(tokenized_dataset, str(tokenized_data_path))
        
        # Save individual splits
        for split_name, dataset in datasets.items():
            split_path = tokenized_data_path / split_name
            save_tokenized_dataset(dataset, str(split_path))
        
        logger.info("Data preprocessing pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Original data shape: {adata.shape}")
        print(f"Number of genes: {len(genes)}")
        print(f"Number of cells: {adata.n_obs}")
        print(f"Train dataset size: {len(datasets['train'])}")
        print(f"Test dataset size: {len(datasets['test'])}")
        if 'small_train' in datasets:
            print(f"Small train dataset size: {len(datasets['small_train'])}")
            print(f"Small test dataset size: {len(datasets['small_test'])}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in data preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 