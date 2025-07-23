#!/usr/bin/env python3
"""
Transformer Immune Cells - Inference Pipeline
=============================================

This script handles inference with trained models for immune cell analysis:
1. Loading trained model and tokenizer
2. Loading new data for prediction
3. Making predictions
4. Saving results and generating reports

Usage:
    python transformer_immune_cells_INFERENCE.py [options]

Options:
    --model-dir PATH      Directory containing trained model
    --data-path PATH      Path to input data (.h5ad file or tokenized dataset)
    --output-dir PATH     Output directory for results
    --data-type STR       Type of input data: 'adata' or 'tokenized' (default: adata)
    --label-key STR       Label key for evaluation (optional)
    --batch-size INT      Batch size for inference (default: 128)
    --verbose             Enable verbose logging
    --help                Show this help message
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

# Import utility functions
from src.utils.data_processing import load_adata, preprocess_adata
from src.utils.tokenization import tokenize_dataset, load_tokenized_dataset
from src.utils.model_utils import (
    load_saved_model, load_tokenizer_dict, setup_device
)
from transformers import Trainer


class InferencePipeline:
    """
    Inference pipeline for transformer immune cell analysis.
    """
    
    def __init__(self, config):
        """
        Initialize the inference pipeline.
        
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
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized inference pipeline")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "logs" / "inference.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_model_and_tokenizer(self):
        """
        Load trained model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer_dict, num_labels)
        """
        self.logger.info("Step 1: Loading model and tokenizer")
        
        model_dir = Path(self.config['model_dir'])
        
        # Load model configuration
        config_path = model_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            num_labels = model_config.get('num_labels', 3)
        else:
            num_labels = 3  # Default fallback
            self.logger.warning("Could not load model config, using default num_labels=3")
        
        # Load model
        self.logger.info(f"Loading model from: {model_dir}")
        model = load_saved_model(str(model_dir), num_labels)
        
        # Load tokenizer dictionary
        self.logger.info("Loading tokenizer dictionary")
        tokenizer_dict = load_tokenizer_dict(str(model_dir))
        
        self.logger.info(f"Loaded model with {num_labels} labels")
        return model, tokenizer_dict, num_labels
    
    def load_and_prepare_data(self, tokenizer_dict):
        """
        Load and prepare data for inference.
        
        Args:
            tokenizer_dict: Tokenizer dictionary
            
        Returns:
            Tokenized dataset
        """
        self.logger.info("Step 2: Loading and preparing data")
        
        data_type = self.config.get('data_type', 'adata')
        
        if data_type == 'tokenized':
            # Load pre-tokenized dataset
            self.logger.info(f"Loading tokenized dataset from: {self.config['data_path']}")
            dataset = load_tokenized_dataset(self.config['data_path'])
            
        else:
            # Load AnnData and tokenize
            self.logger.info(f"Loading AnnData from: {self.config['data_path']}")
            adata = load_adata(self.config['data_path'])
            
            # Preprocess if needed
            if self.config.get('preprocess', True):
                self.logger.info("Preprocessing data")
                adata = preprocess_adata(adata)
            
            # Create gene tokens
            genes = adata.var_names.to_list()
            gene_tokens = np.array([tokenizer_dict.get(g, 0) for g in genes], dtype=int)
            
            # Tokenize dataset
            self.logger.info("Tokenizing dataset")
            dataset = tokenize_dataset(
                adata,
                gene_tokens,
                max_len=self.config.get('max_len', 512),
                label_key=self.config.get('label_key'),
                n_jobs=self.config.get('n_jobs', 4)
            )
        
        self.logger.info(f"Prepared dataset with {len(dataset)} samples")
        return dataset
    
    def run_inference(self, model, dataset):
        """
        Run inference on the dataset.
        
        Args:
            model: Loaded model
            dataset: Tokenized dataset
            
        Returns:
            Inference results dictionary
        """
        self.logger.info("Step 3: Running inference")
        
        # Create trainer for inference
        trainer = Trainer(model=model)
        
        # Run predictions
        self.logger.info("Making predictions")
        predictions = trainer.predict(dataset)
        
        # Extract results
        pred_labels = predictions.predictions.argmax(-1)
        pred_probs = predictions.predictions
        
        results = {
            'predictions': pred_labels,
            'prediction_probs': pred_probs,
            'dataset_size': len(dataset)
        }
        
        # Add true labels if available
        if 'labels' in dataset.features:
            true_labels = dataset['labels']
            results['true_labels'] = true_labels
            
            # Calculate accuracy
            accuracy = np.mean(pred_labels == true_labels)
            results['accuracy'] = accuracy
            
            self.logger.info(f"Prediction accuracy: {accuracy:.4f}")
        
        self.logger.info(f"Completed inference on {len(dataset)} samples")
        return results
    
    def save_results(self, results):
        """
        Save inference results.
        
        Args:
            results: Inference results dictionary
        """
        self.logger.info("Step 4: Saving results")
        
        # Save predictions
        predictions_path = self.output_dir / "predictions" / "predictions.json"
        with open(predictions_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results_to_save[key] = value.tolist()
                else:
                    results_to_save[key] = value
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"Saved predictions to: {predictions_path}")
        
        # Save prediction probabilities as CSV
        if 'prediction_probs' in results:
            probs_df = pd.DataFrame(
                results['prediction_probs'],
                columns=[f'class_{i}_prob' for i in range(results['prediction_probs'].shape[1])]
            )
            probs_path = self.output_dir / "predictions" / "prediction_probabilities.csv"
            probs_df.to_csv(probs_path, index=False)
            self.logger.info(f"Saved prediction probabilities to: {probs_path}")
        
        # Save prediction labels as CSV
        pred_df = pd.DataFrame({
            'sample_id': range(len(results['predictions'])),
            'predicted_label': results['predictions']
        })
        
        if 'true_labels' in results:
            pred_df['true_label'] = results['true_labels']
            pred_df['correct'] = pred_df['predicted_label'] == pred_df['true_label']
        
        labels_path = self.output_dir / "predictions" / "prediction_labels.csv"
        pred_df.to_csv(labels_path, index=False)
        self.logger.info(f"Saved prediction labels to: {labels_path}")
    
    def generate_inference_report(self, results, inference_time):
        """
        Generate inference report.
        
        Args:
            results: Inference results dictionary
            inference_time: Total inference time
        """
        self.logger.info("Generating inference report")
        
        report = {
            "inference_summary": {
                "model_directory": self.config['model_dir'],
                "data_path": self.config['data_path'],
                "output_directory": str(self.output_dir),
                "inference_time_seconds": inference_time,
                "inference_time_formatted": f"{inference_time//60:.0f}m {inference_time%60:.1f}s"
            },
            "dataset_info": {
                "total_samples": results['dataset_size'],
                "data_type": self.config.get('data_type', 'adata')
            },
            "prediction_summary": {
                "prediction_distribution": {}
            }
        }
        
        # Add prediction distribution
        unique_preds, counts = np.unique(results['predictions'], return_counts=True)
        for pred, count in zip(unique_preds, counts):
            report["prediction_summary"]["prediction_distribution"][f"class_{pred}"] = {
                "count": int(count),
                "percentage": float(count / len(results['predictions']) * 100)
            }
        
        # Add accuracy if available
        if 'accuracy' in results:
            report["prediction_summary"]["accuracy"] = float(results['accuracy'])
        
        # Save report
        report_path = self.output_dir / "inference_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"Model directory: {self.config['model_dir']}")
        print(f"Data path: {self.config['data_path']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Inference time: {inference_time//60:.0f}m {inference_time%60:.1f}s")
        print(f"\nDataset info:")
        print(f"  Total samples: {results['dataset_size']:,}")
        print(f"  Data type: {self.config.get('data_type', 'adata')}")
        print(f"\nPrediction distribution:")
        for pred, count in zip(unique_preds, counts):
            percentage = count / len(results['predictions']) * 100
            print(f"  Class {pred}: {count:,} samples ({percentage:.1f}%)")
        
        if 'accuracy' in results:
            print(f"\nAccuracy: {results['accuracy']:.4f}")
        
        print("="*60)
        
        self.logger.info(f"Inference report saved to: {report_path}")
    
    def run(self):
        """
        Run the complete inference pipeline.
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting inference pipeline")
            
            # Step 1: Load model and tokenizer
            model, tokenizer_dict, num_labels = self.load_model_and_tokenizer()
            
            # Step 2: Load and prepare data
            dataset = self.load_and_prepare_data(tokenizer_dict)
            
            # Step 3: Run inference
            results = self.run_inference(model, dataset)
            
            # Step 4: Save results
            self.save_results(results)
            
            # Generate inference report
            inference_time = time.time() - start_time
            self.generate_inference_report(results, inference_time)
            
            self.logger.info("Inference pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Inference pipeline failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer Immune Cells - Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transformer_immune_cells_INFERENCE.py --model-dir model_results/model/final_model --data-path new_data.h5ad --output-dir inference_results
  python transformer_immune_cells_INFERENCE.py --model-dir model_results/model/final_model --data-path tokenized_data --data-type tokenized --output-dir inference_results
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing trained model'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to input data (.h5ad file or tokenized dataset)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['adata', 'tokenized'],
        default='adata',
        help='Type of input data: adata or tokenized (default: adata)'
    )
    
    parser.add_argument(
        '--label-key',
        type=str,
        help='Label key for evaluation (optional)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference (default: 128)'
    )
    
    parser.add_argument(
        '--max-len',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='Number of parallel jobs for tokenization (default: 4)'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        default=True,
        help='Preprocess AnnData before tokenization (default: True)'
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
        'model_dir': args.model_dir,
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'data_type': args.data_type,
        'label_key': args.label_key,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'n_jobs': args.n_jobs,
        'preprocess': args.preprocess,
        'verbose': args.verbose
    }
    
    # Run pipeline
    pipeline = InferencePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main() 