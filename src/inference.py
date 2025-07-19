"""
Inference script for making predictions with the trained model.
"""
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_processing import load_preprocessed_data
from utils.tokenization import tokenize_dataset, load_tokenized_dataset
from utils.model_utils import (
    load_saved_model, load_tokenizer_dict, setup_device
)
from transformers import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ImmuneCellPredictor:
    """
    Class for making predictions on immune cell data.
    """
    
    def __init__(self, model_path: str, num_labels: int = NUM_LABELS):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model
            num_labels: Number of classification labels
        """
        self.model_path = model_path
        self.num_labels = num_labels
        self.device = setup_device()
        
        # Load model and tokenizer
        self.model = load_saved_model(model_path, num_labels)
        self.tokenizer_dict = load_tokenizer_dict(model_path)
        
        # Create trainer for predictions
        self.trainer = Trainer(model=self.model)
        
        logger.info(f"Initialized predictor with model from {model_path}")
    
    def predict_from_adata(self, adata, label_key: str = None) -> dict:
        """
        Make predictions from AnnData object.
        
        Args:
            adata: AnnData object
            label_key: Optional key for true labels
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Create gene tokens
        genes = adata.var_names.to_list()
        gene_tokens = np.array([self.tokenizer_dict.get(g, 0) for g in genes], dtype=int)
        
        # Tokenize dataset
        tokenized_dataset = tokenize_dataset(
            adata, 
            gene_tokens, 
            max_len=MAX_SEQUENCE_LENGTH,
            label_key=label_key,
            n_jobs=N_JOBS
        )
        
        # Make predictions
        predictions = self.trainer.predict(tokenized_dataset)
        
        # Extract results
        pred_labels = predictions.predictions.argmax(-1)
        
        results = {
            'predictions': pred_labels,
            'prediction_probs': predictions.predictions,
            'dataset_size': len(tokenized_dataset)
        }
        
        # Add true labels if available
        if label_key and label_key in adata.obs.columns:
            true_labels = adata.obs[label_key].astype("category").cat.codes.values
            results['true_labels'] = true_labels
            
            # Calculate accuracy
            accuracy = np.mean(pred_labels == true_labels)
            results['accuracy'] = accuracy
            
            logger.info(f"Prediction accuracy: {accuracy:.4f}")
        
        logger.info(f"Made predictions for {len(tokenized_dataset)} cells")
        return results
    
    def predict_from_tokenized_dataset(self, dataset_path: str) -> dict:
        """
        Make predictions from pre-tokenized dataset.
        
        Args:
            dataset_path: Path to tokenized dataset
            
        Returns:
            Dictionary with predictions
        """
        # Load dataset
        dataset = load_tokenized_dataset(dataset_path)
        
        # Make predictions
        predictions = self.trainer.predict(dataset)
        
        # Extract results
        pred_labels = predictions.predictions.argmax(-1)
        
        results = {
            'predictions': pred_labels,
            'prediction_probs': predictions.predictions,
            'dataset_size': len(dataset)
        }
        
        # Add true labels if available
        if 'labels' in dataset.features:
            true_labels = dataset['labels']
            results['true_labels'] = true_labels
            
            # Calculate accuracy
            accuracy = np.mean(pred_labels == true_labels)
            results['accuracy'] = accuracy
            
            logger.info(f"Prediction accuracy: {accuracy:.4f}")
        
        logger.info(f"Made predictions for {len(dataset)} cells")
        return results
    
    def save_predictions(self, results: dict, save_path: str) -> None:
        """
        Save prediction results to file.
        
        Args:
            results: Prediction results dictionary
            save_path: Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_to_save[key] = value.tolist()
            else:
                results_to_save[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Saved predictions to {save_path}")


def main():
    """
    Main inference pipeline.
    """
    logger.info("Starting inference pipeline")
    
    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        adata, genes, gene_tokens = load_preprocessed_data(DATA_DIR)
        
        # Initialize predictor
        model_path = MODELS_DIR / "trained_model"
        predictor = ImmuneCellPredictor(str(model_path))
        
        # Make predictions on test dataset
        logger.info("Making predictions on test dataset...")
        test_dataset_path = DATA_DIR / "tokenized_datasets" / "small_test"
        
        if test_dataset_path.exists():
            results = predictor.predict_from_tokenized_dataset(str(test_dataset_path))
        else:
            # Fallback to using AnnData
            results = predictor.predict_from_adata(adata, label_key='subject.ageGroup')
        
        # Save predictions
        predictions_save_path = RESULTS_DIR / "predictions.json"
        predictor.save_predictions(results, str(predictions_save_path))
        
        # Print summary
        print("\n" + "="*50)
        print("INFERENCE SUMMARY")
        print("="*50)
        print(f"Model path: {model_path}")
        print(f"Dataset size: {results['dataset_size']}")
        if 'accuracy' in results:
            print(f"Prediction accuracy: {results['accuracy']:.4f}")
        print(f"Predictions saved to: {predictions_save_path}")
        print("="*50)
        
        # Show prediction distribution
        unique_preds, counts = np.unique(results['predictions'], return_counts=True)
        print("\nPREDICTION DISTRIBUTION:")
        for pred, count in zip(unique_preds, counts):
            print(f"  Class {pred}: {count} cells ({count/len(results['predictions'])*100:.1f}%)")
        
        logger.info("Inference pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 