"""
Main training script for the Transformer Immune Cells project.
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.tokenization import load_tokenized_dataset, clear_cuda_memory
from utils.model_utils import (
    setup_device, load_pretrained_model, create_training_arguments,
    create_trainer, freeze_base_layers, print_trainable_parameters,
    save_model_and_tokenizer, evaluate_model, plot_training_history
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Main model training pipeline.
    """
    logger.info("Starting model training pipeline")
    
    try:
        # Step 1: Setup device
        logger.info("Setting up device...")
        device = setup_device()
        
        # Step 2: Load tokenized datasets
        logger.info("Loading tokenized datasets...")
        tokenized_data_path = DATA_DIR / "tokenized_datasets"
        
        # Load small datasets for faster training/debugging
        small_train_path = tokenized_data_path / "small_train"
        small_test_path = tokenized_data_path / "small_test"
        
        train_dataset = load_tokenized_dataset(str(small_train_path))
        test_dataset = load_tokenized_dataset(str(small_test_path))
        
        logger.info(f"Loaded datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Step 3: Load pre-trained model
        logger.info("Loading pre-trained model...")
        model = load_pretrained_model(GENEFORMER_MODEL_NAME, NUM_LABELS)
        
        # Step 4: Create training arguments
        logger.info("Creating training arguments...")
        training_args = create_training_arguments(
            output_dir=str(MODELS_DIR / "geneformer_clf"),
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            logging_dir=str(LOGS_DIR),
            report_to="none"
        )
        
        # Step 5: Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        # Step 6: Freeze/unfreeze layers
        if FREEZE_BASE_LAYERS:
            logger.info("Freezing base layers...")
            freeze_base_layers(model, UNFREEZE_LAST_N_LAYERS)
            print_trainable_parameters(model)
        
        # Step 7: Clear CUDA memory before training
        logger.info("Clearing CUDA memory...")
        clear_cuda_memory()
        
        # Step 8: Train model
        logger.info("Starting model training...")
        trainer.train()
        
        # Step 9: Save model and tokenizer
        logger.info("Saving model and tokenizer...")
        model_save_path = MODELS_DIR / "trained_model"
        save_model_and_tokenizer(
            trainer=trainer,
            save_path=str(model_save_path),
            tokenizer_dict=trainer.tokenizer.gene_token_dict if hasattr(trainer, 'tokenizer') else {}
        )
        
        # Step 10: Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = evaluate_model(trainer, test_dataset)
        
        # Step 11: Plot training history
        logger.info("Plotting training history...")
        plot_save_path = RESULTS_DIR / "training_history.png"
        plot_training_history(trainer, str(plot_save_path))
        
        # Step 12: Save evaluation results
        logger.info("Saving evaluation results...")
        import json
        results_save_path = RESULTS_DIR / "evaluation_results.json"
        with open(results_save_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {
                'predictions': evaluation_results['predictions'].tolist(),
                'true_labels': evaluation_results['true_labels'].tolist(),
                'classification_report': evaluation_results['classification_report'],
                'metrics': evaluation_results['metrics']
            }
            json.dump(results_to_save, f, indent=2)
        
        logger.info("Model training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model: {GENEFORMER_MODEL_NAME}")
        print(f"Number of labels: {NUM_LABELS}")
        print(f"Training epochs: {NUM_EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Model saved to: {model_save_path}")
        print(f"Results saved to: {RESULTS_DIR}")
        print("="*50)
        
        # Print evaluation metrics
        if 'metrics' in evaluation_results:
            print("\nEVALUATION METRICS:")
            for metric, value in evaluation_results['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 