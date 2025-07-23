#!/usr/bin/env python3
"""
Transformer Immune Cells - Model Training Pipeline
==================================================

This script handles the complete model training pipeline for immune cell analysis:
1. Loading preprocessed and tokenized data
2. Setting up transformer model (Geneformer)
3. Configuring training parameters
4. Training the model with proper evaluation
5. Saving trained model and results
6. Generating training reports and visualizations

Usage:
    python transformer_immune_cells_MODEL.py [options]

Options:
    --data-dir PATH       Directory containing preprocessed data
    --output-dir PATH     Output directory for model and results
    --model-name STR      Pre-trained model name (default: ctheodoris/Geneformer)
    --num-labels INT      Number of classification labels (default: 3)
    --batch-size INT      Batch size for training (default: 128)
    --learning-rate FLOAT Learning rate (default: 3e-5)
    --num-epochs INT      Number of training epochs (default: 10)
    --max-len INT         Maximum sequence length (default: 512)
    --freeze-layers       Freeze base model layers
    --unfreeze-layers INT Number of last layers to unfreeze (default: 1)
    --use-small-data      Use small datasets for faster training
    --logging-steps INT   Number of steps between logging (default: 50)
    --eval-steps INT      Number of steps between evaluation (default: 100)
    --save-steps INT      Number of steps between saving (default: 500)
    --warmup-steps INT    Number of warmup steps (default: 100)
    --weight-decay FLOAT  Weight decay (default: 0.01)
    --gradient-accumulation-steps INT Gradient accumulation steps (default: 1)
    --max-grad-norm FLOAT Maximum gradient norm (default: 1.0)
    --fp16                Use mixed precision training
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
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility functions
from src.utils.tokenization import load_tokenized_dataset, clear_cuda_memory
from src.utils.model_utils import (
    setup_device, load_pretrained_model, create_training_arguments,
    create_trainer, freeze_base_layers, print_trainable_parameters,
    save_model_and_tokenizer, evaluate_model, plot_training_history
)
from transformers import Trainer


class ModelTrainingPipeline:
    """
    Complete model training pipeline for transformer immune cell analysis.
    """
    
    def __init__(self, config):
        """
        Initialize the model training pipeline.
        
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
        (self.output_dir / "model").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized model training pipeline")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "logs" / "model_training.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_training_data(self):
        """
        Load preprocessed and tokenized training data.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        self.logger.info("Step 1: Loading training data")
        
        data_dir = Path(self.config['data_dir'])
        tokenized_dir = data_dir / "tokenized"
        
        # Determine which datasets to load
        if self.config.get('use_small_data', False):
            train_path = tokenized_dir / "small_train"
            test_path = tokenized_dir / "small_test"
            self.logger.info("Using small datasets for faster training")
        else:
            train_path = tokenized_dir / "train"
            test_path = tokenized_dir / "test"
            self.logger.info("Using full datasets")
        
        # Load datasets
        self.logger.info(f"Loading train dataset from: {train_path}")
        train_dataset = load_tokenized_dataset(str(train_path))
        
        self.logger.info(f"Loading test dataset from: {test_path}")
        test_dataset = load_tokenized_dataset(str(test_path))
        
        self.logger.info(f"Loaded datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_dataset, test_dataset
    
    def setup_model_and_training(self, train_dataset, test_dataset):
        """
        Setup model and training configuration.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (model, trainer, training_args)
        """
        self.logger.info("Step 2: Setting up model and training")
        
        # Setup device
        self.logger.info("Setting up device")
        device = setup_device()
        
        # Load pre-trained model
        self.logger.info(f"Loading pre-trained model: {self.config['model_name']}")
        model = load_pretrained_model(
            self.config['model_name'], 
            self.config['num_labels']
        )
        
        # Freeze/unfreeze layers if specified
        if self.config.get('freeze_layers', False):
            self.logger.info("Freezing base model layers")
            freeze_base_layers(model, self.config.get('unfreeze_layers', 1))
            print_trainable_parameters(model)
        
        # Create training arguments
        self.logger.info("Creating training arguments")
        training_args = create_training_arguments(
            output_dir=str(self.output_dir / "model" / "checkpoints"),
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            num_train_epochs=self.config['num_epochs'],
            learning_rate=self.config['learning_rate'],
            logging_steps=self.config.get('logging_steps', 50),
            logging_dir=str(self.output_dir / "logs"),
            report_to="none",
            evaluation_strategy="steps" if self.config.get('eval_steps') else "epoch",
            eval_steps=self.config.get('eval_steps'),
            save_strategy="steps" if self.config.get('save_steps') else "epoch",
            save_steps=self.config.get('save_steps'),
            warmup_steps=self.config.get('warmup_steps', 100),
            weight_decay=self.config.get('weight_decay', 0.01),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            fp16=self.config.get('fp16', False)
        )
        
        # Create trainer
        self.logger.info("Creating trainer")
        trainer = create_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        return model, trainer, training_args
    
    def train_model(self, trainer):
        """
        Train the model.
        
        Args:
            trainer: Configured trainer instance
        """
        self.logger.info("Step 3: Training model")
        
        # Clear CUDA memory before training
        self.logger.info("Clearing CUDA memory")
        clear_cuda_memory()
        
        # Start training
        self.logger.info("Starting model training")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time//60:.0f}m {training_time%60:.1f}s")
        
        return training_time
    
    def save_model_and_results(self, trainer, training_time):
        """
        Save the trained model and results.
        
        Args:
            trainer: Trained trainer instance
            training_time: Total training time
        """
        self.logger.info("Step 4: Saving model and results")
        
        # Save model
        model_save_path = self.output_dir / "model" / "final_model"
        trainer.save_model(str(model_save_path))
        self.logger.info(f"Saved model to: {model_save_path}")
        
        # Save tokenizer dictionary if available
        try:
            tokenizer_dict_path = Path(self.config['data_dir']) / "preprocessed" / "gene_token_dict.json"
            if tokenizer_dict_path.exists():
                import shutil
                shutil.copy(tokenizer_dict_path, model_save_path / "gene_token_dict.json")
                self.logger.info("Saved tokenizer dictionary")
        except Exception as e:
            self.logger.warning(f"Could not save tokenizer dictionary: {e}")
        
        # Save training configuration
        config_save_path = self.output_dir / "model" / "training_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.info(f"Saved training configuration to: {config_save_path}")
        
        # Save training time
        time_info = {
            'training_time_seconds': training_time,
            'training_time_formatted': f"{training_time//60:.0f}m {training_time%60:.1f}s"
        }
        time_save_path = self.output_dir / "model" / "training_time.json"
        with open(time_save_path, 'w') as f:
            json.dump(time_info, f, indent=2)
    
    def evaluate_model(self, trainer, test_dataset):
        """
        Evaluate the trained model.
        
        Args:
            trainer: Trained trainer instance
            test_dataset: Test dataset
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Step 5: Evaluating model")
        
        # Run evaluation
        evaluation_results = evaluate_model(trainer, test_dataset)
        
        # Save evaluation results
        results_save_path = self.output_dir / "results" / "evaluation_results.json"
        with open(results_save_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {}
            for key, value in evaluation_results.items():
                if isinstance(value, np.ndarray):
                    results_to_save[key] = value.tolist()
                else:
                    results_to_save[key] = value
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to: {results_save_path}")
        return evaluation_results
    
    def generate_plots(self, trainer, evaluation_results):
        """
        Generate training plots and visualizations.
        
        Args:
            trainer: Trained trainer instance
            evaluation_results: Evaluation results dictionary
        """
        self.logger.info("Step 6: Generating plots and visualizations")
        
        # Plot training history
        plot_save_path = self.output_dir / "plots" / "training_history.png"
        plot_training_history(trainer, str(plot_save_path))
        
        # Create confusion matrix plot if labels are available
        if 'true_labels' in evaluation_results and 'predictions' in evaluation_results:
            self.create_confusion_matrix_plot(evaluation_results)
        
        # Create metrics summary plot
        self.create_metrics_summary_plot(evaluation_results)
    
    def create_confusion_matrix_plot(self, evaluation_results):
        """Create confusion matrix plot."""
        from sklearn.metrics import confusion_matrix
        
        true_labels = evaluation_results['true_labels']
        predictions = evaluation_results['predictions']
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = self.output_dir / "plots" / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved confusion matrix plot to: {plot_path}")
    
    def create_metrics_summary_plot(self, evaluation_results):
        """Create metrics summary plot."""
        if 'classification_report' not in evaluation_results:
            return
        
        report = evaluation_results['classification_report']
        
        # Extract metrics for plotting
        metrics_data = []
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                metrics_data.append({
                    'class': class_name,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score']
                })
        
        if not metrics_data:
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(['precision', 'recall', 'f1-score']):
            axes[i].bar(df['class'], df[metric])
            axes[i].set_title(f'{metric.capitalize()} by Class')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / "metrics_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved metrics summary plot to: {plot_path}")
    
    def generate_training_report(self, train_dataset, test_dataset, training_time, evaluation_results):
        """
        Generate comprehensive training report.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            training_time: Total training time
            evaluation_results: Evaluation results
        """
        self.logger.info("Generating training report")
        
        report = {
            "training_summary": {
                "model_name": self.config['model_name'],
                "output_directory": str(self.output_dir),
                "training_time_seconds": training_time,
                "training_time_formatted": f"{training_time//60:.0f}m {training_time%60:.1f}s"
            },
            "dataset_info": {
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "total_samples": len(train_dataset) + len(test_dataset)
            },
            "model_configuration": {
                "num_labels": self.config['num_labels'],
                "batch_size": self.config['batch_size'],
                "learning_rate": self.config['learning_rate'],
                "num_epochs": self.config['num_epochs'],
                "max_len": self.config.get('max_len', 512),
                "freeze_layers": self.config.get('freeze_layers', False),
                "unfreeze_layers": self.config.get('unfreeze_layers', 1)
            },
            "evaluation_results": {
                "metrics": evaluation_results.get('metrics', {}),
                "classification_report": evaluation_results.get('classification_report', {})
            }
        }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"Model: {self.config['model_name']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Training time: {training_time//60:.0f}m {training_time%60:.1f}s")
        print(f"\nDataset info:")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Test samples: {len(test_dataset):,}")
        print(f"\nModel configuration:")
        print(f"  Number of labels: {self.config['num_labels']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Number of epochs: {self.config['num_epochs']}")
        print(f"\nEvaluation metrics:")
        if 'metrics' in evaluation_results:
            for metric, value in evaluation_results['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        print("="*60)
        
        self.logger.info(f"Training report saved to: {report_path}")
    
    def run(self):
        """
        Run the complete model training pipeline.
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting model training pipeline")
            
            # Step 1: Load training data
            train_dataset, test_dataset = self.load_training_data()
            
            # Step 2: Setup model and training
            model, trainer, training_args = self.setup_model_and_training(train_dataset, test_dataset)
            
            # Step 3: Train model
            training_time = self.train_model(trainer)
            
            # Step 4: Save model and results
            self.save_model_and_results(trainer, training_time)
            
            # Step 5: Evaluate model
            evaluation_results = self.evaluate_model(trainer, test_dataset)
            
            # Step 6: Generate plots
            self.generate_plots(trainer, evaluation_results)
            
            # Generate training report
            self.generate_training_report(train_dataset, test_dataset, training_time, evaluation_results)
            
            total_time = time.time() - start_time
            self.logger.info(f"Model training pipeline completed successfully in {total_time//60:.0f}m {total_time%60:.1f}s!")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training pipeline failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer Immune Cells - Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transformer_immune_cells_MODEL.py --data-dir processed_data --output-dir model_results
  python transformer_immune_cells_MODEL.py --data-dir processed_data --output-dir model_results --use-small-data --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing preprocessed data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for model and results'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='ctheodoris/Geneformer',
        help='Pre-trained model name (default: ctheodoris/Geneformer)'
    )
    
    parser.add_argument(
        '--num-labels',
        type=int,
        default=3,
        help='Number of classification labels (default: 3)'
    )
    
    # Training configuration
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training (default: 128)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate (default: 3e-5)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--max-len',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    # Layer freezing
    parser.add_argument(
        '--freeze-layers',
        action='store_true',
        help='Freeze base model layers'
    )
    
    parser.add_argument(
        '--unfreeze-layers',
        type=int,
        default=1,
        help='Number of last layers to unfreeze (default: 1)'
    )
    
    # Data options
    parser.add_argument(
        '--use-small-data',
        action='store_true',
        help='Use small datasets for faster training'
    )
    
    # Training optimization
    parser.add_argument(
        '--logging-steps',
        type=int,
        default=50,
        help='Number of steps between logging (default: 50)'
    )
    
    parser.add_argument(
        '--eval-steps',
        type=int,
        help='Number of steps between evaluation (default: epoch)'
    )
    
    parser.add_argument(
        '--save-steps',
        type=int,
        help='Number of steps between saving (default: epoch)'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='Number of warmup steps (default: 100)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )
    
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm (default: 1.0)'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training'
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
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'num_labels': args.num_labels,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_len': args.max_len,
        'freeze_layers': args.freeze_layers,
        'unfreeze_layers': args.unfreeze_layers,
        'use_small_data': args.use_small_data,
        'logging_steps': args.logging_steps,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_grad_norm': args.max_grad_norm,
        'fp16': args.fp16,
        'verbose': args.verbose
    }
    
    # Run pipeline
    pipeline = ModelTrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

