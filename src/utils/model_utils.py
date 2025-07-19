"""
Model utilities for transformer training and evaluation.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report
)

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Setup and return the appropriate device (GPU/CPU).
    
    Returns:
        torch.device object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_pretrained_model(model_name: str, num_labels: int) -> AutoModelForSequenceClassification:
    """
    Load a pre-trained transformer model for sequence classification.
    
    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of classification labels
        
    Returns:
        Loaded model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    logger.info(f"Loaded pre-trained model: {model_name}")
    return model


def create_training_arguments(
    output_dir: str,
    per_device_train_batch_size: int = 128,
    per_device_eval_batch_size: int = 128,
    num_train_epochs: int = 10,
    learning_rate: float = 3e-5,
    logging_steps: int = 50,
    logging_dir: str = "./logs",
    report_to: str = "none"
) -> TrainingArguments:
    """
    Create training arguments for the trainer.
    
    Args:
        output_dir: Directory to save model outputs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        logging_steps: Number of steps between logging
        logging_dir: Directory for logs
        report_to: Reporting backend
        
    Returns:
        TrainingArguments object
    """
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        logging_strategy="steps",
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        logging_dir=logging_dir,
        report_to=report_to
    )
    
    logger.info("Created training arguments")
    return args


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    logger.info(f"Computed metrics: {metrics}")
    return metrics


def create_trainer(
    model: AutoModelForSequenceClassification,
    args: TrainingArguments,
    train_dataset,
    eval_dataset,
    compute_metrics_func=compute_metrics
) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        compute_metrics_func: Function to compute metrics
        
    Returns:
        Trainer instance
    """
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_func
    )
    
    logger.info("Created trainer instance")
    return trainer


def freeze_base_layers(model: AutoModelForSequenceClassification, 
                      unfreeze_last_n_layers: int = 1) -> None:
    """
    Freeze base model layers and unfreeze the last N layers.
    
    Args:
        model: The model to modify
        unfreeze_last_n_layers: Number of last layers to unfreeze
    """
    # Freeze all parameters
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze the last N transformer layers
    for name, param in model.base_model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num >= (model.base_model.config.num_hidden_layers - unfreeze_last_n_layers):
                param.requires_grad = True
    
    logger.info(f"Froze base layers and unfroze last {unfreeze_last_n_layers} layers")


def print_trainable_parameters(model: AutoModelForSequenceClassification) -> None:
    """
    Print which parameters are trainable.
    
    Args:
        model: The model to inspect
    """
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    logger.info(f"Trainable parameters: {len(trainable_params)}")
    for param in trainable_params:
        logger.info(f"  {param}")


def save_model_and_tokenizer(
    trainer: Trainer, 
    save_path: str, 
    tokenizer_dict: Dict[str, int]
) -> None:
    """
    Save the trained model and tokenizer dictionary.
    
    Args:
        trainer: Trained trainer instance
        save_path: Path to save the model
        tokenizer_dict: Tokenizer dictionary to save
    """
    # Save the model
    trainer.save_model(save_path)
    
    # Save the tokenizer dictionary
    tokenizer_path = Path(save_path) / "gene_token_dict.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_dict, f)
    
    logger.info(f"Saved model and tokenizer to {save_path}")


def evaluate_model(trainer: Trainer, test_dataset) -> Dict[str, Any]:
    """
    Evaluate the trained model on test dataset.
    
    Args:
        trainer: Trained trainer instance
        test_dataset: Test dataset
        
    Returns:
        Dictionary with evaluation results
    """
    # Get predictions
    preds_output = trainer.predict(test_dataset)
    
    # Extract predictions and labels
    pred_labels = preds_output.predictions.argmax(-1)
    true_labels = preds_output.label_ids
    
    # Get unique classes
    unique_classes = sorted(list(set(np.unique(true_labels)) | set(np.unique(pred_labels))))
    target_names = [f"class{i}" for i in unique_classes]
    
    # Generate classification report
    report = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)
    
    results = {
        "predictions": pred_labels,
        "true_labels": true_labels,
        "classification_report": report,
        "metrics": preds_output.metrics
    }
    
    logger.info("Model evaluation completed")
    return results


def plot_training_history(trainer: Trainer, save_path: Optional[str] = None) -> None:
    """
    Plot training history from trainer state.
    
    Args:
        trainer: Trained trainer instance
        save_path: Optional path to save the plot
    """
    history = trainer.state.log_history
    df_logs = pd.DataFrame(history)
    
    # Plot training loss
    df_loss = df_logs[df_logs["loss"].notna()]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df_loss["step"], df_loss["loss"], label="Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Steps")
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation metrics if available
    if "eval_loss" in df_logs.columns:
        df_eval = df_logs[df_logs["eval_loss"].notna()]
        plt.subplot(1, 2, 2)
        plt.plot(df_eval["step"], df_eval["eval_loss"], label="Evaluation Loss", color='red')
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss over Steps")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.show()


def load_saved_model(model_path: str, num_labels: int) -> AutoModelForSequenceClassification:
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model
        num_labels: Number of classification labels
        
    Returns:
        Loaded model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels
    )
    logger.info(f"Loaded saved model from {model_path}")
    return model


def load_tokenizer_dict(model_path: str) -> Dict[str, int]:
    """
    Load tokenizer dictionary from saved model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tokenizer dictionary
    """
    tokenizer_path = Path(model_path) / "gene_token_dict.json"
    with open(tokenizer_path, "r") as f:
        tokenizer_dict = json.load(f)
    
    logger.info(f"Loaded tokenizer dictionary from {tokenizer_path}")
    return tokenizer_dict 