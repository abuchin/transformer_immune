#!/usr/bin/env python3
"""
Main pipeline script for the Transformer Immune Cells project.
This script orchestrates the entire workflow from data preprocessing to training to inference.
"""
import argparse
import logging
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import *
from src.data_preprocessing import main as run_data_preprocessing
from src.train_model import main as run_training
from src.inference import main as run_inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def run_full_pipeline():
    """
    Run the complete pipeline from data preprocessing to inference.
    """
    start_time = time.time()
    
    print("="*60)
    print("TRANSFORMER IMMUNE CELLS PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Data Preprocessing
        print("\n" + "="*40)
        print("STEP 1: DATA PREPROCESSING")
        print("="*40)
        run_data_preprocessing()
        
        # Step 2: Model Training
        print("\n" + "="*40)
        print("STEP 2: MODEL TRAINING")
        print("="*40)
        run_training()
        
        # Step 3: Inference
        print("\n" + "="*40)
        print("STEP 3: INFERENCE")
        print("="*40)
        run_inference()
        
        # Calculate total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Logs saved to: {LOGS_DIR}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPIPELINE FAILED: {e}")
        sys.exit(1)


def run_data_preprocessing_only():
    """
    Run only the data preprocessing step.
    """
    print("="*40)
    print("DATA PREPROCESSING ONLY")
    print("="*40)
    
    try:
        run_data_preprocessing()
        print("\nData preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        print(f"\nData preprocessing failed: {e}")
        sys.exit(1)


def run_training_only():
    """
    Run only the model training step.
    """
    print("="*40)
    print("MODEL TRAINING ONLY")
    print("="*40)
    
    try:
        run_training()
        print("\nModel training completed successfully!")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        print(f"\nModel training failed: {e}")
        sys.exit(1)


def run_inference_only():
    """
    Run only the inference step.
    """
    print("="*40)
    print("INFERENCE ONLY")
    print("="*40)
    
    try:
        run_inference()
        print("\nInference completed successfully!")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(f"\nInference failed: {e}")
        sys.exit(1)


def main():
    """
    Main function to parse arguments and run the appropriate pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Transformer Immune Cells Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --preprocess-only  # Run only data preprocessing
  python run_pipeline.py --train-only       # Run only model training
  python run_pipeline.py --inference-only   # Run only inference
        """
    )
    
    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Run only data preprocessing step'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Run only model training step'
    )
    
    parser.add_argument(
        '--inference-only',
        action='store_true',
        help='Run only inference step'
    )
    
    args = parser.parse_args()
    
    # Check for mutually exclusive arguments
    exclusive_args = [args.preprocess_only, args.train_only, args.inference_only]
    if sum(exclusive_args) > 1:
        parser.error("Only one of --preprocess-only, --train-only, or --inference-only can be specified")
    
    # Run appropriate pipeline
    if args.preprocess_only:
        run_data_preprocessing_only()
    elif args.train_only:
        run_training_only()
    elif args.inference_only:
        run_inference_only()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main() 