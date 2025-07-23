# Transformer Immune Cells - Production Pipelines

This repository contains production-ready pipelines for analyzing immune cell data using transformer models (Geneformer). The code is organized into three main pipelines:

1. **Data Preparation Pipeline** (`transformer_immune_cells_DATA.py`)
2. **Model Training Pipeline** (`transformer_immune_cells_MODEL.py`)
3. **Inference Pipeline** (`transformer_immune_cells_INFERENCE.py`)

## 🏗️ Project Structure

```
transformer_immune/
├── notebook/
│   ├── transformer_immune_cells_DATA.ipynb   # Data preparation notebook
│   └── transformer_immune_cells_MODEL.ipynb  # Model training notebook
├── transformer_immune_cells_DATA.py          # Data preparation pipeline
├── transformer_immune_cells_MODEL.py         # Model training pipeline
├── transformer_immune_cells_INFERENCE.py     # Inference pipeline
├── src/                                  # Utility modules
│   ├── config.py                        # Configuration settings
│   ├── utils/                           # Utility functions
│   │   ├── data_processing.py          # Data processing utilities
│   │   ├── scvi_utils.py               # scVI model utilities
│   │   ├── tokenization.py             # Tokenization utilities
│   │   └── model_utils.py              # Model training utilities
├── requirements.txt                     # Python dependencies
├── requirements_data.txt                # Data processing dependencies
├── requirements_model.txt               # Model training dependencies
├── README.md                           # Original README
└── README_PIPELINES.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Geneformer (if not already installed)
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
pip install ./Geneformer
```

### 2. Data Preparation

```bash
# Prepare data for training
python transformer_immune_cells_DATA.py \
    --input-path data/raw_data.h5ad \
    --output-dir processed_data \
    --n-genes 2000 \
    --verbose
```

### 3. Model Training

```bash
# Train the model
python transformer_immune_cells_MODEL.py \
    --data-dir processed_data \
    --output-dir model_results \
    --num-epochs 10 \
    --batch-size 128 \
    --freeze-layers \
    --use-small-data \
    --verbose
```

### 4. Inference

```bash
# Run inference on new data
python transformer_immune_cells_INFERENCE.py \
    --model-dir model_results/model/final_model \
    --data-path new_data.h5ad \
    --output-dir inference_results \
    --verbose
```

## 📋 Pipeline Details

### 1. Data Preparation Pipeline (`transformer_immune_cells_DATA.py`)

**Purpose**: Preprocess raw single-cell RNA-seq data and prepare it for transformer model training.

**Features**:
- Load and preprocess AnnData objects
- Gene symbol to Ensembl ID mapping using MyGene.info
- Highly variable gene selection
- scVI latent space analysis
- Tokenization for transformer models
- Dataset splitting (train/test)
- Comprehensive logging and reporting

**Output Structure**:
```
processed_data/
├── preprocessed/
│   ├── adata_processed.h5ad          # Preprocessed AnnData
│   ├── adata_scvi.h5ad              # scVI results
│   ├── genes.npy                    # Gene names
│   ├── gene_tokens.npy              # Gene tokens
│   └── gene_token_dict.json         # Tokenizer dictionary
├── tokenized/
│   ├── full_dataset/                # Complete tokenized dataset
│   ├── train/                       # Training dataset
│   ├── test/                        # Test dataset
│   ├── small_train/                 # Small training dataset (for debugging)
│   └── small_test/                  # Small test dataset (for debugging)
├── dataset_info.json                # Dataset statistics
├── data_preparation_report.json     # Processing report
└── logs/
    └── data_preparation.log         # Processing logs
```

**Usage Examples**:
```bash
# Basic usage
python transformer_immune_cells_DATA.py \
    --input-path data/raw_data.h5ad \
    --output-dir processed_data

# Advanced usage with custom parameters
python transformer_immune_cells_DATA.py \
    --input-path data/raw_data.h5ad \
    --output-dir processed_data \
    --n-genes 3000 \
    --max-len 1024 \
    --test-split 0.3 \
    --species mouse \
    --batch-key sample_id \
    --label-key cell_type \
    --scvi-epochs 20 \
    --n-jobs 8 \
    --verbose
```

### 2. Model Training Pipeline (`transformer_immune_cells_MODEL.py`)

**Purpose**: Train transformer models on preprocessed data for immune cell classification.

**Features**:
- Load preprocessed and tokenized data
- Setup Geneformer model with custom configurations
- Layer freezing/unfreezing strategies
- Comprehensive training with evaluation
- Model checkpointing and saving
- Training visualization and reporting
- GPU/CPU support with automatic device detection

**Output Structure**:
```
model_results/
├── model/
│   ├── final_model/                  # Trained model
│   ├── checkpoints/                  # Training checkpoints
│   ├── training_config.json          # Training configuration
│   └── training_time.json            # Training time info
├── results/
│   └── evaluation_results.json       # Evaluation metrics
├── plots/
│   ├── training_history.png          # Training curves
│   ├── confusion_matrix.png          # Confusion matrix
│   └── metrics_summary.png           # Metrics by class
├── training_report.json              # Comprehensive report
└── logs/
    └── model_training.log            # Training logs
```

**Usage Examples**:
```bash
# Basic training
python transformer_immune_cells_MODEL.py \
    --data-dir processed_data \
    --output-dir model_results

# Advanced training with custom parameters
python transformer_immune_cells_MODEL.py \
    --data-dir processed_data \
    --output-dir model_results \
    --model-name ctheodoris/Geneformer \
    --num-labels 5 \
    --batch-size 64 \
    --learning-rate 1e-5 \
    --num-epochs 20 \
    --freeze-layers \
    --unfreeze-layers 2 \
    --use-small-data \
    --eval-steps 100 \
    --save-steps 500 \
    --fp16 \
    --verbose
```

### 3. Inference Pipeline (`transformer_immune_cells_INFERENCE.py`)

**Purpose**: Make predictions on new data using trained models.

**Features**:
- Load trained models and tokenizers
- Support for both AnnData and pre-tokenized data
- Batch prediction with configurable batch sizes
- Comprehensive result saving (JSON, CSV)
- Prediction visualization and reporting
- Accuracy calculation (if true labels available)

**Output Structure**:
```
inference_results/
├── predictions/
│   ├── predictions.json              # Raw predictions
│   ├── prediction_probabilities.csv  # Prediction probabilities
│   └── prediction_labels.csv         # Prediction labels
├── inference_report.json             # Inference summary
└── logs/
    └── inference.log                 # Inference logs
```

**Usage Examples**:
```bash
# Inference on new AnnData
python transformer_immune_cells_INFERENCE.py \
    --model-dir model_results/model/final_model \
    --data-path new_data.h5ad \
    --output-dir inference_results

# Inference on pre-tokenized data
python transformer_immune_cells_INFERENCE.py \
    --model-dir model_results/model/final_model \
    --data-path processed_data/tokenized/test \
    --data-type tokenized \
    --output-dir inference_results

# Inference with evaluation
python transformer_immune_cells_INFERENCE.py \
    --model-dir model_results/model/final_model \
    --data-path test_data.h5ad \
    --output-dir inference_results \
    --label-key cell_type \
    --batch-size 256 \
    --verbose
```

## ⚙️ Configuration Options

### Data Preparation Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-path` | Required | Path to input .h5ad file |
| `--output-dir` | Required | Output directory for processed data |
| `--n-genes` | 2000 | Number of highly variable genes |
| `--max-len` | 512 | Maximum sequence length |
| `--test-split` | 0.2 | Test split ratio |
| `--small-fraction` | 0.25 | Fraction for small datasets |
| `--n-jobs` | 4 | Number of parallel jobs |
| `--species` | human | Species for gene mapping |
| `--batch-key` | subject.ageGroup | Batch key for scVI |
| `--label-key` | subject.ageGroup | Label key for classification |
| `--scvi-epochs` | 10 | Number of scVI training epochs |

### Model Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | Required | Directory containing preprocessed data |
| `--output-dir` | Required | Output directory for model and results |
| `--model-name` | ctheodoris/Geneformer | Pre-trained model name |
| `--num-labels` | 3 | Number of classification labels |
| `--batch-size` | 128 | Batch size for training |
| `--learning-rate` | 3e-5 | Learning rate |
| `--num-epochs` | 10 | Number of training epochs |
| `--freeze-layers` | False | Freeze base model layers |
| `--unfreeze-layers` | 1 | Number of last layers to unfreeze |
| `--use-small-data` | False | Use small datasets for faster training |
| `--fp16` | False | Use mixed precision training |

### Inference Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-dir` | Required | Directory containing trained model |
| `--data-path` | Required | Path to input data |
| `--output-dir` | Required | Output directory for results |
| `--data-type` | adata | Type of input data (adata/tokenized) |
| `--label-key` | None | Label key for evaluation |
| `--batch-size` | 128 | Batch size for inference |

## 🔧 Advanced Usage

### Custom Configuration

You can modify the default parameters in `src/config.py` or pass them as command-line arguments.

### Parallel Processing

For large datasets, increase the `--n-jobs` parameter in data preparation:

```bash
python transformer_immune_cells_DATA.py \
    --input-path large_dataset.h5ad \
    --output-dir processed_data \
    --n-jobs 16
```

### GPU Training

The model training pipeline automatically detects and uses available GPUs. For mixed precision training:

```bash
python transformer_immune_cells_MODEL.py \
    --data-dir processed_data \
    --output-dir model_results \
    --fp16 \
    --batch-size 256
```

### Debug Mode

Use small datasets for faster development and testing:

```bash
# Data preparation with small datasets
python transformer_immune_cells_DATA.py \
    --input-path data/raw_data.h5ad \
    --output-dir processed_data \
    --small-fraction 0.1

# Model training with small datasets
python transformer_immune_cells_MODEL.py \
    --data-dir processed_data \
    --output-dir model_results \
    --use-small-data
```

## 📊 Output and Reports

Each pipeline generates comprehensive reports and logs:

### Data Preparation Reports
- Dataset statistics (cell count, gene count, etc.)
- Processing time and configuration
- scVI analysis results
- Tokenization statistics

### Model Training Reports
- Training configuration and hyperparameters
- Training time and performance metrics
- Evaluation results (accuracy, precision, recall, F1)
- Training curves and visualizations

### Inference Reports
- Prediction distribution across classes
- Accuracy metrics (if labels available)
- Processing time and configuration

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use smaller datasets
2. **CUDA Out of Memory**: Enable mixed precision (`--fp16`) or reduce batch size
3. **Gene Mapping Failures**: Check gene symbol format and species setting
4. **Missing Dependencies**: Ensure all packages are installed from requirements.txt

### Debug Mode

For debugging, use the `--verbose` flag and small datasets:

```bash
python transformer_immune_cells_DATA.py \
    --input-path data/raw_data.h5ad \
    --output-dir processed_data \
    --small-fraction 0.1 \
    --verbose
```

### Log Files

All pipelines generate detailed log files in their respective output directories:
- `processed_data/logs/data_preparation.log`
- `model_results/logs/model_training.log`
- `inference_results/logs/inference.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 📞 Contact

[Add contact information here] 