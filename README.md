# Transformer Immune Cells

A production-ready pipeline for analyzing immune cell data using transformer models, specifically Geneformer. This project provides a complete workflow from data preprocessing to model training and inference.

## Project Structure

```
transformer_immune/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data_preprocessing.py     # Main data preprocessing script
│   ├── train_model.py           # Main training script
│   ├── inference.py             # Inference script
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py   # Data processing utilities
│       ├── scvi_utils.py        # scVI model utilities
│       ├── tokenization.py      # Tokenization utilities
│       └── model_utils.py       # Model training utilities
├── data/                        # Data directory (created automatically)
├── models/                      # Saved models (created automatically)
├── logs/                        # Log files (created automatically)
├── results/                     # Results and plots (created automatically)
├── requirements.txt             # Python dependencies
├── run_pipeline.py             # Main pipeline script
└── README.md                   # This file
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for different tasks
- **Production Ready**: Comprehensive logging, error handling, and configuration management
- **Flexible Pipeline**: Can run individual steps or the complete pipeline
- **scVI Integration**: Includes scVI for variational inference and latent space analysis
- **Geneformer Integration**: Uses the Geneformer pre-trained model for gene expression analysis
- **Parallel Processing**: Efficient tokenization with parallel processing
- **Comprehensive Evaluation**: Detailed metrics and visualization of results

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd transformer_immune
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Geneformer** (if not already installed):
   ```bash
   git lfs install
   git clone https://huggingface.co/ctheodoris/Geneformer
   pip install ./Geneformer
   ```

## Configuration

The project uses a centralized configuration file (`src/config.py`) where you can modify:

- **Data paths**: Paths to input data and output directories
- **Model parameters**: Learning rate, batch size, number of epochs
- **Processing parameters**: Number of highly variable genes, sequence length
- **Training parameters**: Layer freezing strategy, evaluation settings

## Usage

### Running the Complete Pipeline

To run the entire pipeline from data preprocessing to inference:

```bash
python run_pipeline.py
```

### Running Individual Steps

You can also run individual steps of the pipeline:

```bash
# Data preprocessing only
python run_pipeline.py --preprocess-only

# Model training only
python run_pipeline.py --train-only

# Inference only
python run_pipeline.py --inference-only
```

### Running Individual Scripts

You can also run the individual scripts directly:

```bash
# Data preprocessing
python src/data_preprocessing.py

# Model training
python src/train_model.py

# Inference
python src/inference.py
```

## Data Requirements

The pipeline expects:

1. **Input Data**: AnnData object (.h5ad file) with single-cell RNA-seq data
2. **Gene Information**: Gene symbols that can be mapped to Ensembl IDs
3. **Cell Annotations**: Optional cell type or condition labels in `adata.obs`

## Pipeline Steps

### 1. Data Preprocessing (`src/data_preprocessing.py`)

- Loads raw AnnData object
- Preprocesses data (normalization, log transformation)
- Maps gene symbols to Ensembl IDs using MyGene.info
- Selects highly variable genes
- Runs scVI pipeline for latent space analysis
- Tokenizes data for transformer models
- Splits data into train/test sets

### 2. Model Training (`src/train_model.py`)

- Loads pre-trained Geneformer model
- Sets up training configuration
- Freezes/unfreezes appropriate layers
- Trains the model on tokenized data
- Evaluates model performance
- Saves trained model and results

### 3. Inference (`src/inference.py`)

- Loads trained model
- Makes predictions on new data
- Calculates accuracy and other metrics
- Saves prediction results

## Output Files

The pipeline generates several output files:

- **Preprocessed Data**: `data/adata_nk_hvg.h5ad`
- **Tokenized Datasets**: `data/tokenized_datasets/`
- **Trained Model**: `models/trained_model/`
- **Training Logs**: `logs/model_training.log`
- **Results**: `results/evaluation_results.json`
- **Plots**: `results/training_history.png`

## Key Components

### Data Processing Utilities (`src/utils/data_processing.py`)

- Functions for loading and preprocessing AnnData objects
- Gene symbol to Ensembl ID mapping
- Highly variable gene selection
- Data saving and loading utilities

### scVI Utilities (`src/utils/scvi_utils.py`)

- scVI model setup and training
- Latent space extraction
- UMAP visualization
- Complete scVI pipeline

### Tokenization Utilities (`src/utils/tokenization.py`)

- Geneformer tokenizer initialization
- Parallel tokenization of cells
- Dataset splitting and management
- Memory management utilities

### Model Utilities (`src/utils/model_utils.py`)

- Model loading and setup
- Training configuration
- Layer freezing strategies
- Evaluation and visualization
- Model saving and loading

## Configuration Options

### Data Processing
- `MAX_SEQUENCE_LENGTH`: Maximum token sequence length (default: 512)
- `N_TOP_GENES`: Number of highly variable genes (default: 2000)
- `TARGET_SUM`: Normalization target sum (default: 1e6)

### Model Training
- `NUM_LABELS`: Number of classification labels (default: 3)
- `LEARNING_RATE`: Learning rate (default: 3e-5)
- `BATCH_SIZE`: Batch size (default: 128)
- `NUM_EPOCHS`: Number of training epochs (default: 10)

### Training Strategy
- `FREEZE_BASE_LAYERS`: Whether to freeze base model layers (default: True)
- `UNFREEZE_LAST_N_LAYERS`: Number of last layers to unfreeze (default: 1)

## Logging

The pipeline includes comprehensive logging:

- **File Logs**: Saved to `logs/` directory
- **Console Output**: Real-time progress updates
- **Error Handling**: Detailed error messages and stack traces

## Performance Optimization

- **Parallel Processing**: Tokenization uses joblib for parallel execution
- **Memory Management**: CUDA memory clearing and garbage collection
- **Small Datasets**: Option to use smaller datasets for faster debugging
- **GPU Support**: Automatic GPU detection and utilization

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use smaller datasets
2. **CUDA Out of Memory**: Clear CUDA cache or reduce model size
3. **Gene Mapping Failures**: Check gene symbol format and species setting
4. **Missing Dependencies**: Ensure all packages are installed from requirements.txt

### Debug Mode

For debugging, the pipeline creates smaller datasets (25% of original size) by default. You can modify this in the configuration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

[Add contact information here]
