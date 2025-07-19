!pip install torch
!pip install transformers
!pip install scanpy
!pip install datasets anndata



# Geneformer installation
!git lfs install
!git clone https://huggingface.co/ctheodoris/Geneformer
!pip install ./Geneformer




# install mygene
!pip install mygene



# install scvi
!pip install scvi-tools



# get the standard libraries
import scanpy as sc
import pandas as pd
import numpy as np

import os
import torch
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import json

# transformer libraries
from torch.utils.data import Dataset, random_split
import scvi
from tqdm.notebook import tqdm
from datasets import load_from_disk

# get the standard tokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification




# code to mount google drive
from google.colab import drive
drive.mount('/content/drive')



# read adata: nk file
adata_nk=sc.read_h5ad('/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/data/human_immune_health_atlas_nk-ilc.h5ad')



# save raw data counts
adata_nk.X=adata_nk.raw.X

# keep origina counts in another layer
adata_nk.layers['counts']=adata_nk.raw.X



# normalize and log-transform the data
sc.pp.normalize_total(adata_nk, target_sum=1e6)
sc.pp.log1p(adata_nk)  # log-transform



# Step 1: Setup AnnData for scVI
adata_scvi = adata_nk.copy()  # make a separate copy to not modify the original
scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key='subject.ageGroup')

# normalize
sc.pp.normalize_total(adata_scvi, target_sum=1e6)
sc.pp.log1p(adata_scvi)  # log-transform

# use 2K highly variable genes
sc.pp.highly_variable_genes(adata_scvi, n_top_genes=2000)



# Step 2: Train scVI model
model_scvi = scvi.model.SCVI(adata_scvi)
model_scvi.train(max_epochs=10)

# Step 3: Extract latent representation
adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()

# Step 4: Run neighbors and UMAP using the latent space
sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
sc.tl.umap(adata_scvi)



# Step 3: Extract latent representation
adata_scvi.obsm["X_scVI"] = model.get_latent_representation()

# Step 4: Run neighbors and UMAP using the latent space
sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
sc.tl.umap(adata_scvi)



# show age group
sc.pl.umap(adata_scvi, color=['subject.ageGroup','cohort.cohortGuid'])



# if GPU available - move to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Get the tokenizer
from geneformer.tokenizer import tokenize_cell
import numpy as np
from datasets import Dataset as HFDataset
import torch
from geneformer.tokenizer import TranscriptomeTokenizer
import mygene



# Get the list of gene symbols from your AnnData object
gene_symbols = adata_nk.var.index.tolist()

# Instantiate the MyGene.info client
mg = mygene.MyGeneInfo()

# Query for Ensembl IDs
# Specify species='mouse' if you are working with mouse data
gene_info = mg.querymany(gene_symbols,
                         scopes='symbol',
                         fields='ensembl.gene',
                         species='human',
                         as_dataframe=True)



# Step 0: Map genes to the right names

# Create a clean Series for mapping, dropping genes that were not found
ensembl_map = gene_info.reset_index().dropna(subset=['ensembl.gene'])

# Create a dictionary for mapping from symbol ('query') to Ensembl ID
id_map = pd.Series(ensembl_map['ensembl.gene'].values, index=ensembl_map['query']).to_dict()

# Add the new 'ensembl_id' column to adata.var
adata_nk.var['ensembl_id'] = adata_nk.var.index.map(id_map)

# Display the result to verify
print(adata_nk.var.head())



# keep original gene names
adata_nk.var['gene_name_HGCN'] = adata_nk.var.index.values

# convert cells to ensemble IDs
adata_nk.var['gene_name'] = adata_nk.var['ensembl_id']

# convert to ENSIDs
adata_nk.var_names=adata_nk.var['ensembl_id']



# make ENS ids the main names
valid_var_mask = pd.notnull(adata_nk.var_names)

# subset adata to these values
adata_nk = adata_nk[:, valid_var_mask]



# Step 2: initialize
tk = TranscriptomeTokenizer()  # init with defaults
gene_token_dict = tk.gene_token_dict  # maps Ensembl ID → token index

# create genes and gene tokens
genes = adata_nk.var_names.to_list()
gene_tokens = np.array([gene_token_dict.get(g, 0) for g in genes], dtype=int)



# Step 3: align adata
# adata_nk.var_names should be Ensembl IDs
genes = adata_nk.var_names.to_list()
gene_tokens = np.array([gene_token_dict.get(g, 0) for g in genes], dtype=int)



# cut this to only highly variable genes: 2K features
sc.pp.highly_variable_genes(adata_nk, n_top_genes=2000, inplace=True)

# subset the adata to only HVG
adata_nk = adata_nk[:, adata_nk.var.highly_variable].copy()



# adata file
adata_nk.write_h5ad("/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/data/preprocessed/full_data/adata_nk_hvg.h5ad")

# save genes and tokens
np.save("/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/data/preprocessed/full_data/genes.npy", genes)
np.save("/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/data/preprocessed/full_data/gene_tokens.npy", gene_tokens)



# Parallel implementation
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import numpy as np

# max token length
MAX_LEN = 512

# prepare labels once if available
if "subject.ageGroup" in adata_nk.obs.columns:
    labels_array = adata_nk.obs["subject.ageGroup"].astype("category").cat.codes.values
else:
    labels_array = None

# define the per-cell processing function
def process_cell(idx):
    gene_vector = adata_nk.X[idx].toarray().flatten() if hasattr(adata_nk.X[idx], "toarray") else adata_nk.X[idx]

    # get ranked token ids
    ranked_tokens = tokenize_cell(gene_vector, gene_tokens)

    # pad or truncate
    input_ids = list(ranked_tokens[:MAX_LEN])
    input_ids += [0] * (MAX_LEN - len(input_ids))

    attention_mask = [1 if id != 0 else 0 for id in input_ids]

    rec = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    if labels_array is not None:
        rec["labels"] = int(labels_array[idx])

    return rec

# parallel execution: adjust n_jobs based on available CPU
records = Parallel(n_jobs=4)(
    delayed(process_cell)(i) for i in tqdm(range(adata_nk.n_obs), desc="Tokenizing cells (parallel)")
)




# Step 5: build HF dataset
hf_ds = HFDataset.from_list(records)



# save to disk
hf_ds.save_to_disk("/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/data/preprocessed/full_data/tokenized_adata_nk")



# save to disk
hf_ds = load_from_disk("/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/data/preprocessed/full_data/tokenized_adata_nk")



# Assuming hf_ds is your Hugging Face Dataset
split_dataset = hf_ds.train_test_split(test_size=0.2, seed=42)

# create train and test ds
train_ds = split_dataset["train"]
test_ds = split_dataset["test"]

# create a small train and test datasets for debugging
small_train_ds = train_ds.shuffle(seed=42).select(range(int(0.25 * len(train_ds))))
small_test_ds = test_ds.shuffle(seed=42).select(range(int(0.25 * len(test_ds))))




# clean up CUDA memory (just in case)

import torch, gc

# Clear Python variables
gc.collect()

# Clear CUDA cache
torch.cuda.empty_cache()




# get the trainer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# number of unique categories
NUM_LABELS=3

# take pre-trained model: Geneformer
model_class = AutoModelForSequenceClassification.from_pretrained(
    "ctheodoris/Geneformer",
    num_labels=NUM_LABELS
)


# Define the training

args = TrainingArguments(
    output_dir="./geneformer_clf",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    #max_steps=150,
    num_train_epochs=10,
    logging_strategy="steps",
    learning_rate=3e-5,
    logging_steps=50,
    logging_dir="./logs",
    report_to="none"  # or "wandb", "tensorboard" if you want logging
)



# define compute metrics function

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):

    # define predictions
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # clear predictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # model accuracy
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



# Instantieate the trainer
trainer = Trainer(
    model=model_class,
    args=args,
    # using small train and test datasets
    train_dataset=small_train_ds,
    eval_dataset=small_test_ds,
    compute_metrics=compute_metrics
)



# 1. Freeze all parameters
for name, param in model_class.base_model.named_parameters():
    param.requires_grad = False

# 2. Unfreeze the last 1 transformer layers
for name, param in model_class.base_model.named_parameters():
    if "encoder.layer" in name:  # works for BERT-style models
        # extract the layer number
        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
        if layer_num >= (model_class.base_model.config.num_hidden_layers - 1):
            param.requires_grad = True



# check model parameters needed for training
for name, param in model_class.named_parameters():
    if param.requires_grad:
        print(f"{name}: trainable")



# start training!
trainer.train()



# save trained model
#trainer.save_model("/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/model/trained_model")

# make sure tokenizer is here
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# path where you saved the model
model_path = "/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer/model/trained_model"

# save the model
# save the tokenizer using json
with open(f"{model_path}/gene_token_dict.json", "w") as f:
    json.dump(tk.gene_token_dict, f)




# get accuracy score
from sklearn.metrics import accuracy_score



# use small part of the dataset: temporarl
small_test_ds = test_ds.shuffle(seed=42).select(range(int(0.2 * len(test_ds))))

# evalute results on test dataset
preds_output = trainer.predict(small_test_ds)

# create labels
pred_labels = preds_output.predictions.argmax(-1)
true_labels = preds_output.label_ids




# show classificaiton performance
from sklearn.metrics import classification_report

# get all unique classes in true and predicted labels
unique_classes = sorted(list(set(np.unique(true_labels)) | set(np.unique(pred_labels))))
target_names = [f"class{i}" for i in unique_classes]

print(classification_report(true_labels, pred_labels, target_names=target_names))



# get a history
history = trainer.state.log_history

# log loss
df_logs = pd.DataFrame(trainer.state.log_history)

df_loss = df_logs[df_logs["loss"].notna()]

plt.figure(figsize=(8, 4))
plt.plot(df_loss["step"], df_loss["loss"], label="Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.grid(True)
plt.show()



df_logs = pd.DataFrame(trainer.state.log_history)
print(df_logs.columns)
print(df_logs.head())



