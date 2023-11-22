# Message Clustering using BERT and HDBSCAN

This Python script clusters messages related to a specific dataset using BERT for tokenization and HDBSCAN for clustering.

## Description

The script processes messages from a CSV file, tokenizes them using BERT language model, reduces their dimensionality, and performs clustering using HDBSCAN. The primary goal is to group similar messages together based on their content.

## Usage

1. **Prepare Input Data:**
   - Place the messages in a CSV file named `Messages.csv` (adjust file name if needed).

2. **Run the Script:**
   - Execute the Python script `message_clustering.py` to perform message clustering.

   ```bash
   python message_clustering.py
