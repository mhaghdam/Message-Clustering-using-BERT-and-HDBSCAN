import pandas as pd
import re
from sklearn import cluster
import time
from transformers import AutoTokenizer, AutoModel
from scipy import sparse
import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def read_messages():
    file_name = 'Messages.csv'
    # Read the CSV file containing messages
    db = pd.read_csv(file_name, encoding='utf_8', error_bad_lines=False)
    # Extract messages from the first column of the CSV
    messages = list(db.iloc[:, 0])
    print("Reading time:", time.time() - start_time, "seconds")
    return messages


def tokenize_and_build_model(messages):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    model = AutoModel.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    print("Tokenizing messages and building model...")
    start = time.time()
    # Create a sparse matrix to store model output
    last_layer = sparse.csr_matrix((len(messages), 768)).tolil()

    for i, message in enumerate(messages):
        # Preprocess message text
        message = re.sub('ئ', 'ی', str(message))
        message2 = re.sub('[\W_]+', ' ', message)
        # Tokenize and encode the preprocessed message
        wrapped_input = tokenizer(message2, truncation=True, max_length=20, padding='max_length', return_tensors="pt")
        # Get model output for the encoded message
        output = model(**wrapped_input)
        # Store model output in the sparse matrix
        last_layer[i, :] = output[1].detach().numpy()[0]

    print("Building model time:", time.time() - start, "seconds")
    print("Last layer dimensions:", last_layer.shape)

    last_layer = last_layer.tocsr()
    # Save the sparse matrix
    sparse.save_npz("sparse_matrix.npz", last_layer)
    return last_layer


def reduce_dimensions(dim):
    # Load the previously saved sparse matrix
    data = sparse.load_npz("sparse_matrix.npz")
    start = time.time()
    # Reduce dimensionality using Gaussian Random Projection
    Reduced_data = GaussianRandomProjection(n_components=dim).fit_transform(data)
    print('Shape of data matrix client * features(Broker, Trader, Location):', data.shape)
    print("Dimension reduction time:", time.time() - start, "seconds")
    return Reduced_data


def perform_clustering(data, messages):
    print("\nClustering...")
    start = time.time()
    # Perform clustering using HDBSCAN algorithm
    clustering = cluster.HDBSCAN(alpha=1.0, cluster_selection_method='eom', n_jobs=4, leaf_size=40,
                                 metric='euclidean', min_cluster_size=2, min_samples=2,
                                 cluster_selection_epsilon=0.0000001).fit(data)

    print("Clustering time:", time.time() - start, "seconds")
    print("Number of messages:", len(messages))
    unique, counts = np.unique(clustering.labels_, return_counts=True)
    print("Number of noisy (without labeling) data:", dict(zip(unique, counts))[-1])
    print("Number of clusters:", max(unique) + 1)
    print("Minimum cluster size:", min(counts))
    print("Maximum cluster size:", max(counts[1:]))

    m = list(counts).index(max(counts[1:])) - 1
    print("Label of maximum cluster size:", m)
    print('\n Sizes:\n', counts, '\n Labels:\n', unique)
    print("\n Maximum cluster messages:")

    # Create a DataFrame with messages and their corresponding cluster labels
    df = pd.DataFrame({'Messages': messages, 'Labels': clustering.labels_})
    df.to_csv(f"Messages_Labels.csv", index=False)
    return clustering


start_time = time.time()
messages = read_messages()
data = tokenize_and_build_model(messages)
data = sparse.load_npz("sparse_matrix.npz")

clustering = perform_clustering(data, messages)

## Reduced dimension
# Reduced_data = reduce_dimensions(50)
# clustering = perform_clustering(Reduced_data, messages)

print("Final time:", time.time() - start_time, "seconds")