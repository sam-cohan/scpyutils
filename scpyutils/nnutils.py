"""
Utilities related nearest neighbors search.

Author: Sam Cohan
"""

import time

import faiss  # pip install faiss-cpu or pip install faiss-gpu
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def compute_brute_force_neighbors(
    index_df: pd.DataFrame,
    query_df: pd.DataFrame,
    embedding_fld: str,
    encoding_hash_fld: str,
    top_k: int = 100,
    distance_metric: str = "L2",
) -> pd.DataFrame:
    """Calculates the nearest neighbors using brute force search.

    Args:
        index_df: The DataFrame containing the training embeddings.
        query_df: The DataFrame containing the test embeddings.
        embeding_fld: Name of embedding field.
        encoding_hash_fld: Name of the field containing the unique hash
            of the encoding (this will be used as the id field for source
            and neighbor records).
        top_k: The number of nearest neighbors to find for each test instance.
        distance_metric: The metric used to compute distances between embeddings.
            Supports 'L2' and 'cosine'.

    Returns:
        A DataFrame containing the nearest neighbors for each test instance,
        including columns for source_id, neighbor_id, rank, score, distance,
        and group columns.

    Raises:
        ValueError: If an unsupported distance metric is provided.
    """

    if index_df.empty or query_df.empty:
        raise ValueError("Empty DataFrame provided.")

    print(
        f"Index data size: {len(index_df)} | Query data size: {len(query_df)}",
        flush=True,
    )

    start_time = time.time()
    test_embeddings = np.stack(index_df[embedding_fld].to_numpy())
    train_embeddings = np.stack(query_df[embedding_fld].to_numpy())
    convert_time = time.time() - start_time
    print(f"Conversion time: {convert_time:.2f} seconds", flush=True)

    d = test_embeddings.shape[1]

    # Faiss index selection based on the distance metric
    # https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/#spaces
    if distance_metric == "L2":
        index = faiss.IndexFlatL2(d)

        def get_score(distance):
            return 1 / (1 + distance)

    elif distance_metric == "cosine":
        faiss.normalize_L2(test_embeddings)
        faiss.normalize_L2(train_embeddings)
        index = faiss.IndexFlatIP(d)

        def get_score(distance):
            return 2 - distance

    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    start_time = time.time()
    print("Creating index...", flush=True)
    index.add(train_embeddings)
    index_time = time.time() - start_time
    print(f"Index creation time: {index_time:.2f} seconds", flush=True)

    start_time = time.time()
    print("Searching...", flush=True)
    distances, indices = index.search(test_embeddings, top_k)
    search_time = time.time() - start_time
    print(f"Search time: {search_time:.2f} seconds", flush=True)

    source_ids = index_df[encoding_hash_fld].to_numpy()
    neighbor_ids = query_df[encoding_hash_fld].to_numpy()

    results_list = []
    for i in tqdm(range(len(indices)), desc="Building final neighbor DF ..."):
        idxs = indices[i]
        source_id = source_ids[i]
        valid_mask = idxs >= 0
        ranks = np.arange(1, top_k + 1)[valid_mask]
        distances_i = distances[i][valid_mask]
        neighbor_ids_i = neighbor_ids[idxs[valid_mask]]

        results_list.append(
            pd.DataFrame(
                {
                    "source_id": source_id,
                    "neighbor_id": neighbor_ids_i,
                    "rank": ranks,
                    "distance": distances_i,
                    "score": get_score(distances_i),
                }
            )
        )

    results_df = pd.concat(results_list, ignore_index=True)
    return results_df
