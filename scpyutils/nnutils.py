"""
Utilities related nearest neighbors search.

Author: Sam Cohan
"""

import time

import faiss  # pip install faiss-cpu or pip install faiss-gpu
import numpy as np
import pandas as pd


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

    # Directly convert embeddings to numpy arrays
    start_time = time.time()
    test_embeddings = np.array(index_df[embedding_fld].tolist()).astype("float32")
    train_embeddings = np.array(query_df[embedding_fld].tolist()).astype("float32")
    convert_time = time.time() - start_time
    print(f"Conversion time: {convert_time:.2f} seconds", flush=True)

    d = test_embeddings.shape[1]

    # Faiss index selection based on the distance metric
    # https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/#spaces
    if distance_metric == "L2":
        index = faiss.IndexFlatL2(d)

        def get_score(distance: float) -> float:
            return 1 / (1 + distance)

    elif distance_metric == "cosine":
        faiss.normalize_L2(test_embeddings)
        faiss.normalize_L2(train_embeddings)
        index = faiss.IndexFlatIP(d)

        def get_score(distance: float) -> float:
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

    results_list = []
    for i, idxs in enumerate(indices):
        source_id = index_df.iloc[i][encoding_hash_fld]
        for rank, train_idx in enumerate(idxs, start=1):
            if (
                train_idx >= 0
            ):  # Faiss returns -1 if there are fewer than top_k neighbors
                distance = distances[i][rank - 1]

                results_list.append(
                    {
                        "source_id": source_id,
                        "neighbor_id": query_df.iloc[train_idx][encoding_hash_fld],
                        "rank": rank,
                        "distance": distance,
                        "score": get_score(distance),
                    }
                )

    results_df = pd.DataFrame(results_list)
    return results_df
