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
    index_embeddings = np.stack(index_df[embedding_fld].to_numpy())
    query_embeddings = np.stack(query_df[embedding_fld].to_numpy())
    convert_time = time.time() - start_time
    print(f"Conversion time: {convert_time:.2f} seconds", flush=True)

    d = query_embeddings.shape[1]

    # Faiss index selection based on the distance metric
    # https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/#spaces
    if distance_metric == "L2":
        index = faiss.IndexFlatL2(d)

        def get_score(distance):
            return 1 / (1 + distance)

    elif distance_metric == "cosine":
        faiss.normalize_L2(index_embeddings)
        index = faiss.IndexFlatIP(d)

        def get_score(distance):
            return 2 - distance

    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    start_time = time.time()
    print("Creating index...", flush=True)
    index.add(index_embeddings)
    index_time = time.time() - start_time
    print(f"Index creation time: {index_time:.2f} seconds", flush=True)

    start_time = time.time()
    print("Searching...", flush=True)
    distances, indices = index.search(query_embeddings, top_k)
    search_time = time.time() - start_time
    print(f"Search time: {search_time:.2f} seconds", flush=True)

    results_list = []
    for query_idx, neighbor_idxs in tqdm(
        enumerate(indices), total=len(indices), desc="Building neighbors"
    ):
        source_id = query_df.iloc[query_idx][encoding_hash_fld]
        # source_embedding = query_embeddings[query_idx]
        results_list.extend(
            [
                {
                    "source_id": source_id,
                    "neighbor_id": index_df.iloc[neighbor_idx][encoding_hash_fld],
                    "rank": rank,
                    "distance": distances[query_idx][rank - 1],
                    # "distance": np.sqrt(distances[query_idx][rank-1]),
                    # "distance_": np.linalg.norm(
                    #     source_embedding - index_embeddings[neighbor_idx]),
                }
                for rank, neighbor_idx in enumerate(neighbor_idxs, start=1)
                if neighbor_idx >= 0
            ]
        )
    results_df = pd.DataFrame(results_list)
    results_df["score"] = get_score(results_df["distance"])
    return results_df
