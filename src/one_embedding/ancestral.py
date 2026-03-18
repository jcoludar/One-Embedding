"""Ancestral embedding reconstruction under Brownian motion.

Given a phylogenetic tree with per-residue embeddings at the leaves,
reconstructs ancestral embeddings at internal nodes using maximum
likelihood under a Brownian motion model.

The ML ancestral state at each internal node is the weighted mean of
descendant values, with weights proportional to inverse branch lengths.
This is the continuous-character analog of Felsenstein pruning.

Based on phytools::fastAnc (R) and Draupnir (ICLR 2022) concepts,
implemented for our 512d codec embeddings.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def reconstruct_ancestral_embeddings(
    tree_nodes: list,
    leaf_embeddings: Dict[str, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Reconstruct ML ancestral embeddings at all internal nodes.

    Under Brownian motion, the ML estimate at an internal node is:
    mu = (mu_L * s_R + mu_R * s_L) / (s_L + s_R)
    where s_L = v_L + t_L, s_R = v_R + t_R (partial variances + branch lengths)

    This is the same computation as Felsenstein pruning, but we keep
    the partial means instead of discarding them.

    Args:
        tree_nodes: list of node dicts from postorder traversal, each with:
            - "id": int
            - "name": str (for leaves)
            - "is_leaf": bool
            - "children_ids": list of child node ids
            - "branch_length": float (to parent)
        leaf_embeddings: {leaf_name: (D,) or (L, D) embedding array}

    Returns:
        {node_id: embedding_array} for ALL nodes (leaves + internal)
    """
    D = next(iter(leaf_embeddings.values())).shape

    # Build lookup
    node_by_id = {n["id"]: n for n in tree_nodes}

    # Felsenstein pruning: postorder traversal
    partials: Dict[int, Tuple[np.ndarray, float]] = {}  # id -> (mu, variance)

    for node in tree_nodes:
        if node["is_leaf"]:
            emb = leaf_embeddings.get(node["name"])
            if emb is None:
                raise ValueError(f"No embedding for leaf '{node['name']}'")
            partials[node["id"]] = (emb.astype(np.float64), 0.0)
        else:
            children = node["children_ids"]
            if len(children) != 2:
                raise ValueError(
                    f"Node {node['id']} has {len(children)} children "
                    f"(expected 2 for binary tree)"
                )
            left_id, right_id = children
            mu_L, v_L = partials[left_id]
            mu_R, v_R = partials[right_id]

            t_L = node_by_id[left_id]["branch_length"]
            t_R = node_by_id[right_id]["branch_length"]

            s_L = v_L + t_L
            s_R = v_R + t_R
            s_total = s_L + s_R

            # Weighted mean (ML estimate under BM)
            mu = (mu_L * s_R + mu_R * s_L) / s_total

            # Combined partial variance
            v = (s_L * s_R) / s_total

            partials[node["id"]] = (mu, v)

    # Extract embeddings for all nodes
    result = {}
    for node_id, (mu, _) in partials.items():
        result[node_id] = mu.astype(np.float32) if mu.dtype == np.float64 else mu

    return result


def tree_to_node_list(tree) -> list:
    """Convert an experiment-35 Tree object to a list of node dicts.

    Compatible with the Tree class from experiments/35_embedding_phylogenetics.py.

    Args:
        tree: Tree object with .postorder(), .nodes, etc.

    Returns:
        list of node dicts in postorder
    """
    nodes = []
    for node in tree.postorder():
        nodes.append({
            "id": node.id,
            "name": node.name,
            "is_leaf": node.is_leaf(),
            "is_root": node.is_root(),
            "children_ids": [c.id for c in node.children],
            "branch_length": node.branch_length,
        })
    return nodes


def ancestral_distance_matrix(
    ancestral_embeddings: Dict[int, np.ndarray],
    node_names: Dict[int, str],
) -> Tuple[np.ndarray, list]:
    """Compute distance matrix including ancestral nodes.

    Args:
        ancestral_embeddings: {node_id: embedding} for all nodes
        node_names: {node_id: name} (empty string for internal nodes)

    Returns:
        (distance_matrix, names) where names include "anc_N" for internal nodes
    """
    ids = sorted(ancestral_embeddings.keys())
    names = []
    for nid in ids:
        name = node_names.get(nid, "")
        if not name:
            name = f"anc_{nid}"
        names.append(name)

    vecs = np.array([ancestral_embeddings[nid] for nid in ids])

    # Handle both 1D (protein-level) and 2D (per-residue) embeddings
    if vecs.ndim == 3:
        # (N, L, D) -> flatten to (N, L*D)
        N = vecs.shape[0]
        vecs = vecs.reshape(N, -1)

    # Pairwise Euclidean distances
    n = len(ids)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(vecs[i] - vecs[j])
            dist[i, j] = d
            dist[j, i] = d

    return dist, names


def embedding_to_nearest_aa(
    embedding: np.ndarray,
    aa_embeddings: Dict[str, np.ndarray],
) -> str:
    """Map an ancestral embedding back to the nearest amino acid.

    For each position in the ancestral embedding, find the AA whose
    embedding is most similar (cosine similarity).

    Args:
        embedding: (L, D) ancestral per-residue embedding
        aa_embeddings: {amino_acid: (D,) average embedding per AA type}

    Returns:
        Reconstructed amino acid sequence string
    """
    aa_list = sorted(aa_embeddings.keys())
    aa_matrix = np.array([aa_embeddings[aa] for aa in aa_list])  # (20, D)

    # Normalize
    aa_norms = np.linalg.norm(aa_matrix, axis=1, keepdims=True) + 1e-10
    aa_normed = aa_matrix / aa_norms

    sequence = []
    for i in range(embedding.shape[0]):
        emb_norm = embedding[i] / (np.linalg.norm(embedding[i]) + 1e-10)
        sims = aa_normed @ emb_norm
        best_aa = aa_list[np.argmax(sims)]
        sequence.append(best_aa)

    return "".join(sequence)
