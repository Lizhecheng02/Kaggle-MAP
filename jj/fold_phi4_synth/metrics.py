import numpy as np
import torch


def compute_map_k(eval_pred, k):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    topk = np.argsort(-probs, axis=1)[:, :k]
    score = 0.0
    for i, label in enumerate(labels):
        ranks = topk[i]
        for rank_idx, predicted_label in enumerate(ranks):
            if predicted_label == label:
                score += 1.0 / (rank_idx + 1)
                break  # Found the label, no need to check further ranks
    return {f"map@{k}": score / len(labels)}


def calculate_aggregate_map_k(all_predictions, all_labels, k=3):
    """
    Calculate MAP@3 for all predictions combined
    """
    if len(all_predictions) == 0:
        return 0.0

    # Convert to numpy arrays (if not already)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # Create eval_pred tuple in the format expected by compute_map_k
    eval_pred = (predictions, labels)

    # Use compute_map_k and extract the score
    result = compute_map_k(eval_pred, k)

    return result[f"map@{k}"]
