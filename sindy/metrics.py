import numpy as np


def rel_fro_err(
        eval_mat: np.ndarray,
        true_mat: np.ndarray
) -> float:
    """
    Calculates the relative Frobenius norm error between two matrices, which
    provides a measure for the element wise error between two matrices.

    The relative Frobenius norm error is defined as the Frobenius norm of the
    difference between `eval_mat` and `true_mat`, divided by the Frobenius
    norm of `true_mat`. This metric quantifies the overall difference between
    the two matrices, normalized by the magnitude of the true matrix.

    Args:
        eval_mat (np.ndarray): The evaluated matrix.
        true_mat (np.ndarray): The true matrix.

    Returns:
        float: The relative Frobenius norm error. A value of 0 indicates
            perfect agreement between the two matrices.
    """
    
    return np.linalg.norm(eval_mat - true_mat, ord='fro') / \
        np.linalg.norm(true_mat, ord='fro')


def precision(
        eval_mat: np.ndarray,
        true_mat: np.ndarray
) -> float:
    """
    Calculates the true positive ratio between two matrices which provides a
    measure of how many correct non-zero terms were identified.

    It is defined by the ratio between true positives, and the sum of true
    positives and false negatives.

    Args:
        eval_mat (np.ndarray): The evaluated matrix, where non-zero elements
            indicate selected features or active terms.
        true_mat (np.ndarray): The true matrix, where non-zero elements
            indicate the actual selected features or active terms.

    Returns:
        float: The true positive ratio. A value of 1 indicates perfect
            recovery of the sparsity pattern. A value of 0 indicates no
            overlap in the sparsity patterns.
    """
    
    eval_mask: np.ndarray = (eval_mat != 0.0)
    true_mask: np.ndarray = (true_mat != 0.0)

    true_positive: int = np.sum(eval_mask & true_mask)
    false_positive: int = np.sum(eval_mask & ~true_mask)

    return true_positive / (true_positive+false_positive)