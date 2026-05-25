import numpy as np
from typing import Optional, List, Sequence

from .utils import get_combinations


def poly_lib(
        x: np.ndarray,
        power: int = 2,
        include_bias: bool = True,
        feature_names: Optional[List[str]] = None
) -> tuple[np.ndarray, list[str]]:
    """
    Generates the candidate function library Theta matrix for polynomial basis
    functions.

    Args:
        x (np.ndarray): Matrix of x data
        power (int, optional): Highest polynomial order. Defaults to 2.
        include_bias (bool, optional): Whether to include bias term. Defaults
            to True.
        feature_names (Optional[List[str]], optional): List of feature names.
            Defaults to None.

    Returns:
        tuple[np.ndarray, list[str]]: Tuple containing the final Theta matrix
            and the list of library function names.
    """
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"x_{i+1:d}" for i in range(x.shape[1])]

    # Initialise a list of theta vectors and corresponding function names
    list_Theta: list[np.ndarray] = list()
    library_names: list[str] = list()

    # Include bias term if specified
    if include_bias:
        list_Theta.append(np.ones(x.shape[0]))
        library_names.append("1")
    
    # Loop through each polynomial degree
    for deg in range(1, power+1):

        # Get all combinations of feature indices for the current degree
        combos = get_combinations(x.shape[1], deg)

        # Loop through each combination within the current degree
        for combo in combos:

            # Multiply relevant data columns together
            list_Theta.append(np.prod(x[:, combo], axis=1))
            
            # Create the corresponding library name
            present_features: list[int] = np.unique(combo).tolist()
            counts_features: list[int] = [combo.count(i)
                                          for i in present_features]
            library_names.append(' '.join([
                f"{feature_names[i]}^{counts_features[j]}"
                if counts_features[j] > 1
                else feature_names[i]
                for j, i in enumerate(present_features)
            ]))

    # Return final theta matrix and list of library names
    return np.stack(list_Theta, axis=1), library_names


def trig_lib(
        x: np.ndarray, n_frequencies: int = 1,
        include_sin: bool = True, include_cos: bool = True,
        feature_names: Optional[List[str]] = None
) -> tuple[np.ndarray, list[str]]:
    """
    Generates the candidate function library Theta matrix for Fourier basis
    functions.

    Args:
        x (np.ndarray): Matrix of x data.
        n_frequencies (int, optional): Number of frequencies to include.
            Defaults to 1.
        include_sin (bool, optional): Whether to include sine terms. Defaults
            to True.
        include_cos (bool, optional): Whether to include cosine terms. Defaults
            to True.
        feature_names (Optional[List[str]], optional): Feature names of the
            states. Defaults to None.

    Returns:
        tuple[np.ndarray, list[str]]: Tuple containing the final Theta matrix
            and the list of library function names.
    """
    
    # Get total number of states present
    n: int = x.shape[1]

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"x_{i+1:d}" for i in range(x.shape[1])]

    # Initialise a list of theta vectors and corresponding function names
    list_Theta: list[np.ndarray] = list()
    library_names: list[str] = list()

    # Loop through all frequencies
    for freq_idx in range(n_frequencies):

        freq = freq_idx+1

        # Loop through all states
        for n_idx in range(n):

            if include_sin:
                list_Theta.append(np.sin(freq*x[:, n_idx]))
                if freq == 1:
                    library_names.append(f'sin({feature_names[n_idx]})')
                else:
                    library_names.append(f'sin({freq:d} {feature_names[n_idx]})')

            if include_cos:
                list_Theta.append(np.cos(freq*x[:, n_idx]))
                if freq == 1:
                    library_names.append(f"cos({feature_names[n_idx]})")
                else:
                    library_names.append(f"cos({freq:d} {feature_names[n_idx]})")

    return np.stack(list_Theta, axis=1), library_names


def tensor_libs(
        libs: Sequence[np.ndarray], lib_names: Sequence[list[str]]
) -> tuple[np.ndarray, list[str]]:
    
    # Validate that each provided library has matching dimensions
    n_samples: int = libs[0].shape[0]
    for lib_idx, lib in enumerate(libs):
        assert lib.shape[0] == n_samples, (
            f"Library {lib_idx} has mismatched dimensions!"
        )

    # Helper function to tensor a pair of libraries together
    def _tensor_pair(
            lib1: np.ndarray, lib2: np.ndarray,
            lib1_names: list[str], lib2_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:

        tensored_lib: np.ndarray = (
            lib1[:,:,None] * lib2[:,None,:]
        ).reshape(n_samples, -1)

        tensored_names: list[str] = [
            f"{name1} {name2}"
            for name1 in lib1_names
            for name2 in lib2_names
        ]

        return tensored_lib, tensored_names

    # Initialise the tensor library
    tensor_lib: np.ndarray = libs[0].copy()
    tensor_names: list[str] = lib_names[0].copy()

    # Tensor together the remaining libraries
    for lib_idx in range(1, len(libs)):
        tensor_lib, tensor_names = _tensor_pair(
            tensor_lib, libs[lib_idx],
            tensor_names, lib_names[lib_idx]
        )

    return tensor_lib, tensor_names


def combine_libs(
        libs: Sequence[np.ndarray], lib_names: Sequence[list[str]]
) -> tuple[np.ndarray, list[str]]:
    
    # Validate that each provided library has matching dimensions
    n_samples: int = libs[0].shape[0]
    for lib_idx, lib in enumerate(libs):
        assert lib.shape[0] == n_samples, (
            f"Library {lib_idx} has mismatched dimensions!"
        )
    
    # Concatenate library matrix
    combined_lib = np.concatenate(libs, axis=1)

    # Concatenate library names
    combined_lib_names: list[str] = list()
    for lib_name in lib_names:
        combined_lib_names += lib_name

    return combined_lib, combined_lib_names