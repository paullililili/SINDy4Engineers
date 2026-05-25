import numpy as np
import random
import os


def get_combinations(
        n_features: int,
        degree: int,
        start_idx: int = 0) -> list[tuple[int, ...]]:
    """
    Generate all combinations of feature indices for a given polynomial degree.
    Used for polynomial library generation.

    Args:
        n_features (int): The number of features.
        degree (int): The polynomial degree power.
        start_idx (int, optional): The starting index for combinations.
            Defaults to 0.

    Returns:
        list[tuple[int, ...]]: List of tuples representing the combination of
            feature indices.
    """

    # If degree is 0, return a list with an empty tuple
    if degree == 0:
        return [()]
    
    combinations: list[tuple[int,...]] = []
    
    # Loops through top level indicies starting from start_idx to ensure
    # combinations are not repeated.
    for i in range(start_idx, n_features):
        remaining_combo: list[tuple[int,...]] = get_combinations(
            n_features,
            degree-1,
            i)
        # Append the sub-level combinations to the list
        for combo in remaining_combo:
            combinations.append((i,) + combo)

    return combinations
    

def add_gauss_noise(
        clean_x: np.ndarray,
        noise_ratio: float) -> np.ndarray:
    """
    Adds Gaussian noise to the dataset using the noise ratio definition from
    Messenger et al, 2021.

    Args:
        clean_x (np.ndarray): Matrix of data.
        noise_ratio (float): Noise ratio.

    Returns:
        np.ndarray: Data with added Gaussian noise.
    """
    
    # Calculate standard deviation of noise
    std: float = noise_ratio * np.linalg.norm(clean_x) / \
        np.sqrt(clean_x.size)
    
    # Generate noisy data
    noisy_x: np.ndarray = clean_x + std * np.random.randn(*clean_x.shape)

    return noisy_x


def bi_piecewise_regression(
        x: np.ndarray, y: np.ndarray
) -> tuple[int, float]:
    
    # Ensure x and y are vectors
    x = x.reshape(-1)
    y = y.reshape(-1)

    # Define all possible segmentation point
    corner_pts: np.ndarray = np.arange(1, x.shape[0]-1)

    # Loss function
    loss: np.ndarray = np.empty(corner_pts.shape)

    for idx, corner in enumerate(corner_pts):

        # Obtain segmented x
        x1: np.ndarray = x[:corner+1]
        x2: np.ndarray = x[corner:]

        # Obtain segmented y
        y1: np.ndarray = y[:corner+1]
        y2: np.ndarray = y[corner:]

        # Obtain gradients of segmented lines
        m1: np.ndarray = (y1[-1] - y1[0]) / (x1[-1] - x1[0])
        m2: np.ndarray = (y2[-1] - y2[0]) / (x2[-1] - x2[0])

        # Obtain the fitted lines
        l1: np.ndarray = m1 * (x1 - x1[0]) + y1[0]
        l2: np.ndarray = m2 * (x2 - x2[0]) + y2[0]
        l: np.ndarray = np.concatenate((l1, l2[1:]), axis=0)

        # Compute loss
        loss[idx] = np.linalg.norm((l - y)/y)

    # Get corner index based on minimisation of the loss function
    corner_opt_idx: int = np.argmin(loss) + 1
    corner_opt_x: float = x[corner_opt_idx]
    
    return corner_opt_idx, corner_opt_x


def print_latex_eqn(
        xi: np.ndarray,
        state_names: list[str],
        lib_names: list[str],
        precision: int = 3,
        is_ode: bool = True,
        dt_order: int = 1
) -> str:
    
    # Validate dimensions of the provided arguements
    assert xi.shape[0] == len(lib_names), (
        f"Expecting {xi.shape[1]:d} number of library functions, received "
        + f"{len(lib_names):d} number of library names instead!"
    )

    # Initialise latex string
    latex_str: str = r"\begin{align}" + "\n"

    # Iterate through each state
    for i, state in enumerate(state_names):
        
        # Start the line: \dot{x} &= ...
        rhs_terms = []
        lhs_diff_symbol: str = "d" if is_ode else r"\partial "
        lhs_power: str = "" if dt_order == 1 else f"^{dt_order:d}"
        lhs = fr"\frac{{{lhs_diff_symbol}{lhs_power}{state}}}" \
            + fr"{{{lhs_diff_symbol}t{lhs_power}}}"
        
        # Iterate through each library term (column)
        for j, term in enumerate(lib_names):
            coef = xi[j, i]
            
            # Skip if coefficient is zero
            if coef == 0.0:
                continue

            # Determine sign
            sign = "+" if coef >= 0 else "-"
            val = abs(coef)

            # Format the coefficient
            coef_str = f"{val:.{precision}f}"

            # Format the term
            if term == "1":
                term_str = ""
            else:
                term_str = term.replace("*", " ")
                
            # Combine
            full_term = f"{sign} {coef_str} {term_str}".strip()
            rhs_terms.append(full_term)

        # Re-assemble the Right Hand Side
        if not rhs_terms:
            rhs_str = f"{0.0:.{precision}f}"
        else:
            rhs_str = " ".join(rhs_terms)
            
            # Clean up leading plus sign if it exists
            if rhs_str.startswith("+"):
                rhs_str = rhs_str[1:].strip()

        # Add line to latex string
        latex_str += fr"{lhs} &= {rhs_str} \\" + "\n"

    latex_str += r"\end{align}"
    
    return latex_str


def set_seed(seed: int = 0):
    """
    Fixes the seed of the RNG generators used.

    Args:
        seed (int, optional): Random seed. Defaults to 0.
    """

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)