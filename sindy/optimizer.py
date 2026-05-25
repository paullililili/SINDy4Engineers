import numpy as np
from typing import Optional
import warnings


def stlsq(
        theta: np.ndarray,
        x_dot: np.ndarray,
        lam: float = 0.01,
        max_iter: int = 20) -> np.ndarray:
    """
    Sequentially Thresholded Least Squares algorithm based on (Brunton et al,
    2016) implementation.

    Args:
        theta (np.ndarray): Candidate function library matrix.
        x_dot (np.ndarray): Matrix of time derivative data.
        lam (float, optional): Lambda threshold value. Defaults to 0.01.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.

    Returns:
        np.ndarray: Sparse coefficient matrix.
    """

    # Reshape x_dot to ensure shape consistency
    if x_dot.ndim == 1:
        x_dot = x_dot.reshape(-1, 1)
    
    # Initial least squares fit
    xi: np.ndarray = least_squares(x_dot, theta)

    # Create coefficient mask and set coefficients below threshold to zero
    mask_xi: np.ndarray = np.abs(xi) >= lam
    xi[mask_xi == False] = 0.0

    # Create a copy of mask to track convergence
    prev_mask_xi: np.ndarray = np.copy(mask_xi)

    # Perform least squares fit per state
    for state in range(x_dot.shape[1]):
            
        # Sequentially iterate through the algorithm
        for _ in range(max_iter):

            try:

                # Perform least squares
                xi[mask_xi[:, state], state] = least_squares(
                    x_dot[:, state],
                    theta[:, mask_xi[:, state]])
                
            except np.linalg.LinAlgError:
                warnings.warn('Singular matrix found! Stopping STLSQ early.')
                break
            
            # Update mask and sparsify coefficients
            mask_xi[:, state] = np.abs(xi[:, state]) >= lam
            xi[mask_xi == False] = 0.0

        # Check for convergence
        if np.array_equal(mask_xi, prev_mask_xi):
            break
        else:
            prev_mask_xi = np.copy(mask_xi)
            
    return xi


def stridge(
        theta: np.ndarray,
        x_dot: np.ndarray,
        lam: float = 0.01,
        l2_norm: float = 1e-3,
        max_iter: int = 20) -> np.ndarray:
    """
    Sequential Threshold Ridge regression (STRidge) from (Rudy et al, 2017).

    Args:
        theta (np.ndarray): Candidate function library matrix.
        x_dot (np.ndarray): Matrix of time derivative data.
        lam (float, optional): Lambda threshold value. Defaults to 0.01.
        l2_norm (float, optional): Ridge regression regularization parameter.
            Defaults to 1e-3.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.

    Returns:
        np.ndarray: Sparse coefficient matrix.
    """

    # Reshape x_dot to ensure shape consistency
    if x_dot.ndim == 1:
        x_dot = x_dot.reshape(-1, 1)
    
    # Initial least squares with ridge regression fit
    xi: np.ndarray = least_squares(x_dot, theta, l2_norm)

    # Create coefficient mask and set coefficients below threshold to zero
    mask_xi: np.ndarray = np.abs(xi) >= lam
    xi[mask_xi == False] = 0.0

    # Create a copy of mask to track convergence
    prev_mask_xi: np.ndarray = np.copy(mask_xi)
    
    # Perform least squares fit per state
    for state in range(x_dot.shape[1]):

        # Sequentially iterate through the algorithm
        for _ in range(max_iter):
                
            try:

                # Perform least squares with ridge regression
                xi[mask_xi[:, state], state] = least_squares(
                    x_dot[:, state],
                    theta[:, mask_xi[:, state]],
                    l2_norm
                )

            except np.linalg.LinAlgError:
                warnings.warn(f'Singular matrix found! Stopping STRidge early.')
                break
            
            # Update mask and sparsify coefficients
            mask_xi[:, state] = np.abs(xi[:, state]) >= lam
            xi[mask_xi == False] = 0.0

        # Check for convergence
        if np.array_equal(mask_xi, prev_mask_xi):
            break
        else:
            prev_mask_xi = np.copy(mask_xi)
            
    return xi


def stcls(
        theta: np.ndarray, x_dot: np.ndarray,
        lam: float = 0.01, l2_norm: float = 0.0,
        constraint_lhs: Optional[np.ndarray] = None,
        constraint_rhs: Optional[np.ndarray] = None,
        max_iter: int = 20
) -> np.ndarray:
    r"""
    Performs Sequentially Thresholded Constrained Least Squares regression,
    where given a regression problem of $\dot{X} = \Theta \Xi$ subjected to a
    linear equality constraint matrix $C \Xi(:) = d$.

    Args:
        theta (np.ndarray): Candidate function library matrix.
        x_dot (np.ndarray): Matrix of time derivative data.
        lam (float, optional): L0 lambda threshold value. Defaults to 0.01.
        l2_norm (float, optional): L2 ridge regularizer strength. Defaults to
            0.0.
        constraint_lhs (Optional[np.ndarray], optional): LHS of constraint
            equation, constraint matrix $C$. Defaults to None.
        constraint_rhs (Optional[np.ndarray], optional): RHS of constraint
            equation, constraint vector $d$. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.

    Returns:
        np.ndarray: Sparse coefficient matrix.
    """
    
    # Check for constraints
    if constraint_lhs is None and constraint_rhs is None:
        return stridge(
            theta, x_dot,
            lam, l2_norm,
            max_iter
        )
    else:
        assert constraint_lhs is not None, "Both constraints must be given."
        assert constraint_rhs is not None, "Both constraints must be given."

    # Reshape x_dot to ensure shape consistency
    if x_dot.ndim == 1:
        x_dot = x_dot.reshape(-1, 1)

    # Define dimensions
    n_targets: int = x_dot.shape[1]
    n_features: int = theta.shape[1]
    n_coefs: int = n_targets*n_features

    # Reshape constraints
    constraint_lhs = constraint_lhs.reshape((-1, n_coefs))
    constraint_rhs = constraint_rhs.flatten()
    n_constraints: int = constraint_lhs.shape[0]
    
    # Initial least squares with ridge regression fit
    xi: np.ndarray = np.empty((n_coefs,))

    # Create coefficient mask and set coefficients below threshold to zero
    mask_xi: np.ndarray = np.ones(xi.shape, dtype=bool)

    # Create a copy of mask to track convergence
    prev_mask_xi: np.ndarray = np.copy(mask_xi)

    # Define constraint mask to override sparsification on actively constrained
    # terms
    constrained_mask: np.ndarray = np.any(
        constraint_lhs != 0.0,
        axis=0
    )

    # Identify constraints that enforce sparsity
    sparsity_constraints: np.ndarray = (
        (np.count_nonzero(constraint_lhs, axis=1) == 1)
        & (constraint_rhs == 0.0)
    )

    # Create mask that forces sparsity on coefficient matrix
    sparsity_mask = np.any(constraint_lhs[sparsity_constraints,:], axis=0)

    # Define modified xdot and theta
    x_dot_mod = x_dot.flatten(order='F')
    theta_mod = np.kron(np.eye(n_targets), theta)

    # Sequentially iterate through the algorithm
    for _ in range(max_iter):

        try:
            # Perform least squares with regression
            if np.sum(sparsity_constraints) == n_constraints:
                xi[mask_xi] = least_squares(
                    x_dot_mod, theta_mod[:, mask_xi], l2_norm
                )
            else:
                xi[mask_xi] = constrained_least_squares(
                    x_dot_mod, theta_mod[:, mask_xi],
                    constraint_lhs[~sparsity_constraints][:, mask_xi],
                    constraint_rhs[~sparsity_constraints],
                    l2_norm
                )

        except np.linalg.LinAlgError:
            warnings.warn('Singular matrix found! Stopping STCLS early.')
            break
        
        # Update mask and sparsify coefficients
        mask_xi = (
            (np.abs(xi) >= lam) | constrained_mask
        ) & ~sparsity_mask
        xi[mask_xi == False] = 0.0

        # Check for convergence
        if np.array_equal(mask_xi, prev_mask_xi):
            break
        else:
            prev_mask_xi = np.copy(mask_xi)
            
    return xi.reshape((n_features, n_targets), order='F')


def least_squares(
        y: np.ndarray,
        A: np.ndarray,
        l2_norm: float = 0.0
    ) -> np.ndarray:
    """
    Solves the least squares problem Ax = y for x.

    Args:
        y (np.ndarray): Vector y
        A (np.ndarray): Matrix A
        l2_norm (float): Ridge regularization parameter. Defaults to 0.0.

    Returns:
        np.ndarray: Vector x
    """
    
    return np.linalg.inv(
        A.transpose() @ A + l2_norm*np.eye(A.shape[1])
    ) @ A.transpose() @ y
    

def constrained_least_squares(
        y: np.ndarray, A: np.ndarray,
        C: np.ndarray, d: np.ndarray,
        l2_norm: float = 0.0
) -> np.ndarray:
    """
    Solves the linearly constrained ridge regression problem of Ax = y subject
    to Cy = d.

    Args:
        y (np.ndarray): Vector y.
        A (np.ndarray): Matrix A.
        C (np.ndarray): Constraint matrix LHS.
        d (np.ndarray): Constraint vector RHS.
        l2_norm (float, optional): Ridge regularization parameter. Defaults to
            0.0.

    Returns:
        np.ndarray: Vector x.
    """
    
    # Define total number of features
    n_features: int = A.shape[1]

    # Define total constraints
    n_constraints: int = C.shape[0]
    
    # Compute the Hessian
    hessian: np.ndarray = A.T @ A + l2_norm*np.eye(n_features)
    
    # Construct the linear equation to be solved
    lhs: np.ndarray = np.block([
        [hessian, 0.5 * C.T],
        [C, np.zeros((n_constraints, n_constraints))]
    ])
    rhs: np.ndarray = np.concatenate((A.T @ y, d), axis=0)

    # Solve the constructed linear equation
    return np.linalg.solve(lhs, rhs)[:n_features]