import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Iterable


def explicit_euler(
        func: Callable,
        tFinal: float,
        y0: np.ndarray,
        dt: float = 0.01,
        args: Optional[Iterable] = (),
        **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    A fixed time step explicit Euler integrator. Only accepts ODEs with the
    the solution field shape of (time_dim, state_dim).

    Args:
        func (Callable): A function that takes in (t, y, *args, **kwargs), then
            outputs the time derivative of the system.
        tFinal (float): The final time to integrate up to (inclusive).
        y0 (np.ndarray): Initial state vector.
        dt (float, optional): Time step. Defaults to 0.01.
        args (Optional[Iterable], optional): Function parameters. Defaults to ().

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of time vector and solution matrix.
    """

    # Create time vector    
    tVec: np.ndarray = np.arange(0, tFinal+dt, dt)
    
    # Initialise solution matrix
    yMat: np.ndarray = np.empty((tVec.shape[0], y0.shape[0]))

    # Set initial condition
    yMat[0, :] = y0

    # Time-stepping loop
    for n in range(0, tVec.shape[0]-1):
        yMat[n+1, :] = yMat[n, :] + \
            dt * np.array(func(tVec[n], yMat[n, :], *args, **kwargs))

    return tVec, yMat


def rk4(
        func: Callable,
        tFinal: float,
        y0: np.ndarray,
        dt: float = 0.01,
        args: Optional[Iterable] = (),
        progress_bar: bool = False,
        **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    A fixed time step RK4 integrator. This integrator is generalized for
    n-dimensional PDEs which may have shapes of
    (*space_dim, time_dim, state_dim).

    Args:
        func (Callable): A function that takes in (t, y, *args, **kwargs), which
            then outputs the time derivative of the states.
        tFinal (float): The final time to integrate up to (inclusive).
        y0 (np.ndarray): Initial state vector.
        dt (float, optional): Time step. Defaults to 0.01.
        args (Optional[Iterable], optional): Function parameters. Defaults to ().
        progress_bar (bool, optional): Whether to display a progress bar.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns a tuple containing the time
            vector and solution matrix.
    """

    # Create time vector
    tVec: np.ndarray = np.arange(0, tFinal+dt, dt)

    # Get total number of variables
    sol_dim: int = y0.shape[-1]

    # Get total number of time grids
    t_dim: int = tVec.shape[0]

    # Get total number of spatial dimensions
    x_dims: np.ndarray = y0.shape[:-1]

    # Define solution shape
    sol_shape: tuple = x_dims + (t_dim, sol_dim)
    
    # Initialise solution matrix
    yMat: np.ndarray = np.empty(sol_shape)

    # Set initial condition
    yMat[..., 0, :] = y0

    # Time-stepping loop
    for n in tqdm(range(0, t_dim-1), disable=not progress_bar):
        k1: np.ndarray = np.array(
            func(tVec[n], yMat[..., n, :], *args, **kwargs))
        k2: np.ndarray = np.array(
            func(tVec[n] + dt/2, yMat[..., n, :] + dt/2 * k1, *args, **kwargs))
        k3: np.ndarray = np.array(
            func(tVec[n] + dt/2, yMat[..., n, :] + dt/2 * k2, *args, **kwargs))
        k4: np.ndarray = np.array(
            func(tVec[n] + dt, yMat[..., n, :] + dt * k3, *args, **kwargs))
        yMat[..., n+1, :] = yMat[..., n, :] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return tVec, yMat