import numpy as np
from typing import Callable, Optional

from .utils import add_gauss_noise
from .ivp_solvers import rk4


def generate_ode_data(
        M: int, dt: float,
        x0: np.ndarray,
        ode: Callable, ode_params: list = [], ivp_solver: Callable = rk4,
        noise_ratios: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Function to generate time vector, as well as clean and noisy state data
    using a given ODE function.

    Args:
        M (int): Number of time points to generate.
        dt (float): Time step dt.
        x0 (np.ndarray): Initial conditions.
        ode (Callable): ODE function.
        ode_params (list): Parameters of the ODE function.
        ivp_solver (Callable): An initial value problem solver. Defaults to
            `rk4`.
        noise_ratios (Optional[np.ndarray]): List of noise ratios to generate.
            Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, list[np.ndarray]]: Returns the time
            vector, clean x data, as well as a list of noisy x data.
    """
    
    # Find simulation time
    tFinal: float = dt*(M-1)

    # Solve for clean data
    t, x = ivp_solver(
        ode,
        tFinal=tFinal,
        y0=x0,
        dt=dt,
        args=ode_params
    )

    # Obtain noisy data
    noisy_xs: list[np.ndarray] = list()

    if noise_ratios is not None:
        for noise_ratio in noise_ratios:
            noisy_xs.append(
                add_gauss_noise(x, noise_ratio=noise_ratio))
        
    return t, x, noisy_xs


def duffing(
        t: float, x: np.ndarray,
        delta: float = 0.2,
        alpha: float = 0.05,
        beta: float = 1.) -> np.ndarray:
    """
    Duffing ODE.

    Args:
        t (float): Time variable t
        x (np.ndarray): Current x(t) state
        delta (float, optional): Delta parameter. Defaults to 0.2
        alpha (float, optional): Alpha parameter. Defaults to 0.05
        beta (float, optional): Beta parameter. Defaults to 1.0

    Returns:
        np.ndarray: x_dot = f(x(t))
    """
    return np.array([x[1], -delta * x[1] - alpha * x[0] - beta * x[0] ** 3])


def van_der_pol(
        t: float, x: np.ndarray,
        mu: float = 0.5) -> np.ndarray:
    """
    Duffing ODE.

    Args:
        t (float): Time variable t
        x (np.ndarray): Current x(t) state
        mu (float, optional): Mu parameter. Defaults to 0.5

    Returns:
        np.ndarray: x_dot = f(x(t))
    """
    return np.array([x[1], mu * (1-x[0]**2) * x[1] - x[0]])


def lorenz(
        t: float, x: np.ndarray,
        sigma: float = 10, beta: float = 8/3, rho: float = 28
) -> np.ndarray:
    """
    Lorenz ODE.

    Args:
        t (float): Time variable t
        x (np.ndarray): Current x(t) state
        sigma (float, optional): Sigma parameter. Defaults to 10.
        beta (float, optional): Beta parameter. Defaults to 8/3.
        rho (float, optional): Rho parameter. Defaults to 28.

    Returns:
        np.ndarray: x_dot = f(x(t))
    """
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def lorenz_control(
        t: float, x: np.ndarray, u_func: Callable[[float], np.ndarray],
        sigma: float = 10, beta: float = 8/3, rho: float = 28
) -> np.ndarray:
    """
    Lorenz ODE with forcing function on x term.

    Args:
        t (float): Time variable t.
        x (np.ndarray): Current x(t) state.
        u_func (Callable): A callable function that takes in time and outputs
            the control/forcing scalar value in numpy array.
        sigma (float, optional): Sigma parameter. Defaults to 10.
        beta (float, optional): Beta parameter. Defaults to 8/3.
        rho (float, optional): Rho parameter. Defaults to 28.

    Returns:
        np.ndarray: x_dot = f(x(t))
    """
    
    u: np.ndarray = u_func(t)
    return np.array([
        sigma * (x[1] - x[0]) + u.item(0),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def f8_crusader(
        t: float, x: np.ndarray, u_func: Callable[[float], np.ndarray]
) -> np.ndarray:
    """
    F-8 Crusader example ODE used by (Kaiser et al, 2018), adapted from
    (Garrard and Jordan, 1977)

    Args:
        t (float): Time variable t.
        x (np.ndarray): Current state vector [x1, x2, x3].
        u_func (Callable): A callable function that takes in time and outputs
            the control scalar value in a numpy array.

    Returns:
        np.ndarray: x_dot = f(x(t))
    """
    
    u: float = u_func(t).item(0)

    return np.array([
        # dx1/dt
        -0.877*x[0] + x[2] - 0.088*x[0]*x[2] + 0.47*x[0]**2 - 0.019*x[1]**2
        - x[0]**2*x[2] + 3.846*x[0]**3 - 0.215*u + 0.28*(x[0]**2)*u
        + 0.47*x[0]*u**2 + 0.63*u**3,

        # dx2/dt
        x[2],

        # dx3/dt
        -4.208*x[0] - 0.396*x[2] - 0.47*x[0]**2 - 3.564*x[0]**3 - 20.967*u
        + 6.265*(x[0]**2)*u + 46*x[0]*u**2 + 61.1*u**3
    ])