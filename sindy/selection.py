import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pysindy as ps
from pysindy.feature_library.base import BaseFeatureLibrary
from typing import Callable, Optional, Union
from tqdm import tqdm

from .ivp_solvers import rk4
from .optimizer import stcls


def aic(
        t: np.ndarray, xs: list[np.ndarray],
        theta_func: Callable, xis: list[np.ndarray],
        ivp_sovler: Callable = rk4
) -> tuple[list[float], list[int], list[float]]:
    
    # Obtain number of trajectories
    p: int = len(xs)

    # Obtain number of candidate models
    j: int = len(xis)

    # Initialise list of AIC scores
    aics: list[float] = list()

    # Initialise list of active terms and their average MAEs
    ks: list[int] = list()
    mses: list[float] = list()

    # Loop through each candidate model
    for j_idx in tqdm(range(j), desc="Candidate models assessed"):

        # Initialise list of mean absolute errors for all trajectories with jth
        # candidate model
        mae_list: list[float] = list()

        # Loop through each trajectory
        for p_idx in range(p):

            # Obtain IVP solver parameters
            dt: float = np.mean(np.diff(t))
            tFinal: float = t[-1]

            # Forward solve using the candidate model
            f: Callable = lambda t, x: theta_func(x.reshape(1,-1)) @ xis[j_idx]
            pred_x: np.ndarray = ivp_sovler(f, tFinal, xs[p_idx][0,:], dt)[1]

            # Compute mean absolute error
            mae_list.append(np.mean(np.abs(pred_x - xs[p_idx])))

        # Obtain number of nonzero terms
        k: int = np.count_nonzero(xis[j_idx])

        # Get MSE
        mse: np.ndarray = np.mean(np.array(mae_list)**2)

        # Compute the AIC score for this candidate model
        aic: float = 2*k + p * np.log(mse)

        # Store result for the jth candidate model
        aics.append(aic)
        ks.append(k)
        mses.append(mse)

    return aics, ks, mses


def stability_selection(
        lambdas: list[float], B: int,
        theta: np.ndarray, xdot: np.ndarray,
        state_names: list[str],
        l2_norm: float = 0.0,
        constraint_lhs: Optional[np.ndarray] = None,
        constraint_rhs: Optional[np.ndarray] = None,
        normalize_columns: bool = False
):
    """
    Visualizes the stability selection method, by performing a lambda
    hyperparameter sweep with an ensemble of models built using own SINDy
    implementation.

    Args:
        lambdas (list[float]): Lambdas to sweep.
        B (int): Number of models to use for each lambda value.
        theta (np.ndarray): Candidate function library matrix.
        xdot (np.ndarray): Derivative of time series data matrix.
        state_names (list[str]): Names of states.
        l2_norm (float, optional): L2 ridge regularizer strength. Defaults to
            0.0.
        constraint_lhs (Optional[np.ndarray], optional): LHS of constraint
            equation, if applicable. Defaults to None.
        constraint_rhs (Optional[np.ndarray], optional): RHS of constraint
            equation, if applicable. Defaults to None.
        normalize_columns (bool, optional): Whether to normalize the library
            matrix. Defaults to False.
    """
    
    # Get dimensional information of the dataset
    m: int = theta.shape[0]
    n: int = xdot.shape[1]
    l: int = theta.shape[1]

    # Initialise coefficient inclusion counter for each lambda
    term_counter: list[np.ndarray] = list()

    # Normalize the library if enabled
    if normalize_columns:
        norm_inv = np.linalg.inv(
            np.linalg.norm(theta, axis=0) * np.eye(theta.shape[1])
        )
        theta = theta @ norm_inv

        if constraint_lhs is not None:
            for idx in range(n):
                constraint_lhs[:, idx*l:(idx+1)*l] = \
                    constraint_lhs[:, idx*l:(idx+1)*l] @ norm_inv
    
    # Sweep through all thresholds
    for lam in lambdas:

        term_counter.append(np.zeros((l,n)))

        # Loop through all subsets of data
        for _ in range(B):

            # Sample a subset sized m/2
            sample_idx = np.random.choice(range(m), (int(m/2),))

            # Fit a new model
            xi = stcls(
                theta[sample_idx], xdot[sample_idx],
                lam, l2_norm,
                constraint_lhs, constraint_rhs
            )

            # Add active terms to counter
            term_counter[-1] += xi != 0.

    # Compute importance measure (inclusion probability)
    im: np.ndarray = np.array([counter/B for counter in term_counter])

    # Visualize importance measure
    _, axs = plt.subplots(n, 1, sharex=True, figsize=(14,10))

    # Plot stability selection plot
    for n_idx in range(n):
        axs[n_idx].plot(lambdas, im[:,:,n_idx], 'k', linewidth=1, alpha=0.5)
        axs[n_idx].axhline(0.8, linestyle='--', color='red')
        axs[n_idx].set_xscale('log')
        axs[n_idx].set_ylabel(
            'Inclusion probability\n' + fr'for $\dot{{{state_names[n_idx]}}}$'
        )
        axs[n_idx].minorticks_on()
        axs[n_idx].xaxis.set_major_locator(LogLocator(
            base=10.0, subs=[1.0], numticks=100
        ))
        axs[n_idx].xaxis.set_minor_locator(LogLocator(
            base=10.0, subs='auto', numticks=100
        ))

    axs[-1].set_xlabel(r'Sparsification threshold $\lambda$');


def ps_stability_selection(
        x: Union[np.ndarray, list[np.ndarray]],
        t: Union[np.ndarray, list[np.ndarray]],
        u: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        lambdas: np.ndarray = np.logspace(-1, 8), l2_norm: float = 0.05,
        B: int = 50,
        ps_library: BaseFeatureLibrary = ps.PolynomialLibrary(),
        ps_fd: ps.BaseDifferentiation = ps.FiniteDifference(4),
        normalize_columns: bool = False,
        feature_names: Optional[list[str]] = None
):
    """
    Visualizes the stability selection method, by performing a lambda
    hyperparameter sweep with an ensemble of models built PySINDy.

    Args:
        x (Union[np.ndarray, list[np.ndarray]]): Time series data matrix.
        t (Union[np.ndarray, list[np.ndarray]]): Time vector.
        u (Optional[Union[np.ndarray, list[np.ndarray]]], optional): Control
            inputs or parameters. Defaults to None.
        lambdas (np.ndarray, optional): Lambda values tos weep. Defaults to
            `np.logspace(-1, 8)`.
        l2_norm (float, optional): L2 ridge regularizer strength. Defaults to
            0.05.
        B (int, optional): Number of models to use for each lambda value.
            Defaults to 50.
        ps_library (BaseFeatureLibrary, optional): PySINDy library to use.
            Defaults to ps.PolynomialLibrary().
        ps_fd (ps.BaseDifferentiation, optional): PySINDy differentiator to
            use. Defaults to ps.FiniteDifference(4).
        normalize_columns (bool, optional): Whether to normalize library matrix.
            Defaults to False.
        feature_names (Optional[list[str]], optional): Feature names. Defaults
            to None.
    """
    
    # Check if data is multi-trajectory
    if isinstance(x, list):
        
        # Differentiate each trajectory individually and offset time to ensure
        # monotonicity
        xdot_list = list()
        t_list = list()
        for idx in range(len(x)):
            xdot_list.append(ps_fd._differentiate(x[idx], t[idx]))
            t_offset = t_list[-1][-1] if len(t_list) > 0 else 0.0
            t_list.append(t[idx] + t_offset + 1.0)
        
        # Concatenate data
        x = np.vstack(x)
        t = np.concatenate(t_list)
        if u is not None:
            u = np.vstack(u)
        xdot = np.vstack(xdot_list)

    else:
        xdot: np.ndarray = ps_fd._differentiate(x, t)
    
    # Get data dimensions
    n_lambdas: int = len(lambdas)
    m: int = x.shape[-2]
    n: int = x.shape[-1]

    # Test fit a function to find library size
    model = ps.SINDy(feature_library=ps_library)
    model.fit(
        x[..., :1, :], t[:1], xdot[..., :1, :],
        u[..., :1, :] if u is not None else u
    )
    l: int = model.coefficients().shape[1]

    # Initialise importance measure array (inclusion probability)
    im: np.ndarray = np.zeros((n_lambdas, l, n))

    for idx, lam in enumerate(tqdm(lambdas)):

        for _ in range(B):

            # Sample a subset sized m/2
            sample_idx = np.sort(np.random.choice(
                range(m), (int(m/2),), replace=False
            ))

            # Set up PySINDy model
            opti = ps.STLSQ(lam, l2_norm, normalize_columns=normalize_columns)
            model = ps.SINDy(opti, ps_library, ps_fd)

            # Fit PySINDy model using subset of data
            model.fit(
                x[..., sample_idx, :], t[sample_idx],
                x_dot=xdot[..., sample_idx, :],
                u=u[..., sample_idx, :] if u is not None else u,
                feature_names=feature_names
            )

            # Add to im
            im[idx] += model.coefficients().T != 0.0

    # Obtain inclusion probability
    im /= B

    # Plot stability selection
    _, axs = plt.subplots(n, 1, sharex=True, figsize=(14,3*n))

    for n_idx in range(n):
        axs[n_idx].plot(lambdas, im[:,:,n_idx], 'k', linewidth=1, alpha=0.5)
        axs[n_idx].axhline(0.8, linestyle='--', color='red')
        axs[n_idx].set_xscale('log')
        if feature_names is not None:
            axs[n_idx].set_ylabel(
                'Inclusion probability\n' + fr'for $\dot{{{feature_names[n_idx]}}}$'
            )
        axs[n_idx].minorticks_on()
        axs[n_idx].xaxis.set_major_locator(LogLocator(
            base=10.0, subs=[1.0], numticks=100
        ))
        axs[n_idx].xaxis.set_minor_locator(LogLocator(
            base=10.0, subs='auto', numticks=100
        ))

    axs[-1].set_xlabel(r'Sparsification threshold $\lambda$');