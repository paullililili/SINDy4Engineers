import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional

from .optimizer import stridge
from .metrics import precision, rel_fro_err


def ensemble(
        theta: np.ndarray, xdot: np.ndarray,
        lam: float = 0.01, l2_norm: float = 1e-3,
        bootstrap: bool = False, library_ensemble: bool = False,
        replace: bool = True, n_candidate_to_drop: Optional[int] = None,
        n_models: int = 20, n_subset: Optional[int] = None
) -> np.ndarray:
    """
    Function that produces SINDy model ensembles.

    Args:
        theta (np.ndarray): Candidate function library matrix.
        xdot (np.ndarray): Time derivative matrix.
        lam (float, optional): Regularization parameter. Defaults to 0.01.
        l2_norm (float, optional): L2 norm for regularization. Defaults to 1e-3.
        bootstrap (bool, optional): Whether to use bootstrap sampling on the
            data. Defaults to False.
        library_ensemble (bool, optional): Whether to use library ensembling.
            Defaults to False.
        replace (bool, optional): Whether to sample with replacement. Defaults
            to True.
        n_candidate_to_drop (Optional[int], optional): Number of candidate
            terms to drop for library ensemble. Defaults to None.
        n_models (int, optional): Number of models to ensemble. Defaults to 20.
        n_subset (Optional[int], optional): Number of samples to include in the
            subset. Defaults to None.

    Returns:
        np.ndarray: Ensemble of SINDy models.
    """
    
    # Get data dimensions
    m: int = theta.shape[0]
    n: int = xdot.shape[1]
    l: int = theta.shape[1]

    # Get subset size
    if n_subset is None:
        n_subset: int = m if replace else int(0.5 * m)
    else:
        assert n_subset <= m, (
            f"Subset size must be smaller than total samples provided! "
            + f"Received data with {m:d} samples but requesting a subset "
            + f"size of {n_subset:d}."
        )

    # Validate candidate drop size
    if n_candidate_to_drop is None:
        n_candidate_to_drop = int(np.round(0.1 * l))
    else:
        assert n_candidate_to_drop < l, (
            f"Not possible to drop {n_candidate_to_drop:d} from a library "
            + f"with {l:d} candidates!"
        )

    # Initialize list of models
    xis: list[np.ndarray] = list()

    # Iterate through the specified number of models
    for _ in range(n_models):

        # Bootstrap the dataset if bootstrap is enabled
        if bootstrap:
            theta_en, xdot_en = subsample_data(
                theta, xdot,
                replace, n_subset
            )
        else:
            theta_en, xdot_en = theta, xdot

        # Drop candidate terms if library ensembling is enabled
        if library_ensemble:
            theta_en, theta_idx = subsample_library(
                theta_en, n_candidate_to_drop
            )
        else:
            theta_idx = range(l)

        # Fit SINDy model with STRidge
        xi = np.zeros((l,n))
        xi[theta_idx, :] = stridge(theta_en, xdot_en, lam, l2_norm)
        xis.append(xi)

    return np.stack(xis, axis=-1)


def subsample_data(
        theta: np.ndarray, xdot: np.ndarray,
        replace: bool, n_subset: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function subsamples data randomly with or with replacement from both
    the candidate function library and the time derivatives.

    Args:
        theta (np.ndarray): Candidate function library matrix.
        xdot (np.ndarray): Time derivative matrix.
        replace (bool): Whether to sample with replacement.
        n_subset (int): Number of samples to include in the subset.

    Returns:
        tuple[np.ndarray, np.ndarray]: Subsampled candidate function library
            and time derivative matrices.
    """
        
    # Get total number of samples
    m: int = theta.shape[0]

    # Check if the number of subset specified is valid
    assert n_subset <= m, (
        f"Subset size must be smaller than total samples provided! "
        + f"Received data with {m:d} samples but requesting a subset "
        + f"size of {n_subset:d}."
    )
    
    # Create subsamples of data
    idx_samples: np.ndarray = np.random.choice(
        range(m), n_subset, replace=replace
    )
    
    return theta[idx_samples], xdot[idx_samples]


def subsample_library(
        theta: np.ndarray, n_candidate_to_drop: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function subsamples functions from the candidate function library.

    Args:
        theta (np.ndarray): The full candidate function library.
        n_candidate_to_drop (int): Number of candidate functions to drop when
            subsampling.

    Returns:
        tuple[np.ndarray, np.ndarray]: Subsampled candidate function library
            and the corresponding indices of the retained functions.
    """
    
    # Get original library dimension
    l: int = theta.shape[1]
    
    # Validate candidate drop size
    assert n_candidate_to_drop < l, (
        f"Not possible to drop {n_candidate_to_drop:d} from a library with "
        + f"{l:d} candidates!"
    )

    # Drop terms the specified number of candidate terms
    theta_idx: np.ndarray = np.random.choice(
        range(l), l - n_candidate_to_drop,
        replace=False
    )

    return theta[:, theta_idx], theta_idx


def get_model_metric(
        xi: np.ndarray, true_xi: np.ndarray,
        theta_validation: np.ndarray,
        xdot_validation: np.ndarray
) -> tuple[float, float, float]:
    """
    Helper function to compute both the precision metric as well as the
        Frobenius error of the model and its prediction.

    Args:
        xi (np.ndarray): Coefficient matrix of the fitted model.
        true_xi (np.ndarray): True coefficient matrix of the system.
        theta_validation (np.ndarray): Candidate function library built from
            validation data.
        xdot_validation (np.ndarray): Time derivatives built from validation
            data.

    Returns:
        tuple[float, float, float]: Model precision, model error, and
            trajectory error.
    """
    
    # Compute the precision
    prec: float = precision(xi, true_xi)
    model_err: float = rel_fro_err(xi, true_xi)

    # Compute a trajectory using validation data and its error
    xdot_predict: np.ndarray = theta_validation @ xi
    traj_err: float = rel_fro_err(xdot_predict, xdot_validation)

    return prec, model_err, traj_err


def benchmark_ensemble(
        noise_ratios: np.ndarray, data_use_ratios: np.ndarray,
        thetas: list[np.ndarray], xdots: list[np.ndarray],
        theta_validation: np.ndarray, xdot_validation: np.ndarray,
        lam: float, l2_norm: float, incl_threshold: float,
        n_models: int, n_candidate_to_drop: int,
        true_xi: np.ndarray
) -> tuple[dict, dict]:
    """
    A function to benchmark the different methods of ensembling in SINDy: 
    vanilla SINDy, bragging, library ensembling and double bragging with
    inclusion thresholding. This function returns dictionaries of the model
    ensembles and their corresponding performance metrics.

    Args:
        noise_ratios (np.ndarray): Noise ratio to sweep through for
            benchmarking.
        data_use_ratios (np.ndarray): Percentages of the training data to use
            when sweeping through data availability levels.
        thetas (list[np.ndarray]): List of candidate function libraries with
            corresponding noise levels.
        xdots (list[np.ndarray]): List of time derivative matrices with
            corresponding noise levels.
        theta_validation (np.ndarray): Candidate function library built from
            validation data.
        xdot_validation (np.ndarray): Time derivatives built from validation
            data.
        lam (float): Regularization parameter.
        l2_norm (float): L2 norm parameter.
        incl_threshold (float): Inclusion threshold parameter for double
            bragging.
        n_models (int): Number of models to ensemble for each ensemble method.
        n_candidate_to_drop (int): Number of candidate functions to drop when
            subsampling.
        true_xi (np.ndarray): True coefficient matrix of the system.

    Returns:
        tuple[dict, dict]: Dictionaries of the model ensembles and their
            corresponding performance metrics.
    """
    
    # Get data dimensions
    l: int = theta_validation.shape[1]
    n: int = xdot_validation.shape[1]
    
    # Define metric shape
    heatmap_shape: tuple = (noise_ratios.shape[0], data_use_ratios.shape[0])

    # Define ensemble methods and metrics
    methods = [
        'SINDy', 'Bragging', 'Library Ensemble', 'Inclusion Thresholding'
    ]
    metrics_keys = ['Precision', 'Model Error', 'Trajectory Error']

    # Initialise metrics and coefficient matrix fields
    metrics_data = dict()
    xis = dict()
    for method in methods:
        metrics_data[method] = dict()
        xis[method] = np.zeros((
            noise_ratios.shape[0], data_use_ratios.shape[0], l, n, n_models
        ))
        for metrics_key in metrics_keys:
            metrics_data[method][metrics_key] = np.zeros(heatmap_shape)

    # Loop through specified noise ratios and data use percentage
    for noiseIdx in range(noise_ratios.shape[0]):
        for dataIdx in range(data_use_ratios.shape[0]):

            data_cutoff = int(thetas[0].shape[0] * data_use_ratios[dataIdx])
            
            # ----------- Fit SINDy model ----------- #

            sindy_xi = stridge(
                thetas[noiseIdx][:data_cutoff,:],
                xdots[noiseIdx][:data_cutoff,:],
                lam, l2_norm
            )
            xis['SINDy'][noiseIdx, dataIdx] = sindy_xi[..., None]

            # Get SINDy model metrics
            prec, model_err, traj_err = get_model_metric(
                sindy_xi, true_xi, theta_validation, xdot_validation
            )
            metrics_data['SINDy']['Precision'][noiseIdx, dataIdx] = prec
            metrics_data['SINDy']['Model Error'][noiseIdx, dataIdx] = model_err
            metrics_data['SINDy']['Trajectory Error'][noiseIdx, dataIdx] = traj_err

            # ----------- Fit ensemble model ----------- #

            ensemble_xis = ensemble(
                thetas[noiseIdx][:data_cutoff,:],
                xdots[noiseIdx][:data_cutoff,:],
                lam, l2_norm,
                bootstrap=True, library_ensemble=False, replace=False,
                n_candidate_to_drop=n_candidate_to_drop, n_models=n_models
            )
            xis['Bragging'][noiseIdx, dataIdx] = ensemble_xis

            # Obtain model bragging metrics
            bragging_xi: np.ndarray = np.median(ensemble_xis, axis=-1)
            bragging_xi[np.abs(bragging_xi) < lam] = 0.0
            prec, model_err, traj_err = get_model_metric(
                bragging_xi, true_xi, theta_validation, xdot_validation
            )
            metrics_data['Bragging']['Precision'][noiseIdx, dataIdx] = prec
            metrics_data['Bragging']['Model Error'][noiseIdx, dataIdx] = model_err
            metrics_data['Bragging']['Trajectory Error'][noiseIdx, dataIdx] = traj_err

            # ----------- Fit library ensemble model ----------- #

            lib_ensemble_xis = ensemble(
                thetas[noiseIdx][:data_cutoff,:],
                xdots[noiseIdx][:data_cutoff,:],
                lam, l2_norm,
                bootstrap=True, library_ensemble=True, replace=False,
                n_candidate_to_drop=n_candidate_to_drop, n_models=n_models
            )
            xis['Library Ensemble'][noiseIdx, dataIdx] = lib_ensemble_xis

            # Obtain library ensemble bragging metrics
            lib_bragging_xi: np.ndarray = np.median(lib_ensemble_xis, axis=-1)
            lib_bragging_xi[np.abs(lib_bragging_xi) < lam] = 0.0
            prec, model_err, traj_err = get_model_metric(
                lib_bragging_xi, true_xi, theta_validation, xdot_validation
            )
            metrics_data['Library Ensemble']['Precision'][
                noiseIdx, dataIdx] = prec
            metrics_data['Library Ensemble']['Model Error'][
                noiseIdx, dataIdx] = model_err
            metrics_data['Library Ensemble']['Trajectory Error'][
                noiseIdx, dataIdx] = traj_err

            # ----------- Fit double bragged model ----------- #
            
            # Find the inclusion probability from library ensembling
            incl_prob: np.ndarray = np.count_nonzero(
                xis['Library Ensemble'][noiseIdx, dataIdx], axis=-1
            ) / n_models

            # Get thresholded indices using inclusion probability
            threshold_idx: np.ndarray = incl_prob > incl_threshold

            # Carry out second library ensembling just with data bootstrapping
            # for each state
            for idx in range(xdots[noiseIdx].shape[1]):
                xis['Inclusion Thresholding'][
                    noiseIdx, dataIdx, threshold_idx[:, idx], idx:idx+1, :
                ] = ensemble(
                    thetas[noiseIdx][:data_cutoff, threshold_idx[:, idx]],
                    xdots[noiseIdx][:data_cutoff, idx:idx+1],
                    lam, l2_norm,
                    bootstrap=True, library_ensemble=False, replace=False,
                    n_models=n_models
                )

            # Obtain library ensemble bragging metrics
            double_brag_xi: np.ndarray = np.median(
                xis['Inclusion Thresholding'][noiseIdx, dataIdx],
                axis=-1
            )
            double_brag_xi[np.abs(double_brag_xi) < lam] = 0.0
            prec, model_err, traj_err = get_model_metric(
                double_brag_xi, true_xi, theta_validation, xdot_validation
            )
            metrics_data['Inclusion Thresholding']['Precision'][
                noiseIdx, dataIdx] = prec
            metrics_data['Inclusion Thresholding']['Model Error'][
                noiseIdx, dataIdx] = model_err
            metrics_data['Inclusion Thresholding']['Trajectory Error'][
                noiseIdx, dataIdx] = traj_err

    return xis, metrics_data


def plot_heatmap(
        noise_ratios: np.ndarray, data_use_ratios: np.ndarray,
        metrics_data: dict
):
    """
    A heatmap plotting function that visualizes the metrics returned from
    `benchmark_ensemble`.

    Args:
        noise_ratios (np.ndarray): Array of noise ratios.
        data_use_ratios (np.ndarray): Array of data use ratios.
        metrics_data (dict): Dictionary containing the metrics data.
    """

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16,10))

    row_titles = ['Precision', 'Log of Model Error', 'Log of Trajectory Error']
    methods = metrics_data.keys()
    
    # Tick formatting
    x_ticks = np.arange(len(data_use_ratios))
    y_ticks = np.arange(len(noise_ratios))
    xticklabels = [f"{x*100:.2f}%" for x in data_use_ratios]
    yticklabels = [f"{y*100:.2f}%" for y in noise_ratios]

    # Loop through each metric to plot
    for row_idx, metric_key in enumerate(metrics_data[list(methods)[0]].keys()):

        # Obtain min and max values across the methods for each metric
        row_vals = []
        for method in methods:
            row_vals.append(metrics_data[method][metric_key])
        row_vals = np.array(row_vals)
        
        # Log scale handling for errors
        if metric_key != 'Precision':
            row_vals = np.log10(row_vals + 1e-16)
            g_min, g_max = np.min(row_vals), np.max(row_vals)
        else:
            g_min, g_max = 0.0, 1.0

        # Loop through each method for the given metric
        for col_idx, method in enumerate(methods):
            ax = axs[row_idx, col_idx]
            
            # Log transform the data if needed
            data = metrics_data[method][metric_key]
            if metric_key != 'Precision':
                data = np.log10(data + 1e-16)

            cmap = 'viridis_r' if metric_key == 'Precision' else 'inferno_r'
            im = ax.imshow(
                data,
                origin='lower',
                aspect='auto',
                vmin=g_min, vmax=g_max,
                cmap=cmap
            )

            # Titles and Labels
            if row_idx == 0:
                ax.set_title(method, fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=12)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(yticklabels)

            if row_idx == 2:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(xticklabels, rotation=45)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    # Global Labels
    fig.supxlabel("Data Use Ratio", fontsize=16)
    fig.supylabel("Noise Ratio", fontsize=16)


def plot_coef_distribution(
        xis: np.ndarray,
        state_names: list[str], lib_names: list[str],
        useMedian: bool = True
):
    """
    Plotting function to visualize the coefficient distribution across the
    ensembles.

    Args:
        xis (np.ndarray): The ensemble of models.
        state_names (list[str]): The state variable names.
        lib_names (list[str]): The names of candidate functions.
        useMedian (bool, optional): Whether to plot median or mean. Defaults to
            True.
    """

    # Get data dimensions
    l: int = xis.shape[0]
    n: int = xis.shape[1]
    n_models: int = xis.shape[2]

    # Initialise plot
    fig, axs = plt.subplots(
        l, n,
        sharex=False, sharey=False,
        figsize=(2.5*n, 1.5*l)
    )

    # Compute distribution properties
    if useMedian:
        xi_avg = np.median(xis, axis=-1)
    else:
        xi_avg = np.mean(xis, axis=-1)
    incl_prob = np.count_nonzero(xis, axis=-1) / n_models

    # Loop through each term
    for lIdx in range(l):
        for nIdx in range(n):

            # Obtain coefficients
            coefs = xis[lIdx, nIdx, :]

            # Plot distribution
            if not np.allclose(coefs, coefs[0]):
                kde = stats.gaussian_kde(coefs)
                xi_min = np.min(coefs)
                xi_max = np.max(coefs)
                plot_values = np.linspace(xi_min, xi_max)
                axs[lIdx, nIdx].fill_between(
                    plot_values,
                    np.zeros(plot_values.shape), kde(plot_values),
                    color=f'C{nIdx:d}', alpha=0.5, linewidth=1
                )
            else:
                axs[lIdx, nIdx].set_xlim(
                    [xi_avg[lIdx, nIdx]-10, xi_avg[lIdx, nIdx]+10]
                )

            # Plot average
            axs[lIdx, nIdx].axvline(
                xi_avg[lIdx, nIdx], color=f'C{nIdx:d}'
            )
            axs[lIdx, nIdx].set_xlabel(
                f'Inclusion probability of \n'
                + fr'$\xi_{{{lIdx+1:d}, {nIdx+1:d}}}$ '
                + fr'= {incl_prob[lIdx, nIdx]:.2f}',
                fontsize=7
            )
            axs[lIdx, nIdx].set_yticklabels([])
            
            if nIdx == 0:
                axs[lIdx, nIdx].set_ylabel(f'${lib_names[lIdx]}$', fontsize=12)

            if lIdx == 0:
                axs[lIdx, nIdx].set_title(
                    rf'$\dot{{{state_names[nIdx]}}}$',
                    fontsize=12
                )

    fig.tight_layout()


def plot_predict_trajectory(
        xis: np.ndarray,
        validation_theta: np.ndarray, validation_xdot: np.ndarray,
        t: np.ndarray, state_names: list[str]
):
    """
    Function to visualise the distribution of trajectories generated by the
    ensembled models.

    Args:
        xis (np.ndarray): Ensemble of models.
        validation_theta (np.ndarray): Validation candidate function library.
        validation_xdot (np.ndarray): Validation time derivatives.
        t (np.ndarray): Time points for plotting.
        state_names (list[str]): State variable names.
    """
    
    # Get dimensions
    m: int = validation_theta.shape[0]
    n: int = xis.shape[1]
    n_models: int = xis.shape[2]

    # Initialise predicted trajectories
    xdot_predicts: np.ndarray = np.zeros((m, n, n_models))

    # Predict using provided models
    for idx in range(n_models):
        xdot_predicts[..., idx] = validation_theta @ xis[..., idx]

    # Find distribution statistics
    xdot_mean = np.mean(xdot_predicts, axis=-1)
    lower_bound = np.percentile(xdot_predicts, 2.5, axis=-1)
    upper_bound = np.percentile(xdot_predicts, 97.5, axis=-1)

    # Plot prediction with uncertainty interval
    _, axs = plt.subplots(n, 1, sharex=True, figsize=(10, 3*n))

    for idx in range(n):

        axs[idx].fill_between(
            t, lower_bound[:,idx], upper_bound[:,idx],
            label=r'95% percentile', alpha=0.5
        )
        axs[idx].plot(t, xdot_mean[:, idx], label='Prediction mean')
        axs[idx].plot(
            t, validation_xdot[:, idx],
            '--k', label='True trajectory'
        )
        axs[idx].legend()
        axs[idx].set_ylabel(state_names[idx])
        
    axs[-1].set_xlabel('Time')