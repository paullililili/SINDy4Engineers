import copy
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.stats.qmc import LatinHypercube
from typing import Optional

from ..library import poly_lib
from ..optimizer import stridge as _stridge


def calculate_rms(signal: np.ndarray) -> float:
    """Root Mean Square of a signal array."""
    if signal is None or signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def sample_candidates(
        n: int, iteration: int,
        n_dims: int, state_bounds: np.ndarray
) -> np.ndarray:
    """
    Latin Hypercube sampling of candidate initial conditions.

    Args:
        n (int): Number of candidate points to generate.
        iteration (int): Current iteration index, used as LHS seed.
        n_dims (int): State-space dimensionality.
        state_bounds (np.ndarray): Shape (n_dims, 2) bounds per dimension.

    Returns:
        np.ndarray: Shape (n, n_dims) array of candidate initial conditions.
    """
    sampler = LatinHypercube(d=n_dims, seed=iteration)
    unit = sampler.random(n=n)
    return state_bounds[:, 0] + (state_bounds[:, 1] - state_bounds[:, 0]) * unit


def _differentiate(x: np.ndarray, t: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Savitzky-Golay derivative — matches pysindy's
    ``SmoothedFiniteDifference(smoother_kws={'window_length': window})``.

    Args:
        x (np.ndarray): State trajectory (M, n_dims).
        t (np.ndarray): Time vector (M,), assumed uniform spacing.
        window (int): Savitzky-Golay window length (must be odd, <= M).

    Returns:
        np.ndarray: Time derivatives (M, n_dims).
    """
    dt = t[1] - t[0]
    return savgol_filter(x, window_length=window, polyorder=3, deriv=1, delta=dt, axis=0)


def _build_theta_xdot(
        x_list: list, t_list: list,
        poly_degree: int, feature_names: list,
        smoother_window: int = 7
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Build concatenated library matrix and derivative matrix from a list of
    trajectories, then vertically stack them.

    Args:
        x_list (list[np.ndarray]): State trajectories, each (M_i, n_dims).
        t_list (list[np.ndarray]): Corresponding time vectors, each (M_i,).
        poly_degree (int): Polynomial degree for the library.
        feature_names (list[str]): State variable names.
        smoother_window (int): Savitzky-Golay window length. Defaults to 7.

    Returns:
        tuple:
            theta (np.ndarray): Concatenated library matrix (M_total, l).
            xdot (np.ndarray): Concatenated derivative matrix (M_total, n_dims).
            lib_names (list[str]): Library term labels.
    """
    thetas, xdots = [], []
    lib_names = None

    for x, t in zip(x_list, t_list):
        xdot_i = _differentiate(x, t, window=smoother_window)
        theta_i, lib_names = poly_lib(
            x, power=poly_degree,
            include_bias=False, feature_names=feature_names
        )
        thetas.append(theta_i)
        xdots.append(xdot_i)

    return np.vstack(thetas), np.vstack(xdots), lib_names


def _run_ensemble(
        theta: np.ndarray, xdot: np.ndarray,
        lam: float, l2_norm: float, n_models: int,
        n_candidates_to_drop: int = 1,
) -> np.ndarray:
    """
    Mirrors pysindy's ``EnsembleOptimizer(STLSQ(threshold=lam, alpha=l2_norm),
    bagging=True, library_ensemble=True, replace=False, n_models=n_models)``.

    Per pysindy source:
      - Row subset = int(0.6 * m), sampled without replacement.
      - Library subset = drop ``n_candidates_to_drop`` columns at random.
      - Optimizer = STRidge (ridge + sequential thresholding), which is
        mathematically identical to pysindy's STLSQ with alpha > 0.
      - Result stored as zeros in dropped positions (same as pysindy's
        ``new_coefs[:, keep_inds] = coef_``).

    Args:
        theta (np.ndarray): Library matrix (M, l).
        xdot (np.ndarray): Derivative matrix (M, n).
        lam (float): Sparsity threshold (= STLSQ threshold).
        l2_norm (float): Ridge regularisation (= STLSQ alpha).
        n_models (int): Number of ensemble models.
        n_candidates_to_drop (int): Library columns dropped per model.
            Defaults to 1 — pysindy's ``n_candidates_to_drop`` default.

    Returns:
        np.ndarray: Shape (l, n, n_models).
    """
    m, l = theta.shape
    n = xdot.shape[1]

    # pysindy: n_subset = int(0.6 * n_samples) when replace=False
    n_subset = min(int(0.6 * m), m)
    n_keep = l - n_candidates_to_drop

    xis = []
    for _ in range(n_models):
        row_idx = np.random.choice(m, n_subset, replace=False)
        keep_inds = np.sort(np.random.choice(l, n_keep, replace=False))

        xi_sub = _stridge(
            theta[np.ix_(row_idx, keep_inds)],
            xdot[row_idx],
            lam, l2_norm,
        )

        xi = np.zeros((l, n))
        xi[keep_inds, :] = xi_sub
        xis.append(xi)

    return np.stack(xis, axis=-1)  # (l, n, n_models)


def run_active_sindy(
        x_init: list, t_init: list, params: dict
) -> tuple[list, list, list]:
    """
    Active SINDy loop: iteratively selects the highest-uncertainty initial
    condition using ensemble variance to guide new data acquisition.

    Uses the custom ``sindy`` package throughout (``poly_lib``, ``ensemble``).
    Differentiation matches the original script's
    ``SmoothedFiniteDifference(smoother_kws={'window_length': 7})``.

    Args:
        x_init (list[np.ndarray]): Seed trajectory arrays, each (M_i, n_dims).
        t_init (list[np.ndarray]): Seed time vectors, each (M_i,).
        params (dict): Configuration dict with keys:

            ode (Callable): ODE right-hand side f(t, x, *ode_params).
            ode_params (tuple): Parameters passed to ode.
            n_dims (int): State-space dimensionality.
            feature_names (list[str]): State variable names.
            poly_degree (int): Polynomial library degree.
            smoother_window (int): Savitzky-Golay window length (default 7).
            lam (float): STRidge sparsity threshold.
            l2_norm (float): STRidge L2 regularisation.
            n_ensemble (int): Number of ensemble models per iteration.
            n_candidate_to_drop (int, optional): Library terms dropped per model.
            state_bounds (np.ndarray): (n_dims, 2) candidate sampling bounds.
            t_span_query (tuple): (t0, tf) for new trajectories.
            dt (float): Time step for t_eval of new trajectories.
            n_candidates (int): Candidate ICs per iteration.
            max_iter (int): Maximum iterations.
            patience (int): Sparsity-stable iterations before MAD check.
            conv_thresh_coef (float): Max rel-MAD (%) for convergence.
            relative_noise_factor (float): Noise amplitude relative to RMS.
            initial_conditions_lhs (np.ndarray): Seed IC array.

    Returns:
        tuple:
            coef_mad_history (list[np.ndarray]): Rel-MAD matrix (l, n_dims)
                per iteration.
            coef_estimated_history (list[np.ndarray]): Median coefficient
                matrix (l, n_dims) per iteration.
            points_ic (list): Initial conditions queried at each iteration.
    """
    x_train_list = copy.deepcopy(x_init)
    t_train_list = copy.deepcopy(t_init)

    l0_history         = []   # L0 norm  (sparsity of median coefficients)
    l2_history         = []   # L2 norm  (Frobenius error from truth, or MAD norm)
    data_count_history = []   # cumulative training samples at each fit

    converged = False
    xis = None

    sparsity_last_iteration = float('inf')
    sparsity_equal_itcount  = 0

    true_coeffs = params.get('true_coeffs', None)   # (l, n_dims) or None
    points_ic   = [params['initial_conditions_lhs']]

    num_points_query = int(round(
        (params['t_span_query'][1] - params['t_span_query'][0]) / params['dt']
    )) + 1
    t_eval_query = np.linspace(
        params['t_span_query'][0], params['t_span_query'][1], num_points_query
    )
    integrator_kws = dict(method='LSODA', rtol=1e-12, atol=1e-12, max_step=1e-3)
    smoother_window      = params.get('smoother_window', 7)
    n_candidates_to_drop = params.get('n_candidates_to_drop', 1)

    for iteration in range(params['max_iter']):

        if iteration % 10 == 0:
            print(f"  Active iteration {iteration + 1}/{params['max_iter']}")

        data_count = sum(len(t) for t in t_train_list)

        theta, xdot, _ = _build_theta_xdot(
            x_train_list, t_train_list,
            params['poly_degree'], params['feature_names'],
            smoother_window=smoother_window
        )

        # xis shape: (l, n_dims, n_ensemble)
        xis = _run_ensemble(
            theta, xdot,
            lam=params['lam'],
            l2_norm=params['l2_norm'],
            n_models=params['n_ensemble'],
            n_candidates_to_drop=n_candidates_to_drop,
        )

        coeffs_estimated = np.median(xis, axis=-1)          # (l, n_dims)
        sparsity = int(np.count_nonzero(coeffs_estimated))

        if sparsity == sparsity_last_iteration:
            sparsity_equal_itcount += 1
        else:
            sparsity_equal_itcount = 0

        coef_mad = np.median(
            np.abs(xis - coeffs_estimated[:, :, np.newaxis]), axis=-1
        )
        rel_mad = 100.0 * (coef_mad / (np.abs(coeffs_estimated) + 1e-6))

        active_mask = np.abs(coeffs_estimated) > 1e-6
        if active_mask.any():
            max_mad = float(np.max(rel_mad[active_mask]))
        else:
            max_mad = float('inf')

        if sparsity_equal_itcount >= params['patience'] and max_mad <= params['conv_thresh_coef']:
            converged = True
        else:
            sparsity_last_iteration = sparsity

        # --- Record metrics ---
        data_count_history.append(data_count)
        l0_history.append(sparsity)
        if true_coeffs is not None:
            l2_history.append(float(np.linalg.norm(coeffs_estimated - true_coeffs, 'fro')))
        else:
            l2_history.append(float(np.linalg.norm(coef_mad, 'fro')))

        if converged:
            break

        # --- Active sampling ---
        candidate_points = sample_candidates(
            params['n_candidates'], iteration,
            params['n_dims'], params['state_bounds']
        )

        existing_ics = [
            row.tolist()
            for arr in points_ic
            for row in (arr if arr.ndim == 2 else [arr])
        ]
        mask = ~np.any(
            np.all(np.isclose(candidate_points[:, None], existing_ics, atol=1e-2), axis=2),
            axis=1
        )
        candidate_points = candidate_points[mask]

        while len(candidate_points) < params['n_candidates']:
            needed = params['n_candidates'] - len(candidate_points)
            new_pts = sample_candidates(
                needed, iteration * 10,
                params['n_dims'], params['state_bounds']
            )
            mask = ~np.any(
                np.all(np.isclose(new_pts[:, None], candidate_points, atol=1e-2), axis=2),
                axis=1
            )
            candidate_points = np.vstack([candidate_points, new_pts[mask]])

        # Evaluate library at candidates  →  (n_cands, l)
        theta_cands, _ = poly_lib(
            candidate_points,
            power=params['poly_degree'],
            include_bias=False,
            feature_names=params['feature_names']
        )

        # Ensemble predictions  →  (n_ensemble, n_cands, n_dims)
        n_models = xis.shape[2]
        all_preds = np.stack(
            [theta_cands @ xis[:, :, i] for i in range(n_models)], axis=0
        )
        variance = np.var(all_preds, axis=0)                # (n_cands, n_dims)
        uncertainties = np.sum(variance, axis=1)            # (n_cands,)

        query_ic = candidate_points[np.argmax(uncertainties)]

        sol = solve_ivp(
            params['ode'], params['t_span_query'], query_ic,
            t_eval=t_eval_query, args=params['ode_params'], **integrator_kws
        )

        if sol.status == 0 and sol.y.shape[1] == len(t_eval_query):
            x_query = sol.y.T
            noise_std = calculate_rms(x_query) * params['relative_noise_factor']
            x_train_list.append(x_query + np.random.normal(0, noise_std, x_query.shape))
            t_train_list.append(sol.t)
        else:
            print(f"  Warning: solver failed at iteration {iteration + 1}. Skipping.")

        points_ic.append(query_ic)

    status = "converged" if converged else f"reached max iterations ({params['max_iter']})"
    print(f"\nActive SINDy finished — {status}.")

    return l0_history, l2_history, data_count_history


def run_random_sindy(
        x_init: list, t_init: list, params: dict
) -> tuple[list, list, list]:
    """
    Random-sampling SINDy baseline: same loop as :func:`run_active_sindy`
    but selects new ICs uniformly at random instead of by ensemble uncertainty.

    Args / Returns: identical signature to :func:`run_active_sindy`.
    """
    x_train_list = copy.deepcopy(x_init)
    t_train_list = copy.deepcopy(t_init)

    l0_history         = []
    l2_history         = []
    data_count_history = []

    converged = False

    sparsity_last_iteration = float('inf')
    sparsity_equal_itcount  = 0

    true_coeffs = params.get('true_coeffs', None)
    points_ic   = [params['initial_conditions_lhs']]

    num_points_query = int(round(
        (params['t_span_query'][1] - params['t_span_query'][0]) / params['dt']
    )) + 1
    t_eval_query = np.linspace(
        params['t_span_query'][0], params['t_span_query'][1], num_points_query
    )
    integrator_kws = dict(method='LSODA', rtol=1e-12, atol=1e-12, max_step=1e-3)
    smoother_window      = params.get('smoother_window', 7)
    n_candidates_to_drop = params.get('n_candidates_to_drop', 1)

    for iteration in range(params['max_iter']):

        if iteration % 10 == 0:
            print(f"  Random iteration {iteration + 1}/{params['max_iter']}")

        data_count = sum(len(t) for t in t_train_list)

        theta, xdot, _ = _build_theta_xdot(
            x_train_list, t_train_list,
            params['poly_degree'], params['feature_names'],
            smoother_window=smoother_window
        )

        xis = _run_ensemble(
            theta, xdot,
            lam=params['lam'],
            l2_norm=params['l2_norm'],
            n_models=params['n_ensemble'],
            n_candidates_to_drop=n_candidates_to_drop,
        )

        coeffs_estimated = np.median(xis, axis=-1)
        sparsity = int(np.count_nonzero(coeffs_estimated))

        if sparsity == sparsity_last_iteration:
            sparsity_equal_itcount += 1
        else:
            sparsity_equal_itcount = 0

        coef_mad = np.median(
            np.abs(xis - coeffs_estimated[:, :, np.newaxis]), axis=-1
        )
        rel_mad = 100.0 * (coef_mad / (np.abs(coeffs_estimated) + 1e-6))

        active_mask = np.abs(coeffs_estimated) > 1e-6
        if active_mask.any():
            max_mad = float(np.max(rel_mad[active_mask]))
        else:
            max_mad = float('inf')

        if sparsity_equal_itcount >= params['patience'] and max_mad <= params['conv_thresh_coef']:
            converged = True
        else:
            sparsity_last_iteration = sparsity

        # --- Record metrics ---
        data_count_history.append(data_count)
        l0_history.append(sparsity)
        if true_coeffs is not None:
            l2_history.append(float(np.linalg.norm(coeffs_estimated - true_coeffs, 'fro')))
        else:
            l2_history.append(float(np.linalg.norm(coef_mad, 'fro')))

        if converged:
            break

        # --- Random sampling ---
        candidate_points = sample_candidates(
            params['n_candidates'], iteration + 99999,
            params['n_dims'], params['state_bounds']
        )
        query_ic = candidate_points[np.random.randint(0, len(candidate_points))]

        sol = solve_ivp(
            params['ode'], params['t_span_query'], query_ic,
            t_eval=t_eval_query, args=params['ode_params'], **integrator_kws
        )

        if sol.status == 0 and sol.y.shape[1] == len(t_eval_query):
            x_query = sol.y.T
            noise_std = calculate_rms(x_query) * params['relative_noise_factor']
            x_train_list.append(x_query + np.random.normal(0, noise_std, x_query.shape))
            t_train_list.append(sol.t)
        else:
            print(f"  Warning: solver failed at iteration {iteration + 1}. Skipping.")

        points_ic.append(query_ic)

    status = "converged" if converged else f"reached max iterations ({params['max_iter']})"
    print(f"\nRandom SINDy finished — {status}.")

    return l0_history, l2_history, data_count_history


def run_stats(
        x_init: list, t_init: list,
        params: dict,
        cand_list: list,
        n_repetitions: int,
        run_fn
) -> dict:
    """
    Repeat an Active or Random SINDy run ``n_repetitions`` times for each
    value of ``n_candidates`` in ``cand_list`` and collect the results.

    Args:
        x_init (list[np.ndarray]): Seed trajectory arrays.
        t_init (list[np.ndarray]): Seed time vectors.
        params (dict): Base params dict (``n_candidates`` is overwritten —
            the original dict is not modified).
        cand_list (list[int]): Candidate pool sizes to sweep.
        n_repetitions (int): Independent repetitions per pool size.
        run_fn (callable): :func:`run_active_sindy` or :func:`run_random_sindy`.

    Returns:
        dict: Keyed by ``n_candidates``, each value is a dict with:
            ``n_iter`` (list[int]): iterations until convergence per rep.
            ``mad_histories`` (list[list]): per-iteration MAD matrices per rep.
    """
    results = {}

    for cand in cand_list:
        print(f"\nn_candidates = {cand}")
        results[cand] = {
            'n_iter':          [],
            'l0_histories':    [],
            'l2_histories':    [],
            'data_histories':  [],
        }
        p = {**params, 'n_candidates': cand}

        for rep in range(n_repetitions):
            if (rep + 1) % 10 == 0 or rep == 0:
                print(f"  Rep {rep + 1}/{n_repetitions}")
            l0_hist, l2_hist, data_hist = run_fn(x_init, t_init, p)
            results[cand]['n_iter'].append(len(l0_hist))
            results[cand]['l0_histories'].append(l0_hist)
            results[cand]['l2_histories'].append(l2_hist)
            results[cand]['data_histories'].append(data_hist)

    return results


def plot_convergence_statistics(
        stats_active: dict,
        stats_random: dict,
        cand_list: list,
        true_sparsity: Optional[int] = None,
        xlim: Optional[tuple] = None,
) -> None:
    """
    Three-panel figure plotted against cumulative training samples.

    Left   — Active SINDy L0 bubble chart: bubble area proportional to the
              fraction of repetitions that produced that sparsity at that
              sample count.
    Centre — Random Sampling L0 bubble chart (same encoding).
    Right  — L2 violin: Active (blue) and Random (orange) violins placed
              side by side at each sample count within ``xlim``.

    Uses the middle ``n_candidates`` value from ``cand_list``.

    Args:
        stats_active (dict): Output of :func:`run_stats` for active SINDy.
        stats_random (dict): Output of :func:`run_stats` for random SINDy.
        cand_list (list[int]): Candidate pool sizes swept.
        true_sparsity (int, optional): Dashed reference line on L0 panels.
        xlim (tuple, optional): (xmin, xmax) in training-sample units applied
            to all three panels.  ``None`` shows all data.
    """
    cand_mid = cand_list[len(cand_list) // 2]
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    def _in_range(data_counts):
        if xlim is None:
            return list(range(len(data_counts))), list(data_counts)
        idx = [i for i, dc in enumerate(data_counts) if xlim[0] <= dc <= xlim[1]]
        return idx, [data_counts[i] for i in idx]

    for panel, stats, label, color in [
        (0, stats_active, 'Active SINDy',    'C0'),
        (1, stats_random, 'Random Sampling', 'C1'),
    ]:
        ax = axs[panel]
        l0_hists   = stats[cand_mid]['l0_histories']
        data_hists = stats[cand_mid]['data_histories']
        n_reps     = len(l0_hists)
        min_len    = min(len(h) for h in data_hists)
        base_dc    = data_hists[0][:min_len]

        valid_idx, dc_plot = _in_range(base_dc)
        max_s = 600

        for src_i, dc in zip(valid_idx, dc_plot):
            vals = [l0_hists[r][src_i]
                    for r in range(n_reps) if src_i < len(l0_hists[r])]
            for l0_val, cnt in Counter(vals).items():
                ax.scatter(dc, l0_val,
                           s=cnt / n_reps * max_s,
                           color=color, alpha=0.7, edgecolors='none')

        ax.set_xlabel('# data points')
        ax.set_ylabel('l0 norm')
        ax.set_title(f'{label}\n(n_cand={cand_mid})')
        if xlim is not None:
            ax.set_xlim(xlim)

    ax = axs[2]

    def _l2_arrays(stats):
        l2_hists   = stats[cand_mid]['l2_histories']
        data_hists = stats[cand_mid]['data_histories']
        min_len    = min(len(h) for h in data_hists)
        base_dc    = data_hists[0][:min_len]
        valid_idx, dc_plot = _in_range(base_dc)
        arrays = [
            [l2_hists[r][i]
             for r in range(len(l2_hists)) if i < len(l2_hists[r])]
            for i in valid_idx
        ]
        return dc_plot, arrays

    dc_a, l2_a = _l2_arrays(stats_active)
    dc_r, l2_r = _l2_arrays(stats_random)

    step   = (dc_a[1] - dc_a[0]) if len(dc_a) > 1 else 11
    width  = step * 0.38
    offset = step * 0.20

    def _violin(positions, data_arrays, color):
        pos_v, dat_v, pos_s, dat_s = [], [], [], []
        for pos, vals in zip(positions, data_arrays):
            if len(vals) >= 2 and len(set(vals)) > 1:
                pos_v.append(pos)
                dat_v.append(vals)
            elif vals:
                pos_s.append(pos)
                dat_s.append(float(np.mean(vals)))
        if pos_v:
            parts = ax.violinplot(dat_v, positions=pos_v,
                                  widths=width, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_edgecolor(color)
                    parts[key].set_linewidth(1.2)
        if pos_s:
            ax.scatter(pos_s, dat_s, color=color, marker='D',
                       s=40, alpha=0.8, zorder=3)

    _violin([dc - offset for dc in dc_a], l2_a, 'C0')
    _violin([dc + offset for dc in dc_r], l2_r, 'C1')

    ax.legend(handles=[
        mpatches.Patch(color='C0', alpha=0.6,
                       label=f'Active SINDy (n_cand={cand_mid})'),
        mpatches.Patch(color='C1', alpha=0.6,
                       label=f'Random Sampling (n_cand={cand_mid})'),
    ], fontsize=8)
    ax.set_xlabel('# data points')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'Active vs Random\n(n_cand={cand_mid})')
    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()
