import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from IPython.display import display, Math, HTML
from typing import Optional


def init_plt_format():
    """
    Initialise plot formatting for matplotlib.
    """

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "xtick.minor.visible": True,
        "xtick.minor.visible": True,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 150,
    })


def plot_colored_line(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: Optional[np.ndarray] = None,
        cmap: str = 'viridis',
        linewidth: float = 2.0
) -> LineCollection:
    """
    Utility function for plotting colored lines, where the color of each line
    segment encodes a value from the data's third dimension.

    Args:
        ax (Axes): Plot `Axes` object.
        x (np.ndarray): x data.
        y (np.ndarray): y data.
        z (np.ndarray): z data. Used for color encoding.
        t (Optional[np.ndarray], optional): Time vector as an alternate third
            dimension data axis to plot, will overwrite z data for plotting.
            Defaults to None.
        cmap (str, optional): Matplotlib color map. Defaults to 'viridis'.
        linewidth (float, optional): Line width. Defaults to 2.0.

    Returns:
        LineCollection: LineCollection object of the collection of colored line
            segments.
    """

    # Create segments for color-coded line
    if t is None:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    else:
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection for color-coded line
    if t is None:
        lc = LineCollection(segments, cmap=cmap)
        lc.set_array(z)
    else:
        lc = Line3DCollection(segments, cmap=cmap)
        lc.set_array(t)
    lc.set_linewidth(linewidth)

    # Add invisible line plot to set axis limits
    if t is not None:
        ax.plot(x, y, z, alpha=0)
    else:
        ax.plot(x, y, alpha=0)

    # Add line collection
    ax.add_collection(lc)

    return lc


def display_matrix(arr: np.ndarray, precision: int = 4):
    """
    Displays a 2D NumPy array as a formatted LaTeX matrix in Jupyter.
    
    Args:
        arr (np.ndarray): The 2D matrix to display.
        precision (int): Number of decimal places for floating-point numbers.
    """
    if arr.ndim != 2:
        print("Error: Input is not a 2D array.")
        display(arr)
        return

    # Start the LaTeX string with a bmatrix (matrix with brackets)
    latex_str = r"\begin{bmatrix}"
    
    # Iterate over each row
    for row in arr:
        # Format each element in the row
        # 'g' formats as general number, trimming trailing zeros
        formatted_row = [f"{x:.{precision}g}" for x in row]
        
        # Join elements with '&' (LaTeX column separator)
        latex_str += " & ".join(formatted_row)
        
        # Add '\\' (LaTeX new line)
        latex_str += r" \\ "
        
    # Close the bmatrix environment
    latex_str += r"\end{bmatrix}"
    
    # Display the rendered LaTeX
    display(Math(latex_str))


def visualize_ode_2d(
        x: np.ndarray, t: np.ndarray, noisy_x: Optional[np.ndarray] = None,
        title: Optional[str] = None
):
    """
    Function to visualise a given 2D time-series dataset.

    Args:
        x (np.ndarray): x data matrix, must be shaped (m, 2).
        t (np.ndarray): Corresponding time vector, must be shaped (m,).
        noisy_x (Optional[np.ndarray], optional): Optional noisy data that can
            be plotted for comparison, must have matching shape as `x`.
            Defaults to None.
        title (Optional[str], optional): Title of the plot. Defaults to None.
    """
    
    fig = plt.figure(figsize=(14,6))
    gs: GridSpec = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])

    labels: list[str] = ['$x_1$', '$x_2$']

    # Plot x-t
    for i in range(2):
        ax = fig.add_subplot(gs[i,0])
        ax.plot(t, x[:,i])

        if noisy_x is not None:
            ax.plot(t, noisy_x[:,i], 'r.', markersize=0.2)

        ax.set_ylabel(labels[i])

    ax.set_xlabel('Time')

    # Plot x states
    ax = fig.add_subplot(gs[:,1])
    lc: LineCollection = plot_colored_line(
        ax,
        x[:, 0], x[:, 1],
        t)
    plt.colorbar(lc, ax=ax, label='Time (s)')

    if noisy_x is not None:
        ax.plot(
            noisy_x[:, 0], noisy_x[:, 1],
            'r.', markersize=0.2
        )
        
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()


def visualize_ode_3d(
        x: np.ndarray, t: np.ndarray, noisy_x: Optional[np.ndarray] = None,
        title: Optional[str] = None, labels: Optional[str] = None
):
    """
    Function to visualize a 3D time-series dataset.

    Args:
        x (np.ndarray): x data matrix, must be shaped (m, 3).
        t (np.ndarray): Corresponding time vector, must be shaped (m,).
        noisy_x (Optional[np.ndarray], optional): Optional noisy data that can
            be plotted for comparison, must have matching shape as `x`.
            Defaults to None.
        title (Optional[str], optional): Title of the plot. Defaults to None.
        labels (Optional[str], optional): List of string to be used for the
            variable names. Defaults to `['$x$', '$y$', '$z$']`.
    """
    
    fig = plt.figure(figsize=(14,7))
    gs: GridSpec = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])

    if labels is None:
        labels: list[str] = ['$x$', '$y$', '$z$']
    else:
        labels = labels.copy()
        for idx in range(len(labels)):
            labels[idx] = '$' + labels[idx] + '$'

    # Plot 2D plots
    for i in range(3):
        ax = fig.add_subplot(gs[i,0])
        ax.plot(t, x[:,i])

        if noisy_x is not None:
            ax.plot(t, noisy_x[:,i], 'r.', markersize=0.3)

        ax.set_ylabel(labels[i])

    ax.set_xlabel('Time')

    # Plot 3D plot
    ax = fig.add_subplot(gs[:,1], projection='3d')
    lc: LineCollection = plot_colored_line(
        ax,
        x[:,0], x[:,1], x[:,2],
        t
    )
    plt.colorbar(lc, ax=ax, label='Time')

    if noisy_x is not None:
        ax.plot(
            noisy_x[:, 0], 
            noisy_x[:, 1], 
            noisy_x[:, 2],
            'r.',
            markersize=0.3
        )
        
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()


def visualize_pde_1d(
        u: np.ndarray, t: np.ndarray, x: np.ndarray,
        title: Optional[str] = None,
        noisy_u: Optional[np.ndarray] = None,
        variable_names: Optional[list[str]] = None,
        x_label: str = '$x$',
        time_label: str = '$t$',
        animated: bool = False,
        frame_interval: Optional[int] = None,
        log_scale: bool = False,
        save_path: Optional[str] = None
) -> Optional[HTML]:
    """
    Visualise a 1D spatiotemporal PDE solution on a surface plot.

    Args:
        u (np.ndarray): PDE solution.
        t (np.ndarray): Vector of time grid points.
        x (np.ndarray): Vector of spatial grid points.
        title (Optional[str]): Title of the plot.
        noisy_u (Optional[np.ndarray]): Noisy PDE solution for comparison.
        variable_names (Optional[list[str]]): Names of the solution variables.
            Defaults to None.
        x_label (str): Label of the spatial variable. Defaults to '$x$'.
        time_label (str): Label of the time variable. Defaults to '$t$'.
        animated (bool): Controls whether to plot a static 2D contour plot or
            an animated time series plot. Defaults to False.
        frame_interval (int, optional): Interval between in data between each
            plot. Defaults to None which automatically sets the interval to
            create 60 frames.
        log_scale (bool): Sets the scale to log scale. Defaults to False.
        save_path (Optional[str]): If specified, will save the figure to the
            given file path. Defaults to None.

    Returns:
        HTML: Returns HTML video player object if `animated` is True.
    """

    udim: int = u.shape[-1]
    mdim: int = t.shape[0]

    if variable_names is None:
        variable_names = [f'$u_{idx+1:d}$' for idx in range(udim)]

    if log_scale is True and animated is False:
        variable_names = [f'Log of {name}' for name in variable_names]

    if not animated:

        # Set up figure
        fig, axs = plt.subplots(
            udim, 1 if noisy_u is None else 2,
            figsize=(6 if noisy_u is None else 12, 4*udim),
            squeeze=False
        )
        
        # Get mesh grid
        [X,T] = np.meshgrid(x, t, indexing='ij')

        # Set to log scale
        if log_scale:
            u = np.log(u)
            noisy_u = np.log(noisy_u) if noisy_u is not None else None

        for uIdx in range(udim):

            levels = MaxNLocator(100).tick_values(
                np.min(u[..., uIdx]), np.max(u[..., uIdx])
            )
            cmap = plt.get_cmap('viridis')
            norm = BoundaryNorm(levels, cmap.N, clip=True)
            cplt = axs[uIdx, 0].pcolormesh(
                X, T, u[:,:,uIdx],
                cmap=cmap, norm=norm
            )
            axs[uIdx, 0].set_xlabel(x_label)
            axs[uIdx, 0].set_ylabel(time_label)
            fig.colorbar(cplt, ax=axs[uIdx, 0], label=variable_names[uIdx])

            if noisy_u is not None:
                levels = MaxNLocator(100).tick_values(
                    np.min(noisy_u[..., uIdx]), np.max(noisy_u[..., uIdx])
                    )
                norm = BoundaryNorm(levels, cmap.N, clip=True)
                cplt = axs[uIdx, 1].pcolormesh(
                    X, T, noisy_u[:,:,uIdx],
                    cmap=cmap, norm=norm
                )
                axs[uIdx, 1].set_xlabel(x_label)
                axs[uIdx, 1].set_ylabel(time_label)
                fig.colorbar(
                    cplt, ax=axs[uIdx, 1],
                    label='Noisy ' + variable_names[uIdx]
                )

        if title is not None:
            fig.suptitle(title)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

    
    else:

        fig, axs = plt.subplots(
            udim, 1,
            figsize=(10, 1.5*udim),
            squeeze=False
        )

        if frame_interval is None:
            frame_interval: int = int(np.ceil(mdim/60))

        uMax = np.max(u, axis=(0, 1))
        uMin = np.min(u, axis=(0, 1))

        # Initialise plots
        for uIdx in range(udim):

            axs[uIdx, 0].plot(x, u[:, 0, uIdx])
            if noisy_u is not None:
                axs[uIdx, 0].plot(x, noisy_u[:, 0, uIdx], 'r.', markersize=0.2)
            axs[uIdx, 0].set_ylabel(variable_names[uIdx])
            axs[uIdx, 0].set_ylim([uMin[uIdx], uMax[uIdx]])

        axs[-1, 0].set_xlabel(x_label)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.close()

        def update(frame):

            for uIdx in range(udim):

                axs[uIdx, 0].clear()
                axs[uIdx, 0].plot(x, u[:, frame, uIdx])
                if noisy_u is not None:
                    axs[uIdx, 0].plot(x, noisy_u[:, frame, uIdx], 'r.', markersize=0.2)
                axs[uIdx, 0].set_ylabel(variable_names[uIdx])
                axs[uIdx, 0].set_ylim([uMin[uIdx], uMax[uIdx]])
                if log_scale:
                    axs[uIdx, 0].set_yscale('log')

            axs[-1, 0].set_xlabel(x_label)

            if title is not None:
                _title: str = title + f" | {time_label} = {t[frame]:.3f}"
            else:
                _title: str = f"{time_label} = {t[frame]:.3f}"
            fig.suptitle(_title) 

        anim = FuncAnimation(
            fig, update,
            frames=range(0, t.shape[0], frame_interval)
        )

        if save_path is not None:
            anim.save(save_path)

        return HTML(anim.to_jshtml())


def visualize_pde_2d(
        u: np.ndarray, t: np.ndarray, x_grids: list[np.ndarray],
        title: Optional[str] = None,
        variable_names: Optional[list[str]] = None,
        xlabels: Optional[list[str]] = None,
        ylabels: Optional[list[str]] = None,
        noisy_u: Optional[str] = None,
        contour_lvls: int = 50,
        equal_axis: bool = False,
        frame_interval: Optional[int] = None,
        polar_projection: bool = False,
        save_path: Optional[str] = None
) -> HTML:
    """
    Visualizes a 2D spatiotemporal PDE solution using Jupyter's built in
    player.

    Args:
        u (np.ndarray): PDE solution.
        t (np.ndarray): Vector of time grid points.
        x_grids (list[np.ndarray]): List of vectors of x grid points.
        title (Optional[str], optional): Title of figure. Defaults to None.
        variable_names (Optional[list[str]], optional): Names of individual
            solution variable. Defaults to None.
        xlabels (Optional[list[str]]): x labels of the plot. Defaults to None.
        ylabels (Optional[list[str]]): y labels of the plot. Defaults to None.
        noisy_u (Optional[str], optional): Noisy solution to plot beside the
            clean one for comnparison. Defaults to None.
        contour_lvls (int, optional): Number of contours to show. Defaults to
            50.
        equal_axis (bool, optional): Whether or not to equalize axis. Defaults
            to False.
        frame_interval (int, optional): Interval between in data between each
            plot. Defaults to None which automatically sets the interval to
            create 60 frames.
        polar_projection (bool, optional): Whether to plot using a polar
            projection with the radius starting at 0.
        save_path: (str, optional): If specified, will save the plot to the
            given file path. Defaults to None.

    Returns:
        HTML: HTML video player object. Must be returned in the last line of a
            cell or passed to `display()`.
    """
    
    udim: int = u.shape[-1]
    mdim: int = t.shape[0]

    # If frame interval is not provided, will try to compute an interval that
    # results in around 60 frames.
    if frame_interval is None:
        frame_interval: int = int(np.ceil(mdim/60))

    subplot_kw = {'projection': 'polar'} if polar_projection else {}

    # Set up figure and axes
    if noisy_u is None:
        fig, axs = plt.subplots(
            udim, 1, figsize=(6,4*udim), subplot_kw=subplot_kw)
    else:
        fig, axs = plt.subplots(
            udim, 2, figsize=(12,4*udim), subplot_kw=subplot_kw)

    if not isinstance(axs, np.ndarray):
        axs: np.ndarray[Axes] = np.array([[axs]])
    axs = axs.reshape((udim, (noisy_u is not None) + 1))

    cmap = plt.get_cmap('viridis')

    x_mesh: np.ndarray = np.meshgrid(*x_grids, indexing='ij')
    norms: list[BoundaryNorm] = list()
    
    # Initialise plots
    for uIdx in range(udim):

        levels = MaxNLocator(contour_lvls).tick_values(
            np.min(u[..., uIdx]),
            np.max(u[..., uIdx])
        )
        norms.append(BoundaryNorm(levels, cmap.N, clip=True))

        cplt = axs[uIdx, 0].pcolormesh(
            *x_mesh, u[..., 0, uIdx],
            cmap=cmap, norm=norms[-1]
        )
        
        if variable_names is not None:
            axs[uIdx, 0].set_title(variable_names[uIdx])

        if xlabels is not None:
            axs[uIdx, 0].set_xlabel(xlabels[uIdx])

        if ylabels is not None:
            axs[uIdx, 0].set_ylabel(ylabels[uIdx])

        if noisy_u is not None:
            axs[uIdx, 1].contourf(
                *x_mesh, noisy_u[..., 0, uIdx],
                cmap=cmap, norm=norms[-1]
            )
            if variable_names is not None:
                axs[uIdx, 1].set_title(variable_names[uIdx])

            if xlabels is not None:
                axs[uIdx, 1].set_xlabel(xlabels[uIdx])

            if ylabels is not None:
                axs[uIdx, 1].set_ylabel(ylabels[uIdx])

        fig.colorbar(cplt, ax=axs[uIdx,0])

    if title is not None:
        _title: str = title + f" | t = {t[0]:.2f}"
    else:
        _title: str = f"t = {t[0]:.2f}"
    fig.suptitle(_title)

    if equal_axis:
        for ax in axs.flatten():
            ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.close()
    
    def update(frame):

        for uIdx in range(udim):
            axs[uIdx, 0].clear()
            axs[uIdx, 0].pcolormesh(
                *x_mesh, u[..., frame, uIdx],
                cmap=cmap, norm=norms[uIdx]
            )
            
            if polar_projection:
                axs[uIdx, 0].set_ylim([0, x_grids[1][-1]])

            if variable_names is not None:
                axs[uIdx, 0].set_title(variable_names[uIdx])

            if xlabels is not None:
                axs[uIdx, 0].set_xlabel(xlabels[uIdx])

            if ylabels is not None:
                axs[uIdx, 0].set_ylabel(ylabels[uIdx])

            if noisy_u is not None:

                axs[uIdx, 1].clear()
                axs[uIdx, 1].pcolormesh(
                    *x_mesh, noisy_u[..., frame, uIdx],
                    cmap=cmap, norm=norms[uIdx]
                )

                if polar_projection:
                    axs[uIdx, 1].set_ylim([0, x_grids[1][-1]])

                if variable_names is not None:
                    axs[uIdx, 1].set_title(variable_names[uIdx])

                if xlabels is not None:
                    axs[uIdx, 1].set_xlabel(xlabels[uIdx])

                if ylabels is not None:
                    axs[uIdx, 1].set_ylabel(ylabels[uIdx])

        if title is not None:
            _title: str = title + f" | t = {t[frame]:.3f}"
        else:
            _title: str = f"t = {t[frame]:.3f}"
        fig.suptitle(_title)

    anim = FuncAnimation(
        fig, update,
        frames=range(0, t.shape[0], frame_interval)
    )

    if save_path is not None:
        anim.save(save_path)

    return HTML(anim.to_jshtml())


def visualize_mpc(
        x: np.ndarray, r: np.ndarray, t: np.ndarray,
        cost: np.ndarray, x_hat_hist: list[np.ndarray], u_hist: list[np.ndarray],
        state_names: Optional[list[str]] = None,
        control_names: Optional[list[str]] = None,
        title: Optional[str] = None,
        frame_interval: int = 20,
        save_path: Optional[str] = None
) -> HTML:
    """
    Plots an animated running plot using data from a trajectory.

    Args:
        x (np.ndarray): The state trajectory history.
        r (np.ndarray): The reference trajectory that was tracked.
        t (np.ndarray): Time vector.
        cost (np.ndarray): Running cost of the MPC at each step.
        x_hat_hist (list[np.ndarray]): History of predicted states at each MPC
            prediction step.
        u_hist (list[np.ndarray]): History of optimal control sequences at each
            MPC control step.
        state_names (Optional[list[str]], optional): Names of the states.
            Defaults to None.
        control_names (Optional[list[str]], optional): Names of the control.
            Defaults to None.
        title (Optional[str], optional): Title of the plot. Defaults to None.
        frame_interval (int, optional): Fram interval between plotting steps.
            Defaults to 20.
        save_path (Optional[str], optional): Path to save the animation.
            Defaults to None.

    Returns:
        HTML: Display data of the animated plot for Jupyter notebook.
    """
    
    # Define dimensions
    m: int = t.shape[0]
    n: int = x.shape[1]
    q: int = u_hist[0].shape[1]

    # Get MPC horizons
    mp: int = x_hat_hist[0].shape[0]
    mc: int = u_hist[0].shape[0]

    # Initialise figure
    total_plots: int = 1+n+q
    fig, axs = plt.subplots(
        total_plots, 1,
        sharex=True, figsize=(12, 2*total_plots)
    )

    # Set state names if not defined
    if state_names is None:
        state_names = [f'x_{idx+1:d}' for idx in range(n)]

    # Set control names if not defined
    if control_names is None:
        control_names = [f'u_{idx+1:d}' for idx in range(q)]

    def update(frame):

        # Plot cost
        axs[0].clear()
        axs[0].plot(t[:frame], cost[:frame], 'k-')
        axs[0].set_ylabel('MPC Cost')
        axs[0].set_xlim([t[0], t[-1]])
        axs[0].set_ylim([-max(cost)*0.2, max(cost)*1.2])

        # Plot states
        for idx in range(n):
            axs[1+idx].clear()
            axs[1+idx].plot(t[:frame], x[:frame, idx], label='Trajectory')
            axs[1+idx].plot(t, r[:,idx], ':', label='Reference')
            axs[1+idx].plot(t, x[:, idx], alpha=0)
            horizon_end: int = min(m, frame+mp)
            axs[1+idx].plot(
                t[frame:horizon_end],
                x_hat_hist[frame][:min(m-frame, mp), idx],
                '--', label='Predicted'
            )
            axs[1+idx].set_ylabel(f'${state_names[idx]}$')
            axs[1+idx].set_xlim([t[0], t[-1]])
            axs[1+idx].legend()

        # Plot control
        for idx in range(q):
            axs[1+n+idx].clear()
            axs[1+n+idx].plot(
                t[:frame], [u[0, idx] for u in u_hist[:frame]],
                label='Applied control'
            )
            horizon_end: int = min(m, frame+mc)
            axs[1+n+idx].plot(
                t[frame:horizon_end],
                u_hist[frame][:min(m-frame, mc), idx],
                '--', label='Planned control'
            )
            axs[1+n+idx].plot(t[:-1], [u[0, idx] for u in u_hist], alpha=0)
            axs[1+n+idx].set_ylabel(f'${control_names[idx]}$')
            axs[1+n+idx].set_xlim([t[0], t[-1]])
            axs[1+n+idx].legend()

        if title is not None:
            fig.suptitle(title)
    
    anim = FuncAnimation(
        fig, update,
        frames=range(0, t.shape[0], frame_interval)
    )
    plt.close(fig)

    if save_path is not None:
        anim.save(save_path)

    return HTML(anim.to_jshtml())