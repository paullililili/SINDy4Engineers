import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from typing import Optional, Callable

from ..library import poly_lib
from ..optimizer import stridge, least_squares, stlsq
from ..metrics import precision, rel_fro_err
from ..utils import print_latex_eqn

from pysindy import FiniteDifference

class WSINDy_ODE():

    def __init__(
            self,
            poly_order: int = 2, include_bias: bool = True,
            lam: float = 1e-2, gamma: float = 0.0,
            useAG: bool = False, useGLS: bool = False,
            L: Optional[int] = None, p: int = 16, s: float = 0.5,
            K: int = 250, r_whm: Optional[float] = None
    ):
        """
        This class implements the weak form SINDy method from (Messenger and
        Bortz, 2021) for ODEs.

        Args:
            poly_order (int, optional): Power of the polynomial order library
                to use for the candidate function library. Defaults to 2.
            include_bias (bool, optional): Whether to include a bias term in
                the candidate function library. Defaults to True.
            lam (float, optional): Lambda threshold for STLSQ. Defaults to 
                1e-2.
            gamma (float, optional): L2 ridge regression parameter. Defaults to
                0.0.
            useAG (bool, optional): Whether to use adaptive grid or uniform
                grid for test functions. Defaults to False.
            useGLS (bool, optional): Whether to use generalized least squares
                regression as opposed to ordinary least squares. Defaults to
                False.
            L (Optional[int], optional): Uniform grid parameter to prescribe
                total number of discrete support points for the test function.
                Defaults to None.
            p (int, optional): Uniform grid parameter. Polynomial power of the
                test function. Defaults to 16.
            s (float, optional): Uniform grid parameter. Value of the test
                function at the intersection with neighbouring test functions.
                Defaults to 0.5.
            K (int, optional): Adaptive grid parameter to describe total number
                of test functions desired. Defaults to 250.
            r_whm (Optional[float], optional): Adaptive grid width at half max
                parameter. Defaults to None.
        """
        
        # Store variables relating to candidate function library
        self.poly_order: int = poly_order
        self.include_bias: bool = include_bias

        # Store variables relating to the optimizer
        self.lam: float = lam
        self.gamma: float = gamma

        # Store variables relating to the formulation of WSINDy
        self.useAG: bool = useAG
        self.useGLS: bool = useGLS

        # Store parameters of the uniform grid
        self.L: int = L
        self.p: int = p
        self.s: float = s

        # Store parameters of relating to adaptive grid
        self.K: int = K
        self.r_whm: float = r_whm

        # Fitted flag
        self.fitted: bool = False


    def fit(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Function to discover the sparse coefficient matrix.

        Args:
            x (np.ndarray): Data matrix x.
            t (np.ndarray): Time vector t. Assumes uniform time sampling.

        Returns:
            np.ndarray: The discovered coefficient matrix.
        """

        # Get total number of data points
        self.M: int = x.shape[0]

        # Get total number of states
        self.N: int = x.shape[1]
        
        # Set default span of test functions if none provided
        if self.L is None:
            self.L = int(np.floor(self.M/25))

        # Set default half height width in points if none provided
        if self.r_whm is None:
            self.r_whm = int(np.floor(self.M/100))

        # Obtain candidate function library
        self.theta, self.lib_names = poly_lib(x, self.poly_order, self.include_bias)

        # Initialise sparse weight matrix
        self.coef: np.ndarray = np.empty((self.theta.shape[1], self.N))

        # Loop through each target state
        for state_idx in range(self.N):

            # Generate test functions
            if self.useAG:
                V, Vp = self._adaptive_grid(
                    t, x[:, state_idx],
                    self.K, r_whm=self.r_whm
                )
            else:
                # For uniform grid, grid only needs to be generated once
                if state_idx == 0:
                    V, Vp = self._uniform_grid(t, self.L, self.s, self.p)

            # Compute Gram matrix and RHS vector
            self.mat_G: np.ndarray = V @ self.theta
            self.vec_b: np.ndarray = (-Vp @ x[:, state_idx]).reshape(-1,1)

            if self.useGLS:
                # Compute covariance matrix and its Cholesky decomposition
                self.mat_covar: np.ndarray = Vp @ Vp.transpose() \
                    + np.eye(Vp.shape[0])*1e-12
                self.mat_C: np.ndarray = np.linalg.cholesky(self.mat_covar)

                # Apply Cholesky whitening transformation to the GLS problem
                self.mat_G: np.ndarray = least_squares(self.mat_G, self.mat_C)
                self.vec_b: np.ndarray = least_squares(self.vec_b, self.mat_C)

            # Apply STLSQ on the transformed OLS problem
            self.coef[:, state_idx] = stridge(
                self.mat_G, self.vec_b,
                lam=self.lam, l2_norm=self.gamma
            ).reshape(-1)

        self.fitted = True

        return self.coef


    def _adaptive_grid(
            self,
            t: np.ndarray, x: np.ndarray,
            K: int, r_whm: int,
            Lw: int = 17, pw: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the integration matrices V and V' using adaptive grid
        method.

        Args:
            t (np.ndarray): Time vector.
            x (np.ndarray): Vector of a single state x data.
            K (int): Number of test functions.
            r_whm (int): Width at half max parameter.
            Lw (int, optional): Number of support points for the test function
                used in weak differentiation. Defaults to 17.
            pw (int, optional): Degree of test function polynomial used in weak
                differentiation. Defaults to 2.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of V and V' matrices.
        """

        # Obtain M
        M: int = x.shape[0]
        
        # Step 1: Get weak derivative
        weak_der: np.ndarray = self._weak_deriv_approx(t, x, Lw, pw)

        # Step 2: Find centres of test functions
        ck = self._get_centres(weak_der, K)
        
        # Step 3: Construct test functions
        p, half_interval = self._get_test_func_param(t, r_whm)

        # Initialise V and V' matrices
        V = np.empty((K, M))
        Vp = np.empty((K, M))

        # Find mean dt for trapezoidal integration (1/2 factor on boundary 
        # stencil points are ignored due to assumption of compactness of test
        # function)
        dt: float = np.mean(np.diff(t))

        # Get test function matrices (bounds of test functions must still be
        # adjusted to maintain its compactness assumption)
        for idx, c in enumerate(ck):
            a: float = max(t[c] - half_interval, t[0])
            b: float = min(t[c] + half_interval, t[-1])
            phi, phip = self._unimodal_poly(t, a, b, p, p)
            V[idx,:] = phi * dt
            Vp[idx,:] = phip * dt

        return V, Vp


    def _weak_deriv_approx(
            self,
            t: np.ndarray, x: np.ndarray,
            Lw: int = 17, pw: int = 2
    ) -> np.ndarray:
        """
        Compute the weak derivative of the given data.

        Args:
            t (np.ndarray): Time vector t.
            x (np.ndarray): Vector of a single x state.
            Lw (int, optional): Number of support points for the test function
                used in the weak derivative. Defaults to 17.
            pw (int, optional): Polynomial power of the test function used in
                the weak derivative. Defaults to 2.

        Returns:
            np.ndarray: The weak derivative of the given x state.
        """
        
        # Get total number of grid points
        M: int = t.shape[0]

        # Get dimension of weak derivative
        W: int = M - Lw + 1

        # Find mean dt for trapezoidal integration (1/2 factor on boundary stencil
        # points are ignored due to assumption of compactness of test function)
        dt: float = np.mean(np.diff(t))

        # Get phi' matrix
        phip: np.ndarray = np.empty((W, M))
        for idx in range(W):
            a: float = t[idx]
            b: float = t[idx+Lw-1]
            phip[idx,:] = self._unimodal_poly(
                t,
                a, b,
                pw, pw
            )[1] * dt

        # Return weak derivative approximation
        weak_deriv: np.ndarray = -phip @ x

        # Pad matrix with zeroes
        offset: int = int(np.floor((M-W)/2))
        weak_deriv_padded: np.ndarray = np.zeros((M,))
        weak_deriv_padded[offset:W+offset] = weak_deriv

        return weak_deriv_padded


    def _get_centres(self, weak_der: np.ndarray, K: int) -> list[int]:
        """
        Obtain the placements of test functions by finding the indices that the
        test function should centre around.

        Args:
            weak_der (np.ndarray): Weak derivative of a target x state.
            K (int): Number of test functions to place.

        Returns:
            list[int]: Returns a list of centre indices.
        """
        
        # Find unnormalised psi
        psi: np.ndarray = np.cumsum(np.abs(weak_der))

        # Normalise psi
        psi /= psi[-1]

        # Compute U
        U: np.ndarray = np.linspace(0, 1, K+2)

        # Find c_k vector
        ck: list[int] = list()
        for k in range(K):
            ck.append(int(np.argmin(np.abs(psi - U[k+1]))))
            psi[ck[-1]] = np.inf

        return ck


    def _get_test_func_param(
            self,
            t: np.ndarray, r_whm: int
    ) -> tuple[float, float]:
        """
        Obtain the width of the test function and polynomial degree to be used
        by the test function.

        Args:
            t (np.ndarray): Time vector.
            r_whm (int): Width at half max parameter.

        Returns:
            tuple[float, float]: A tuple of the polynomial degree followed by
            half the width of the test function.
        """

        dt: float = np.mean(np.diff(t))

        # Intermediate variable
        temp: float = 16*np.log2(10)

        # Convert r_whm from index to domain width
        r_whm_x: float = r_whm * dt

        def func2solve(x: float) -> float:
            """
            Function to be solved in step 3.

            Args:
                x (np.ndarray): Vector of [p, half_interval]

            Returns:
                np.ndarray: Vector of outputs
            """

            return (2*x - dt)*dt - x**2 * (1 - (r_whm_x/x)**2)**temp

        # Solve for degree p and half interval of test function
        half_interval = brentq(func2solve, r_whm_x, r_whm_x*np.sqrt(temp)+dt)
        p: float = -(np.log2(1 - (r_whm_x/half_interval)**2))**-1

        # Set limit on the lowest interval possible (5 grid points)
        return np.ceil(p), max(half_interval, dt*5)


    def _uniform_grid(
            self,
            t: np.ndarray,
            L: int, s: float, p: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the integration matrices V and V' using uniform grid method.

        Args:
            t (np.ndarray): Time vector t.
            L (int): Number of discrete points supporting the test function.
            s (float): Value of phi at point of intersection with neighboring
                test functions.
            p (int): Degree of polynomial to be used for the test function.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of V and V'.
        """

        # Determine the number of time discretization points
        M = len(t)

        # Determine the number of overlapping points between neighboring test
        # functions
        overlap: float = int(np.floor(L * (1 - np.sqrt(1-s**(1/p)))))

        # Create grid
        interval_list: list[list[int]] = list()
        idx_a: int = 0
        idx_b: int = L-1
        interval_list.append([idx_a,idx_b])

        while idx_b-overlap+L <= M-1:
            idx_a = idx_b - overlap + 1
            idx_b = idx_a + L - 1
            interval_list.append([idx_a,idx_b])
        grid: np.ndarray = np.array(interval_list)

        # Determine the number of test functions
        K: int = len(interval_list)

        # Initialise V and V' matrices
        V = np.empty((K, M))
        Vp = np.empty((K, M))

        # Find mean dt for trapezoidal integration (1/2 factor on boundary stencil
        # points are ignored due to assumption of compactness of test function)
        dt: float = np.mean(np.diff(t))

        # Compute V and V' matrices
        for k in range(K):
            a: float = t[grid[k,0]]
            b: float = t[grid[k,1]]
            phi, phip = self._unimodal_poly(t, a, b, p, p)
            V[k,:] = phi*dt
            Vp[k,:] = phip*dt

        # Parameter \rho that estimates the ratio between standard deviations
        # of the errors left and right side of the weak formulation.
        self.rho: float = np.linalg.norm(Vp) / np.linalg.norm(V)

        return V, Vp
    

    def _unimodal_poly(
            self,
            t: np.ndarray,
            a: float, b: float,
            p: float, q: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Return the piecewise unimodal polynomial function \phi(t).

        Args:
            t (np.ndarray): Input vector t.
            a (float): Left bound of the test function.
            b (float): Right bound of the test function.
            p (float): Polynomial power 1.
            q (float): Polynomial power 2. Will skew the test function if
                different from polynomial power 1.

        Returns:
            tuple[np.ndarray, np.ndarray]: Returns the test function itself and
                its first order derivative.
        """
        
        # Normalization coefficient to set max of phi = 1
        coef_C: float = 1/(p**p * q**q) * ((p+q)/(b-a))**(p+q)

        # Calculate phi and phi'
        phi: np.ndarray = coef_C * (t-a)**p * (b-t)**q
        phip: np.ndarray = coef_C * (t-a)**(p-1) * (b-t)**(q-1) * \
            (p*(b-t) - q*(t-a))
        
        # Set phi and phi' to zero outside of interval [a,b]
        phi[t < a] = 0.0
        phi[t > b] = 0.0
        phip[t < a] = 0.0
        phip[t > b] = 0.0

        return phi, phip
    

    def print(
            self,
            state_names: Optional[list[str]] = None, precision: int = 3
    ) -> str:
        """
        Return the test function equation in formatted Latex string.

        Args:
            state_names (Optional[list[str]], optional): The name strings of
                the states. Defaults to None.
            precision (int, optional): Precision of printed values. Defaults to
                3.

        Returns:
            str: Formatted Latex string.
        """

        assert self.fitted, 'No data fitted to yet!'

        if state_names is None:
            state_names = [f"x_{idx+1:d}" for idx in range(self.N)]
        
        return print_latex_eqn(
            self.coef,
            state_names, self.lib_names,
            precision
        )
    

def benchmark_wsindy_ode(
        xs: list[np.ndarray], t: np.ndarray, noise_ratios: np.ndarray,
        poly_order: int, include_bias: bool, lam: float,
        true_xi: np.ndarray
):
    """
    Helper function to help benchmark various WSINDy methods against SINDy
    (which uses 2nd and 4th order finite difference methods).

    Args:
        xs (list[np.ndarray]): List of noisy x data.
        t (np.ndarray): Time vector t.
        noise_ratios (np.ndarray): List of corresponding noise ratios.
        poly_order (int): Highest order of polynomial term in the candidate
            function library.
        include_bias (bool): Whether or not to include a bias term in the
            library.
        lam (float): Lambda threshold for sequential thresholding.
        true_xi (np.ndarray): True coefficient matrix.
    """
    
    # Define finite difference methods to be used
    diff_func_1: Callable = FiniteDifference(2)._differentiate
    diff_func_2: Callable = FiniteDifference(4)._differentiate
    
    # Initialise list of sparse coefficients matrix from different methods
    sindy_1_xis: list[np.ndarray] = list()
    sindy_2_xis: list[np.ndarray] = list()
    wsindy_uniform_ols_xis: list[np.ndarray] = list()
    wsindy_ag_ols_xis: list[np.ndarray] = list()

    # Initialise list of coefficient error metrics
    sindy_1_xi_errors: list[np.ndarray] = list()
    sindy_2_xi_errors: list[np.ndarray] = list()
    wsindy_uniform_ols_xi_errors: list[np.ndarray] = list()
    wsindy_ag_ols_xi_errors: list[np.ndarray] = list()

    # Initialise list of precision metrics
    sindy_1_precs: list[np.ndarray] = list()
    sindy_2_precs: list[np.ndarray] = list()
    wsindy_uniform_ols_precs: list[np.ndarray] = list()
    wsindy_ag_ols_precs: list[np.ndarray] = list()
    
    # Loop through all x data
    for x in xs:
        
        # Fit data using SINDy (2nd order finite difference)
        xi, xi_err, prec = fit_sindy(
            x=x, t=t,
            diff_func=diff_func_1,
            lam=lam, poly_order=poly_order, include_bias=include_bias,
            true_xi=true_xi)
        
        sindy_1_xis.append(xi)
        sindy_1_xi_errors.append(xi_err)
        sindy_1_precs.append(prec)

        # Fit data using SINDy (4th order finite difference)
        xi, xi_err, prec = fit_sindy(
            x=x, t=t,
            diff_func=diff_func_2,
            lam=lam, poly_order=poly_order, include_bias=include_bias,
            true_xi=true_xi)
        
        sindy_2_xis.append(xi.copy())
        sindy_2_xi_errors.append(xi_err.copy())
        sindy_2_precs.append(prec.copy())

        # Fit data using WSINDy using uniform grid with OLS
        xi, xi_err, prec = fit_wsindy(
            x=x, t=t,
            lam=lam, poly_order=poly_order, include_bias=include_bias,
            useAG=False, useGLS=False,
            true_w=true_xi)
        
        wsindy_uniform_ols_xis.append(xi.copy())
        wsindy_uniform_ols_xi_errors.append(xi_err.copy())
        wsindy_uniform_ols_precs.append(prec.copy())
        
        # Fit data using WSINDy using adaptive grid with OLS
        xi, xi_err, prec = fit_wsindy(
            x=x, t=t,
            lam=lam, poly_order=poly_order, include_bias=include_bias,
            useAG=True, useGLS=False,
            true_w=true_xi)
        
        wsindy_ag_ols_xis.append(xi.copy())
        wsindy_ag_ols_xi_errors.append(xi_err.copy())
        wsindy_ag_ols_precs.append(prec.copy())

    # Plot coefficient errors
    _, axs = plt.subplots(2,1,figsize=(6,8))
    axs[0].plot(noise_ratios, sindy_1_xi_errors, '-', label='SINDy 2nd Order FD')
    axs[0].plot(noise_ratios, sindy_2_xi_errors, '-', label='SINDy 4th Order FD')
    axs[0].plot(noise_ratios, wsindy_uniform_ols_xi_errors, '--', label='WSINDy Uniform OLS')
    axs[0].plot(noise_ratios, wsindy_ag_ols_xi_errors, '-.', label='WSINDy Adaptive OLS')
    axs[0].set_xlabel(r'$\sigma_{NR}$')
    axs[0].set_ylabel('$E_2$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend()

    # Plot precision metric
    axs[1].plot(noise_ratios, sindy_1_precs, '-', label='SINDy 2nd Order FD')
    axs[1].plot(noise_ratios, sindy_2_precs, '-', label='SINDy 4th Order FD')
    axs[1].plot(noise_ratios, wsindy_uniform_ols_precs, '--', label='WSINDy Uniform OLS')
    axs[1].plot(noise_ratios, wsindy_ag_ols_precs, '-.', label='WSINDy Adaptive OLS')
    axs[1].set_xlabel(r'$\sigma_{NR}$')
    axs[1].set_ylabel('Precision')
    axs[1].set_xscale('log')
    axs[1].legend()


def fit_sindy(
        x: np.ndarray, t: np.ndarray,
        diff_func: Callable,
        lam: float, poly_order: int, include_bias: bool,
        true_xi: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float, float]:
    """
    Helper SINDy fitter function which also computes the coefficient errors and
    the precision metric.

    Args:
        x (np.ndarray): State matrix x.
        t (np.ndarray): Time vector t.
        diff_func (Callable): Callable numerical differentiation function.
        lam (float): Lambda parameter for sequential hard thresholding.
        poly_order (int): Highest polynomial order to be used by the polynomial
            candidate function library.
        include_bias (bool): Whether or not to include a bias term in the
            library.
        true_xi (Optional[np.ndarray], optional): The true coefficient matrix.
            Defaults to None.

    Returns:
        tuple[np.ndarray, float, float]: A tuple of the coefficient matrix,
            error in coefficient matrix and the precision metric.
    """
    
    x_dot: np.ndarray = diff_func(x, t)
    
    theta = poly_lib(x, poly_order, include_bias)[0]
    xi: np.ndarray = stlsq(theta, x_dot, lam)

    if true_xi is not None:
        coef_err: float = rel_fro_err(xi, true_xi)
        prec: float = precision(xi, true_xi)
    else:
        coef_err = np.nan
        prec = np.nan
    
    return xi, coef_err, prec


def fit_wsindy(
        x: np.ndarray, t: np.ndarray,
        lam: float, poly_order: int, include_bias: bool,
        useAG: bool, useGLS: bool,
        true_w: np.ndarray,
        wsindy_kwargs: dict = dict()
) -> tuple[np.ndarray, float, float]:
    """
    Helper WSINDy fitter function that computes the coefficient errors and
    the precision metric.

    Args:
        x (np.ndarray): x state matrix.
        t (np.ndarray): Time vector.
        lam (float): Lambda threshold for STLSQ.
        poly_order (int): Highest order of polynomial to be used for candidate
            functions.
        include_bias (bool): Whether or not to include a bias term in the
            candidate function library.
        useAG (bool): Whether to use adaptive grid.
        useGLS (bool): Whether to use generalized least squares method.
        true_w (np.ndarray): The true coefficient matrix to evaluate against.
        wsindy_kwargs (dict, optional): Keyword arguments for `WSINDy_ODE`.
            Defaults to dict().

    Returns:
        tuple[np.ndarray, float, float]: Returns the coefficient matrix,
            coefficient error and the precision metric.
    """
    
    wsindy: WSINDy_ODE = WSINDy_ODE(
        poly_order, include_bias, lam, 0.0,
        useAG, useGLS,
        **wsindy_kwargs
    )
    
    w: np.ndarray = wsindy.fit(x, t)
    
    coef_err: float = rel_fro_err(w, true_w)
    prec: float = precision(w, true_w)
    
    return w, coef_err, prec


def wsindy_param_plot(
        xs: list[np.ndarray], t: np.ndarray,
        s_vars: np.ndarray, rho_vars: np.ndarray,
        lam: float, poly_order: int, include_bias: bool,
        L: int,
        true_w: np.ndarray
):
    
    # Pre-calculate values needed for p calculation
    dt: float = np.mean(np.diff(t))
    interval: float = L*dt

    # Check that rho is within valid bounds
    rho_min: float = np.sqrt((5+2*np.sqrt(6))/interval**2)
    rho_vars[rho_vars<rho_min] = rho_min
    
    # Initialise gridded data
    [rho_VARS, s_VARS] = np.meshgrid(rho_vars, s_vars)
    coef_errs: np.ndarray = np.empty((len(xs), s_VARS.shape[0], s_VARS.shape[1]))

    # Loop through given s and rho variables
    for x_idx, x in enumerate(xs):
        for s_idx, s in enumerate(s_vars):
            for rho_idx, rho in enumerate(rho_vars):
                # Calculate p
                p: int = int(np.floor(1/8 * (
                    (interval**2 * rho**2 - 1)
                    + np.sqrt(
                        (interval**2 * rho**2 - 1)**2
                            - 8 * interval**2 * rho**2
                    )
                )))
                # Fit WSINDy
                _, xi_err, _ = fit_wsindy(
                    x=x, t=t,
                    lam=lam, poly_order=poly_order, include_bias=include_bias,
                    useAG=False, useGLS=True,
                    true_w=true_w,
                    wsindy_kwargs={'L':L, 'p':p, 's':s}
                )
                coef_errs[x_idx, s_idx, rho_idx] = xi_err

    # Find mean and standard deviation coefficient error
    mean_coef_errs: np.ndarray = np.mean(coef_errs, axis=0)
    std_coef_errs: np.ndarray = np.std(coef_errs, axis=0)
    
    # Plotting mean
    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        rho_VARS, s_VARS,
        np.log10(mean_coef_errs),
        cmap='viridis',
        edgecolor='white',
        linewidth=0.5
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$log_{10}$ of Mean Coefficient Error')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$s$')
    ax.grid(False, which='both')

    # Plotting standard deviation
    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        rho_VARS, s_VARS,
        np.log10(std_coef_errs),
        cmap='viridis',
        edgecolor='white',
        linewidth=0.5
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$log_{10}$ of Standard Deviation of Coefficient Error')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$s$')
    ax.grid(False, which='both')