import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.optimize import brentq

from typing import Union, Optional, Callable
from math import comb, perm
from itertools import product

from ..library import poly_lib
from ..optimizer import stlsq
from ..utils import bi_piecewise_regression, print_latex_eqn
from ..metrics import precision, rel_fro_err


class WSINDy_PDE():

    def __init__(
            self,
            poly_order: int = 2, include_bias: bool = True, lam: float = 1e-2,
            max_dx: int = 2, max_dt: int = 1,
            m_x: Optional[int] = None, m_t: Optional[int] = None,
            K: int = 100,
            tau: float = 1e-10, tau_hat: float = 3.0
    ):
        """
        This class implements the weak form SINDy method for PDEs based on
        (Messenger and Bortz, 2021).

        Args:
            poly_order (int): Highest polynomial order for candidate function
                library. Defaults to 2.
            include_bias (bool): Whether to include bias term in the candidate
                function library. Defaults to True.
            lam (float): Threshold parameter to be used for STLSQ. Defaults to
                1e-2.
            max_dx (int): Highest spatial partial derivative order. Defaults to
                2.
            max_dt (int): The temporal partial derivative order desired.
                Defaults to 1.
            m_x (Optional[int], optional): Number of support points for test
                functions. Defaults to None, which computes a value based on
                the active wave number.
            m_t (Optional[int], optional): Number of support points for test
                functions. Defaults to None, which computes a value based on
                the active wave number.
            K (int, optional): Number of test functions to use for each
                dimension. Note that this defines the lower bound of test
                functions to be used. Defaults to 100.
            tau (float, optional): Parameter that controls the real and
                spectral decay of the test functions through control of the
                polynomial degree used in the test function. Defaults to 1e-10.
            tau_hat (float, optional): Parameter that controls the automated
                selections of m. Defaults to 3.0.
        """
        
        # Store candidate function library parameters
        self.poly_order: int = poly_order
        self.include_bias: bool = include_bias

        # Store parameters relating to the sparse regressor
        self.lam: float = lam

        # Store parameters relating to WSINDy
        self.m_x: int = m_x
        self.m_t: int = m_t

        self.max_dx: int = max_dx
        self.max_dt: int = max_dt

        self.K: int = K
        
        self.tau: int = tau
        self.tau_hat: float = tau_hat

        # Fitted flag
        self.fitted: bool = False


    def fit(
            self,
            u_sol: np.ndarray,
            x_grids: Union[list[np.ndarray], np.ndarray], t_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Function to call to discover sparse PDE/ODE models using weak form
        SINDy.

        Args:
            u_sol (np.ndarray): Solution matrix in the shape
                [x_0, ..., x_D, t, u]
            x_grids (Union[list[np.ndarray], np.ndarray]): The x spatial grid
                or list of x spatial grids.
            t_grid (np.ndarray): Temporal grid points.

        Returns:
            np.ndarray: Sparse coefficient matrix.
        """
        
        # Get total number of solution variables
        self.udim: int = u_sol.shape[-1]
        
        # If only a single spatial dimension grid point is provided, reshape it
        # such that it adheres to the data structure of multi-spatial
        # dimension.
        if not isinstance(x_grids, list):
            x_grids = [x_grids]

        # Define a list of total number of grid points specified by each
        # dimension of the spatial-temporal grid.
        # List is sized D+1 where D is the total number of spatial dimensions.
        # The last element is the time dimension.
        self.dims: list[int] = [x.shape[0] for x in x_grids] \
            + [t_grid.shape[0]]
        
        # Validate the dimensions of the provided grids against the solution.
        self._validate_dims(u_sol)

        # Compute support points if not provided or else validate the provided
        # support point number.
        if self.m_x is None or self.m_t is None:
            default_mx, default_mt = self._get_support_pts(u_sol)
            if self.m_x is None:
                self.m_x = default_mx
            if self.m_t is None:
                self.m_t = default_mt

        # Validate support point number
        for dim_idx, dim in enumerate(self.dims[:-1]):
            assert dim >= self.m_x*2+1, (
                f"Spatial dimension {dim_idx+1:d} "
                + f"does not contain sufficient number of grid points to "
                + f"support a test function spanning {self.m_x*2+1:d} "
                + f"grid points!"
            )
        assert self.dims[-1] >= self.m_t*2+1, (
            f"Temporal dimension does not contain sufficient number of "
            + f"grid points to support a test function spanning "
            + f"{self.m_t*2+1:d} grid points!"
        )

        # Obtain test function basis (spatial test function is the same across
        # all spatial dimensions)
        if self.m_x > 0:
            self.phi_x_basis, self.p_x = self._get_test_func_basis(
                self.m_x, self.max_dx, self.tau)
        else:
            self.phi_x_basis, self.p_x = np.array([[1]]), 0
        self.phi_t_basis, self.p_t = self._get_test_func_basis(
            self.m_t, self.max_dt, self.tau)
        
        # Get the spatial derivatives in Fourier space
        self.phi_x_hats = self._get_fft_phi(
            self.phi_x_basis,
            self.dims[:-1],
            self.m_x
        )

        # Get the temporal derivatives in Fourier space
        self.phi_t_hat = self._get_fft_phi(
            self.phi_t_basis,
            self.dims[-1],
            self.m_t
        )[0]

        # Obtain grid intervals (assumes uniform grid)
        grid_intervals: np.ndarray = np.empty((len(self.dims),))
        for dim_idx in range(len(self.dims)-1):
            grid_intervals[dim_idx] = np.mean(np.diff(x_grids[dim_idx]))
        grid_intervals[-1] = np.mean(np.diff(t_grid))

        # Rescale from basis domain to PDE grid domain
        for xIdx, dx in enumerate(grid_intervals[:-1]):
            if not np.isnan(dx):
                self.phi_x_hats[xIdx] *= (self.m_x*dx)**-np.arange(
                    0, self.max_dx+1).reshape(-1,1)
        self.phi_t_hat *= (self.m_t*grid_intervals[-1])**-np.arange(
            0, self.max_dt+1).reshape(-1,1)
        
        # Obtain subsample mask
        subsample_mask = self._get_subsample_mask()

        # Construct left and right side of regression problem
        lhs_b, rhs_G = self._build_linear_eqn(
            u_sol, self.phi_x_hats, self.phi_t_hat,
            subsample_mask
        )

        # Initialise coeffient matrix
        self.coef: np.ndarray = np.empty((rhs_G.shape[1], self.udim))

        for uIdx in range(self.udim):

            # Carry out sparse regression
            self.coef[:, uIdx] = stlsq(rhs_G, lhs_b[:,uIdx], lam=self.lam).reshape(-1)

        self.fitted = True

        return self.coef
    

    def print(self, precision: int = 3) -> str:
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

        state_names = [f"u_{idx+1:d}" for idx in range(self.udim)]
        
        return print_latex_eqn(
            self.coef,
            state_names, self.lib_names,
            precision, False, self.max_dt
        )
    

    def _get_support_pts(self, u_sol: np.ndarray) -> tuple[int, int]:
        """
        Computes the optimal number of discrete support points for both the
        spatial and temporal test functions based on active frequency vs noise
        floor analysis.

        Args:
            u_sol (np.ndarray): Solution matrix.

        Returns:
            tuple[int, int]: Returns the optimal support points for spatial and
                temporal test functions.
        """
    
        # Get total number of solution variables
        udims: int = u_sol.shape[-1]

        # Initialise list of candidate support points
        m_xs: list[int] = list()
        m_ts: list[int] = list()
        
        # Loop through solution variables
        for uIdx in range(udims):

            # Loop through each dimension
            for dimIdx, dim in enumerate(self.dims):

                # Skip if total dimension is only 1 (case for ODEs)
                if dim > 1:

                    # Get active wave number
                    active_k: int = self._get_active_wavenumber(
                        u_sol[..., uIdx], dimIdx
                    )

                    # Set lower bound for k for numerical stability
                    active_k = max(4, active_k)

                    # Compute support point
                    support_pt: int = brentq(
                        lambda m: np.log((2*m-1)/m**2) \
                            * (4 * np.pi**2 * active_k**2 * m**2 \
                            - 3 * dim**2 * self.tau_hat**2) \
                            - 2 * dim**2 * self.tau_hat**2 * np.log(self.tau),
                        2.0,
                        dim/2
                    )

                    if dimIdx < (len(self.dims)-1):
                        m_xs.append(support_pt)
                    else:
                        m_ts.append(support_pt)

                else:

                    if dimIdx < (len(self.dims)-1):
                        m_xs.append(0)
                    else:
                        m_ts.append(0)


        # Get support points
        m_x: int = int(min(
            np.ceil(np.mean(m_xs)),
            np.floor((np.min(self.dims[:-1])-1)/2)
        ))
        m_t: int = int(min(
            np.ceil(np.mean(m_ts)),
            np.floor((self.dims[-1]-1)/2)
        ))
        
        return m_x, m_t
    

    def _get_active_wavenumber(self, u_var: np.ndarray, dim: int) -> int:
        """
        Get the active wave number using frequency analysis for a specified
        dimension.

        Args:
            u_var (np.ndarray): Solution matrix for a given solution variable.
            dim (int): The specified dimension.

        Returns:
            int: The frequency bin number that separates active frequencies
                against the noise floor.
        """
        
        # Get total number of spatial and temporal dimensions
        ndims: int = u_var.ndim
        
        # Get gain of solution in frequency space
        u_hat: np.ndarray = np.fft.rfft(u_var, axis=dim)
        u_hat_gain: np.ndarray = np.abs(u_hat)

        # Average across all other dimensions
        u_hat_gain_avg: np.ndarray = np.mean(
            u_hat_gain,
            axis=tuple([d for d in range(ndims) if d != dim])
        )

        # Truncate from peak signal strength
        max_idx: int = np.argmax(u_hat_gain_avg)

        # Apply cumulative sum
        gain_cumsum: np.ndarray = np.cumsum(u_hat_gain_avg[max_idx:])

        # Find corner point
        active_k = bi_piecewise_regression(
            np.arange(max_idx, u_hat_gain_avg.shape[0]),
            gain_cumsum
        )[1]
        
        return active_k
    

    def _get_test_func_basis(
            self, m: int, d: int, tau: float = 1e-10
    ) -> tuple[np.ndarray, int]:
        """
        Obtain the basis piecewise unimodal test function supported on 2m+1
        discrete grid points on a domain [-1,1].

        Args:
            m (int): Parameter to define the number of grid points to be used.
            d (int): Highest derivative of test function to compute.
            tau (float, optional): Parameter that controls spectral decay
                by controlling the polynomial power to be used by the test
                function. Defaults to 1e-10.

        Returns:
            tuple[np.ndarray, int]: A tuple consisting of an array of the test
                function and its derivatives (where the dth derivative is
                on the dth row of the matrix) and the polynomial power computed
                to be used.
        """
        
        # Compute the degree of test function in accordance with equation 4.3
        p: int = max(
            int(np.ceil(np.log(tau) / np.log((2*m-1)/m**2))),
            d+1
        )
        
        # Define v grid over the domain [-1, 1] supported by 2m+1 grid points
        v: np.ndarray = np.linspace(-1, 1, 2*m+1).reshape(1,-1)

        # Define list of derivative orders to compute.
        derivs: np.ndarray = np.arange(0, d+1, 1, dtype=np.int32).reshape(-1,1)

        # Define vectorised permutation function
        perm_vec: Callable = lambda n, arr: np.array([
            perm(n,k) for k in arr.reshape(-1)
        ]).reshape(arr.shape)

        # Define test function \phi = (1-v^2)^p = fg where f = (1+v)^p and
        # g = (1-v)^p.
        f: np.ndarray = perm_vec(p, derivs) * (1+v)**(p-derivs)
        g: np.ndarray = (-1)**derivs * perm_vec(p, derivs) * (1-v)**(p-derivs)

        # Initialise derivatives \phi.
        phi: np.ndarray = np.zeros((d+1, 2*m+1))

        # Compute the test function
        phi[0,:] = (1-v**2)**p
        
        # Compute the derivatives of \phi.
        for deriv in map(int, derivs[1:,0]):
            for k in range(deriv+1):
                phi[deriv,:] += comb(deriv, k) * f[deriv-k,:] * g[k,:]

        return phi, p


    def _get_fft_phi(
            self, phi_basis: np.ndarray,
            dims: Union[list[int], int], m: int
    ) -> list[np.ndarray]:
        """
        Conducts FFT on the provided test function basis for the specified
        dimensions. For each dimension, it pads and shifts the test function
        accordingly to prepare it for convolution through multiplication in the
        frequency space.

        Args:
            phi_basis (np.ndarray): The basis test function.
            dims (Union[list[int], int]): The dimensions to be computed.
            m (int): The parameter used to define the number of support grid
                points.

        Returns:
            list[np.ndarray]: List of Fourier transformed test functions for
                the list of dimensions.
        """
        
        # If time dimension provided, ensure that it remains a list.
        if not isinstance(dims, list):
            dims = [dims]
        
        phi_hats: list[np.ndarray] = list()

        # Loop through each dimension
        for dim in dims:

            # Pad the basis function with zeros to match dimension length
            phi_padded: np.ndarray = np.concatenate((
                phi_basis,
                np.zeros((phi_basis.shape[0], dim-phi_basis.shape[1]))
            ), axis=1)

            # Centre the test function at the zeroth index
            shift_idx: int = -m
            phi_shifted = np.roll(phi_padded, shift_idx, axis=1)

            # Transform via FFT
            phi_hat = np.fft.fft(phi_shifted, axis=1)

            # Append to list of test functions in Fourier space
            phi_hats.append(phi_hat)

        return phi_hats


    def _get_subsample_mask(self) -> np.ndarray:
        """
        Defines the subsample access mask. Note that instead of defining a
        subsample interval as detailed in Messenger's implementation, it is now
        controlled by K, the lower bound limit on the number of test functions.

        Returns:
            np.ndarray: The final subsample access index array for each
                nd-array solution variable.
        """
        
        # Define subsample mask paddings
        mask_pad = [self.m_x for _ in range(len(self.dims))]
        mask_pad[-1] = self.m_t

        # Define subsample mask intervals
        mask_interval = [
            int(max(np.floor((dim - 2*pad)/self.K), 1))
            for (dim, pad) in zip(self.dims, mask_pad)
        ]

        # Define subsample mask indicies
        mask_idx = tuple(
            slice(pad, -pad if pad > 0 else None, interval)
            for (pad, interval) in zip(mask_pad, mask_interval)
        )

        # Define subsample mask
        mask = np.zeros(self.dims, dtype=bool)
        mask[mask_idx] = True
        
        return mask


    def _build_linear_eqn(
            self,
            u_sol: np.ndarray,
            phi_x_hats: list[np.ndarray], phi_t_hat: np.ndarray,
            subsample_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Function that builds LHS and RHS matrices by convolving the appropriate
        test functions with the solution variables and candidate function
        library respectively.

        Args:
            u_sol (np.ndarray): The solution array.
            phi_x_hats (list[np.ndarray]): The spatial test functions.
            phi_t_hat (np.ndarray): The temporal test function.
            subsample_mask (Optional[np.ndarray], optional): Subsample mask to
                be used for querying.

        Returns:
            tuple[np.ndarray, np.ndarray, list[str]]: A tuple of LHS array, RHS
                array and the list of library names.
        """

        # Obtain list of all alpha vectors (vectors of partial derivative
        # order)
        self.alphas: list[tuple[int, ...]] = self._get_alpha()

        # Compute total subsampled query points
        total_samples: int = np.sum(subsample_mask)


        # ----------------------- LHS Computation ----------------------- #

        # Initialise LHS b
        lhs_b: np.ndarray = np.empty((total_samples, self.udim))

        # Initialise LHS kernel
        lhs_kernel = np.ones(self.dims, dtype=np.complex128)

        # Loop through all dimensions
        for dimIdx in range(len(self.dims)):

            # Obtain desired shape of phi for broadcasting
            phi_shape: list[int] = [1] * len(self.dims)
            phi_shape[dimIdx] = -1

            if dimIdx + 1 < len(self.dims):
                # Multiply spatial phi(x_1)...phi(x_D)
                lhs_kernel *= phi_x_hats[dimIdx][0,:].reshape(phi_shape)
            else:
                # Multiply temporal phi(t) derivative
                lhs_kernel *= phi_t_hat[-1,:].reshape(phi_shape)

        # Loop through all solution variables to construct LHS
        for uIdx in range(self.udim):

            # Fourier transform the solution
            u_hat: np.ndarray = np.fft.fftn(u_sol[..., uIdx])

            # Multiply LHS in Fourier space
            b_hat: np.ndarray = u_hat * lhs_kernel

            # Store in G and b variables
            lhs_b[:, uIdx] = np.fft.ifftn(b_hat).real[subsample_mask]


        # ----------------------- RHS Computation ----------------------- #

        # Obtain candidate function library
        self.theta, self.theta_names = self._build_theta(u_sol)

        # Obtain total number of candidate functions
        self.theta_dim: int = self.theta.shape[-1]

        # Initialise RHS matrix
        lib_dim: int = len(self.alphas)*self.theta_dim
        if self.include_bias:
            lib_dim += 1
        rhs_G: np.ndarray = np.empty((
            total_samples, lib_dim
        ))

        # Initialise list of library names
        self.lib_names: list[str] = list()

        # Loop through all partial derivatives 
        for alpha_idx, alpha in enumerate(self.alphas):

            # Initialise RHS kernels of D^{alpha} Phi(x,t)
            rhs_kernel = np.ones(self.dims, dtype=np.complex128)

            # Loop through spatial dimension to construct RHS kernel
            for dimIdx in range(len(self.dims)):

                # Obtain desired shape of phi for broadcasting
                phi_shape: list[int] = [1] * len(self.dims)
                phi_shape[dimIdx] = -1

                # Multiply phi^{alpha^s_d}
                if dimIdx + 1 < len(self.dims):
                    rhs_kernel *= phi_x_hats[dimIdx][alpha[dimIdx],:].reshape(phi_shape)
                else:
                    rhs_kernel *= phi_t_hat[alpha[dimIdx],:].reshape(phi_shape)

            # Adding on bias term
            if self.include_bias and alpha_idx == 0:
                feature_hat = np.fft.fftn(np.ones(self.dims))
                G_feature_hat = np.fft.ifftn(feature_hat*rhs_kernel)
                rhs_G[:, 0] = G_feature_hat[subsample_mask].real

                self.lib_names.append('1')

            # Define partial derivative latex string
            deriv_order: int = np.sum(alpha)
            deriv_name: str = ''

            if deriv_order > 0:
                if deriv_order > 1:
                    deriv_name += rf"\frac{{\partial^{deriv_order:d}}}"
                else:
                    deriv_name += rf"\frac{{\partial}}"
                deriv_name += "{"
                for dimIdx, deriv in enumerate(alpha):
                    if deriv > 1:
                        deriv_name += rf"\partial x_{dimIdx+1:d}^{deriv:d} "
                    elif deriv == 1:
                        deriv_name += rf"\partial x_{dimIdx+1:d} "
                deriv_name += "}"

            # Multiply each RHS theta term in Fourier space
            for thetaIdx in range(self.theta_dim):
                feature_hat = np.fft.fftn(self.theta[..., thetaIdx])
                G_feature_hat = np.fft.ifftn(feature_hat*rhs_kernel)
                rhs_G[
                    :, alpha_idx*self.theta_dim + thetaIdx + self.include_bias
                ] = G_feature_hat[subsample_mask].real

                if deriv_name != '':
                    self.lib_names.append(
                        deriv_name \
                            + rf'\left( {self.theta_names[thetaIdx]} \right)'
                    )
                else:
                    self.lib_names.append(
                        f'{self.theta_names[thetaIdx]}'
                    )

        return lhs_b, rhs_G


    def _get_alpha(self) -> list[tuple[int, ...]]:
        """
        Return all alphas for the given dimensions and desired derivative
        orders.

        Returns:
            list[tuple[int, ...]]: List of alpha vectors prescribing the
                derivatives to use.
        """

        # Obtain total number of spatial dimensions
        xdims: int = len(self.dims)-1

        # Get all vectors of alpha (all combinations of derivatives)
        # Spatial and temporal cross terms currently removed
        alphas: list[tuple[int, ...]] = list(product(
            *[list(range(self.max_dx+1)) for _ in range(xdims)],
            [0]
        ))

        # Sort by time derivative
        return sorted(alphas, key=lambda x: x[-1])
    

    def _build_theta(self, u_sol: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Build the polynomial library for a PDE solution matrix. This adds an
        additional dimension to the solution matrix such that it becomes
        [x_0, ..., x_D, t, u, f].

        Args:
            u_sol (np.ndarray): The input solution matrix.

        Returns:
            tuple[np.ndarray, list[str]]: Returns the candidate function
                library matrix and the list of library names to be used.
        """
        
        # Obtain the number of solution variables
        udim: int = u_sol.shape[-1]
        
        # Apply nonlinearities by first flattening the solution data into shape
        u_sol_flattened: np.ndarray = u_sol.reshape(-1, udim)
        
        # Apply polynomial library
        theta_modified, lib_names = poly_lib(
            u_sol_flattened,
            self.poly_order, False,
            [f'u_{u_idx+1:d}' for u_idx in range(udim)]
        )
        
        # Reshape theta library back into original shape with an added first
        # dimension corresponding to a different candidate function
        theta: np.ndarray = theta_modified.reshape((
            *self.dims, theta_modified.shape[1]
        ))

        return theta, lib_names


    def _validate_dims(self, u_sol: np.ndarray):
        """
        Validate the dimensions of the solution against the provided grid point
        information.

        Args:
            u_sols (Union[list[np.ndarray], np.ndarray]): Spatial-temporal
                solution.
        """

        # Verify that at least a spatial, temporal and solution dimensions are
        # provided
        assert len(u_sol.shape) >= 3, (
            f"The spatiotemporal solution must be provided in the shape of "
            + f"(x_1, ..., x_D, t, u)."
        )

        # Verify that the provided spatial temporal solution matches the
        # provided grid
        assert len(self.dims) == len(u_sol.shape)-1, (
            f"Defined {len(self.dims)-1:d} spatial dimensions but the provided "
            + f"solution has {len(u_sol.shape)-1} number of spatial dimension!"
        )

        # Check that the size of the temporal dimension matches
        assert u_sol.shape[-2] == self.dims[-1], (
            f"Defined {self.dims[-1]:d} number of time grid points, but the "
            + f"provided solution has {u_sol.shape[-2]} number of time grid "
            + f"points!"
        )

        # Check that the size of each spatial dimension matches
        for x_idx, x_dim in enumerate(self.dims[:-1]):
            assert u_sol.shape[x_idx] == x_dim, (
                f"Defined {x_dim:d} number of spatial grid points, but the "
                + f"provided solution has {u_sol.shape[x_idx]} number of "
                + f"spatial grid points on dimension {x_idx+1:d}!"
            )


def benchmark_wsindy_pde(
        us: list[np.ndarray], noise_ratios: np.ndarray,
        x_grids: Union[list[np.ndarray], np.ndarray], t_grid: np.ndarray,
        xi_sindy: np.ndarray, xi_wsindy: np.ndarray,
        wsindy_kwargs: dict = dict(),
        sindy_lib_kwargs: dict = dict, sindy_opti_kwargs: dict = dict(),
):
    """
    Benchmarks weak form SINDy with PDE-FIND from PySINDy.

    Args:
        us (list[np.ndarray]): List of noisy solutions.
        noise_ratios (np.ndarray): Noise ratios in that solution.
        x_grids (Union[list[np.ndarray], np.ndarray]): The list of spatial
            grids.
        t_grid (np.ndarray): Temporal grid.
        xi_sindy (np.ndarray): Correct coefficient matrix for PySINDy.
        xi_wsindy (np.ndarray): Correct coefficient matrix for WSINDy.
        wsindy_kwargs (dict, optional): Keyword arguments for `WSINDy_PDE`.
            Defaults to dict().
        sindy_lib_kwargs (dict, optional): Keyword arguments for the library
            function of PySINDy. Defaults to dict.
        sindy_opti_kwargs (dict, optional): Keyword arguments for the optimizer
            of PySINDy. Defaults to dict().
    """
    
    # Initialise list of sparse coefficients matrix from different methods
    sindy_xis: list[np.ndarray] = list()
    wsindy_xis: list[np.ndarray] = list()

    # Initialise list of coefficient error metrics
    sindy_xi_errors: list[np.ndarray] = list()
    wsindy_xi_errors: list[np.ndarray] = list()

    # Initialise list of precisions
    sindy_precs: list[np.ndarray] = list()
    wsindy_precs: list[np.ndarray] = list()
    
    # Loop through all u data
    for idx, u in enumerate(us):
        
        # Fit data using SINDy (2nd order finite difference)
        xi, xi_err, prec = fit_pde_find(
            u, x_grids, t_grid,
            xi_sindy, sindy_lib_kwargs, sindy_opti_kwargs
        )
        
        sindy_xis.append(xi)
        sindy_xi_errors.append(xi_err)
        sindy_precs.append(prec)

        # Fit data using WSINDy using uniform grid with OLS
        xi, xi_err, prec = fit_wsindy(
            u, x_grids, t_grid,
            xi_wsindy, wsindy_kwargs
        )
        
        wsindy_xis.append(xi)
        wsindy_xi_errors.append(xi_err)
        wsindy_precs.append(prec)

    # Plot coefficient errors
    _, axs = plt.subplots(2,1,figsize=(6,8))
    axs[0].plot(noise_ratios, sindy_xi_errors, '-', label='PDE-FIND')
    axs[0].plot(noise_ratios, wsindy_xi_errors, '--', label='Weak Form SINDy')
    axs[0].set_xlabel(r'$\sigma_{NR}$')
    axs[0].set_ylabel('$E_2$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend()

    # Plot precisions
    axs[1].plot(noise_ratios, sindy_precs, '-', label='PDE-FIND')
    axs[1].plot(noise_ratios, wsindy_precs, '--', label='Weak Form SINDy')
    axs[1].set_xlabel(r'$\sigma_{NR}$')
    axs[1].set_ylabel('Precision')
    axs[1].set_xscale('log')
    axs[1].legend()


def fit_wsindy(
        u: np.ndarray,
        x_grids: Union[list[np.ndarray], np.ndarray], t_grid: np.ndarray,
        true_w: np.ndarray,
        wsindy_kwargs: dict = dict()
) -> tuple[np.ndarray, float, float]:
    """
    A helper function to run weak form SINDy and evaluate it.

    Args:
        u (np.ndarray): Solution array.
        x_grids (Union[list[np.ndarray], np.ndarray]): Spatial grid or a list
            of spatial grids.
        t_grid (np.ndarray): Temporal grid.
        true_w (np.ndarray): Correct coefficient matrix expected for weak form
            SINDy.
        wsindy_kwargs (dict, optional): Keyword arguments for weak form
            SINDy. Defaults to dict().

    Returns:
        tuple[np.ndarray, float, float]: Returns the fitted coefficient matrix,
            coefficient error and the precision metric.
    """
    
    wsindy: WSINDy_PDE = WSINDy_PDE(**wsindy_kwargs)
    
    w: np.ndarray = wsindy.fit(u, x_grids, t_grid)
    
    coef_err: float = rel_fro_err(w, true_w)
    prec: float = precision(w, true_w)
    
    return w, coef_err, prec


def fit_pde_find(
        u: np.ndarray,
        x_grids: Union[list[np.ndarray], np.ndarray], t_grid: np.ndarray,
        true_w: np.ndarray,
        lib_kwargs: dict = dict(),
        opti_kwargs: dict = dict()
) -> tuple[np.ndarray, float, float]:
    """
    Helper function to fit using PDE-FIND.

    Args:
        u (np.ndarray): Solution array.
        x_grids (Union[list[np.ndarray], np.ndarray]): Spatial grid or a list
            of spatial grids.
        t_grid (np.ndarray): Temporal grid.
        true_w (np.ndarray): Correct coefficient matrix expected for PySINDy.
        lib_kwargs (dict, optional): Keyword arguments for PySINDy library.
            Defaults to dict().
        opti_kwargs (dict, optional): Keyword arguments for PySINDy optimizer.
            Defaults to dict().

    Returns:
        tuple[np.ndarray, float, float]: Returns the fitted coefficient matrix,
            coefficient error and the precision metric.
    """
    
    # Check if x_grids is in list form
    if not isinstance(x_grids, list):
        x_grids = [x_grids]

    mesh_X = np.meshgrid(*x_grids, indexing='ij')
    mesh_X = np.transpose(mesh_X, [i+1 for i in range(len(x_grids))] + [0])
    lib_kwargs['spatial_grid'] = mesh_X

    pde_lib = ps.PDELibrary(**lib_kwargs)
    opti = ps.STLSQ(**opti_kwargs)

    model = ps.SINDy(opti, pde_lib)
    model.fit(u, t_grid)

    w: np.ndarray = model.coefficients()

    coef_err: float = rel_fro_err(w, true_w)
    prec: float = precision(w, true_w)
    
    return w, coef_err, prec