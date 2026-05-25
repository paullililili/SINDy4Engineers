import numpy as np
import casadi as ca
import pysindy as ps
from typing import Callable, Optional, Union

from .ivp_solvers import rk4
from .library import poly_lib
from .optimizer import stridge
from .utils import get_combinations, print_latex_eqn


class SINDycMPC():

    def __init__(
            self,
            Q: np.ndarray, Qmp: np.ndarray,
            R: np.ndarray, Rdu: np.ndarray,
            mp: int, mc: int,
            u_min: np.ndarray, u_max: np.ndarray,
            du_min: np.ndarray, du_max: np.ndarray,
            lam: float = 1e-2, l2_norm: float = 0.0,
            poly_order: int = 3, include_bias: bool = True,
            dt: Optional[float] = None,
            fd_func: Callable[
                    [np.ndarray, np.ndarray],
                    np.ndarray
                ] = ps.FiniteDifference(4)._differentiate,
            verbose: int = 0
    ):
        """
        A SINDy-MPC class based off of the method proposed by (Kaiser et al,
        2018). The class allows the user to fit a SINDy model using data
        trajectory via `fit()`. Once fitted, a MPC is also set up and control
        inputs can be retrieved via `query()`.

        Args:
            Q (np.ndarray): MPC running cost matrix.
            Qmp (np.ndarray): MPC terminal cost matrix.
            R (np.ndarray): MPC control input cost matrix.
            Rdu (np.ndarray): MPC control step cost matrix.
            mp (int): MPC prediction horizon.
            mc (int): MPC control horizon.
            u_min (np.ndarray): Minimum control input value vector.
            u_max (np.ndarray): Maximum control input value vector.
            du_min (np.ndarray): Minimum control step input vector.
            du_max (np.ndarray): Maximum control step input vector.
            lam (float, optional): Sparsity threshold for STRidge. Defaults to
                1e-2.
            l2_norm (float, optional): Regularization threshold for STRidge.
                Defaults to 0.0.
            poly_order (int, optional): Order of polynomial used by the
                polynomial library. Defaults to 3.
            include_bias (bool, optional): Whether to include a bias term
                inside the library. Defaults to True.
            dt (Optional[float], optional): Controller time step. Defaults to
                None which inherits the time step from the provided data.
            fd_func (Callable[ [np.ndarray, np.ndarray], np.ndarray ], optional):
                The numerical differentiator function to be used. Defaults to
                ps.FiniteDifference(4)._differentiate.
            verbose (int, optional): Verbose level of the MPC optimizer.
                Defaults to 0.
        """
        
        # Store penalty weight matrices
        self.Q: np.ndarray = Q
        self.Qmp: np.ndarray = Qmp
        self.R: np.ndarray = R
        self.Rdu: np.ndarray = Rdu

        # Extract expected data dimension information from weight matrices
        self.n: int = Q.shape[0]
        self.q: int = R.shape[0]

        # Store horizon information
        self.mp: int = mp
        self.mc: int = mc

        # Store information relating to control constraints
        self.u_min: np.ndarray = u_min
        self.u_max: np.ndarray = u_max
        self.du_min: np.ndarray = du_min
        self.du_max: np.ndarray = du_max

        # Store SINDy related parameters
        self.lam: float = lam
        self.l2_norm: float = l2_norm
        self.poly_order: int = poly_order
        self.include_bias: bool = include_bias

        # Store dt option
        self.dt: Union[float, None] = dt

        # Store finite difference differentiator function
        self.fd_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = fd_func

        # Set verbose level
        self.verbosity: int = verbose

        # Set fitted flag
        self.fitted: bool = False


    def fit(
            self,
            x: np.ndarray, t: np.ndarray, u: np.ndarray,
            normalize_columns: bool = False
    ) -> np.ndarray:
        """
        Fit a SINDy model to the provided data.

        Args:
            x (np.ndarray): State data matrix.
            t (np.ndarray): Time vector.
            u (np.ndarray): Control input matrix.
            normalize_columns (bool, optional): Determine whether the library
                should be normalized or not before fitting. Defaults to False.

        Returns:
            np.ndarray: Returns the SINDy coefficient matrix.
        """

        # Validate that the given data matches expected dimensions
        assert x.shape[1] == self.n, f'Expecting data with {self.n:d} states!'
        assert u.shape[1] == self.q, f'Expecting data with {self.q:d} inputs!'

        # Obtain time step from training data (assumes uniform time step)
        if self.dt is None:
            self.dt = float(np.mean(np.diff(t)))

        # Theta library
        lib, self.lib_names = poly_lib(
            np.hstack((x, u)),
            self.poly_order, self.include_bias,
            [f'x_{idx+1:d}' for idx in range(self.n)] \
                + [f'u_{idx+1:d}' for idx in range(self.q)]
        )

        # Compute time derivative
        xdot = self.fd_func(x,t)

        if normalize_columns:
            norm_inv = np.linalg.inv(np.linalg.norm(lib, axis=0) * np.eye(lib.shape[1]))
            lib = lib @ norm_inv

        self.xi = stridge(lib, xdot, self.lam, self.l2_norm)

        if normalize_columns:
            self.xi = norm_inv @ self.xi

        # Create callable fitted function
        self.fitted_func = lambda x, u: self._poly_lib(
            x, u,
            self.poly_order, self.include_bias
        ) @ self.xi

        # Initialise MPC solver
        self._init_solver()

        # Initialise last control input as 0
        self.opti.set_value(self.last_u, np.zeros((1,self.q)))

        # Set fitted flag
        self.fitted = True

        return self.xi
    

    def query(
            self,
            x: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Function to return the next control input step given current system
        state and reference trajectory to be tracked.

        Args:
            x (np.ndarray): Current state vector.
            r (np.ndarray): Reference trajectory to be tracked. Must span the
                specified prediction horizon.

        Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray]: Returns a tuple
                of next controller input, cost at current step, predicted
                future states within the prediction horizon, and the predicted
                optimal control inputs within the control horizon.
        """
        
        assert self.fitted, 'Model not yet fitted!'

        # Update parameters
        self.opti.set_value(self.x, x)
        self.opti.set_value(self.r, r)

        # Solve the MPC optimization problem
        sol = self.opti.solve()
        u_opt = sol.value(self.u)
        u_next = u_opt[0]

        # Retrieve cost and predicted trajectory
        cost = float(sol.value(self.opti.f))
        x_hat = sol.value(self.x_hat)

        # Update previous step u parameter
        self.opti.set_value(self.last_u, u_next)

        return u_next, cost, x_hat, u_opt
    

    def print_str(self) -> str:
        """
        Returns the SINDy model equation in Latex string.

        Returns:
            str: SINDy model.
        """

        return print_latex_eqn(
            self.xi,
            [f'x_{idx+1:d}' for idx in range(self.n)],
            self.lib_names
        )
    

    def _init_solver(self):
        """
        Initializes the MPC solver.
        """

        # Initialise the optimizer
        self.opti: ca.Opti = ca.Opti()

        # Define sequence of controls to optimise
        self.u = self.opti.variable(self.mc, self.q)

        # Define the control input during the last step
        self.last_u = self.opti.parameter(1, self.q)

        # Define sequence of relative steps
        self.delta_u = self.u - ca.vertcat(self.last_u, self.u[:-1,:])

        # Define sequence of states to predict
        self.x_hat = self.opti.variable(self.mp, self.n)

        # Set reference tracking trajectory
        self.r = self.opti.parameter(self.mp, self.n)

        # Set current state
        self.x = self.opti.parameter(1, self.n)

        # Obtain the cost function
        self._get_cost(self.x_hat, self.r, self.u, self.delta_u)

        # Set the constraints
        self._get_constraints(self.x, self.x_hat, self.u, self.delta_u)

        # Set solver verbosity
        opts: dict = {
            'ipopt.print_level': self.verbosity,
            'print_time': 0 if self.verbosity == 0 else 1,
            'ipopt.sb': 'yes' if self.verbosity == 0 else 'no'
        }

        # Set solver
        self.opti.solver('ipopt', opts)
    

    def _get_cost(
            self,
            x_hat: ca.MX, r: ca.MX,
            u: ca.MX, delta_u: ca.MX
    ):
        """
        Sets up the cost function for the MPC.

        Args:
            x_hat (ca.MX): Predicted state variable matrix.
            r (ca.MX): Reference state trajectory variable matrix.
            u (ca.MX): Input variable matrix.
            delta_u (ca.MX): Input step variable matrix.
        """
        
        # Compute state deviation
        delta_x: ca.MX = x_hat - r
        
        # Final predicted state cost
        cost: ca.MX = delta_x[-1,:] @ self.Qmp @ delta_x[-1,:].T

        # Predicted cost for remaining states
        for k in range(self.mp-1):
            cost += delta_x[k,:] @ self.Q @ delta_x[k,:].T

        # Control input penalty
        for k in range(self.mc):
            cost += u[k,:] @ self.R @ u[k,:].T

        # Control movement penalty
        for k in range(self.mc):
            cost += delta_u[k,:] @ self.Rdu @ delta_u[k,:].T

        self.opti.minimize(cost)
    

    def _get_constraints(
            self,
            x: ca.MX, x_hat:ca.MX,
            u:ca.MX, delta_u:ca.MX
    ):
        """
        Set up constraints for the MPC problem. Specifically, bound constraints
        for input and input step, as well as system dynamics constraint.

        Args:
            x (ca.MX): State variable matrix.
            x_hat (ca.MX): Predicted state variable matrix.
            u (ca.MX): Control input variable matrix.
            delta_u (ca.MX): Control input step variable matrix.
        """

        # System dynamics constraint
        next_x = self._rk4(self.fitted_func, self.dt, x, u[0,:])
        for k in range(self.mp):
            self.opti.subject_to(x_hat[k,:] == next_x)

            # If prediction horizon exceeds control horizon, take last control
            # step in the sequence
            if k < self.mc-1:
                next_x = self._rk4(
                    self.fitted_func, self.dt, x_hat[k,:], u[k+1,:])
            else:
                next_x = self._rk4(
                    self.fitted_func, self.dt, x_hat[k,:], u[-1,:])

        # Control input constraints
        self.opti.subject_to(self.opti.bounded(
            self.u_min, u, self.u_max
        ))

        # Relative control step constraint
        self.opti.subject_to(self.opti.bounded(
            self.du_min, delta_u, self.du_max
        ))


    def _poly_lib(
            self,
            x: ca.MX, u: ca.MX,
            poly_order: int, include_bias: bool
    ) -> ca.MX:
        """
        Adapted polynomial library function from `sindy.poly_lib` which allows
        Casadi MX variables to be used.

        Args:
            x (ca.MX): State variable matrix.
            u (ca.MX): Input variable matrix.
            poly_order (int): Polynomial order to be used.
            include_bias (bool): Whether or not to include bias.

        Returns:
            ca.MX: Theta library matrix in variable form.
        """
        
        # Concatenate state and control
        xu = ca.horzcat(x, u)

        # Initialise a list of theta functions
        list_Theta: list[np.ndarray] = list()

        # Include bias term if specified
        if include_bias:
            list_Theta.append(1.0)
        
        # Loop through each polynomial degree
        for deg in range(1, poly_order+1):

            # Get all combinations of feature indices for the current degree
            combos = get_combinations(self.n+self.q, deg)

            # Loop through each combination within the current degree
            for combo in combos:

                # Multiply features according to prescribed powers
                term = 1.0
                for idx in combo:
                    term *= xu[0, idx]

                # Multiply relevant data columns together
                list_Theta.append(term)

        # Return final theta matrix and list of library names
        return ca.horzcat(*list_Theta)
    

    def _rk4(
            self,
            func: Callable,
            dt: float,
            x0: ca.MX, u: ca.MX
    ) -> ca.MX:
        """
        Adapted RK4 method that allows Casadi MX variable to be passed through
        and integrated for a single step.

        Args:
            func (Callable): xdot = f(x) function to be used for system
                dynamics.
            dt (float): Time step.
            x0 (ca.MX): Initial condition to be used for integration.
            u (ca.MX): Control input.

        Returns:
            ca.MX: Returns the next step of the system trajectory.
        """

        # Compute next step
        k1 = func(x0, u)
        k2 = func(x0 + dt/2 * k1, u)
        k3 = func(x0 + dt/2 * k2, u)
        k4 = func(x0 + dt*k3, u)

        return x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)