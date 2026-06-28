import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from typing import Iterable, Callable, Optional, NamedTuple
from pysindy import FiniteDifference
from tqdm import tqdm
from functools import partial
import warnings

from sindy.ivp_solvers import rk4


def kuramato_sivashinsky(
        tFinal: float, dt: float,
        total_x_pts: int, xbounds: Iterable,
        u0_func: Callable,
        total_unity_roots: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python implementation of (Kassam and Trefethen, 2005) modified exponential
    time differencing RK4 solver for Kuramato Sivashinky equation.

    Args:
        tFinal (float): Final time in computation.
        dt (float): Time step.
        total_x_pts (int): Total number of spatial grid points.
        xbounds (Iterable): Bounds of spatial domain.
        u0_func (Callable): Initial condition.
        total_unity_roots (int, optional): Total number of points used in complex integration. Defaults to 16.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Returns solution with time and spatial grid point vectors.
    """
    
    # Check if `xbounds` is specified correctly
    assert len(xbounds) == 2, (
        "xbounds should be provided in the form of (xmin, xmax)")
    assert xbounds[1] > xbounds[0], (
        "The maximum bound should be higher than the minimum bound!")

    # Create spatial grid
    x_span: float = xbounds[1] - xbounds[0]
    x: np.ndarray = np.linspace(1, total_x_pts, total_x_pts) / total_x_pts \
        * x_span + xbounds[0]
    
    # Create temporal grid
    t: np.ndarray = np.arange(0, tFinal+dt, dt)
    
    # Obtain initial condition
    u0: np.ndarray = u0_func(x)
    
    # Define wavenumber
    k: np.ndarray = np.fft.fftfreq(
        total_x_pts,
        d=(x_span / (2*np.pi*total_x_pts))
    )

    # Define linear operator L (which is a vector in this case due to diagonal
    # matrix)
    l: np.ndarray = k**2 - k**4

    # Define matrix exponents
    e: np.ndarray = np.exp(dt*l)
    e2: np.ndarray = np.exp(dt*l/2)

    # Define linear operator multiplied by roots of unity
    r: np.ndarray = np.exp(
        1j * np.pi
        * (np.linspace(1, total_unity_roots, total_unity_roots) - 0.5)
        / total_unity_roots
    ).reshape(1,-1)
    lr: np.ndarray = dt * l.reshape(-1, 1) + r

    # Obtain ETDRK4 coefficients
    q: np.ndarray = dt * np.mean(
        (np.exp(lr/2)-1)/lr,
        axis=1
    )
    f1: np.ndarray = dt * np.mean(
        (-4 - lr + np.exp(lr) * (4 - 3*lr + lr**2)) / lr**3,
        axis=1
    )
    f2: np.ndarray = dt * np.mean(
        (2 + lr + np.exp(lr) * (-2+lr)) / lr**3,
        axis=1
    )
    f3: np.ndarray = dt * np.mean(
        (-4 - 3*lr - lr**2 + np.exp(lr)*(4-lr)) / lr**3,
        axis=1
    )

    # Define multiplier for nonlinear operator
    g: np.ndarray = -0.5 * 1j * k

    # Initialise solution
    u = np.empty((t.shape[0], total_x_pts))
    u[0,:] = u0
    v: np.ndarray = np.fft.fft(u0)

    # Time step loop
    for tIdx in range(t.shape[0]-1):

        Nv: np.ndarray = g * np.fft.fft(np.fft.ifft(v).real**2)

        a: np.ndarray = e2*v + q*Nv
        Na: np.ndarray = g * np.fft.fft(np.fft.ifft(a).real**2)

        b: np.ndarray = e2*v + q*Na
        Nb: np.ndarray = g * np.fft.fft(np.fft.ifft(b).real**2)

        c: np.ndarray = e2*a + q*(2*Nb - Nv)
        Nc: np.ndarray = g * np.fft.fft(np.fft.ifft(c).real**2)

        v = e*v + f1*Nv + f2*2*(Na+Nb) + f3*Nc

        u[tIdx+1,:] = np.fft.ifft(v).real.reshape(-1)

    # Reshape u into [x_1, ..., x_D, t, u] shape
    return u.T.reshape((*u.T.shape, 1)), t, x


def korteweg_de_vries(
        tFinal: float, dt: float,
        total_x_pts: int, xbounds: Iterable,
        u0_func: Callable,
        total_unity_roots: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python implementation of (Kassam and Trefethen, 2005) modified exponential
    time differencing RK4 solver for Kortewag de Vries equation.

    Args:
        tFinal (float): Final time in computation.
        dt (float): Time step.
        total_x_pts (int): Total number of spatial grid points.
        xbounds (Iterable): Bounds of spatial domain.
        u0_func (Callable): Initial condition.
        total_unity_roots (int, optional): Total number of points used in complex integration. Defaults to 16.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Returns solution with time and spatial grid point vectors.
    """
    
    # Check if `xbounds` is specified correctly
    assert len(xbounds) == 2, (
        "xbounds should be provided in the form of (xmin, xmax)")
    assert xbounds[1] > xbounds[0], (
        "The maximum bound should be higher than the minimum bound!")

    # Create spatial grid
    x_span: float = xbounds[1] - xbounds[0]
    x: np.ndarray = np.linspace(1, total_x_pts, total_x_pts) / total_x_pts \
        * x_span + xbounds[0]
    
    # Create temporal grid
    t: np.ndarray = np.arange(0, tFinal+dt, dt)
    
    # Obtain initial condition
    u0: np.ndarray = u0_func(x)
    
    # Define wavenumber
    k: np.ndarray = np.fft.fftfreq(
        total_x_pts,
        d=(x_span / (2*np.pi*total_x_pts))
    )

    # Define linear operator L (which is a vector in this case due to diagonal
    # matrix)
    l: np.ndarray = 1j * k**3

    # Define matrix exponents
    e: np.ndarray = np.exp(dt*l)
    e2: np.ndarray = np.exp(dt*l/2)

    # Define linear operator multiplied by roots of unity
    r: np.ndarray = np.exp(
        2j * np.pi
        * (np.linspace(1, total_unity_roots, total_unity_roots) - 0.5)
        / total_unity_roots
    ).reshape(1,-1)
    lr: np.ndarray = dt * l.reshape(-1, 1) + r

    # Obtain ETDRK4 coefficients
    q: np.ndarray = dt * np.mean(
        (np.exp(lr/2)-1)/lr,
        axis=1
    )
    f1: np.ndarray = dt * np.mean(
        (-4 - lr + np.exp(lr) * (4 - 3*lr + lr**2)) / lr**3,
        axis=1
    )
    f2: np.ndarray = dt * np.mean(
        (2 + lr + np.exp(lr) * (-2+lr)) / lr**3,
        axis=1
    )
    f3: np.ndarray = dt * np.mean(
        (-4 - 3*lr - lr**2 + np.exp(lr)*(4-lr)) / lr**3,
        axis=1
    )

    # Define multiplier for nonlinear operator
    g: np.ndarray = -0.5 * 1j * k

    # Initialise solution
    u = np.empty((t.shape[0], total_x_pts))
    u[0,:] = u0
    v: np.ndarray = np.fft.fft(u0)

    # Time step loop
    for tIdx in range(t.shape[0]-1):

        Nv: np.ndarray = g * np.fft.fft(np.fft.ifft(v).real**2)

        a: np.ndarray = e2*v + q*Nv
        Na: np.ndarray = g * np.fft.fft(np.fft.ifft(a).real**2)

        b: np.ndarray = e2*v + q*Na
        Nb: np.ndarray = g * np.fft.fft(np.fft.ifft(b).real**2)

        c: np.ndarray = e2*a + q*(2*Nb - Nv)
        Nc: np.ndarray = g * np.fft.fft(np.fft.ifft(c).real**2)

        v = e*v + f1*Nv + f2*2*(Na+Nb) + f3*Nc

        u[tIdx+1,:] = np.fft.ifft(v).real.reshape(-1)

    # Reshape u into [x_1, ..., x_D, t, u] shape
    return u.T.reshape((*u.T.shape, 1)), t, x


def inviscid_burgers(
        tFinal: float, dt: float,
        total_x_pts: int, xbounds: Iterable,
        A: float = 1000, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of the analytical inviscid Burgers equation using the
    initial conditions specified in (Messenger and Bortz, 2021).

    Args:
        tFinal (float): Final time in computation.
        dt (float): Time step.
        total_x_pts (int): Total number of spatial grid points.
        xbounds (Iterable): Bounds of spatial domain.
        A (float): Max value of ramp initial condition.
        alpha (float): Gradient of ramp initial condition.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Returns solution with time 
            and spatial grid point vectors.
    """
    
    # Check if `xbounds` is specified correctly
    assert len(xbounds) == 2, (
        "xbounds should be provided in the form of (xmin, xmax)")
    assert xbounds[1] > xbounds[0], (
        "The maximum bound should be higher than the minimum bound!")
    
    # Create spatial grid
    x: np.ndarray = np.linspace(xbounds[0], xbounds[1], total_x_pts)
    
    # Create temporal grid
    t: np.ndarray = np.arange(0, tFinal+dt, dt)
    
    # Initialise solution grid
    u: np.ndarray = np.zeros((t.shape[0], total_x_pts))

    # Get analytical solution of a ramp initial condition
    for t_idx, t_val in enumerate(t):
        for x_idx, x_val in enumerate(x):
            if t_val >= max(x_val/A + 1/alpha, 2*x_val/A + 1/alpha):
                u[t_idx, x_idx] = A
            elif x_val <= 0 and x_val > A*(t_val - 1/alpha):
                u[t_idx, x_idx] = -alpha*x_val/(1-alpha*t_val)

    # Reshape u into [x_1, ..., x_D, t, u] shape
    return u.T.reshape((*u.T.shape, 1)), t, x


def shallow_water(
        t: float, u: np.ndarray,
        x_grids: list[np.ndarray],
        g: float = 9.81,
        dx_func: Callable = FiniteDifference(
            6, axis=0, periodic=True)._differentiate,
        dy_func: Callable = FiniteDifference(
            6, axis=1, periodic=True)._differentiate
) -> np.ndarray:
    """
    2D shallow water equation. Variable states are [u, v, h].

    Args:
        t (float): Time variable t
        u (np.ndarray): Current u(x, t) state
        x_grids (list[np.ndarray]): Spatial grid points that the system is on.
        g (float, optional): Gravitational acceleration. Defaults to 9.81.
        dx_func (Callable, optional): Finite difference derivative method.
            Defaults to FiniteDifference( 6, axis=0, periodic=True)._differentiate.
        dy_func (Callable, optional): Finite difference derivative.
            Defaults to FiniteDifference( 6, axis=1, periodic=True)._differentiate.

    Returns:
        np.ndarray: Returns [dudt, dvdt, dhdt].
    """

    # Find spatial derivatives of u
    dudx: np.ndarray = dx_func(u[..., 0], x_grids[0])
    dudy: np.ndarray = dy_func(u[..., 0], x_grids[1])

    # Find spatial derivatives of v
    dvdx: np.ndarray = dx_func(u[..., 1], x_grids[0])
    dvdy: np.ndarray = dy_func(u[..., 1], x_grids[1])

    # Find spatial derivatives of h
    dhdx: np.ndarray = dx_func(u[..., 2], x_grids[0])
    dhdy: np.ndarray = dy_func(u[..., 2], x_grids[1])

    # Find time derivatives of each state
    dudt: np.ndarray = -u[..., 0] * dudx - u[..., 1] * dudy - g * dhdx
    dvdt: np.ndarray = -u[..., 0] * dvdx - u[..., 1] * dvdy - g * dhdy
    dhdt: np.ndarray = -u[..., 2] * dudx - u[..., 0] * dhdx \
        - u[..., 2] * dvdy - u[..., 1] * dhdy
    
    return np.stack((dudt, dvdt, dhdt), axis=-1)


def reaction_diffusion_2d(
        t: float, u: np.ndarray,
        x_grids: list[np.ndarray],
        d2x_func: Callable = FiniteDifference(
            6, 2, axis=0, periodic=True)._differentiate,
        d2y_func: Callable = FiniteDifference(
            6, 2, axis=1, periodic=True)._differentiate,
        F_func: Callable = lambda u, v:
            -u*v**2 - u**3 + v**3 + v*u**2 + u,
        G_func: Callable = lambda u, v:
            v - u*v**2 - u**3 - v**3 - v*u**2
) -> np.ndarray:
    """
    2D reaction-diffusion equation. Variable states are [u, v]. Equation
    implemented based on the application from (Messenger and Bortz, 2021).

    Args:
        t (float): Time variable t
        u (np.ndarray): Current u(x, t) state
        x_grids (list[np.ndarray]): Spatial grid points that the system is on.
        d2x_func (Callable, optional): Finite difference derivative method.
            Defaults to FiniteDifference( 6, 2, axis=0, periodic=True)._differentiate.
        d2y_func (Callable, optional): Finite difference derivative method.
            Defaults to FiniteDifference( 6, 2, axis=1, periodic=True)._differentiate.
        F_func (Callable, optional): Net production rate of u.
        G_func (Callable, optional): Net production rate of v.

    Returns:
        np.ndarray: Returns [dudt, dvdt].
    """
    
    # Find derivative of u
    d2udx2: np.ndarray = d2x_func(u[..., 0], x_grids[0])
    d2udy2: np.ndarray = d2y_func(u[..., 0], x_grids[1])

    # Find derivative of v
    d2vdx2: np.ndarray = d2x_func(u[..., 1], x_grids[0])
    d2vdy2: np.ndarray = d2y_func(u[..., 1], x_grids[1])

    # Find the time derivative
    dudt: np.ndarray = 0.1 * d2udx2 + 0.1 * d2udy2 \
        + F_func(u[..., 0], u[..., 1])
    dvdt: np.ndarray = 0.1 * d2vdx2 + 0.1 * d2vdy2 \
        + G_func(u[..., 0], u[..., 1])
    
    return np.stack((dudt, dvdt), axis=-1)

class CylinderParam(NamedTuple):

    # Mesh domain information
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    dx: float
    dy: float
    x: jnp.ndarray
    y: jnp.ndarray
    XY_grid: list[jnp.ndarray]

    # Flow parameters
    Re: float
    Pr: float
    gamma: float
    speed_sound: float
    Cp: float
    mu: float
    k: float
    
    # Freestream inflow parameters
    rho_in: float
    u_in: float
    T_in: float
    mach_in: float

    # Cylinder parameters
    cy_d: float
    cy_x: float
    cy_y: float
    eps: jnp.ndarray

    # Simulation parameters
    cfl: float
    dt: float


class CylinderFlow:
    
    def __init__(
            self,
            Lx: float = 20., Ly: float = 12.,
            Nx: int = 513, Ny: int = 257,
            cfl: float = 0.25, Re: float = 200, Pr: float = 0.7,
            rho_in: float = 1, mach_in: float = 0.2,
            gamma: float = 1.4, speed_sound: float = 1.0, Cp: float = 1.0,
            cy_d: float = 1, cy_x: Optional[float] = None,
            cy_y: Optional[float] = None,
    ):

        # Obtain inflow velocity
        u_in: float = mach_in * speed_sound

        # Compute inflow temperature
        T_in: float = speed_sound**2 / (gamma-1)

        # Compute viscosity
        mu: float = (rho_in * u_in * cy_d)/Re

        # Compute thermal conductivity
        k: float = mu*Cp/Pr

        # Define mesh
        x: jnp.ndarray = jnp.linspace(0, Lx, Nx)
        y: jnp.ndarray = jnp.linspace(0, Ly, Ny)
        XY_grid: list[jnp.ndarray] = jnp.meshgrid(x, y, indexing='ij')

        # Obtain mesh grid information
        dx: float = float(x[1] - x[0])
        dy: float = float(y[1] - y[0])

        # Simulation parameters
        dt: float = cfl * dx

        # Define cylinder position if otherwise not given
        if cy_x is None:
            cy_x = 0.25 * Lx + x[0]
        if cy_y is None:
            cy_y = 0.5 * Ly + y[0]

        # Define cylinder placement on the mesh grid
        eps: jnp.ndarray = jnp.zeros((Nx, Ny))
        eps = eps.at[jnp.sqrt(
            (XY_grid[0] - cy_x)**2
            + (XY_grid[1] - cy_y)**2
        ) < (cy_d/2)].set(1.0)

        # Store parameters
        self.params: CylinderParam = CylinderParam(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dx=dx, dy=dy,
            x=x, y=y, XY_grid=XY_grid,
            Re=Re, Pr=Pr,
            gamma=gamma, speed_sound=speed_sound, Cp=Cp, mu=mu, k=k,
            rho_in=rho_in, u_in=u_in, T_in=T_in, mach_in=mach_in,
            cy_d=cy_d, cy_x=cy_x, cy_y=cy_y, eps=eps,
            cfl=cfl, dt=dt
        )


    def simulate(
            self,
            tFinal: float, u0: np.ndarray = None,
            progress_bar: bool = True, n_checkpoint: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:

        # Compute time vector
        total_steps: int = int(round(tFinal/self.params.dt))

        # Obtain initial state
        if u0 is None:
            u: jnp.ndarray = self._get_u0()
        else:
            u: jnp.ndarray = jnp.array(u0)

        # Determine main loop block size
        if n_checkpoint is None:
            block_size: int = total_steps
            last_block_size: int = 0
        else:
            block_size: int = n_checkpoint
            last_block_size: int = total_steps % block_size

        # Define total complete blocks
        total_blocks: int = int(np.floor(total_steps/block_size))

        # Initialise solution array
        total_u: int = 1 + total_blocks + int(last_block_size>0)
        u_hist: np.ndarray = np.empty((
            self.params.Nx, self.params.Ny, total_u, 4
        ))
        u_hist[..., 0, :] = np.array(u)

        # Define returned time vector
        ret_t: list[float] = [0.0]

        # Time step through the blocks
        for idx in tqdm(range(total_blocks), disable=not progress_bar):
            u = self._run_block(u, block_size, self.params)
            if jnp.any(jnp.isnan(u)):
                warnings.warn('NaN found in computation. Stopping early.')
                return np.array(ret_t), u_hist[..., :idx+1, :]
            u_hist[..., idx+1, :] = np.array(u)
            ret_t.append((idx+1)*self.params.dt*block_size)

        # Step through the last block if remaining
        if last_block_size > 0:
            u = self._run_block(u, last_block_size, self.params)
            u_hist[..., -1, :] = np.array(u)
            ret_t.append(tFinal)

        return np.array(ret_t), u_hist
    
    
    @staticmethod
    @jax.jit
    def get_vorticity(u: jnp.ndarray, params: CylinderParam) -> jnp.ndarray:

        # Obtain u and v
        state_u: jnp.ndarray = u[..., 1] / u[..., 0]
        state_v: jnp.ndarray = u[..., 2] / u[..., 0]

        # Compute derivatives
        dv_dx: jnp.ndarray = CylinderFlow.ddx(state_v, params.dx)
        du_dy: jnp.ndarray = CylinderFlow.ddy(state_u, params.dy)

        return dv_dx - du_dy
    

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _run_block(
            u: jnp.ndarray, block_size: int,
            params: CylinderParam
    ) -> jnp.ndarray:
        
        def step(u_i: jnp.ndarray, _):
            return CylinderFlow._rk4_step(u_i, params), None
        
        u_final, _ = jax.lax.scan(step, u, None, length=block_size)
        return u_final


    def _get_u0(self) -> jnp.ndarray:

        ini_r: jnp.ndarray = self.params.rho_in * jnp.ones(
            (self.params.Nx, self.params.Ny)
        )
        ini_u: jnp.ndarray = self.params.u_in * jnp.ones(
            (self.params.Nx, self.params.Ny)
        )
        ini_v: jnp.ndarray = 0.01 * (
            jnp.sin(4*jnp.pi*self.params.XY_grid[0]/self.params.Lx)
            + jnp.sin(7*jnp.pi*self.params.XY_grid[0]/self.params.Lx)
            * jnp.exp(-(self.params.XY_grid[1] - self.params.Ly/2)**2)
        )
        ini_e: jnp.ndarray = self.params.Cp/self.params.gamma * self.params.T_in \
            + 0.5 * (ini_u**2 + ini_v**2)

        return jnp.stack([
            ini_r,
            ini_r * ini_u,
            ini_r * ini_v,
            ini_r * ini_e
        ], axis=-1)


    @staticmethod
    @jax.jit
    def _compute_dt(u: jnp.ndarray, params: CylinderParam) -> jnp.ndarray:

        # ------------------ Compute primitive states ------------------ #

        # Compute horizontal velocity
        state_u: jnp.ndarray = u[..., 1] / u[..., 0]

        # Compute vertical velocity
        state_v: jnp.ndarray = u[..., 2] / u[..., 0]

        # Compute pressure
        state_p: jnp.ndarray = (params.gamma-1) * (
            u[..., 3]
            - 0.5 * (u[..., 1] * state_u + u[..., 2] * state_v)
        )

        # Compute temperature
        state_T: jnp.ndarray = params.gamma/(params.gamma-1) * state_p \
            / u[..., 0] / params.Cp

        # Compute energy
        state_e: jnp.ndarray = params.Cp/params.gamma * state_T + 0.5*(
            state_u**2 + state_v**2
        )


        # --------------- Enforce inflow boundary conditions --------------- #

        state_u = state_u.at[0,:].set(params.u_in)
        state_v = state_v.at[0,:].set(0.01 * (
            jnp.sin(4*jnp.pi/params.Nx)
            + jnp.sin(7*jnp.pi/params.Nx)
            * jnp.exp(-(params.y - params.Ly/2)**2)
        ))
        state_T = state_T.at[0,:].set(params.T_in)
        state_e = state_e.at[0,:].set(1/params.gamma * state_T[0,:] + 0.5*(
            state_u[0,:]**2 + state_v[0,:]**2
        ))

        u = u.at[0,:,0].set(params.rho_in)
        u = u.at[0,:,1].set(params.rho_in * state_u[0,:])
        u = u.at[0,:,2].set(params.rho_in * state_v[0,:])
        u = u.at[0,:,3].set(params.rho_in * state_e[0,:])


        # ------------------ Compute flux time derivatives ------------------ #

        # Compute time derivative of mass flux
        dru_dx: jnp.ndarray = CylinderFlow.ddx(u[..., 1], params.dx)
        drv_dy: jnp.ndarray = CylinderFlow.ddy(u[..., 2], params.dy)

        dr_dt: jnp.ndarray = -dru_dx - drv_dy

        # Compute time derivative of horizontal momentum flux
        dp_dx: jnp.ndarray = CylinderFlow.ddx(state_p, params.dx)

        druu_dx: jnp.ndarray = CylinderFlow.ddx(u[..., 1] * state_u, params.dx)
        druv_dy: jnp.ndarray = CylinderFlow.ddy(u[..., 1] * state_v, params.dy)

        d2u_dx2: jnp.ndarray = CylinderFlow.d2dx2(state_u, params.dx)
        d2u_dy2: jnp.ndarray = CylinderFlow.d2dy2(state_u, params.dy)
        d2v_dxdy: jnp.ndarray = CylinderFlow.d2dxdy(
            state_v, params.dx, params.dy)

        dru_dt: jnp.ndarray = -dp_dx - druu_dx - druv_dy + params.mu * (
            4/3 * d2u_dx2 + d2u_dy2 + 1/3 * d2v_dxdy
        ) - params.eps * u[..., 1]

        # Compute time derivative of vertical momentum flux
        dp_dy: jnp.ndarray = CylinderFlow.ddy(state_p, params.dy)

        druv_dx: jnp.ndarray = CylinderFlow.ddx(u[..., 1] * state_v, params.dx)
        drvv_dy: jnp.ndarray = CylinderFlow.ddy(u[..., 2] * state_v, params.dy)

        d2v_dx2: jnp.ndarray = CylinderFlow.d2dx2(state_v, params.dx)
        d2v_dy2: jnp.ndarray = CylinderFlow.d2dy2(state_v, params.dy)
        d2u_dxdy: jnp.ndarray = CylinderFlow.d2dxdy(
            state_u, params.dx, params.dy)

        drv_dt: jnp.ndarray = -dp_dy - druv_dx - drvv_dy + params.mu * (
            d2v_dx2 + 4/3 * d2v_dy2 + 1/3 * d2u_dxdy
        ) - params.eps * u[..., 2]

        # Compute time derivative of energy flux
        dreu_dx: jnp.ndarray = CylinderFlow.ddx(u[..., 3] * state_u, params.dx)
        dpu_dx: jnp.ndarray = CylinderFlow.ddx(state_p * state_u, params.dx)

        drev_dy: jnp.ndarray = CylinderFlow.ddy(u[..., 3] * state_v, params.dy)
        dpv_dy: jnp.ndarray = CylinderFlow.ddy(state_p * state_v, params.dy)

        d2T_dx2: jnp.ndarray = CylinderFlow.d2dx2(state_T, params.dx)
        d2T_dy2: jnp.ndarray = CylinderFlow.d2dy2(state_T, params.dy)

        du_dx: jnp.ndarray = CylinderFlow.ddx(state_u, params.dx)
        du_dy: jnp.ndarray = CylinderFlow.ddy(state_u, params.dy)
        dv_dx: jnp.ndarray = CylinderFlow.ddx(state_v, params.dx)
        dv_dy: jnp.ndarray = CylinderFlow.ddy(state_v, params.dy)

        dre_dt: jnp.ndarray = -dreu_dx - dpu_dx - drev_dy - dpv_dy \
            + params.k * (d2T_dx2 + d2T_dy2) \
            + params.mu * state_u * (4/3 * d2u_dx2 + d2u_dy2 + 1/3 * d2v_dxdy) \
            + params.mu * state_v * (d2v_dx2 + 4/3 * d2v_dy2 + 1/3 * d2u_dxdy) \
            + 2 * params.mu * (du_dx**2 + dv_dy**2) \
            - 2/3 * params.mu * (du_dx + dv_dy)**2 \
            + params.mu * (du_dy + dv_dx)**2
        

        # --------------- Enforce outflow boundary conditions --------------- #

        flux_out = -(params.u_in / params.dx) * (u[-1,:,:] - u[-2,:,:])
        dr_dt = dr_dt.at[-1,:].set(flux_out[:,0])
        dru_dt = dru_dt.at[-1,:].set(flux_out[:,1])
        drv_dt = drv_dt.at[-1,:].set(flux_out[:,2])
        dre_dt = dre_dt.at[-1,:].set(flux_out[:,3])


        return jnp.stack((dr_dt, dru_dt, drv_dt, dre_dt), axis=-1)
    

    @staticmethod
    @jax.jit
    def _rk4_step(u: jnp.ndarray, params: CylinderParam) -> jnp.ndarray:

        k1: jnp.ndarray = CylinderFlow._compute_dt(u, params)
        k2: jnp.ndarray = CylinderFlow._compute_dt(
            u + 0.5*params.dt*k1, params
        )
        k3: jnp.ndarray = CylinderFlow._compute_dt(
            u + 0.5*params.dt*k2, params
        )
        k4: jnp.ndarray = CylinderFlow._compute_dt(
            u + params.dt*k3, params
        )

        return u + params.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    
    @staticmethod
    @jax.jit
    def ddx(u: jnp.ndarray, dx: float) -> jnp.ndarray:

        interior: jnp.ndarray = (u[2:,:] - u[:-2,:]) / (2*dx)
        left: jnp.ndarray = (-3*u[0:1,:] + 4*u[1:2,:] - u[2:3,:]) / (2*dx)
        right: jnp.ndarray = (3*u[-1:,:] - 4*u[-2:-1,:] + u[-3:-2,:]) / (2*dx)

        return jnp.concatenate([left, interior, right], axis=0)
    

    @staticmethod
    @jax.jit
    def d2dx2(u: jnp.ndarray, dx: float) -> jnp.ndarray:

        interior: jnp.ndarray = (u[2:,:] - 2*u[1:-1,:] + u[:-2,:]) / dx**2
        left: jnp.ndarray = (
            2*u[0:1,:]
            - 5*u[1:2,:]
            + 4*u[2:3,:]
            - u[3:4,:]
        ) / dx**2
        right: jnp.ndarray = (
            2*u[-1:,:]
            - 5*u[-2:-1,:]
            + 4*u[-3:-2,:]
            - u[-4:-3,:]
        ) / dx**2

        return jnp.concatenate([left, interior, right], axis=0)
    

    @staticmethod
    @jax.jit
    def ddy(u: jnp.ndarray, dy: float) -> jnp.ndarray:

        return (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2*dy)
    

    @staticmethod
    @jax.jit
    def d2dy2(u: jnp.ndarray, dy: float) -> jnp.ndarray:

        return (
            jnp.roll(u, -1, axis=1)
            - 2. * u
            + jnp.roll(u, 1, axis=1)
        ) / dy**2
    
    
    @staticmethod
    @jax.jit
    def d2dxdy(u: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:

        interior: jnp.ndarray = (u[2:,:] - u[:-2,:]) / (2*dx)
        left: jnp.ndarray = (-3*u[0:1,:] + 4*u[1:2,:] - u[2:3,:]) / (2*dx)
        right: jnp.ndarray = (3*u[-1:,:] - 4*u[-2:-1,:] + u[-3:-2,:]) / (2*dx)

        dudx: jnp.ndarray = jnp.concatenate([left, interior, right], axis=0)

        return (
            jnp.roll(dudx, -1, axis=1)
            - jnp.roll(dudx, 1, axis=1)
        ) / (2*dy)
    

class AnnulusParam(NamedTuple):

    # Mesh domain information
    R1: float
    R2: float
    Ntheta: int
    Nr: int
    dtheta: float
    dr: float
    theta: jnp.ndarray
    r: jnp.ndarray
    ThetaR_grid: list[jnp.ndarray]

    # Flow parameters
    Ra: float
    Pr: float

    # Store flow boundary conditions
    T1: float
    T2: float

    # Simulation parameters
    cfl: float
    dt: float

    # Pre-computed Poisson inverse matrices
    po_inv: jnp.ndarray

    # Pre-compute wall temperature boundary condition
    bc_temp: jnp.ndarray


class AnnulusFlow:
    
    def __init__(
            self,
            R1: float = 0.4, R2: float = 0.5,
            T1: float = 1., T2: float = 0.,
            Ntheta: int = 512, Nr: int = 32,
            cfl: float = 0.5, Ra: float = 100e6, Pr: float = 4
    ):
        """
        Initialize the annulus flow simulation.

        Args:
            R1 (float, optional): Inner radius. Defaults to 0.4.
            R2 (float, optional): Outer radius. Defaults to 0.5.
            T1 (float, optional): Wall tempearture at the top. Defaults to 1.0.
            T2 (float, optional): Wall temperature at the bottom. Defaults to 0.0.
            Ntheta (int, optional): Number of azimuthal grid points. Defaults to 512.
            Nr (int, optional): Number of radial grid points. Defaults to 32.
            cfl (float, optional): CFL number. Defaults to 0.5.
            Ra (float, optional): Rayleigh number. Defaults to 100e6.
            Pr (float, optional): Prandtl number. Defaults to 4.
        """

        # Define mesh
        theta: jnp.ndarray = jnp.linspace(0, 2*jnp.pi, Ntheta, endpoint=False)
        r: jnp.ndarray = jnp.linspace(R1, R2, Nr)
        ThetaR_grid: list[jnp.ndarray] = jnp.meshgrid(theta, r, indexing='ij')

        # Obtain mesh grid information
        dtheta: float = float(theta[1] - theta[0])
        dr: float = float(r[1] - r[0])

        # Simulation parameters
        char_u: float = jnp.sqrt(Ra)
        dt_adv: float = cfl * min(dtheta*R1, dr) / char_u
        dt_diff: float = cfl * min(dtheta*R1, dr)**2 / 2 / Pr
        dt: float = min(dt_adv, dt_diff)

        # Pre-compute the required Poisson matrix inverse
        po_inv: jnp.ndarray = AnnulusFlow._init_Poisson(
            Ntheta, Nr, dr, r
        )

        # Pre-compute the wall temperature boundary condition
        bc_temp: jnp.ndarray = T1 + (T2-T1)/2 * (
            jnp.sin(theta)+1
        ) * jnp.ones((2, Ntheta))

        # Store parameters
        self.params: AnnulusParam = AnnulusParam(
            R1=R1, R2=R2, Ntheta=Ntheta, Nr=Nr, dtheta=dtheta, dr=dr,
            theta=theta, r=r, ThetaR_grid=ThetaR_grid,
            Ra=Ra, Pr=Pr,
            T1=T1, T2=T2,
            cfl=cfl, dt=dt,
            po_inv=po_inv, bc_temp=bc_temp
        )


    def simulate(
            self,
            tFinal: float,
            u0: Optional[jnp.ndarray] = None, q0: Optional[float] = None,
            progress_bar: bool = True, n_checkpoint: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the annulus flow simulation.

        Args:
            tFinal (float): Total physical time to simulate.
            u0 (Optional[jnp.ndarray], optional): Initial state conditions to
                begin the run from, must take the shape (Ntheta, Nr, 2). If None,
                generates a noise injected steady state flow.
            q0 (Optional[float], optional): Initial outer wall circulation flux.
                Defaults to None.
            progress_bar (bool, optional): Whether to show progress bar.
                Defaults to True.
            n_checkpoint (Optional[int], optional): The interval at which to
                store simulation results. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Returns tuple of 1D time
                vector, state (vorticity and temperature) history with shape
                (Ntheta, Nr, 2), and 1D vector of outer wall circulation flux.
        """

        # Compute time vector
        total_steps: int = int(round(tFinal/self.params.dt))

        # Obtain initial state
        if (u0 is None) or (q0 is None):
            u, q = self._get_u0()
            if (u0 is not None) or (q0 is not None):
                warnings.warn(
                    'Both u0 and q0 must be provided! Defaulting to default' \
                    + ' initial condition.'
                )
        else:
            u = u0
            q = q0

        # Determine main loop block size
        if n_checkpoint is None:
            block_size: int = total_steps
            last_block_size: int = 0
        else:
            block_size: int = n_checkpoint
            last_block_size: int = total_steps % block_size

        # Define total complete blocks
        total_blocks: int = int(np.floor(total_steps/block_size))

        # Initialise solution array
        total_u: int = 1 + total_blocks + int(last_block_size>0)
        u_hist: np.ndarray = np.empty((
            self.params.Ntheta, self.params.Nr, total_u, 2
        ))
        u_hist[..., 0, :] = np.array(u)

        # Initialise q array
        q_hist: list[float] = list()
        q_hist.append(q)

        # Define returned time vector
        ret_t: list[float] = [0.0]

        try:

            # Time step through the blocks
            for idx in tqdm(range(total_blocks), disable=not progress_bar):
                u, q = self._run_block((u, q), block_size, self.params)
                if jnp.any(jnp.isnan(u)) or jnp.isnan(q):
                    warnings.warn('NaN found in computation. Stopping early.')
                    return (
                        np.array(ret_t),
                        u_hist[..., :idx+1, :],
                        np.array(q_hist)
                    )
                u_hist[..., idx+1, :] = np.array(u)
                ret_t.append((idx+1)*self.params.dt*block_size)
                q_hist.append(q)

            # Step through the last block if remaining
            if last_block_size > 0:
                u, q = self._run_block((u, q), last_block_size, self.params)
                u_hist[..., -1, :] = np.array(u)
                ret_t.append(tFinal)
                q_hist.append(q)

        except KeyboardInterrupt:

            warnings.warn('User interruption detected. Stopping early.')
            return (np.array(ret_t), u_hist[..., :idx+1, :], np.array(q_hist))

        return np.array(ret_t), u_hist, np.array(q_hist)
    

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _run_block(
            state: tuple[jnp.ndarray, float], block_size: int,
            params: AnnulusParam
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Jax JIT time simulation block to run through a set number of simulation
        steps.

        Args:
            state (tuple[jnp.ndarray, float]): Current state containing u and q.
            block_size (int): Number of simulation steps to run.
            params (AnnulusParam): Simulation parameters.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Tuple of u and q from the ran
                simulation.
        """
        
        def step(state_i: jnp.ndarray, _):
            return AnnulusFlow._step(state_i, params), None
        
        state_final = jax.lax.scan(step, state, None, length=block_size)[0]
        return state_final


    def _get_u0(self) -> tuple[jnp.ndarray, float]:
        """
        Generates the default initial condition which is the steady state flow
        with additive noise injected.

        Returns:
            tuple[jnp.ndarray, float]: Returns initial condition u and q.
        """

        u_shape = (self.params.Ntheta, self.params.Nr)

        ini_w: jnp.ndarray = jnp.zeros(u_shape)
        
        T_avg = 0.5 * (self.params.T1 + self.params.T2)
        T_diff = 0.5 * (self.params.T2 - self.params.T1)
        ini_T: jnp.ndarray = jnp.ones(u_shape) * (
            T_diff*(
                jnp.sin(self.params.theta)
            ) + T_avg
        ).reshape((-1,1)) + np.random.normal(0, 0.1 * np.abs(T_diff), u_shape)

        return jnp.stack([ini_w, ini_T], axis=-1), 0.0


    @staticmethod
    @jax.jit
    def _compute_dt(
        u: jnp.ndarray, q: float, params: AnnulusParam
    ) -> tuple[jnp.ndarray, float]:
        """
        Evalutes the time derivatives of each state variable (vorticity and
        temperature). Also computes the time derivative of the outer wall
        circulation flux.

        Args:
            u (jnp.ndarray): Current state array (vorticity and temperature).
            q (float): Current outer wall circulation flux.
            params (AnnulusParam): Simulation parameters.

        Returns:
            tuple[jnp.ndarray, float]: Tuple of state time derivatives and
                outer wall circulation flux derivative.
        """

        # -------------- Compute common variables -------------- #

        # 1/r
        r_inv: jnp.ndarray = params.ThetaR_grid[1]**-1
        
        # 1/r**2
        r2_inv: jnp.ndarray = params.ThetaR_grid[1]**-2


        # -------------- Solve streamfunction and enforce B.C. -------------- #

        # Obtain B.C. compliant streamfunction from current step
        psi = AnnulusFlow._compute_streamfunction(u, q, params)

        # Apply Thom's method
        u = u.at[:, 0, 0].set(-2.0 * psi[:, 1] / params.dr**2)
        u = u.at[:, -1, 0].set(2.0 * (q - psi[:, -2]) / params.dr**2)


        # -------------- Inner wall temperature Neumann B.C. -------------- #

        u = u.at[:, 0, 1].set(
            (4.0 * u[:, 1, 1] - u[:, 2, 1]) / 3.0
        )


        # -------------- Compute vorticity derivative -------------- #

        dpsi_dtheta: jnp.ndarray = AnnulusFlow.ddtheta(
            psi, params.dtheta)
        dw_dr: jnp.ndarray = AnnulusFlow.ddr(u[..., 0], params.dr)

        dpsi_dr: jnp.ndarray = AnnulusFlow.ddr(psi, params.dr)
        dw_dtheta: jnp.ndarray = AnnulusFlow.ddtheta(u[..., 0], params.dtheta)

        d2w_dr2: jnp.ndarray = AnnulusFlow.d2dr2(u[..., 0], params.dr)
        d2w_dtheta2: jnp.ndarray = AnnulusFlow.d2dtheta2(
            u[..., 0], params.dtheta)
        
        dT_dr: jnp.ndarray = AnnulusFlow.ddr(u[..., 1], params.dr)
        dT_dtheta: jnp.ndarray = AnnulusFlow.ddtheta(u[..., 1], params.dtheta)

        dw_dt = -r_inv * (
            dpsi_dtheta * dw_dr - dpsi_dr * dw_dtheta
        ) + params.Pr * (
            d2w_dr2
            + r_inv * dw_dr
            + r2_inv * d2w_dtheta2
        ) + params.Pr * params.Ra * (
            dT_dr * jnp.cos(params.ThetaR_grid[0])
            - r_inv * dT_dtheta * jnp.sin(params.ThetaR_grid[0])
        )


        # -------------- Compute temperature derivative -------------- #

        d2T_dr2: jnp.ndarray = AnnulusFlow.d2dr2(u[..., 1], params.dr)
        d2T_dtheta2: jnp.ndarray = AnnulusFlow.d2dtheta2(
            u[..., 1], params.dtheta)
        
        dT_dt: jnp.ndarray = -r_inv * (
            dpsi_dtheta * dT_dr - dpsi_dr * dT_dtheta
        ) + d2T_dr2 + r_inv * dT_dr + r2_inv * d2T_dtheta2


        # -------------- Enforce wall Dirichlet B.C. -------------- #

        dw_dt = dw_dt.at[:, 0].set(0.0)
        dw_dt = dw_dt.at[:, -1].set(0.0)

        dT_dt = dT_dt.at[:, -1].set(0.0)


        # -------------- Compute volumetric flux -------------- #

        dq_dt = AnnulusFlow._compute_flowrate(u, psi, params)


        return jnp.stack((dw_dt, dT_dt), axis=-1), dq_dt
    

    @staticmethod
    @jax.jit
    def _step(
            state: tuple[jnp.ndarray, float], params: AnnulusParam
    ) -> jnp.ndarray:
        """
        Executes a single time step using RK4 time integration method.

        Args:
            state (tuple[jnp.ndarray, float]): Current state containing u and q.
            params (AnnulusParam): Simulation parameters.

        Returns:
            jnp.ndarray: The next state (u, q).
        """
        
        # Unpack state and volumetric angular flux
        u, q = state


        # -------------- RK4 Time-step marching -------------- #

        k1_u, k1_q = AnnulusFlow._compute_dt(u, q, params)
        k2_u, k2_q = AnnulusFlow._compute_dt(
            u + 0.5 * params.dt * k1_u,
            q + 0.5 * params.dt * k1_q,
            params
        )
        k3_u, k3_q = AnnulusFlow._compute_dt(
            u + 0.5 * params.dt * k2_u,
            q + 0.5 * params.dt * k2_q,
            params
        )
        k4_u, k4_q = AnnulusFlow._compute_dt(
            u + params.dt * k3_u,
            q + params.dt * k3_q,
            params
        )
        
        u_next = u + (params.dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        q_next = q + (params.dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)


        # -------------- Enforce wall Dirichlet B.C. -------------- #

        u_next = u_next.at[:, -1, 1].set(params.bc_temp[1])

        return u_next, q_next


    @staticmethod
    @jax.jit
    def _compute_streamfunction(
            u: jnp.ndarray, q: float, params: AnnulusParam
    ) -> jnp.ndarray:
        """
        Solve the Poisson equation to compute streamfunction using a pseudo-
        spectral method.

        Args:
            u (jnp.ndarray): Current state array (vorticity and temperature).
            q (float): Current outer wall streamfunction boundary condition.
            params (AnnulusParam): Simulation parameters.

        Returns:
            jnp.ndarray: Streamfunction at the current time.
        """

        # Compute FFT along theta axis
        rhs: jnp.ndarray = jnp.fft.fft(u[..., 0], axis=0)

        # Set boundary conditions
        rhs = rhs.at[:,0].set(0.0)
        rhs = rhs.at[:,-1].set(0.0)
        rhs = rhs.at[0,-1].set(q * params.Ntheta)

        # Solve the Poisson inverse problem
        psi_hat = jnp.einsum('kij, kj -> ki', params.po_inv, rhs)
        
        return jnp.fft.ifft(psi_hat, axis=0).real
    

    @staticmethod
    @jax.jit
    def _compute_vel(
            psi: jnp.ndarray, params: AnnulusParam
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the azimuthal and radial velocity components from the
        streamfunction.

        Args:
            psi (jnp.ndarray): Streamfunction.
            params (AnnulusParam): Simulation parameters.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Tuple of azimuthal and radial
                velocities.
        """
        
        u_theta: jnp.ndarray = -AnnulusFlow.ddr(psi, params.dr)
        u_r: jnp.ndarray = AnnulusFlow.ddtheta(
            psi, params.dtheta
        ) / params.ThetaR_grid[1]

        return u_theta, u_r
    

    @staticmethod
    @jax.jit
    def get_states(
            u: jnp.ndarray, q: jnp.ndarray, params: AnnulusParam
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray,
        jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """
        Using the state history matrix, obtain the flow velocities as well as
        the low-dimensional states proposed in (Huang et al, 2023).

        Args:
            u (jnp.ndarray): State matrix over time, expecting shape of
                (Ntheta, Nr, Nt, 2).
            q (jnp.ndarray): Corresponding time history vector of outer wall
                cirulation flux.
            params (AnnulusParam): Simulation parameters.

        Returns:
            tuple:
                u_theta (jnp.ndarray): Azimuthal velocity.
                u_r (jnp.ndarray): Radial velocity.
                psi (jnp.ndarray): Streamfunction.
                ang_mom (jnp.ndarray): Time history of average angular momentum
                    around the geometry.
                com_X (jnp.ndarray): X-coordinate of the fluid centre of mass.
                com_Y (jnp.ndarray): Y-coordinate of the fluid centre of mass.
        """
        
        def vel_step(u_i, q_i):
            psi = AnnulusFlow._compute_streamfunction(u_i, q_i, params)
            u_theta, u_r = AnnulusFlow._compute_vel(psi, params)
            return u_theta, u_r, psi
        
        u_theta, u_r, psi = jax.vmap(
            vel_step,
            in_axes=(2, 0),
            out_axes=2
        )(u, q)

        # Area of annulus
        A0 = jnp.pi * (params.R2**2 - params.R1**2)

        # Average angular momentum
        ang_mom = jnp.trapezoid(
            jnp.trapezoid(
                params.ThetaR_grid[1][..., None]**2 * u_theta,
                dx=params.dr, axis=1
            ),
            dx=params.dtheta, axis=0
        )/A0

        # Fluid horizontal centre of mass
        com_X = -jnp.trapezoid(
            jnp.trapezoid(
                params.ThetaR_grid[1][..., None]**2
                * jnp.cos(params.ThetaR_grid[0][..., None]) * u[..., 1],
                dx=params.dr, axis=1
            ),
            dx=params.dtheta, axis=0
        )/A0

        # Fluid vertical centre of mass
        com_Y = -jnp.trapezoid(
            jnp.trapezoid(
                params.ThetaR_grid[1][..., None]**2
                * jnp.sin(params.ThetaR_grid[0][..., None]) * u[..., 1],
                dx=params.dr, axis=1,
            ),
            dx = params.dtheta, axis=0
        )/A0

        return u_theta, u_r, psi, ang_mom, com_X, com_Y
    

    @staticmethod
    @jax.jit
    def _compute_flowrate(
            u: jnp.ndarray, psi: jnp.ndarray, params: AnnulusParam
    ) -> float:
        """
        Compute the time derivative of the outer wall circulation flux.

        Args:
            u (jnp.ndarray): Current state matrix.
            psi (jnp.ndarray): Current streamfunction matrix.
            params (AnnulusParam): Simulation parameters.

        Returns:
            float: The time derivative of Q(t).
        """
        
        u_theta, u_r = AnnulusFlow._compute_vel(psi, params)

        u_theta_mean = jnp.mean(u_theta, axis=0).reshape((1,-1))

        dutheta_dr: jnp.ndarray = AnnulusFlow.ddr(u_theta, params.dr)

        adv_term = -jnp.mean(u_r * u_theta, axis=0)/params.r \
            - jnp.mean(dutheta_dr*u_r, axis=0)
        
        buoy_term = params.Pr * params.Ra * jnp.mean(
            u[..., 1] * jnp.cos(params.ThetaR_grid[0]), axis=0
        )

        visc_term = params.Pr * (
            AnnulusFlow.d2dr2(u_theta_mean, params.dr).reshape((-1,))
            + AnnulusFlow.ddr(u_theta_mean, params.dr).reshape((-1,))
            / params.r
            - u_theta_mean.reshape((-1,)) / params.r**2
        )

        return -jnp.trapezoid(adv_term + buoy_term + visc_term, dx=params.dr)
        

    @staticmethod
    @partial(jax.jit, static_argnums=(0,1))
    def _init_Poisson(
            Ntheta: int, Nr: int,
            dr: float, r: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Pre-compute the inverse of the tridiagonal Poisson matrix for each
        azimuthal Fourier mode (to be used for pseudo-spectral methods during
        simulation).

        Args:
            Ntheta (int): Number of azimuthal grid points.
            Nr (int): Number of radial grid points.
            dr (float): Radial grid point interval.
            r (jnp.ndarray): Vector of radial coordinates.

        Returns:
            jnp.ndarray: Stacked inverse matrices for all k modes.
        """

        k = jnp.fft.fftfreq(Ntheta) * Ntheta

        def _init_single_Poisson(k):

            poisson = jnp.zeros((Nr, Nr))

            # Solving interior points
            j = jnp.arange(1, Nr-1, dtype=jnp.int32)
            interior_coefs: list[jnp.ndarray] = [
                -dr**-2. + 0.5/r[j]/dr,
                2/dr**2. + k**2./r[j]**2,
                -dr**-2. - 0.5/r[j]/dr
            ]
            poisson = poisson.at[j, j-1].set(interior_coefs[0])
            poisson = poisson.at[j, j].set(interior_coefs[1])
            poisson = poisson.at[j, j+1].set(interior_coefs[2])

            # Setting stencils for boundary condition
            poisson = poisson.at[0, 0].set(1)
            poisson = poisson.at[-1, -1].set(1)

            return jnp.linalg.inv(poisson)

        return jax.vmap(_init_single_Poisson)(k)
    
    
    @staticmethod
    @jax.jit
    def ddtheta(u: jnp.ndarray, dtheta: float) -> jnp.ndarray:

        return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dtheta)
    

    @staticmethod
    @jax.jit
    def d2dtheta2(u: jnp.ndarray, dtheta: float) -> jnp.ndarray:

        return (
            jnp.roll(u, -1, axis=0)
            - 2.0 * u
            + jnp.roll(u, 1, axis=0)
        ) / dtheta**2
    

    @staticmethod
    @jax.jit
    def ddr(u: jnp.ndarray, dr: float) -> jnp.ndarray:

        interior = (u[:, 2:] - u[:, :-2]) / (2 * dr)
        inner = (-3 * u[:, 0:1] + 4 * u[:, 1:2] - u[:, 2:3]) / (2 * dr)
        outer = (3 * u[:, -1:] - 4 * u[:, -2:-1] + u[:, -3:-2]) / (2 * dr)

        return jnp.concatenate([inner, interior, outer], axis=1)
    

    @staticmethod
    @jax.jit
    def d2dr2(u: jnp.ndarray, dr: float) -> jnp.ndarray:

        interior = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dr**2
        inner = (
            2 * u[:, 0:1]
            - 5 * u[:, 1:2]
            + 4 * u[:, 2:3]
            - u[:, 3:4]
        ) / dr**2
        outer = (
            2 * u[:, -1:]
            - 5 * u[:, -2:-1]
            + 4 * u[:, -3:-2]
            - u[:, -4:-3]
        ) / dr**2

        return jnp.concatenate([inner, interior, outer], axis=1)