import numpy as np
import scipy as sp
from tqdm import tqdm
from pysindy import FiniteDifference
from typing import Optional

from sindy.ivp_solvers import rk4
from sindy.utils import print_latex_eqn


def dmd_svd(
        x: np.ndarray, y: np.ndarray=None, rank: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    A SVD implementation of (Loiseau, 2020)'s DMD.

    Args:
        x (np.ndarray): The data matrix X.
        y (np.ndarray, optional): The time shifted data matrix of X. Defaults
            to None, which creates a time shifted data matrix from X.
        rank (int, optional): The DMD rank to extract. Defaults to 2.

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns the DMD mode projection matrix
            and the eigenspectrum.
    """

    if y is None:
        x, y = x[:, :-1], x[:, 1:]

    # Obtain the SVD for X
    Ux, Sx, VxH = np.linalg.svd(x, full_matrices=False)

    # Compute Y V_X^H
    YVx = y @ VxH.T

    # Compute the eigendecomposition with SVD
    P, sigma, _ = np.linalg.svd(YVx, full_matrices=False)
    sigma *= sigma

    # Sort and normalize the eigenvalues
    idx = np.argsort(-sigma)
    sigma, P = sigma[idx], P[:, idx]
    sigma /= np.sum(sigma)

    # Truncate the output basis to the desired rank
    P = P[:, :rank]

    # Compute Q
    Q = Ux @ np.diag(Sx**-1) @ YVx.T @ P

    # Compute the eigenvalues on the rank reduced matrix
    _, psi = sp.linalg.eig(Q.T @ P, left=False, right=True)
        
    return P @ psi, sigma


class AESINDy:

    def __init__(
            self, data_dim: int, latent_dim: int,
            lambda_xdot: float, lambda_zdot: float, lambda_l2: float,
            deriv_order: int = 2,
            threshold_reg: float = 0.01, threshold_freq: int = 10
    ):
        """
        AESINDy autoencoder with linear SINDy library.

        Parameters:
        -----------
        data_dim : int
            Dimension of the input data (n)
        latent_dim : int
            Dimension of the latent space (r)
        deriv_order : int
            Order of finite difference scheme (default: 2)
        """
        
        # Store dimensions
        self.n = data_dim
        self.r = latent_dim
        
        # Initialise trainable weights
        self.A_E = self._xavier_init((self.r, self.n))
        self.A_D = self._xavier_init((self.n, self.r))
        self.Xi  = self._xavier_init((self.r, self.r))
        self.Xi_mask = np.ones(self.Xi.shape, dtype=bool)

        # Store derivative parameters
        self.deriv_order = deriv_order
        self.deriv_d = 1

        # Save loss weights
        self.lambda_xdot = lambda_xdot
        self.lambda_zdot = lambda_zdot
        self.lambda_l2 = lambda_l2

        # Save STLSQ parameters
        self.threshold_reg = threshold_reg
        self.threshold_freq = threshold_freq


    def fit(
            self, X: np.ndarray, dt: float,
            batch_size: int = 32, epochs: int = 10, lr: float = 1e-3
    ) -> list[dict]:
        """
        Fit the AESINDy model to data X using finite difference for derivatives.

        Parameters:
        -----------
        X : np.ndarray, shape (n, m)
            Input data
        dt : float
            Time step between samples
        batch_size : int
            Batch size
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        """

        n_samples = X.shape[1]
        losses_hist = list()

        # Compute data time derivative
        X_dot = FiniteDifference(
            self.deriv_order, self.deriv_d, axis=1
        )._differentiate(X, dt)

        for epoch in range(epochs):
            # Shuffle the dataset
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            X_dot_shuffled = X_dot[:, indices]
            
            # Mini-batch training
            progress_bar = tqdm(
                range(0, n_samples, batch_size), desc=f'Epoch {epoch+1:d}'
            )

            for start in progress_bar:
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[:, start:end]
                X_dot_batch = X_dot_shuffled[:, start:end]
                self._train_step(X_batch, X_dot_batch, lr)

                # Compute training loss
                total_loss, losses = self.compute_loss(X, X_dot)
                progress_bar.set_description(
                    f"Epoch {epoch+1:d} | Loss: {total_loss:.6f}"
                )

            losses_hist.append(losses)

            # Carry out thresholding
            if epoch % self.threshold_freq == 0:
                self.Xi_mask[np.abs(self.Xi) < self.threshold_reg] = 0
                self.Xi[~self.Xi_mask] = 0.0

        return losses_hist
            
    
    def _train_step(self, X_batch: np.ndarray, X_dot_batch: np.ndarray, lr: float):
        """
        Perform one gradient descent step on a batch.

        Parameters:
        -----------
        X_batch : np.ndarray, shape (n, batch_size)
            X data batch
        X_dot_batch : np.ndarray, shape (batch_size, batch_size)
            X data time derivative batch
        lr : float
            Learning rate
        """

        # Compute intermediate matrices
        Z = self.A_E @ X_batch
        Z_dot = self.A_E @ X_dot_batch
        Z_dot_hat = self.Xi.T @ Z

        X_hat = self.A_D @ Z
        X_dot_hat = self.A_D @ Z_dot_hat
        
        # Residuals
        R_recon = X_batch - X_hat
        R_zdot = Z_dot - Z_dot_hat
        R_xdot = X_dot_batch - X_dot_hat

        # Get matrix sizes for MSE scaling
        n_recon = R_recon.size
        n_zdot = R_zdot.size
        n_xdot = R_xdot.size

        # Gradients
        grad_A_D = (
            - R_recon @ Z.T / n_recon
            - self.lambda_xdot * R_xdot @ (Z.T @ self.Xi) / n_xdot
        )
        grad_A_E = (
            - self.A_D.T @ R_recon @ X_batch.T / n_recon
            - self.lambda_zdot * (R_zdot @ X_dot_batch.T - self.Xi @ R_zdot @ X_batch.T) / n_zdot
            - self.lambda_xdot * self.Xi @ self.A_D.T @ R_xdot @ X_batch.T / n_xdot
        )
        grad_Xi = (
            - self.lambda_zdot * Z @ R_zdot.T / n_zdot
            - self.lambda_xdot * Z @ R_xdot.T @ self.A_D / n_xdot
        )
        grad_Xi += self.lambda_l2 * self.Xi / self.Xi.size

        # Gradient descent update
        self.A_D -= lr * grad_A_D
        self.A_E -= lr * grad_A_E
        self.Xi[self.Xi_mask]  -= lr * grad_Xi[self.Xi_mask]


    def compute_loss(self, X: np.ndarray, X_dot: np.ndarray) -> tuple[float, dict]:

        Z = self.A_E @ X
        Z_dot = self.A_E @ X_dot
        Z_dot_hat = self.Xi.T @ Z

        X_hat = self.A_D @ Z
        X_dot_hat = self.A_D @ Z_dot_hat
        
        # Residuals
        R_recon = X - X_hat
        R_zdot = Z_dot - Z_dot_hat
        R_xdot = X_dot - X_dot_hat

        # Loss components
        L_recon = 0.5 * np.mean(R_recon ** 2)
        L_zdot = 0.5 * np.mean(R_zdot ** 2) * self.lambda_zdot
        L_xdot = 0.5 * np.mean(R_xdot ** 2) * self.lambda_xdot
        L_l2 = 0.5 * np.mean(self.Xi ** 2) * self.lambda_l2

        total_loss = L_recon + L_zdot + L_xdot + L_l2
        
        losses = {
            'recon': L_recon,
            'zdot': L_zdot,
            'xdot': L_xdot,
            'l2': L_l2
        }

        return total_loss, losses


    @staticmethod
    def _xavier_init(shape: tuple) -> np.ndarray:

        dim_sum: int = sum(shape)
        limit: float = np.sqrt(6 / dim_sum)
        return np.random.uniform(-limit, limit, size=shape)
    

    def get_eqn(
            self, state_names: Optional[list[str]] = None, precision: int = 3
    ) -> str:

        if state_names is None:
            state_names = [f'z_{{{idx+1}}}' for idx in range(self.r)]

        return print_latex_eqn(
            xi=self.Xi.T,
            state_names=state_names, lib_names=state_names,
            precision=precision, dt_order=self.deriv_d
        )
    

    def predict(
            self, x0: np.ndarray, tFinal: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        z0 = self.A_E @ x0

        t, Z = rk4(
            func = lambda t, z: z @ self.Xi,
            tFinal = tFinal, y0=z0, dt=dt
        )

        return t, self.A_D @ Z.T, Z.T
    

    def reconstruct(self, X: np.ndarray) -> np.ndarray:

        return self.A_D @ self.A_E @ X