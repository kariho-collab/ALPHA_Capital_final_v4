"""
03. Market Regime Detection: Gaussian Hidden Markov Model (HMM) & GMM
Pure NumPy vectorized Baum-Welch and Viterbi decoding with Scikit-Learn GaussianMixture support.
Zero external C++ compiler dependencies.
"""
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for financial time-series regime switching.
    Learns transition matrix A, emission parameters (means, covariances), and initial distribution pi.
    """
    def __init__(self, n_states: int = 3, max_iter: int = 50, tol: float = 1e-4, random_state: int = 42):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.trans_mat_: Optional[np.ndarray] = None   # A: (n_states, n_states)
        self.means_: Optional[np.ndarray] = None       # mu: (n_states, n_features)
        self.covars_: Optional[np.ndarray] = None      # sigma: (n_states, n_features, n_features)
        self.start_prob_: Optional[np.ndarray] = None  # pi: (n_states,)
        self.gmm_: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fits HMM parameters on observation matrix X of shape (T, D).
        Uses GMM for robust warm-start initialization followed by Baum-Welch training.
        """
        np.random.seed(self.random_state)
        T, D = X.shape
        K = self.n_states

        # 1. Warm-start with Gaussian Mixture Model
        self.gmm_ = GaussianMixture(
            n_components=K,
            covariance_type="full",
            max_iter=100,
            random_state=self.random_state
        )
        self.gmm_.fit(X)

        self.means_ = self.gmm_.means_
        self.covars_ = self.gmm_.covariances_
        self.start_prob_ = np.ones(K) / K

        # Initialize transition matrix with persistence bias on diagonals
        self.trans_mat_ = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.trans_mat_, 0.9)

        # 2. Baum-Welch EM Algorithm
        prev_ll = -np.inf
        for it in range(self.max_iter):
            # E-Step: Emission densities B
            B = self._compute_emission_matrix(X)

            # Forward pass
            alpha, log_scale = self._forward_pass(B)
            curr_ll = np.sum(log_scale)

            # Check convergence
            if np.abs(curr_ll - prev_ll) < self.tol:
                break
            prev_ll = curr_ll

            # Backward pass
            beta = self._backward_pass(B, log_scale)

            # Compute posteriors gamma (T, K) and xi (T-1, K, K)
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = np.divide(gamma, gamma_sum, out=np.zeros_like(gamma), where=gamma_sum > 0)

            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                numerator = (
                    alpha[t, :, None] *
                    self.trans_mat_ *
                    B[t + 1, None, :] *
                    beta[t + 1, None, :]
                )
                denom = numerator.sum()
                if denom > 0:
                    xi[t] = numerator / denom

            # M-Step: Update parameters
            self.start_prob_ = gamma[0]
            xi_sum = xi.sum(axis=0)
            gamma_t_sum = gamma[:-1].sum(axis=0, keepdims=True).T
            self.trans_mat_ = np.divide(xi_sum, gamma_t_sum, out=np.zeros_like(xi_sum), where=gamma_t_sum > 0)
            # Ensure rows sum to 1
            self.trans_mat_ = self.trans_mat_ / self.trans_mat_.sum(axis=1, keepdims=True)

            # Update means and covariances
            for k in range(K):
                w = gamma[:, k]
                w_sum = w.sum()
                if w_sum > 0:
                    self.means_[k] = (w[:, None] * X).sum(axis=0) / w_sum
                    diff = X - self.means_[k]
                    cov = np.dot(w * diff.T, diff) / w_sum
                    # Regularization for numerical stability
                    cov += np.eye(D) * 1e-6
                    self.covars_[k] = cov

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for global most probable state trajectory.
        """
        if self.trans_mat_ is None:
            raise ValueError("Model not fitted.")

        T, D = X.shape
        K = self.n_states
        B = self._compute_emission_matrix(X)

        # Log probabilities to prevent underflow
        log_trans = np.log(np.maximum(self.trans_mat_, 1e-12))
        log_start = np.log(np.maximum(self.start_prob_, 1e-12))
        log_B = np.log(np.maximum(B, 1e-12))

        viterbi = np.zeros((T, K))
        backpointer = np.zeros((T, K), dtype=int)

        viterbi[0] = log_start + log_B[0]

        for t in range(1, T):
            for k in range(K):
                seq_prob = viterbi[t - 1] + log_trans[:, k] + log_B[t, k]
                best_prev = np.argmax(seq_prob)
                viterbi[t, k] = seq_prob[best_prev]
                backpointer[t, k] = best_prev

        # Backtrack best path
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path

    def _compute_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """Computes multivariate Gaussian density for each state and observation."""
        T, D = X.shape
        K = self.n_states
        B = np.zeros((T, K))

        for k in range(K):
            mu = self.means_[k]
            sigma = self.covars_[k]
            det = np.linalg.det(sigma)
            if det <= 0 or np.isnan(det):
                det = 1e-10
                sigma += np.eye(D) * 1e-4

            inv_sigma = np.linalg.pinv(sigma)
            diff = X - mu
            # Mahalanobis distance
            maha = np.sum(np.dot(diff, inv_sigma) * diff, axis=1)
            norm_const = 1.0 / np.sqrt(((2.0 * np.pi) ** D) * det + 1e-12)
            B[:, k] = norm_const * np.exp(-0.5 * maha)

        # Ensure no exact zeros
        B = np.maximum(B, 1e-20)
        return B

    def _forward_pass(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, K = B.shape
        alpha = np.zeros((T, K))
        log_scale = np.zeros(T)

        # t = 0
        alpha[0] = self.start_prob_ * B[0]
        c0 = alpha[0].sum()
        if c0 > 0:
            alpha[0] /= c0
            log_scale[0] = np.log(c0)

        # Induction
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t - 1], self.trans_mat_) * B[t]
            ct = alpha[t].sum()
            if ct > 0:
                alpha[t] /= ct
                log_scale[t] = np.log(ct)

        return alpha, log_scale

    def _backward_pass(self, B: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = np.dot(self.trans_mat_, (B[t + 1] * beta[t + 1]))
            ct = np.exp(log_scale[t + 1])
            if ct > 0:
                beta[t] /= ct

        return beta
