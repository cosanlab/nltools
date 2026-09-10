#!/usr/bin/env python
# coding: latin-1

"""Shared Response Model (SRM) for multi-subject fMRI alignment.

SRM factorizes each subject's data as `X_i ≈ W_i S`: a shared low-dimensional
response `S` common to all subjects plus a subject-specific orthogonal
transform `W_i`. `SRM` is the probabilistic model fit by
expectation-maximization; `DetSRM` is the deterministic variant fit by block
coordinate descent.

**Algorithm.** Initialize each `W_i` as a random orthogonal matrix (QR of a
random matrix), then iterate: update the shared response `S` from the current
transforms, update each `W_i` by solving an orthogonal Procrustes problem
(`SRM` also re-estimates the per-subject noise variance `rho_i^2` and the
shared-response covariance), for `n_iter` iterations.

**Performance.** Time is O(n_iter × (V T K + V K^2 + K^3)) and memory O(V T),
with V the total voxels across subjects, T samples, and K features (typically
V ≫ T ≫ K). `parallel='cpu'` runs the per-subject transform updates with
joblib. `parallel='gpu'` is not implemented and raises `NotImplementedError`
rather than silently running on CPU.

**When to use.** Cross-subject analyses that need a shared response space and
tolerate dimension reduction. Use `HyperAlignment` when spatial structure and
full dimensionality must be preserved.

**References.** Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J.,
& Ramadge, P. J. (2015). A reduced-dimension fMRI shared response model.
*Advances in Neural Information Processing Systems*, 460-468. Anderson, M. J.,
Capota, M., Turek, J. S., Zhu, X., Willke, T. L., Wang, Y., & Norman, K. A.
(2016). Enabling factor analysis on thousand-subject neuroimaging datasets.
*2016 IEEE International Conference on Big Data*, 1151-1160.

Copyright 2016 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

# Authors: Po-Hsuan Chen (Princeton Neuroscience Institute) and Javier Turek
# (Intel Labs), 2015
import logging

import numpy as np
import scipy
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.exceptions import NotFittedError
import sys

__all__ = ["SRM", "DetSRM"]

logger = logging.getLogger(__name__)


def _validate_srm_parallel(parallel: str | None) -> None:
    """Validate the `parallel=` backend selector for SRM/DetSRM.

    Run-or-raise policy: an explicit GPU request never silently runs on CPU.

    Args:
        parallel (str | None): None, `'cpu'`, or `'gpu'`.

    Raises:
        ValueError: If `parallel` is not one of the accepted values.
        NotImplementedError: If `parallel='gpu'` (no torch port yet).
    """
    if parallel not in (None, "cpu", "gpu"):
        raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel}")
    if parallel == "gpu":
        raise NotImplementedError(
            "parallel='gpu' is not implemented for SRM/DetSRM (a torch port is "
            "tracked on the 0.6.x roadmap). Use parallel='cpu' or parallel=None."
        )


def _init_w_transforms(
    data: list[np.ndarray], n_features: int, random_states: list[Any]
) -> tuple[list[np.ndarray | None], np.ndarray]:
    """Initialize the mappings $W_i$ for the SRM with random orthogonal matrices.

    Each subject's transform is the Q factor of the QR decomposition of a
    random (voxels_i, n_features) matrix drawn from that subject's own
    `RandomState`, so the initial transforms are orthogonal and independent
    across subjects. Subjects whose data is None get a None transform and a
    voxel count of 0.

    Args:
        data (list[np.ndarray | None]): One (voxels_i, samples) array per subject.
        n_features (int): Number of features in the model.
        random_states (list[np.random.RandomState]): One generator per subject.

    Returns:
        tuple[list[np.ndarray | None], np.ndarray]: `(w, voxels)` — the initial
            orthogonal transforms, element i of shape (voxels_i, n_features), and
            an integer array with the number of voxels per subject.
    """
    w = []
    subjects = len(data)
    voxels = np.empty(subjects, dtype=int)

    # Set Wi to a random orthogonal voxels by n_features matrix
    # QR decomposition ensures orthogonality: Q is orthogonal, R is upper triangular
    # This initialization strategy enables efficient Procrustes optimization later
    for subject in range(subjects):
        if data[subject] is not None:
            voxels[subject] = data[subject].shape[0]
            rnd_matrix = random_states[subject].random_sample(
                (voxels[subject], n_features)
            )
            q, r = np.linalg.qr(rnd_matrix)
            w.append(q)
        else:
            voxels[subject] = 0
            w.append(None)

    return w, voxels


class SRM(BaseEstimator, TransformerMixin):
    """Probabilistic Shared Response Model (SRM).

    Factorizes multi-subject data as a shared response S plus one orthogonal
    transform W per subject, so that for every subject i

    $$
    X_i \\approx W_i S, \\forall i=1 \\dots N
    $$

    The model is fit by the expectation-maximization algorithm of Chen et al.
    (2015) with the optimizations of Anderson et al. (2016). Subjects may have
    different numbers of voxels; they must have the same number of samples
    unless `fit(pad_samples=True)` zero-pads the shorter ones. Run time is
    $O(I (V T K + V K^2 + K^3))$ and memory $O(V T)$, with I iterations, V the
    sum of voxels across subjects, T samples, and K features (typically
    $V \\gg T \\gg K$).

    Args:
        n_iter (int): Number of EM iterations. Defaults to 10.
        n_features (int): Number of shared features to compute. Defaults to 50.
        random_state (int): Seed for the random initialization. Defaults to 0.

    Attributes:
        w_ (list[np.ndarray]): Per-subject orthogonal transforms, element i of
            shape (voxels_i, n_features).
        s_ (np.ndarray): The shared response, shape (n_features, samples).
        sigma_s_ (np.ndarray): Covariance of the shared response's Normal
            distribution, shape (n_features, n_features).
        mu_ (list[np.ndarray]): Per-subject voxel means over samples, element i
            of shape (voxels_i,).
        rho2_ (np.ndarray): Estimated noise variance $\\rho_i^2$ per subject,
            shape (subjects,).
        random_state_ (np.random.RandomState): Generator seeded from `random_state`.

    Examples:
        ```python
        import numpy as np
        from nltools.algorithms import SRM

        data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

        srm = SRM(n_iter=10, n_features=50)
        srm.fit(data, parallel="cpu", n_jobs=-1)
        shared_responses = srm.transform(data)  # list of (50, 50) arrays

        w = srm.w_  # subject-specific transforms
        s = srm.s_  # shared response
        ```
    """

    def __init__(
        self, *, n_iter: int = 10, n_features: int = 50, random_state: int = 0
    ) -> None:
        self.n_iter = n_iter
        self.n_features = n_features
        self.random_state = random_state
        return

    def fit(
        self,
        X: list[np.ndarray],
        y: Any | None = None,
        *,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
        pad_samples: bool = True,
    ) -> "SRM":
        """Compute the probabilistic Shared Response Model.

        Args:
            X (list[np.ndarray]): One (voxels_i, samples) array per subject.
                Subjects may differ in the number of samples when
                `pad_samples=True`.
            y (Any | None): Ignored; present for scikit-learn compatibility.
            parallel (str | None): `'cpu'` (default) updates subjects in parallel
                with joblib; None runs single-threaded NumPy; `'gpu'` raises
                `NotImplementedError` (never a silent CPU fallback).
            n_jobs (int): Number of CPU workers when `parallel='cpu'`; -1
                (default) picks a count from available memory.
            pad_samples (bool): If True (default), zero-pad subjects with fewer
                samples up to the longest subject; if False, unequal sample
                counts raise `ValueError`.

        Returns:
            SRM: Fitted model (`self`).
        """
        logger.info("Starting Probabilistic SRM")

        _validate_srm_parallel(parallel)

        # Store parallel settings for use in _srm
        self._parallel = parallel
        self._n_jobs = n_jobs

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError(
                f"There are not enough subjects ({len(X):d}) to train the model."
            )

        # Check for input data sizes
        if X[0].shape[1] < self.n_features:
            raise ValueError(
                "There are not enough samples to train the model with "
                f"{self.n_features:d} features."
            )

        # Handle unequal sample counts via padding
        sample_counts = [subj.shape[1] for subj in X]
        max_samples = max(sample_counts)
        number_subjects = len(X)

        if not all(s == max_samples for s in sample_counts):
            if pad_samples:
                # Zero-pad subjects to match the longest
                X_padded = []
                for subj in X:
                    if subj.shape[1] < max_samples:
                        padding = np.zeros((subj.shape[0], max_samples - subj.shape[1]))
                        X_padded.append(np.hstack([subj, padding]))
                    else:
                        X_padded.append(subj)
                X = X_padded
                logger.info(
                    f"Padded subjects to {max_samples} samples (original: {sample_counts})"
                )
            else:
                raise ValueError(
                    f"Different number of samples between subjects: {sample_counts}. "
                    "Set pad_samples=True to automatically zero-pad to the longest subject."
                )

        # Validate all data is finite
        for subject in range(number_subjects):
            if X[subject] is not None:
                assert_all_finite(X[subject])

        # Run SRM
        self.sigma_s_, self.w_, self.mu_, self.rho2_, self.s_ = self._srm(
            X, parallel=self._parallel, n_jobs=self._n_jobs
        )

        return self

    def transform(
        self,
        X: list[np.ndarray],
        y: Any | None = None,
        *,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
    ) -> list[np.ndarray | None]:
        """Project each subject's data into the shared response space.

        Args:
            X (list[np.ndarray | None]): One (voxels_i, samples_i) array per
                fitted subject, in the same order as `fit`; voxel and sample
                counts may vary across subjects. A None entry yields None.
            y (Any | None): Ignored; present for scikit-learn compatibility.
            parallel (str | None): `'cpu'` (default) transforms subjects in
                parallel with joblib; None runs single-threaded NumPy; `'gpu'`
                raises `NotImplementedError`.
            n_jobs (int): Number of CPU workers when `parallel='cpu'`; -1
                (default) reuses the value from `fit`, itself resolved from
                available memory.

        Returns:
            list[np.ndarray | None]: Shared responses, element i of shape
                (n_features, samples_i).
        """

        _validate_srm_parallel(parallel)

        # Check if the model exist
        if hasattr(self, "w_") is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError(
                "The number of subjects does not match the one in the model."
            )

        # Handle parallelization for transform
        if parallel == "cpu" and len(X) > 1:
            # CPU-parallel transform across subjects
            from joblib import Parallel, delayed

            def _transform_one_subject(subj_idx):
                """Transform one subject."""
                if X[subj_idx] is not None:
                    return self.w_[subj_idx].T.dot(X[subj_idx])
                return None

            # Prefer an explicitly-passed n_jobs; fall back to the fit-time value
            n_jobs_to_use = n_jobs if n_jobs != -1 else getattr(self, "_n_jobs", -1)
            if n_jobs_to_use == -1:
                # Auto-detect based on memory
                from nltools.algorithms.backends import auto_n_jobs_for_arrays

                n_jobs_to_use = auto_n_jobs_for_arrays(X)

            s = Parallel(n_jobs=n_jobs_to_use)(
                delayed(_transform_one_subject)(i) for i in range(len(X))
            )
        else:
            # Single-threaded transform
            s: list[np.ndarray | None] = [None] * len(X)
            for subject in range(len(X)):
                if X[subject] is not None:
                    s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _init_structures(self, data, subjects):
        """Initialize the EM data structures and demean the data.

        Removes each subject's voxel means (subject-specific baselines), sets
        the initial noise variance to 1.0, and precomputes $||X_i||_F^2$ for
        the likelihood computation.

        Args:
            data (list[np.ndarray | None]): One (voxels_i, samples) array per
                subject.
            subjects (int): Number of subjects in `data`.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]: `(x,
                mu, rho2, trace_xtx)` — the demeaned data per subject (element i of
                shape (voxels_i, samples)), the voxel means per subject (element i
                of shape (voxels_i,)), the initial noise variance $\\rho^2$ per
                subject (shape (subjects,)), and the squared Frobenius norm of
                each subject's data (shape (subjects,)).
        """
        x = []
        mu = []
        rho2 = np.zeros(subjects)

        trace_xtx = np.zeros(subjects)

        # Initialize noise variance to 1.0 (unit variance assumption)
        # This will be updated during EM iterations
        for subject in range(subjects):
            rho2[subject] = 1
            if data[subject] is not None:
                mu.append(np.mean(data[subject], 1))
                trace_xtx[subject] = np.sum(data[subject] ** 2)
                x.append(data[subject] - mu[subject][:, np.newaxis])
            else:
                mu.append(None)
                trace_xtx[subject] = 0
                x.append(None)

        return x, mu, rho2, trace_xtx

    def _likelihood(
        self,
        chol_sigma_s_rhos,
        log_det_psi,
        chol_sigma_s,
        trace_xt_invsigma2_x,
        inv_sigma_s_rhos,
        wt_invpsi_x,
        samples,
    ):
        """Calculate the log-likelihood (up to a constant) for convergence logging.

        Log-determinants come from the Cholesky factors rather than explicit
        inverses, for numerical stability.

        Args:
            chol_sigma_s_rhos (np.ndarray): Cholesky factor of
                $(\\Sigma_S + \\sum_i(1/\\rho_i^2) I)$, shape (n_features, n_features).
            log_det_psi (float): Log-determinant of the diagonal matrix Psi
                (each $\\rho_i^2$ repeated voxels_i times).
            chol_sigma_s (np.ndarray): Cholesky factor of $\\Sigma_S$, shape
                (n_features, n_features).
            trace_xt_invsigma2_x (float): $\\sum_i ||X_i||_F^2 / \\rho_i^2$.
            inv_sigma_s_rhos (np.ndarray): Inverse of
                $(\\Sigma_S + \\sum_i(1/\\rho_i^2) I)$, shape (n_features, n_features).
            wt_invpsi_x (np.ndarray): $\\sum_i W_i^T X_i / \\rho_i^2$, shape
                (n_features, samples).
            samples (int): Number of samples in the data.

        Returns:
            float: The log-likelihood value.
        """
        # Compute log-determinant using Cholesky factors (numerically stable)
        log_det = (
            np.log(np.diag(chol_sigma_s_rhos) ** 2).sum()
            + log_det_psi
            + np.log(np.diag(chol_sigma_s) ** 2).sum()
        )
        # Log-likelihood: -0.5 * (determinant terms + trace terms) + quadratic form
        loglikehood = -0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
        loglikehood += 0.5 * np.trace(
            wt_invpsi_x.T.dot(inv_sigma_s_rhos).dot(wt_invpsi_x)
        )
        # + const --> -0.5*nTR*nvoxel*subjects*math.log(2*math.pi)

        return loglikehood

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Update the mapping $W_i$ for one subject.

        Solves the orthogonal Procrustes problem
        $\\min ||X_i - W_i S||_F^2$ subject to $W_i^T W_i = I$: with the SVD
        $U \\Sigma V^T = X_i S^T$, the optimum is $W_i = U V^T$.

        Args:
            Xi (np.ndarray): The subject's data $X_i$, shape (voxels, timepoints).
            S (np.ndarray): The shared response, shape (n_features, timepoints).

        Returns:
            np.ndarray: The orthogonal transform $W_i$, shape (voxels, n_features).
        """
        # Compute cross-covariance: X_i S^T
        A = Xi.dot(S.T)
        # Solve the Procrustes problem via SVD
        # Optimal orthogonal transform: W_i = U V^T where A = U Σ V^T
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X: np.ndarray) -> np.ndarray:
        """Transform a new subject using the existing model.

        The subject is assumed to have received equivalent stimulation.

        Args:
            X (np.ndarray): The new subject's data, shape (voxels, timepoints);
                the timepoints must match the fitted shared response.

        Returns:
            np.ndarray: Orthogonal mapping $W_{new}$ for the new subject, shape
                (voxels, n_features).
        """
        # Check if the model exist
        if hasattr(self, "w_") is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_.shape[1]:
            raise ValueError(
                "The number of timepoints(TRs) does not match the one in the model."
            )

        w = self._update_transform_subject(X, self.s_)

        return w

    def _srm(self, data, parallel: str | None = None, n_jobs: int = -1):
        """Expectation-maximization algorithm for fitting the probabilistic SRM.

        Args:
            data (list[np.ndarray | None]): One (voxels_i, samples) array per
                subject.
            parallel (str | None): None or `'cpu'` (joblib over subjects in the
                M-step).
            n_jobs (int): Number of CPU workers; -1 picks a count from available
                memory.

        Returns:
            tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
                `(sigma_s, w, mu, rho2, s)` — the shared-response covariance
                $\\Sigma_s$ (shape (n_features, n_features)), the per-subject
                orthogonal transforms $W_i$ (element i of shape (voxels_i,
                n_features)), the per-subject voxel means $\\mu_i$ (element i of
                shape (voxels_i,)), the per-subject noise variance $\\rho_i^2$
                (shape (subjects,)), and the shared response (shape (n_features,
                samples)).
        """

        samples = min([d.shape[1] for d in data if d is not None], default=sys.maxsize)
        subjects = len(data)
        self.random_state_ = np.random.RandomState(self.random_state)
        random_states = [
            np.random.RandomState(self.random_state_.randint(2**32 - 1, dtype=np.int64))
            for i in range(len(data))
        ]

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject, and trace_xtx with
        # the ||X_i||_F^2 of each subject.
        w, voxels = _init_w_transforms(data, self.n_features, random_states)
        x, mu, rho2, trace_xtx = self._init_structures(data, subjects)
        shared_response = np.zeros((self.n_features, samples))
        sigma_s = np.identity(self.n_features)

        # Main loop of the algorithm (EM iterations)
        # E-step: Update shared response S given current transforms W_i
        # M-step: Update transforms W_i and noise variances rho_i^2 given S
        for iteration in range(self.n_iter):
            logger.info("Iteration %d", iteration + 1)

            # E-step: Update shared response S

            # Sum the inverted the rho2 elements for computing W^T * Psi^-1 * W
            rho0 = (1 / rho2).sum()

            # Invert Sigma_s using Cholesky factorization
            (chol_sigma_s, lower_sigma_s) = scipy.linalg.cho_factor(
                sigma_s, check_finite=False
            )
            inv_sigma_s = scipy.linalg.cho_solve(
                (chol_sigma_s, lower_sigma_s),
                np.identity(self.n_features),
                check_finite=False,
            )

            # Invert (Sigma_s + rho_0 * I) using Cholesky factorization
            sigma_s_rhos = inv_sigma_s + np.identity(self.n_features) * rho0
            chol_sigma_s_rhos, lower_sigma_s_rhos = scipy.linalg.cho_factor(
                sigma_s_rhos, check_finite=False
            )
            inv_sigma_s_rhos = scipy.linalg.cho_solve(
                (chol_sigma_s_rhos, lower_sigma_s_rhos),
                np.identity(self.n_features),
                check_finite=False,
            )

            # Compute the sum of W_i^T * rho_i^-2 * X_i, and the sum of traces
            # of X_i^T * rho_i^-2 * X_i
            wt_invpsi_x = np.zeros((self.n_features, samples))
            trace_xt_invsigma2_x = 0.0
            for subject in range(subjects):
                if data[subject] is not None:
                    wt_invpsi_x += (w[subject].T.dot(x[subject])) / rho2[subject]
                    trace_xt_invsigma2_x += trace_xtx[subject] / rho2[subject]

            log_det_psi = np.sum(np.log(rho2) * voxels)

            # Update the shared response S (E-step)
            # Weighted average of transformed data: S = Σ_s (I - rho0 * inv(Σ_s + rho0*I)) @ W^T @ Psi^{-1} @ X
            shared_response = sigma_s.dot(
                np.identity(self.n_features) - rho0 * inv_sigma_s_rhos
            ).dot(wt_invpsi_x)

            # M-step: Update transforms W_i and noise variances rho_i^2

            # Update Sigma_s and compute its trace
            sigma_s = (
                inv_sigma_s_rhos + shared_response.dot(shared_response.T) / samples
            )
            trace_sigma_s = samples * np.trace(sigma_s)

            # Update each subject's mapping transform W_i and error variance rho_i^2
            # Each subject's transform is updated independently via Procrustes optimization
            # Noise variance is updated based on residual error after transform update
            # Use CPU parallelization for multi-subject updates if requested
            if parallel == "cpu" and subjects > 1:
                from joblib import Parallel, delayed

                def _update_one_subject(subj_idx):
                    """Update transform and variance for one subject."""
                    if x[subj_idx] is not None:
                        a_subject = x[subj_idx].dot(shared_response.T)
                        perturbation = np.zeros(a_subject.shape)
                        np.fill_diagonal(perturbation, 0.001)
                        u_subject, s_subject, v_subject = np.linalg.svd(
                            a_subject + perturbation, full_matrices=False
                        )
                        w_new = u_subject.dot(v_subject)
                        rho2_new = trace_xtx[subj_idx]
                        rho2_new += -2 * np.sum(w_new * a_subject)
                        rho2_new += trace_sigma_s
                        rho2_new /= samples * voxels[subj_idx]
                        return w_new, rho2_new
                    return None, 0.0

                # Auto-detect n_jobs if needed
                n_jobs_to_use = n_jobs
                if n_jobs_to_use == -1:
                    from nltools.algorithms.backends import auto_n_jobs_for_arrays

                    n_jobs_to_use = auto_n_jobs_for_arrays(
                        [x[i] for i in range(subjects)]
                    )

                # Parallel update
                results = Parallel(n_jobs=n_jobs_to_use)(
                    delayed(_update_one_subject)(i) for i in range(subjects)
                )
                for subject in range(subjects):
                    w[subject], rho2[subject] = results[subject]
            else:
                # Single-threaded update
                for subject in range(subjects):
                    if x[subject] is not None:
                        a_subject = x[subject].dot(shared_response.T)
                        perturbation = np.zeros(a_subject.shape)
                        np.fill_diagonal(perturbation, 0.001)
                        u_subject, s_subject, v_subject = np.linalg.svd(
                            a_subject + perturbation, full_matrices=False
                        )
                        w[subject] = u_subject.dot(v_subject)
                        rho2[subject] = trace_xtx[subject]
                        rho2[subject] += -2 * np.sum(w[subject] * a_subject)
                        rho2[subject] += trace_sigma_s
                        rho2[subject] /= samples * voxels[subject]
                    else:
                        rho2[subject] = 0
            if logger.isEnabledFor(logging.INFO):
                # Calculate and log the current log-likelihood for checking
                # convergence
                loglike = self._likelihood(
                    chol_sigma_s_rhos,
                    log_det_psi,
                    chol_sigma_s,
                    trace_xt_invsigma2_x,
                    inv_sigma_s_rhos,
                    wt_invpsi_x,
                    samples,
                )
                logger.info(f"Objective function {loglike:f}")

        return sigma_s, w, mu, rho2, shared_response


class DetSRM(BaseEstimator, TransformerMixin):
    """Deterministic Shared Response Model (DetSRM).

    Factorizes multi-subject data as a shared response S plus one orthogonal
    transform W per subject, so that for every subject i

    $$
    X_i \\approx W_i S, \\forall i=1 \\dots N
    $$

    The model is fit by the block coordinate descent algorithm of Chen et al.
    (2015). Subjects may have different numbers of voxels but must have the
    same number of samples. Run time is $O(I (V T K + V K^2))$ and memory
    $O(V T)$, with I iterations, V the sum of voxels across subjects, T
    samples, and K features (typically $V \\gg T \\gg K$).

    Args:
        n_iter (int): Number of coordinate-descent iterations. Defaults to 10.
        n_features (int): Number of shared features to compute. Defaults to 50.
        random_state (int): Seed for the random initialization. Defaults to 0.

    Attributes:
        w_ (list[np.ndarray]): Per-subject orthogonal transforms, element i of
            shape (voxels_i, n_features).
        s_ (np.ndarray): The shared response, shape (n_features, samples).
        random_state_ (np.random.RandomState): Generator seeded from `random_state`.

    Examples:
        ```python
        import numpy as np
        from nltools.algorithms import DetSRM

        data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

        detsrm = DetSRM(n_iter=10, n_features=50)
        detsrm.fit(data, parallel="cpu", n_jobs=-1)
        shared_responses = detsrm.transform(data)  # list of (50, 50) arrays

        w = detsrm.w_  # subject-specific transforms
        s = detsrm.s_  # shared response
        ```
    """

    def __init__(
        self, *, n_iter: int = 10, n_features: int = 50, random_state: int = 0
    ) -> None:
        self.n_iter = n_iter
        self.n_features = n_features
        self.random_state = random_state

    def fit(
        self,
        X: list[np.ndarray],
        y: Any | None = None,
        *,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
    ) -> "DetSRM":
        """Compute the Deterministic Shared Response Model.

        Args:
            X (list[np.ndarray]): One (voxels_i, samples) array per subject; all
                subjects must have the same number of samples.
            y (Any | None): Ignored; present for scikit-learn compatibility.
            parallel (str | None): `'cpu'` (default) updates subjects in parallel
                with joblib; None runs single-threaded NumPy; `'gpu'` raises
                `NotImplementedError` (never a silent CPU fallback).
            n_jobs (int): Number of CPU workers when `parallel='cpu'`; -1
                (default) picks a count from available memory.

        Returns:
            DetSRM: Fitted model (`self`).
        """
        logger.info("Starting Deterministic SRM")

        _validate_srm_parallel(parallel)

        # Store parallel settings for use in _srm
        self._parallel = parallel
        self._n_jobs = n_jobs

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError(
                f"There are not enough subjects ({len(X):d}) to train the model."
            )

        # Check for input data sizes
        if X[0].shape[1] < self.n_features:
            raise ValueError(
                "There are not enough samples to train the model with "
                f"{self.n_features:d} features."
            )

        # Check if all subjects have same number of TRs
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of samples between subjects.")

        # Run SRM
        self.w_, self.s_ = self._srm(X, parallel=self._parallel, n_jobs=self._n_jobs)

        return self

    def transform(
        self,
        X: list[np.ndarray],
        y: Any | None = None,
        *,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
    ) -> list[np.ndarray]:
        """Project each subject's data into the shared response subspace.

        Args:
            X (list[np.ndarray]): One (voxels_i, samples_i) array per fitted
                subject, in the same order as `fit`; voxel and sample counts may
                vary across subjects.
            y (Any | None): Ignored; present for scikit-learn compatibility.
            parallel (str | None): `'cpu'` (default) transforms subjects in
                parallel with joblib; None runs single-threaded NumPy; `'gpu'`
                raises `NotImplementedError`.
            n_jobs (int): Number of CPU workers when `parallel='cpu'`; -1
                (default) reuses the value from `fit`, itself resolved from
                available memory.

        Returns:
            list[np.ndarray]: Shared responses, element i of shape
                (n_features, samples_i).
        """

        _validate_srm_parallel(parallel)

        # Check if the model exist
        if hasattr(self, "w_") is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError(
                "The number of subjects does not match the one in the model."
            )

        # Handle parallelization for transform
        if parallel == "cpu" and len(X) > 1:
            # CPU-parallel transform across subjects
            from joblib import Parallel, delayed

            def _transform_one_subject(subj_idx):
                """Transform one subject."""
                return self.w_[subj_idx].T.dot(X[subj_idx])

            # Prefer an explicitly-passed n_jobs; fall back to the fit-time value
            n_jobs_to_use = n_jobs if n_jobs != -1 else getattr(self, "_n_jobs", -1)
            if n_jobs_to_use == -1:
                # Auto-detect based on memory
                from nltools.algorithms.backends import auto_n_jobs_for_arrays

                n_jobs_to_use = auto_n_jobs_for_arrays(X)

            s = Parallel(n_jobs=n_jobs_to_use)(
                delayed(_transform_one_subject)(i) for i in range(len(X))
            )
        else:
            # Single-threaded transform
            s = [self.w_[subject].T.dot(X[subject]) for subject in range(len(X))]

        return s

    def _objective_function(self, data, w, s):
        """Calculate the objective function (mean squared reconstruction error).

        Args:
            data (list[np.ndarray]): One (voxels_i, samples) array per subject.
            w (list[np.ndarray]): Per-subject orthogonal transforms $W_i$, element
                i of shape (voxels_i, n_features).
            s (np.ndarray): The shared response, shape (n_features, samples).

        Returns:
            float: $\\frac{1}{2T} \\sum_i ||X_i - W_i S||_F^2$.
        """
        subjects = len(data)
        objective = 0.0
        for m in range(subjects):
            objective += np.linalg.norm(data[m] - w[m].dot(s), "fro") ** 2

        return objective * 0.5 / data[0].shape[1]

    def _compute_shared_response(self, data, w):
        """Compute the shared response S as the mean of $W_i^T X_i$ over subjects.

        Args:
            data (list[np.ndarray]): One (voxels_i, samples) array per subject.
            w (list[np.ndarray]): Per-subject orthogonal transforms $W_i$, element
                i of shape (voxels_i, n_features).

        Returns:
            np.ndarray: The shared response, shape (n_features, samples).
        """
        s = np.zeros((w[0].shape[1], data[0].shape[1]))
        for m in range(len(w)):
            s = s + w[m].T.dot(data[m])
        s /= len(w)

        return s

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Update the mapping $W_i$ for one subject.

        Solves the orthogonal Procrustes problem
        $\\min ||X_i - W_i S||_F^2$ subject to $W_i^T W_i = I$: with the SVD
        $U \\Sigma V^T = X_i S^T$, the optimum is $W_i = U V^T$.

        Args:
            Xi (np.ndarray): The subject's data $X_i$, shape (voxels, timepoints).
            S (np.ndarray): The shared response, shape (n_features, timepoints).

        Returns:
            np.ndarray: The orthogonal transform $W_i$, shape (voxels, n_features).
        """
        # Compute cross-covariance: X_i S^T
        A = Xi.dot(S.T)
        # Solve the Procrustes problem via SVD
        # Optimal orthogonal transform: W_i = U V^T where A = U Σ V^T
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X: np.ndarray) -> np.ndarray:
        """Transform a new subject using the existing model.

        The subject is assumed to have received equivalent stimulation.

        Args:
            X (np.ndarray): The new subject's data, shape (voxels, timepoints);
                the timepoints must match the fitted shared response.

        Returns:
            np.ndarray: Orthogonal mapping $W_{new}$ for the new subject, shape
                (voxels, n_features).
        """
        # Check if the model exist
        if hasattr(self, "w_") is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_.shape[1]:
            raise ValueError(
                "The number of timepoints(TRs) does not match the one in the model."
            )

        w = self._update_transform_subject(X, self.s_)

        return w

    def _srm(self, data, parallel: str | None = None, n_jobs: int = -1):
        """Block coordinate descent algorithm for fitting the deterministic SRM.

        Args:
            data (list[np.ndarray]): One (voxels_i, samples) array per subject.
            parallel (str | None): None or `'cpu'` (joblib over subjects in the
                transform update).
            n_jobs (int): Number of CPU workers; -1 picks a count from available
                memory.

        Returns:
            tuple[list[np.ndarray], np.ndarray]: `(w, s)` — the per-subject
                orthogonal transforms $W_i$ (element i of shape (voxels_i,
                n_features)) and the shared response (shape (n_features, samples)).
        """

        subjects = len(data)

        self.random_state_ = np.random.RandomState(self.random_state)
        random_states = [
            np.random.RandomState(self.random_state_.randint(2**32 - 1, dtype=np.int64))
            for i in range(len(data))
        ]

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject.
        w, _ = _init_w_transforms(data, self.n_features, random_states)
        shared_response = self._compute_shared_response(data, w)
        if logger.isEnabledFor(logging.INFO):
            # Calculate the current objective function value
            objective = self._objective_function(data, w, shared_response)
            logger.info(f"Objective function {objective:f}")

        # Main loop of the algorithm
        for iteration in range(self.n_iter):
            logger.info("Iteration %d", iteration + 1)

            # Update each subject's mapping transform W_i:
            # Use CPU parallelization for multi-subject updates if requested
            if parallel == "cpu" and subjects > 1:
                from joblib import Parallel, delayed

                def _update_one_subject(subj_idx):
                    """Update transform for one subject."""
                    a_subject = data[subj_idx].dot(shared_response.T)
                    perturbation = np.zeros(a_subject.shape)
                    np.fill_diagonal(perturbation, 0.001)
                    u_subject, _, v_subject = np.linalg.svd(
                        a_subject + perturbation, full_matrices=False
                    )
                    return u_subject.dot(v_subject)

                # Auto-detect n_jobs if needed
                n_jobs_to_use = n_jobs
                if n_jobs_to_use == -1:
                    from nltools.algorithms.backends import auto_n_jobs_for_arrays

                    n_jobs_to_use = auto_n_jobs_for_arrays(
                        [data[i] for i in range(subjects)]
                    )

                # Parallel update
                w = Parallel(n_jobs=n_jobs_to_use)(
                    delayed(_update_one_subject)(i) for i in range(subjects)
                )
            else:
                # Single-threaded update
                for subject in range(subjects):
                    a_subject = data[subject].dot(shared_response.T)
                    perturbation = np.zeros(a_subject.shape)
                    np.fill_diagonal(perturbation, 0.001)
                    u_subject, _, v_subject = np.linalg.svd(
                        a_subject + perturbation, full_matrices=False
                    )
                    w[subject] = u_subject.dot(v_subject)

            # Update the shared response:
            shared_response = self._compute_shared_response(data, w)

            if logger.isEnabledFor(logging.INFO):
                # Calculate the current objective function value
                objective = self._objective_function(data, w, shared_response)
                logger.info(f"Objective function {objective:f}")

        return w, shared_response
