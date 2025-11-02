#!/usr/bin/env python
# coding: latin-1

"""Shared Response Model (SRM)
===========================

The implementations are based on the following publications:

Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J., & Ramadge,
P. J. (2015). A reduced-dimension fMRI shared response model. In Advances
in Neural Information Processing Systems (pp. 460-468).

Anderson, M. J., Capota, M., Turek, J. S., Zhu, X., Willke, T. L., Wang,
Y., & Norman, K. A. (2016, December). Enabling factor analysis on
thousand-subject neuroimaging datasets. In Big Data (Big Data),
2016 IEEE International Conference on (pp. 1151-1160). IEEE.

References
----------
.. [Chen2015] Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J.,
   & Ramadge, P. J. (2015). A reduced-dimension fMRI shared response model.
   In Advances in Neural Information Processing Systems (pp. 460-468).

.. [Anderson2016] Anderson, M. J., Capota, M., Turek, J. S., Zhu, X.,
   Willke, T. L., Wang, Y., & Norman, K. A. (2016, December). Enabling
   factor analysis on thousand-subject neuroimaging datasets. In Big Data
   (Big Data), 2016 IEEE International Conference on (pp. 1151-1160). IEEE.

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
from typing import List, Optional, Any, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.exceptions import NotFittedError
import sys

__all__ = ["SRM", "DetSRM"]

logger = logging.getLogger(__name__)


def _init_w_transforms(
    data: List[np.ndarray], features: int, random_states: List[Any]
) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
    """Initialize the mappings (Wi) for the SRM with random orthogonal matrices.

    Args:
        data (list of 2D arrays, element i has shape=[voxels_i, samples]):
            Each element in the list contains the fMRI data of one subject.
        features (int): The number of features in the model.
        random_states (list of `RandomState`s): One `RandomState` instance per subject.

    Returns:
        w (list of array, element i has shape=[voxels_i, features]):
            The initialized orthogonal transforms (mappings) :math:`W_i` for each
            subject.
        voxels (list of int):
            A list with the number of voxels per subject.

    Note:
        This function assumes that the numpy random number generator was
        initialized.

        Not thread safe.
    """
    w = []
    subjects = len(data)
    voxels = np.empty(subjects, dtype=int)

    # Set Wi to a random orthogonal voxels by features matrix
    for subject in range(subjects):
        if data[subject] is not None:
            voxels[subject] = data[subject].shape[0]
            rnd_matrix = random_states[subject].random_sample(
                (voxels[subject], features)
            )
            q, r = np.linalg.qr(rnd_matrix)
            w.append(q)
        else:
            voxels[subject] = 0
            w.append(None)

    return w, voxels


class SRM(BaseEstimator, TransformerMixin):
    """Probabilistic Shared Response Model (SRM)

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Args:
        n_iter (int, default=10): Number of iterations to run the algorithm.
        features (int, default=50): Number of features to compute.
        rand_seed (int, default=0): Seed for initializing the random number generator.

    Attributes:
        w_ (list of array, element i has shape=[voxels_i, features]):
            The orthogonal transforms (mappings) for each subject.
        s_ (array, shape=[features, samples]):
            The shared response.
        sigma_s_ (array, shape=[features, features]):
            The covariance of the shared response Normal distribution.
        mu_ (list of array, element i has shape=[voxels_i]):
            The voxel means over the samples for each subject.
        rho2_ (array, shape=[subjects]):
            The estimated noise variance :math:`\\rho_i^2` for each subject
        random_state_ (`RandomState`):
            Random number generator initialized using rand_seed

    Note:
        The number of voxels may be different between subjects. However, the
        number of samples must be the same across subjects.

        The probabilistic Shared Response Model is approximated using the
        Expectation Maximization (EM) algorithm proposed in [Chen2015]_. The
        implementation follows the optimizations published in [Anderson2016]_.

        This is a single node version.

        The run-time complexity is :math:`O(I (V T K + V K^2 + K^3))` and the
        memory complexity is :math:`O(V T)` with I - the number of iterations,
        V - the sum of voxels from all subjects, T - the number of samples, and
        K - the number of features (typically, :math:`V \\gg T \\gg K`).

    Examples:
        Basic multi-subject SRM fitting:

        >>> from nltools.algorithms import SRM
        >>> import numpy as np
        >>>
        >>> # Create sample data (3 subjects)
        >>> data = [np.random.randn(100, 50) for _ in range(3)]
        >>>
        >>> # Fit SRM with CPU parallelization (default)
        >>> srm = SRM(n_iter=10, features=50)
        >>> srm.fit(data, parallel="cpu", n_jobs=-1)
        >>>
        >>> # Transform to shared response space
        >>> shared_responses = srm.transform(data)
        >>>
        >>> # Access fitted model components
        >>> w = srm.w_  # Subject-specific transforms
        >>> s = srm.s_  # Shared response
    """

    def __init__(
        self, n_iter: int = 10, features: int = 50, rand_seed: int = 0
    ) -> None:
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        return

    def fit(
        self,
        X: List[np.ndarray],
        y: Optional[Any] = None,
        parallel: Optional[str] = "cpu",
        n_jobs: int = -1,
        max_gpu_memory_gb: float = 4.0,
    ) -> "SRM":
        """Compute the probabilistic Shared Response Model

        Args:
            X (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            y: not used
            parallel (str, optional): Execution backend.
                - None: Single-threaded NumPy (debugging/small problems)
                - "cpu": CPU parallelization via joblib (default, multi-subject processing)
                - "gpu": GPU acceleration (not yet implemented, falls back to CPU)
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect based on memory).
                Only used when parallel="cpu". Defaults to -1.
            max_gpu_memory_gb (float): Maximum GPU memory budget in GB (default: 4.0).
                Only used when parallel="gpu". Defaults to 4.0.

        Returns:
            self (SRM): Fitted model
        """
        logger.info("Starting Probabilistic SRM")
        
        # Validate parallel parameter
        if parallel not in [None, "cpu", "gpu"]:
            raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel}")
        
        # Store parallel settings for use in _srm
        self._parallel = parallel
        self._n_jobs = n_jobs
        self._max_gpu_memory_gb = max_gpu_memory_gb

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError(
                "There are not enough subjects ({0:d}) to train the model.".format(
                    len(X)
                )
            )

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features)
            )

        # Check if all subjects have same number of TRs
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            if X[subject] is not None:
                assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of samples between subjects.")

        # Run SRM
        self.sigma_s_, self.w_, self.mu_, self.rho2_, self.s_ = self._srm(
            X, parallel=self._parallel, n_jobs=self._n_jobs
        )

        return self

    def transform(
        self,
        X: List[np.ndarray],
        y: Optional[Any] = None,
        parallel: Optional[str] = "cpu",
        n_jobs: int = -1,
    ) -> List[np.ndarray]:
        """Use the model to transform matrix to Shared Response space

        Args:
            X (list of 2D arrays, element i has shape=[voxels_i, samples_i]):
                Each element in the list contains the fMRI data of one subject.
                Note that number of voxels and samples can vary across subjects.
            y: not used (as it is unsupervised learning)
            parallel (str, optional): Execution backend.
                - None: Single-threaded NumPy (debugging/small problems)
                - "cpu": CPU parallelization via joblib (default, multi-subject processing)
                - "gpu": GPU acceleration (not yet implemented, falls back to CPU)
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect based on memory).
                Only used when parallel="cpu". Defaults to -1.

        Returns:
            s (list of 2D arrays, element i has shape=[features_i, samples_i]):
                Shared responses from input data (X)
        """
        
        # Validate parallel parameter
        if parallel not in [None, "cpu", "gpu"]:
            raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel}")

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
            
            # Use stored n_jobs from fit if available, otherwise use provided
            n_jobs_to_use = getattr(self, "_n_jobs", n_jobs)
            if n_jobs_to_use == -1:
                # Auto-detect based on memory
                try:
                    from nltools.algorithms.inference.utils import _auto_n_jobs_cpu, _estimate_data_size_mb
                    # Estimate memory for largest subject
                    max_size_mb = max(_estimate_data_size_mb(x) for x in X if x is not None)
                    n_jobs_to_use = _auto_n_jobs_cpu(
                        data_size_mb=max_size_mb,
                        n_permute=len(X),
                        max_memory_gb=8.0,
                        min_jobs=1,
                    )
                except ImportError:
                    # Fallback to single-threaded if inference utils not available
                    n_jobs_to_use = 1
            
            s = Parallel(n_jobs=n_jobs_to_use)(
                delayed(_transform_one_subject)(i) for i in range(len(X))
            )
        else:
            # Single-threaded transform
            s = [None] * len(X)
            for subject in range(len(X)):
                if X[subject] is not None:
                    s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _init_structures(self, data, subjects):
        """Initializes data structures for SRM and preprocess the data.

        Args:
            data (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            subjects (int): The total number of subjects in `data`.

        Returns:
            x (list of array, element i has shape=[voxels_i, samples]):
                Demeaned data for each subject.
            mu (list of array, element i has shape=[voxels_i]):
                Voxel means over samples, per subject.
            rho2 (array, shape=[subjects]):
                Noise variance :math:`\\rho^2` per subject.
            trace_xtx (array, shape=[subjects]):
                The squared Frobenius norm of the demeaned data in `x`.
        """
        x = []
        mu = []
        rho2 = np.zeros(subjects)

        trace_xtx = np.zeros(subjects)

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
        """Calculate the log-likelihood function

        Args:
            chol_sigma_s_rhos (array, shape=[features, features]):
                Cholesky factorization of the matrix (Sigma_S + sum_i(1/rho_i^2) * I)
            log_det_psi (float):
                Determinant of diagonal matrix Psi (containing the rho_i^2 value
                voxels_i times).
            chol_sigma_s (array, shape=[features, features]):
                Cholesky factorization of the matrix Sigma_S
            trace_xt_invsigma2_x (float):
                Trace of :math:`\\sum_i (||X_i||_F^2/\\rho_i^2)`
            inv_sigma_s_rhos (array, shape=[features, features]):
                Inverse of :math:`(\\Sigma_S + \\sum_i(1/\\rho_i^2) * I)`
            wt_invpsi_x (array, shape=[features, samples]):
            samples (int): The total number of samples in the data.

        Returns:
            loglikehood (float): The log-likelihood value.
        """
        log_det = (
            np.log(np.diag(chol_sigma_s_rhos) ** 2).sum()
            + log_det_psi
            + np.log(np.diag(chol_sigma_s) ** 2).sum()
        )
        loglikehood = -0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
        loglikehood += 0.5 * np.trace(
            wt_invpsi_x.T.dot(inv_sigma_s_rhos).dot(wt_invpsi_x)
        )
        # + const --> -0.5*nTR*nvoxel*subjects*math.log(2*math.pi)

        return loglikehood

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Updates the mappings `W_i` for one subject.

        Args:
            Xi (array, shape=[voxels, timepoints]):
                The fMRI data :math:`X_i` for aligning the subject.
            S (array, shape=[features, timepoints]):
                The shared response.

        Returns:
            Wi (array, shape=[voxels, features]):
                The orthogonal transform (mapping) :math:`W_i` for the subject.
        """
        A = Xi.dot(S.T)
        # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X: np.ndarray) -> np.ndarray:
        """Transform a new subject using the existing model.
        The subject is assumed to have recieved equivalent stimulation

        Args:
            X (2D array, shape=[voxels, timepoints]):
                The fMRI data of the new subject.

        Returns:
            w (2D array, shape=[voxels, features]):
                Orthogonal mapping `W_{new}` for new subject
        """
        # Check if the model exist
        if hasattr(self, "w_") is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_.shape[1]:
            raise ValueError(
                "The number of timepoints(TRs) does not match theone in the model."
            )

        w = self._update_transform_subject(X, self.s_)

        return w

    def _srm(self, data, parallel: Optional[str] = None, n_jobs: int = -1):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.

        Args:
            data (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            parallel (str, optional): Execution backend (None, "cpu", or "gpu").
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect).

        Returns:
            sigma_s (array, shape=[features, features]):
                The covariance :math:`\\Sigma_s` of the shared response Normal
                distribution.
            w (list of array, element i has shape=[voxels_i, features]):
                The orthogonal transforms (mappings) :math:`W_i` for each subject.
            mu (list of array, element i has shape=[voxels_i]):
                The voxel means :math:`\\mu_i` over the samples for each subject.
            rho2 (array, shape=[subjects]):
                The estimated noise variance :math:`\\rho_i^2` for each subject
            s (array, shape=[features, samples]):
                The shared response.
        """

        samples = min([d.shape[1] for d in data if d is not None], default=sys.maxsize)
        subjects = len(data)
        self.random_state_ = np.random.RandomState(self.rand_seed)
        random_states = [
            np.random.RandomState(self.random_state_.randint(2**32 - 1, dtype=np.int64))
            for i in range(len(data))
        ]

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject, and trace_xtx with
        # the ||X_i||_F^2 of each subject.
        w, voxels = _init_w_transforms(data, self.features, random_states)
        x, mu, rho2, trace_xtx = self._init_structures(data, subjects)
        shared_response = np.zeros((self.features, samples))
        sigma_s = np.identity(self.features)

        # Main loop of the algorithm (run
        for iteration in range(self.n_iter):
            logger.info("Iteration %d" % (iteration + 1))

            # E-step:

            # Sum the inverted the rho2 elements for computing W^T * Psi^-1 * W
            rho0 = (1 / rho2).sum()

            # Invert Sigma_s using Cholesky factorization
            (chol_sigma_s, lower_sigma_s) = scipy.linalg.cho_factor(
                sigma_s, check_finite=False
            )
            inv_sigma_s = scipy.linalg.cho_solve(
                (chol_sigma_s, lower_sigma_s),
                np.identity(self.features),
                check_finite=False,
            )

            # Invert (Sigma_s + rho_0 * I) using Cholesky factorization
            sigma_s_rhos = inv_sigma_s + np.identity(self.features) * rho0
            chol_sigma_s_rhos, lower_sigma_s_rhos = scipy.linalg.cho_factor(
                sigma_s_rhos, check_finite=False
            )
            inv_sigma_s_rhos = scipy.linalg.cho_solve(
                (chol_sigma_s_rhos, lower_sigma_s_rhos),
                np.identity(self.features),
                check_finite=False,
            )

            # Compute the sum of W_i^T * rho_i^-2 * X_i, and the sum of traces
            # of X_i^T * rho_i^-2 * X_i
            wt_invpsi_x = np.zeros((self.features, samples))
            trace_xt_invsigma2_x = 0.0
            for subject in range(subjects):
                if data[subject] is not None:
                    wt_invpsi_x += (w[subject].T.dot(x[subject])) / rho2[subject]
                    trace_xt_invsigma2_x += trace_xtx[subject] / rho2[subject]

            log_det_psi = np.sum(np.log(rho2) * voxels)

            # Update the shared response
            shared_response = sigma_s.dot(
                np.identity(self.features) - rho0 * inv_sigma_s_rhos
            ).dot(wt_invpsi_x)

            # M-step

            # Update Sigma_s and compute its trace
            sigma_s = (
                inv_sigma_s_rhos + shared_response.dot(shared_response.T) / samples
            )
            trace_sigma_s = samples * np.trace(sigma_s)

            # Update each subject's mapping transform W_i and error variance
            # rho_i^2
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
                        rho2_new += -2 * np.sum(w_new * a_subject).sum()
                        rho2_new += trace_sigma_s
                        rho2_new /= samples * voxels[subj_idx]
                        return w_new, rho2_new
                    else:
                        return None, 0.0
                
                # Auto-detect n_jobs if needed
                n_jobs_to_use = n_jobs
                if n_jobs_to_use == -1:
                    try:
                        from nltools.algorithms.inference.utils import _auto_n_jobs_cpu, _estimate_data_size_mb
                        # Estimate memory for largest subject
                        max_size_mb = max(_estimate_data_size_mb(x[i]) for i in range(subjects) if x[i] is not None)
                        n_jobs_to_use = _auto_n_jobs_cpu(
                            data_size_mb=max_size_mb,
                            n_permute=subjects,
                            max_memory_gb=8.0,
                            min_jobs=1,
                        )
                    except ImportError:
                        n_jobs_to_use = 1
                
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
                        rho2[subject] += -2 * np.sum(w[subject] * a_subject).sum()
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
                logger.info("Objective function %f" % loglike)

        return sigma_s, w, mu, rho2, shared_response


class DetSRM(BaseEstimator, TransformerMixin):
    """Deterministic Shared Response Model (DetSRM)

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Args:
        n_iter (int, default=10): Number of iterations to run the algorithm.
        features (int, default=50): Number of features to compute.
        rand_seed (int, default=0): Seed for initializing the random number generator.

    Attributes:
        w_ (list of array, element i has shape=[voxels_i, features]):
            The orthogonal transforms (mappings) for each subject.
        s_ (array, shape=[features, samples]):
            The shared response.
        random_state_ (`RandomState`):
            Random number generator initialized using rand_seed

    Note:
        The number of voxels may be different between subjects. However, the
        number of samples must be the same across subjects.

        The Deterministic Shared Response Model is approximated using the
        Block Coordinate Descent (BCD) algorithm proposed in [Chen2015]_.

        This is a single node version.

        The run-time complexity is :math:`O(I (V T K + V K^2))` and the memory
        complexity is :math:`O(V T)` with I - the number of iterations, V - the
        sum of voxels from all subjects, T - the number of samples, K - the
        number of features (typically, :math:`V \\gg T \\gg K`), and N - the
        number of subjects.

    Examples:
        Basic multi-subject DetSRM fitting:

        >>> from nltools.algorithms import DetSRM
        >>> import numpy as np
        >>>
        >>> # Create sample data (3 subjects)
        >>> data = [np.random.randn(100, 50) for _ in range(3)]
        >>>
        >>> # Fit DetSRM with CPU parallelization (default)
        >>> detsrm = DetSRM(n_iter=10, features=50)
        >>> detsrm.fit(data, parallel="cpu", n_jobs=-1)
        >>>
        >>> # Transform to shared response space
        >>> shared_responses = detsrm.transform(data)
        >>>
        >>> # Access fitted model components
        >>> w = detsrm.w_  # Subject-specific transforms
        >>> s = detsrm.s_  # Shared response
    """

    def __init__(
        self, n_iter: int = 10, features: int = 50, rand_seed: int = 0
    ) -> None:
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        return

    def fit(
        self,
        X: List[np.ndarray],
        y: Optional[Any] = None,
        parallel: Optional[str] = "cpu",
        n_jobs: int = -1,
        max_gpu_memory_gb: float = 4.0,
    ) -> "DetSRM":
        """Compute the Deterministic Shared Response Model

        Args:
            X (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            y: not used
            parallel (str, optional): Execution backend.
                - None: Single-threaded NumPy (debugging/small problems)
                - "cpu": CPU parallelization via joblib (default, multi-subject processing)
                - "gpu": GPU acceleration (not yet implemented, falls back to CPU)
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect based on memory).
                Only used when parallel="cpu". Defaults to -1.
            max_gpu_memory_gb (float): Maximum GPU memory budget in GB (default: 4.0).
                Only used when parallel="gpu". Defaults to 4.0.

        Returns:
            self (DetSRM): Fitted model
        """
        logger.info("Starting Deterministic SRM")
        
        # Validate parallel parameter
        if parallel not in [None, "cpu", "gpu"]:
            raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel}")
        
        # Store parallel settings for use in _srm
        self._parallel = parallel
        self._n_jobs = n_jobs
        self._max_gpu_memory_gb = max_gpu_memory_gb

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError(
                "There are not enough subjects ({0:d}) to train the model.".format(
                    len(X)
                )
            )

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features)
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
        X: List[np.ndarray],
        y: Optional[Any] = None,
        parallel: Optional[str] = "cpu",
        n_jobs: int = -1,
    ) -> List[np.ndarray]:
        """Use the model to transform data to the Shared Response subspace

        Args:
            X (list of 2D arrays, element i has shape=[voxels_i, samples_i]):
                Each element in the list contains the fMRI data of one subject.
            y: not used
            parallel (str, optional): Execution backend.
                - None: Single-threaded NumPy (debugging/small problems)
                - "cpu": CPU parallelization via joblib (default, multi-subject processing)
                - "gpu": GPU acceleration (not yet implemented, falls back to CPU)
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect based on memory).
                Only used when parallel="cpu". Defaults to -1.

        Returns:
            s (list of 2D arrays, element i has shape=[features_i, samples_i]):
                Shared responses from input data (X)
        """
        
        # Validate parallel parameter
        if parallel not in [None, "cpu", "gpu"]:
            raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel}")

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
            
            # Use stored n_jobs from fit if available, otherwise use provided
            n_jobs_to_use = getattr(self, "_n_jobs", n_jobs)
            if n_jobs_to_use == -1:
                # Auto-detect based on memory
                try:
                    from nltools.algorithms.inference.utils import _auto_n_jobs_cpu, _estimate_data_size_mb
                    # Estimate memory for largest subject
                    max_size_mb = max(_estimate_data_size_mb(x) for x in X)
                    n_jobs_to_use = _auto_n_jobs_cpu(
                        data_size_mb=max_size_mb,
                        n_permute=len(X),
                        max_memory_gb=8.0,
                        min_jobs=1,
                    )
                except ImportError:
                    # Fallback to single-threaded if inference utils not available
                    n_jobs_to_use = 1
            
            s = Parallel(n_jobs=n_jobs_to_use)(
                delayed(_transform_one_subject)(i) for i in range(len(X))
            )
        else:
            # Single-threaded transform
            s = [None] * len(X)
            for subject in range(len(X)):
                s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _objective_function(self, data, w, s):
        """Calculate the objective function

        Args:
            data (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            w (list of 2D arrays, element i has shape=[voxels_i, features]):
                The orthogonal transforms (mappings) :math:`W_i` for each subject.
            s (array, shape=[features, samples]):
                The shared response

        Returns:
            objective (float): The objective function value.
        """
        subjects = len(data)
        objective = 0.0
        for m in range(subjects):
            objective += np.linalg.norm(data[m] - w[m].dot(s), "fro") ** 2

        return objective * 0.5 / data[0].shape[1]

    def _compute_shared_response(self, data, w):
        """Compute the shared response S

        Args:
            data (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            w (list of 2D arrays, element i has shape=[voxels_i, features]):
                The orthogonal transforms (mappings) :math:`W_i` for each subject.

        Returns:
            s (array, shape=[features, samples]):
                The shared response for the subjects data with the mappings in w.
        """
        s = np.zeros((w[0].shape[1], data[0].shape[1]))
        for m in range(len(w)):
            s = s + w[m].T.dot(data[m])
        s /= len(w)

        return s

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Updates the mappings `W_i` for one subject.

        Args:
            Xi (array, shape=[voxels, timepoints]):
                The fMRI data :math:`X_i` for aligning the subject.
            S (array, shape=[features, timepoints]):
                The shared response.

        Returns:
            Wi (array, shape=[voxels, features]):
                The orthogonal transform (mapping) :math:`W_i` for the subject.
        """
        A = Xi.dot(S.T)
        # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X: np.ndarray) -> np.ndarray:
        """Transform a new subject using the existing model.
        The subject is assumed to have recieved equivalent stimulation

        Args:
            X (2D array, shape=[voxels, timepoints]):
                The fMRI data of the new subject.

        Returns:
            w (2D array, shape=[voxels, features]):
                Orthogonal mapping `W_{new}` for new subject
        """
        # Check if the model exist
        if hasattr(self, "w_") is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_.shape[1]:
            raise ValueError(
                "The number of timepoints(TRs) does not match theone in the model."
            )

        w = self._update_transform_subject(X, self.s_)

        return w

    def _srm(self, data, parallel: Optional[str] = None, n_jobs: int = -1):
        """Block Coordinate Descent algorithm for fitting the deterministic SRM.

        Args:
            data (list of 2D arrays, element i has shape=[voxels_i, samples]):
                Each element in the list contains the fMRI data of one subject.
            parallel (str, optional): Execution backend (None, "cpu", or "gpu").
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect).

        Returns:
            w (list of array, element i has shape=[voxels_i, features]):
                The orthogonal transforms (mappings) :math:`W_i` for each subject.
            s (array, shape=[features, samples]):
                The shared response.
        """

        subjects = len(data)

        self.random_state_ = np.random.RandomState(self.rand_seed)
        random_states = [
            np.random.RandomState(self.random_state_.randint(2**32 - 1, dtype=np.int64))
            for i in range(len(data))
        ]

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject.
        w, _ = _init_w_transforms(data, self.features, random_states)
        shared_response = self._compute_shared_response(data, w)
        if logger.isEnabledFor(logging.INFO):
            # Calculate the current objective function value
            objective = self._objective_function(data, w, shared_response)
            logger.info("Objective function %f" % objective)

        # Main loop of the algorithm
        for iteration in range(self.n_iter):
            logger.info("Iteration %d" % (iteration + 1))

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
                    try:
                        from nltools.algorithms.inference.utils import _auto_n_jobs_cpu, _estimate_data_size_mb
                        # Estimate memory for largest subject
                        max_size_mb = max(_estimate_data_size_mb(data[i]) for i in range(subjects))
                        n_jobs_to_use = _auto_n_jobs_cpu(
                            data_size_mb=max_size_mb,
                            n_permute=subjects,
                            max_memory_gb=8.0,
                            min_jobs=1,
                        )
                    except ImportError:
                        n_jobs_to_use = 1
                
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
                logger.info("Objective function %f" % objective)

        return w, shared_response
