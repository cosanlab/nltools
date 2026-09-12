"""Tools to simulate multivariate brain and grid data for testing analysis pipelines."""

__all__ = ["SimulateGrid", "Simulator"]


import os
import numpy as np
import nibabel as nib
from nibabel.affines import voxel_sizes
import matplotlib.pyplot as plt
from nilearn.image.resampling import coord_transform
from nilearn.masking import apply_mask, unmask
from scipy.stats import multivariate_normal, binom, ttest_1samp
from nltools.data import BrainData
from nltools.algorithms.corrections import fdr
from nltools.templates import get_brainspace
import csv
from copy import deepcopy
from sklearn.utils import check_random_state


def _grid_center_world(mask):
    """Return the world (MNI) millimeter coordinate of a mask's grid center."""
    i, j, k = (np.array(mask.shape) // 2).tolist()
    return [float(v) for v in coord_transform(i, j, k, mask.affine)]


class Simulator:
    """Simulate fMRI data with realistic spatial and temporal characteristics.

    This class provides methods for generating synthetic fMRI data with
    controlled signal patterns, including Gaussian blobs, multi-subject
    datasets, and various noise structures. Useful for testing analysis
    pipelines and power analyses.

    Args:
        brain_mask (str | nibabel.Nifti1Image, optional): Path to a NIfTI brain mask
            file, a nibabel image, or None to use the default template mask.
        output_dir (str, optional): Directory for saving generated data. Defaults to
            the current working directory.
        random_state (int | np.random.RandomState, optional): Seed or RandomState for
            reproducibility.

    Attributes:
        brain_mask (nibabel.Nifti1Image): The brain mask image used for simulation.
        output_dir (str): Output directory path.
        random_state (np.random.RandomState): Random state for reproducible simulations.
        data (BrainData | nibabel.Nifti1Image): Most recently simulated data; set by
            the `create_*` methods.
        y (pl.DataFrame | np.ndarray): Outcome values paired with `data`; set by the
            `create_*` methods.
        rep_id (pl.DataFrame | list): Repetition/subject id per observation; set by
            the `create_*` methods.

    Examples:
        ```python
        from nltools.data.simulator import Simulator

        sim = Simulator(random_state=42)
        # Create a dataset with signal in specific regions
        data = sim.create_data(levels=[1, -1, 1, -1], sigma=1, reps=10)
        ```
    """

    def __init__(
        self, *, brain_mask=None, output_dir=None, random_state=None
    ):  # no scoring param
        # self.resource_folder = os.path.join(os.getcwd(),'resources')
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd())
        else:
            self.output_dir = output_dir

        if isinstance(brain_mask, str):
            brain_mask = nib.load(brain_mask)
        elif brain_mask is None:
            brain_mask = nib.load(get_brainspace().mask)
        elif not isinstance(brain_mask, nib.nifti1.Nifti1Image):
            raise ValueError("brain_mask is not a string or a nibabel instance")
        self.brain_mask = brain_mask
        self.random_state = check_random_state(random_state)

    def gaussian(self, mu, sigma, i_tot):
        """Create a 3D gaussian signal normalized to a given intensity.

        Geometry is millimeters: `mu` is a world (MNI) coordinate and `sigma` a
        physical width, both converted to voxel units through the brain mask's
        affine, so the same request describes the same blob on any grid.

        Args:
            mu (array-like): Center of the gaussian `[x, y, z]` in world (MNI)
                millimeters.
            sigma (float | array-like): Standard deviation in millimeters — a scalar
                for an isotropic blob or one width per axis `[sx, sy, sz]`.
            i_tot (float): Total activation; the gaussian is rescaled so its sum
                within the brain mask equals this value.

        Returns:
            np.ndarray: 3-D array the shape of the brain mask.

        Note:
            `sigma` is converted per axis with `nibabel.affines.voxel_sizes`, so the
            millimeter widths map onto world axes only for an axis-aligned affine. On
            an oblique affine the blob's principal axes follow the voxel grid.
        """
        affine = self.brain_mask.affine
        mu_voxel = np.asarray(
            coord_transform(
                float(mu[0]), float(mu[1]), float(mu[2]), np.linalg.inv(affine)
            ),
            dtype=float,
        )
        sigma_voxel = np.broadcast_to(
            np.asarray(sigma, dtype=float), (3,)
        ) / voxel_sizes(affine)

        x, y, z = np.mgrid[
            0 : self.brain_mask.shape[0],
            0 : self.brain_mask.shape[1],
            0 : self.brain_mask.shape[2],
        ]

        # Need an (N, 3) array of (x, y) pairs.
        xyz = np.column_stack([x.flat, y.flat, z.flat])

        covariance = np.diag(sigma_voxel**2)
        g = multivariate_normal.pdf(xyz, mean=mu_voxel, cov=covariance)

        # Reshape back to a 3D grid.
        g = g.reshape(x.shape).astype(float)

        # select only the regions within the brain mask
        g = np.multiply(self.brain_mask.get_fdata(), g)
        # adjust total intensity of gaussian
        g = np.multiply(i_tot / np.sum(g), g)

        return g

    def sphere(self, radius, center):
        """Create a sphere of a given radius at a world coordinate in the brain mask.

        Delegates to `nltools.mask.create_sphere`, so the radius is millimeters and
        the center is a world (MNI) coordinate resolved through the mask's affine.

        Args:
            radius (int | float): Radius of the sphere in millimeters.
            center (array-like): Center of the sphere `[x, y, z]` in world (MNI)
                millimeters.

        Returns:
            np.ndarray: 3-D array the shape of the brain mask, 1 inside the sphere and
                0 elsewhere.
        """
        from nltools.mask import create_sphere

        drawn = create_sphere(
            [float(c) for c in center], radius=radius, mask=self.brain_mask
        )
        return np.asarray(drawn.dataobj, dtype=float)

    def normal_noise(self, mu, sigma):
        """Produce a normal noise distribution for all points in the brain mask.

        Args:
            mu (float): Mean of the noise (usually 0).
            sigma (float): Standard deviation of the noise.

        Returns:
            np.ndarray: 3-D array the shape of the brain mask filled with noise inside
                the mask.
        """

        vlength = int(np.sum(self.brain_mask.get_fdata()))
        if sigma != 0:
            n = self.random_state.normal(mu, sigma, vlength)
        else:
            # float, not a list of Python ints: an int64 array makes nibabel
            # warn and silently downcast the image to int32.
            n = np.full(vlength, float(mu))
        m = unmask(n, self.brain_mask)

        # return the 3D numpy matrix of zeros containing the brain mask filled with noise produced over a normal distribution
        return m.get_fdata()

    def to_nifti(self, m):
        """Convert a numpy array to a NIfTI image with the brain mask's affine.

        Args:
            m (np.ndarray): 3-D (or 4-D) array to convert.

        Returns:
            nibabel.Nifti1Image: The array as a float32 image.
        """
        if not (isinstance(m, np.ndarray) and len(m.shape) >= 3):  # try 4D
            # if not (type(m) == np.ndarray and len(m.shape) == 3):
            raise ValueError(
                "ERROR: need 3D np.ndarray matrix to create the nifti file"
            )
        m = m.astype(np.float32)
        ni = nib.Nifti1Image(m, affine=self.brain_mask.affine)
        return ni

    def n_spheres(self, radius, center=None):
        """Generate a set of spheres in the brain mask space.

        Delegates to `nltools.mask.create_sphere`, so radii are millimeters and
        centers are world (MNI) coordinates resolved through the mask's affine.

        Args:
            radius (int | float | list): Sphere radius in millimeters, or one radius
                per sphere.
            center (list, optional): Sphere center `[x, y, z]` in world (MNI)
                millimeters, or one center per sphere `[[x1, y1, z1], ...]`. None
                places every sphere at the world coordinate of the mask's grid center.

        Returns:
            np.ndarray: 3-D binary array the shape of the brain mask holding the union
                of the requested spheres.
        """
        from nltools.mask import create_sphere

        if center is None:
            n_requested = (
                len(radius) if isinstance(radius, (list, tuple, np.ndarray)) else 1
            )
            center = [_grid_center_world(self.brain_mask)] * n_requested

        drawn = create_sphere(center, radius=radius, mask=self.brain_mask)
        return np.asarray(drawn.dataobj, dtype=float)

    def create_data(
        self, levels, sigma, *, radius=10, center=None, reps=1, output_dir=None
    ):
        """Create simulated data with discrete intensity levels.

        Args:
            levels (list): Intensities or class labels, one per image in a repetition.
            sigma (float): Standard deviation of the added noise.
            radius (int | float | list): Sphere radius in millimeters, or one radius
                per sphere. Default 10.0.
            center (list, optional): Sphere center `[x, y, z]` in world (MNI)
                millimeters, or one center per sphere `[[x1, y1, z1], ...]`. None
                (the default) places every sphere at the world coordinate of the
                mask's grid center.
            reps (int): Number of repetitions (e.g. trials or subjects). Default 1.
            output_dir (str, optional): Directory to write `data.nii.gz`, `y.csv`, and
                `rep_id.csv` into. If None, nothing is written.

        Returns:
            BrainData: The simulated images with `Y` set to the levels.
        """
        import polars as pl

        # Create reps
        nlevels = len(levels)
        y = levels
        rep_id = [1] * len(levels)
        for i in range(reps - 1):
            y = y + levels
            rep_id.extend([i + 2] * nlevels)

        # Initialize Spheres with options for multiple radii and centers of the spheres (or just an int and a 3D list)
        A = self.n_spheres(radius, center)

        # for each intensity
        A_list = []
        for i in y:
            A_list.append(np.multiply(A, i))

        # generate a different gaussian noise profile for each mask
        mu = 0  # values centered around 0
        N_list = []
        for i in range(len(y)):
            N_list.append(self.normal_noise(mu, sigma))

        # add noise and signal together, then convert to nifti files
        NF_list = []
        for i in range(len(y)):
            NF_list.append(self.to_nifti(np.add(N_list[i], A_list[i])))
        NF_list = BrainData(NF_list)

        # Assign variables to object
        self.data = NF_list
        self.y = pl.DataFrame({"y": y})
        self.rep_id = pl.DataFrame({"rep_id": rep_id})

        dat = self.data
        dat.Y = self.y

        # Write Data to files if requested
        if output_dir is not None and isinstance(output_dir, str):
            NF_list.write(os.path.join(output_dir, "data.nii.gz"))
            self.y.write_csv(os.path.join(output_dir, "y.csv"), include_header=False)
            self.rep_id.write_csv(
                os.path.join(output_dir, "rep_id.csv"), include_header=False
            )
        return dat

    def create_cov_data(
        self, cor, cov, sigma, *, mask=None, reps=1, n_sub=1, output_dir=None
    ):
        """Create continuous simulated data with covariance within a single region.

        Results are stored on `self.data` (a 4-D `nibabel.Nifti1Image`), `self.y`, and
        `self.rep_id`.

        Args:
            cor (float): Covariance between each voxel and the outcome `y`.
            cov (float): Covariance between voxels.
            sigma (float): Standard deviation of the added noise.
            mask (nibabel.Nifti1Image, optional): Region where activations are placed.
                Defaults to a 20 mm sphere at the mask's grid center.
            reps (int): Number of repetitions per subject. Default 1.
            n_sub (int): Number of subjects to simulate. Default 1.
            output_dir (str, optional): Directory to write the image, `y.csv`, and
                `rep_id.csv` into. If None, nothing is written.
        """

        if mask is None:
            # Initialize Spheres with options for multiple radii and centers of the spheres (or just an int and a 3D list)
            A = self.n_spheres(20, None)  # parameters are (radius, center)
            mask = nib.Nifti1Image(A.astype(np.float32), affine=self.brain_mask.affine)

        # Create n_reps with cov for each voxel within sphere
        # Build covariance matrix with each variable correlated with y amount 'cor' and each other amount 'cov'
        # apply_mask on a single 3-D mask returns a 1-D vector; the logic below
        # (flat_sphere.shape[1], np.where(...)[1]) assumes a 2-D (1, n_vox)
        # array, so normalize to 2-D (matching create_ncov_data, which wraps its
        # masks in a list and gets a 2-D result).
        flat_sphere = np.atleast_2d(apply_mask(mask, self.brain_mask))

        n_vox = np.sum(flat_sphere == 1)
        cov_matrix = np.ones([n_vox + 1, n_vox + 1]) * cov
        cov_matrix[0, :] = cor  # set covariance with y
        cov_matrix[:, 0] = cor  # set covariance with all other voxels
        np.fill_diagonal(cov_matrix, 1)  # set diagonal to 1
        mv_sim = self.random_state.multivariate_normal(
            np.zeros([n_vox + 1]), cov_matrix, size=reps
        )
        y = mv_sim[:, 0]
        self.y = y
        mv_sim = mv_sim[:, 1:]
        new_dat = np.ones([mv_sim.shape[0], flat_sphere.shape[1]])
        new_dat[:, np.where(flat_sphere == 1)[1]] = mv_sim
        self.data = unmask(
            np.add(
                new_dat, self.random_state.standard_normal(size=new_dat.shape) * sigma
            ),
            self.brain_mask,
        )  # add noise scaled by sigma
        self.rep_id = [1] * len(y)
        if n_sub > 1:
            self.y = list(self.y)
            for s in range(1, n_sub):
                self.data = nib.concat_images(
                    [
                        self.data,
                        unmask(
                            np.add(
                                new_dat,
                                self.random_state.standard_normal(size=new_dat.shape)
                                * sigma,
                            ),
                            self.brain_mask,
                        ),
                    ],
                    axis=3,
                )  # add noise scaled by sigma
                noise_y = list(y + self.random_state.randn(len(y)) * sigma)
                self.y = self.y + noise_y
                self.rep_id = self.rep_id + [s + 1] * len(mv_sim[:, 0])
            self.y = np.array(self.y)

        # # Old method in 4 D space - much slower
        # x,y,z = np.where(A==1)
        # cov_matrix = np.ones([len(x)+1,len(x)+1]) * cov
        # cov_matrix[0,:] = cor # set covariance with y
        # cov_matrix[:,0] = cor # set covariance with all other voxels
        # np.fill_diagonal(cov_matrix,1) # set diagonal to 1
        # mv_sim = self.random_state.multivariate_normal(np.zeros([len(x)+1]),cov_matrix, size=reps) # simulate data from multivariate covar
        # self.y = mv_sim[:,0]
        # mv_sim = mv_sim[:,1:]
        # A_4d = np.resize(A,(reps,A.shape[0],A.shape[1],A.shape[2]))
        # for i in range(len(x)):
        #     A_4d[:,x[i],y[i],z[i]]=mv_sim[:,i]
        # A_4d = np.rollaxis(A_4d,0,4) # reorder shape of matrix so that time is in 4th dimension
        # self.data = self.to_nifti(np.add(A_4d,self.random_state.standard_normal(size=A_4d.shape)*sigma)) # add noise scaled by sigma
        # self.rep_id = ???  # need to add this later

        # Write Data to files if requested
        if output_dir is not None:
            if isinstance(output_dir, str):
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                self.data.to_filename(
                    os.path.join(
                        output_dir,
                        "maskdata_cor"
                        + str(cor)
                        + "_cov"
                        + str(cov)
                        + "_sigma"
                        + str(sigma)
                        + ".nii.gz",
                    )
                )
                with open(os.path.join(output_dir, "y.csv"), "w", newline="") as y_file:
                    wr = csv.writer(y_file, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.y)

                with open(
                    os.path.join(output_dir, "rep_id.csv"), "w", newline=""
                ) as rep_id_file:
                    wr = csv.writer(rep_id_file, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.rep_id)

    def create_ncov_data(
        self, cor, cov, sigma, *, masks=None, reps=1, n_sub=1, output_dir=None
    ):
        """Create continuous simulated data with covariance across multiple regions.

        Results are stored on `self.data` (a 4-D `nibabel.Nifti1Image`), `self.y`, and
        `self.rep_id`.

        Args:
            cor (float | list[float]): Covariance between each region's voxels and the
                outcome `y`; one value per region.
            cov (float | list[list[float]]): Covariance between voxels; a scalar for a
                single region or a region-by-region matrix.
            sigma (float): Standard deviation of the added noise.
            masks (nibabel.Nifti1Image | list[nibabel.Nifti1Image], optional): Region(s)
                where activations are placed. Defaults to a 20 mm sphere at the mask's
                grid center.
            reps (int): Number of repetitions per subject. Default 1.
            n_sub (int): Number of subjects to simulate. Default 1.
            output_dir (str, optional): Directory to write the image, `y.csv`, and
                `rep_id.csv` into. If None, nothing is written.
        """

        if masks is None:
            # Initialize Spheres with options for multiple radii and centers of the spheres (or just an int and a 3D list)
            A = self.n_spheres(20, None)  # parameters are (radius, center)
            masks = nib.Nifti1Image(A.astype(np.float32), affine=self.brain_mask.affine)

        if type(masks) is nib.nifti1.Nifti1Image:
            masks = [masks]
        if type(cor) is float or type(cor) is int:
            cor = [cor]
        if type(cov) is float or type(cov) is int:
            cov = [[cov]]
        if not len(cor) == len(masks):
            raise ValueError(
                "cor matrix has incompatible dimensions for mask list of length "
                + str(len(masks))
            )
        if (
            not len(cov) == len(masks)
            or len(masks) == 0
            or not len(cov[0]) == len(masks)
        ):
            raise ValueError(
                "cov matrix has incompatible dimensions for mask list of length "
                + str(len(masks))
            )

        # Create n_reps with cov for each voxel within sphere
        # Build covariance matrix with each variable correlated with y amount 'cor' and each other amount 'cov'
        flat_masks = apply_mask(masks, self.brain_mask)

        n_vox = np.sum(
            flat_masks == 1, axis=1
        )  # this is a list, each entry contains number voxels for given mask
        if 0 in n_vox:
            raise ValueError(
                "one or more processing mask does not fit inside the brain mask"
            )

        cov_matrix = np.zeros(
            [np.sum(n_vox) + 1, np.sum(n_vox) + 1]
        )  # one big covariance matrix
        for i, nv in enumerate(n_vox):
            cstart = np.sum(n_vox[:i]) + 1
            cstop = cstart + nv
            cov_matrix[0, cstart:cstop] = cor[i]  # set covariance with y
            cov_matrix[cstart:cstop, 0] = cor[i]  # set covariance with all other voxels
            for j in range(len(masks)):
                rstart = np.sum(n_vox[:j]) + 1
                rstop = rstart + nv
                cov_matrix[cstart:cstop, rstart:rstop] = cov[i][
                    j
                ]  # set covariance of this mask's voxels with each of other masks
        np.fill_diagonal(cov_matrix, 1)  # set diagonal to 1

        # these operations happen in one vector that we'll later split into the separate regions
        mv_sim_l = self.random_state.multivariate_normal(
            np.zeros([np.sum(n_vox) + 1]), cov_matrix, size=reps
        )

        self.y = mv_sim_l[:, 0]
        mv_sim = mv_sim_l[:, 1:]
        new_dats = np.ones([mv_sim.shape[0], flat_masks.shape[1]])

        for rep in range(reps):
            for mask_i in range(len(masks)):
                start = int(np.sum(n_vox[:mask_i]))
                stop = int(start + n_vox[mask_i])
                new_dats[rep, np.where(flat_masks[mask_i, :] == 1)] = mv_sim[
                    rep, start:stop
                ]

        noise = self.random_state.standard_normal(size=new_dats.shape[1]) * sigma
        self.data = unmask(
            np.add(new_dats, noise), self.brain_mask
        )  # append 3d simulated data to list
        self.rep_id = [1] * len(self.y)

        if n_sub > 1:
            self.y = list(self.y)
            y = list(self.y)
            for s in range(1, n_sub):
                # ask Luke about this new version
                noise = (
                    self.random_state.standard_normal(size=new_dats.shape[1]) * sigma
                )
                next_subj = unmask(np.add(new_dats, noise), self.brain_mask)
                self.data = nib.concat_images([self.data, next_subj], axis=3)

                y += list(self.y + self.random_state.randn(len(self.y)) * sigma)
                self.rep_id += [s + 1] * len(mv_sim[:, 0])
            self.y = np.array(y)

        if output_dir is not None:
            if type(output_dir) is str:
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                self.data.to_filename(
                    os.path.join(
                        output_dir,
                        "simulated_data_"
                        + str(sigma)
                        + "sigma_"
                        + str(n_sub)
                        + "subj.nii.gz",
                    )
                )
                with open(os.path.join(output_dir, "y.csv"), "w", newline="") as y_file:
                    wr = csv.writer(y_file, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.y)

                with open(
                    os.path.join(output_dir, "rep_id.csv"), "w", newline=""
                ) as rep_id_file:
                    wr = csv.writer(rep_id_file, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.rep_id)


#: Multiple-comparison corrections `SimulateGrid` implements. `None` applies no
#: correction; `'fdr'` requires `threshold_type='q'`.
_SUPPORTED_CORRECTIONS = (None, "fdr")


def _validate_correction(correction):
    """Raise `ValueError` for an unsupported `correction`.

    Args:
        correction: Value passed as `SimulateGrid`'s `correction` argument.

    Raises:
        ValueError: If `correction` is outside `_SUPPORTED_CORRECTIONS`.
    """
    if correction not in _SUPPORTED_CORRECTIONS:
        raise ValueError(
            f"correction must be one of {_SUPPORTED_CORRECTIONS}; got {correction!r}."
        )


class SimulateGrid:
    """Simulate 2D grid data for testing statistical methods.

    Creates a 2D grid (e.g., 100x100 pixels) with optional embedded signal
    regions and Gaussian noise. Useful for testing multiple comparison
    correction methods, threshold selection, and visualization of
    statistical maps.

    Args:
        grid_width (int): Width/height of the square grid. Default 100.
        signal_width (int): Width of the embedded signal region. Default 20.
        n_subjects (int): Number of simulated subjects. Default 20.
        sigma (float): Standard deviation of the Gaussian noise. Default 1.
        signal_amplitude (float, optional): Amplitude of the embedded signal. If None,
            no signal is added.
        random_state (int | np.random.RandomState, optional): Seed or RandomState for
            reproducibility.

    Attributes:
        data (np.ndarray): Simulated data of shape `(grid_width, grid_width, n_subjects)`.
        signal_mask (np.ndarray | None): Binary grid marking the signal region, or None
            when no signal was added.
        t_values (np.ndarray | None): T-statistic map after `fit()`.
        p_values (np.ndarray | None): P-value map after `fit()`.
        thresholded (np.ndarray | None): Thresholded statistical map after
            `threshold_simulation()`.
        isfit (bool): Whether `fit()` has been called.

    Examples:
        ```python
        from nltools.data.simulator import SimulateGrid

        sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
        sim.fit()
        sim.plot_grid_simulation(threshold=0.05, threshold_type="q", correction="fdr")
        ```
    """

    def __init__(
        self,
        *,
        grid_width=100,
        signal_width=20,
        n_subjects=20,
        sigma=1,
        signal_amplitude=None,
        random_state=None,
    ):
        self.isfit = False
        self.thresholded = None
        self.threshold = None
        self.threshold_type = None
        self.correction = None
        self.t_values = None
        self.p_values = None
        self.n_subjects = n_subjects
        self.sigma = sigma
        self.grid_width = grid_width
        self.random_state = check_random_state(random_state)
        self.data = self._create_noise()

        if signal_amplitude is not None:
            self.add_signal(
                signal_amplitude=signal_amplitude, signal_width=signal_width
            )
        else:
            self.signal_amplitude = None
            self.signal_mask = None

    def _create_noise(self):
        """Generate simulated data using object parameters.

        Returns:
            np.ndarray: Simulated noise using object parameters.
        """
        return (
            self.random_state.randn(self.grid_width, self.grid_width, self.n_subjects)
            * self.sigma
        )

    def add_signal(self, signal_width=20, signal_amplitude=1):
        """Add a square signal region, centered in the grid, to `self.data`.

        Args:
            signal_width (int): Width of the signal box in pixels. Default 20.
            signal_amplitude (float): Intensity added inside the box. Default 1.
        """
        if signal_width >= self.grid_width:
            raise ValueError("Signal width must be smaller than total grid.")

        self.signal_amplitude = signal_amplitude
        self.create_mask(signal_width)
        signal = np.repeat(
            np.expand_dims(self.signal_mask, axis=2), self.n_subjects, axis=2
        )
        self.data = deepcopy(self.data) + signal * self.signal_amplitude

    def create_mask(self, signal_width):
        """Create the binary `signal_mask` marking a centered square of the grid.

        Args:
            signal_width (int): Width of the signal box in pixels.
        """

        mask = np.zeros((self.grid_width, self.grid_width))
        mask[
            int(np.floor((self.grid_width / 2) - (signal_width / 2))) : int(
                np.ceil((self.grid_width / 2) + (signal_width / 2))
            ),
            int(np.floor((self.grid_width / 2) - (signal_width / 2))) : int(
                np.ceil((self.grid_width / 2) + (signal_width / 2))
            ),
        ] = 1
        self.signal_width = signal_width
        self.signal_mask = mask

    def _run_ttest(self, data):
        """Run a one-sample t-test on data (helper function)."""
        flattened = data.reshape(self.grid_width * self.grid_width, self.n_subjects)
        t, p = ttest_1samp(flattened.T, 0)
        t = np.reshape(t, (self.grid_width, self.grid_width))
        p = np.reshape(p, (self.grid_width, self.grid_width))
        return (t, p)

    def fit(self):
        """Run a one-sample t-test on self.data."""
        if self.isfit:
            raise ValueError("Can't fit because ttest has already been run.")
        self.t_values, self.p_values = self._run_ttest(self.data)
        self.isfit = True

    def _threshold_simulation(self, t, p, threshold, threshold_type, correction=None):
        """Threshold a simulation (helper function).

        Args:
            threshold (float): threshold to apply to simulation
            threshold_type (str): type of threshold to use can be a specific t-value, p-value, or FDR-corrected q-value ['t', 'p', 'q']

        Returns:
            np.ndarray: Thresholded data.

        Raises:
            ValueError: If `correction` is unsupported (see `_validate_correction`),
                or `correction='fdr'` is paired with a `threshold_type` other than
                `'q'`.
        """
        _validate_correction(correction)
        if correction == "fdr":
            if threshold_type != "q":
                raise ValueError("Must specify a q value when using fdr")

        thresholded = deepcopy(t)
        if threshold_type == "t":
            thresholded[np.abs(t) < threshold] = 0
        elif threshold_type == "p":
            thresholded[p > threshold] = 0
        elif threshold_type == "q":
            fdr_threshold = fdr(p.flatten(), q=threshold)
            if fdr_threshold < 0:
                thresholded = np.zeros(thresholded.shape)
            else:
                thresholded[p > fdr_threshold] = 0
        else:
            raise ValueError("Threshold type must be ['t','p','q']")
        return thresholded

    def threshold_simulation(self, threshold, threshold_type, correction=None):
        """Threshold the fitted simulation and store `thresholded` plus hit rates.

        Args:
            threshold (float): Threshold value to apply.
            threshold_type (str): `'t'` (absolute t-value), `'p'` (p-value), or `'q'`
                (FDR-corrected q-value; requires `correction='fdr'`).
            correction (str, optional): Multiple-comparison correction; `'fdr'` or None.
        """

        if not self.isfit:
            raise ValueError("Must fit model before thresholding.")

        if correction == "fdr":
            self.corrected_threshold = fdr(self.p_values.flatten())

        self.correction = correction
        self.thresholded = self._threshold_simulation(
            self.t_values, self.p_values, threshold, threshold_type, correction
        )
        self.threshold = threshold
        self.threshold_type = threshold_type

        self.fp_percent = self._calc_false_positives(self.thresholded)
        if self.signal_mask is not None:
            self.tp_percent = self._calc_true_positives(self.thresholded)

    def _calc_false_positives(self, thresholded):
        """Calculate percent of grid containing false positives.

        Args:
            thresholded (np.array): thresholded grid
        Returns:
            float: Percentage of grid that contains false positives.
        """

        if self.signal_mask is None:
            fp_percent = np.sum(thresholded != 0) / (self.grid_width**2)
        else:
            fp_percent = np.sum(thresholded[self.signal_mask != 1] != 0) / (
                self.grid_width**2 - self.signal_width**2
            )
        return fp_percent

    def _calc_true_positives(self, thresholded):
        """Calculate percent of mask containing true positives.

        Args:
            thresholded (np.array): thresholded grid
        Returns:
            float: Percentage of grid that contains true positives.
        """

        if self.signal_mask is None:
            raise ValueError("No mask exists, run add_signal() first.")
        tp_percent = np.sum(thresholded[self.signal_mask == 1] != 0) / (
            self.signal_width**2
        )
        return tp_percent

    def _calc_false_discovery_rate(self, thresholded):
        """Calculate percent of activated voxels that are false positives.

        Args:
            thresholded (np.array): thresholded grid
        Returns:
            float: Percentage of activated voxels that are false positives.
        """
        if self.signal_mask is None:
            raise ValueError("No mask exists, run add_signal() first.")
        fp_percent = np.sum(thresholded[self.signal_mask == 0] > 0) / np.sum(
            thresholded > 0
        )
        return fp_percent

    def run_multiple_simulations(
        self, threshold, threshold_type, n_simulations=100, correction=None
    ):
        """Run repeated simulations to estimate the false positive rate.

        Stores per-simulation results on `multiple_thresholded`, `multiple_fp`, and
        `fpr` (plus `multiple_tp` and `multiple_fdr` when a signal is present).

        Args:
            threshold (float): Threshold value to apply to each simulation.
            threshold_type (str): `'t'`, `'p'`, or `'q'` (see `threshold_simulation`).
            n_simulations (int): Number of simulations to run. Default 100.
            correction (str, optional): Multiple-comparison correction; `'fdr'` or None.
        """

        if self.signal_mask is None:
            simulations = [
                self._run_ttest(self._create_noise()) for _ in range(n_simulations)
            ]
        else:
            signal = (
                np.repeat(
                    np.expand_dims(self.signal_mask, axis=2), self.n_subjects, axis=2
                )
                * self.signal_amplitude
            )
            simulations = [
                self._run_ttest(self._create_noise() + signal)
                for _ in range(n_simulations)
            ]

        self.multiple_thresholded = [
            self._threshold_simulation(
                s[0], s[1], threshold, threshold_type, correction=correction
            )
            for s in simulations
        ]
        self.multiple_fp = np.array(
            [self._calc_false_positives(x) for x in self.multiple_thresholded]
        )
        self.fpr = np.mean(np.array(list(self.multiple_fp)) > 0)
        if self.signal_mask is not None:
            self.multiple_tp = np.array(
                [self._calc_true_positives(x) for x in self.multiple_thresholded]
            )
            self.multiple_fdr = np.array(
                [self._calc_false_discovery_rate(x) for x in self.multiple_thresholded]
            )

    def plot_grid_simulation(
        self, threshold, threshold_type, n_simulations=100, correction=None
    ):
        """Plot the t-map, its thresholded version, and the false positive distribution.

        Fits and thresholds the simulation first if needed, then calls
        `run_multiple_simulations`. Adds a signal-recovery histogram when a signal is
        present.

        Args:
            threshold (float): Threshold value to apply.
            threshold_type (str): `'t'`, `'p'`, or `'q'` (see `threshold_simulation`).
            n_simulations (int): Number of simulations to run. Default 100.
            correction (str, optional): Multiple-comparison correction; `'fdr'` or None.
        """
        if not self.isfit:
            self.fit()
        if self.thresholded is None:
            self.threshold_simulation(
                threshold=threshold,
                threshold_type=threshold_type,
                correction=correction,
            )
        self.run_multiple_simulations(
            threshold=threshold,
            threshold_type=threshold_type,
            n_simulations=n_simulations,
            correction=correction,
        )

        if self.signal_mask is None:
            _, a = plt.subplots(ncols=3, figsize=(15, 5))
        else:
            _, a = plt.subplots(ncols=4, figsize=(18, 5))
            a[3].hist(self.multiple_tp)
            a[3].set_ylabel("Frequency", fontsize=18)
            a[3].set_xlabel("Percent Signal Recovery", fontsize=18)
            a[3].set_title("Average Signal Recovery", fontsize=18)

        a[0].imshow(self.t_values)
        a[0].set_title("Random Noise", fontsize=18)
        a[0].axes.get_xaxis().set_visible(False)
        a[0].axes.get_yaxis().set_visible(False)
        a[1].imshow(self.thresholded)
        a[1].set_title(f"Threshold: {threshold_type} = {threshold}", fontsize=18)
        a[1].axes.get_xaxis().set_visible(False)
        a[1].axes.get_yaxis().set_visible(False)
        a[2].plot(
            binom.pmf(
                np.arange(0, n_simulations, 1),
                n_simulations,
                np.mean(self.multiple_fp > 0),
            )
        )
        a[2].axvline(
            x=np.mean(self.fpr) * n_simulations,
            color="r",
            linestyle="dashed",
            linewidth=2,
        )
        a[2].set_title(f"False Positive Rate = {self.fpr:.2f}", fontsize=18)
        a[2].set_ylabel("Probability", fontsize=18)
        a[2].set_xlabel("False Positive Rate", fontsize=18)
        plt.tight_layout()
