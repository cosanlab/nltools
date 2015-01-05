# first line: 181
def compute_epi_mask(epi_img, lower_cutoff=0.2, upper_cutoff=0.85,
                     connected=True, opening=2, exclude_zeros=False,
                     ensure_finite=True,
                     target_affine=None, target_shape=None,
                     memory=None, verbose=0,):
    """
    Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    epi_img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        EPI image, used to compute the mask. 3D and 4D images are accepted.
        If a 3D image is given, we suggest to use the mean image

    lower_cutoff: float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: bool, optional
        if connected is True, only the largest connect component is kept.

    opening: bool or int, optional
        if opening is True, a morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
        If opening is an integer `n`, it is performed via `n` erosions.
        After estimation of the largest connected constituent, 2`n` closing
        operations are performed followed by `n` erosions. This corresponds
        to 1 opening operation of order `n` followed by a closing operator
        of order `n`.
        Note that turning off opening (opening=False) will also prevent
        any smoothing applied to the image during the mask computation.

    ensure_finite: bool
        If ensure_finite is True, the non-finite values (NaNs and infs)
        found in the images will be replaced by zeros

    exclude_zeros: bool, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    memory: instance of joblib.Memory or string
        Used to cache the function call: if this is a string, it
        specifies the directory where the cache will be stored.

    verbose: int, optional
        Controls the amount of verbosity: higher numbers give
        more messages

    Returns
    -------
    mask: nibabel.Nifti1Image
        The brain mask (3D image)
    """
    if verbose > 0:
        print "EPI mask computation"
    # We suppose that it is an img
    # XXX make a is_a_imgs function ?

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean
    mean_epi, affine = cache(_compute_mean, memory)(epi_img,
                                     target_affine=target_affine,
                                     target_shape=target_shape,
                                     smooth=(1 if opening else False))

    if ensure_finite:
        # Get rid of memmapping
        mean_epi = _utils.as_ndarray(mean_epi)
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = int(np.floor(lower_cutoff * len(sorted_input)))
    upper_cutoff = min(int(np.floor(upper_cutoff * len(sorted_input))),
                       len(sorted_input) - 1)

    delta = sorted_input[lower_cutoff + 1:upper_cutoff + 1] \
        - sorted_input[lower_cutoff:upper_cutoff]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + lower_cutoff]
                       + sorted_input[ia + lower_cutoff + 1])

    mask = mean_epi >= threshold

    return _post_process_mask(mask, affine, opening=opening,
        connected=connected, msg="Are you sure that input "
            "data are EPI images not detrended. ")
