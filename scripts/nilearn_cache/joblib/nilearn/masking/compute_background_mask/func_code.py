# first line: 370
def compute_background_mask(data_imgs, border_size=2,
                     connected=False, opening=False,
                     target_affine=None, target_shape=None,
                     memory=None, verbose=0):
    """ Compute a brain mask for the images by guessing the value of the
    background from the border of the image.

    Parameters
    ----------
    data_imgs: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Images used to compute the mask. 3D and 4D images are accepted.
        If a 3D image is given, we suggest to use the mean image

    border_size: integer, optional
        The size, in voxel of the border used on the side of the image
        to determine the value of the background.

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

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    memory: instance of joblib.Memory or string
        Used to cache the function call.

    verbose: int, optional

    Returns
    -------
    mask: nibabel.Nifti1Image
        The brain mask (3D image)
    """
    if verbose > 0:
        print "Background mask computation"
    # We suppose that it is an img
    # XXX make a is_a_imgs function ?

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean
    data, affine = cache(_compute_mean, memory)(data_imgs,
                target_affine=target_affine, target_shape=target_shape,
                smooth=False)

    border_data = np.concatenate([
            data[:border_size, :, :].ravel(), data[-border_size:, :, :].ravel(),
            data[:, :border_size, :].ravel(), data[:, -border_size:, :].ravel(),
            data[:, :, :border_size].ravel(), data[:, :, -border_size:].ravel(),
        ])
    background = np.median(border_data)
    if np.isnan(background):
        # We absolutely need to catter for NaNs as a background:
        # SPM does that by default
        mask = np.logical_not(np.isnan(data))
    else:
        mask = data != background

    return _post_process_mask(mask, affine, opening=opening,
        connected=connected, msg="Are you sure that input "
            "images have a homogeneous background.")
