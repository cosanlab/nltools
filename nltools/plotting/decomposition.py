"""ICA/PCA component viewer — interactive decomposition explorer."""

__all__ = ["component_viewer"]

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from numpy.fft import fft, fftfreq
from nilearn.plotting import plot_stat_map

from nltools.utils import attempt_to_import

# Optional dependencies
ipywidgets = attempt_to_import(
    "ipywidgets",
    name="ipywidgets",
    fromlist=["interact", "fixed", "widgets", "BoundedFloatText", "BoundedIntText"],
)


def component_viewer(output, tr=2.0):
    """This a function to interactively view the results of a decomposition analysis

    Args:
        output: (dict) output dictionary from running BrainData.decompose()
        tr: (float) repetition time of data
    """

    if ipywidgets is None:
        raise ImportError(
            "ipywidgets is required for interactive plotting. Please install this package manually or install nltools with optional arguments: pip install 'nltools[interactive_plots]'"
        )

    def component_inspector(component, threshold):
        """This a function to be used with ipywidgets to interactively view a decomposition analysis

        Make sure you have tr and output assigned to variables.

        Example:

            from ipywidgets import BoundedFloatText, BoundedIntText
            from ipywidgets import interact

            tr = 2.4
            output = data_filtered_smoothed.decompose(method='ica', n_components=30, axis='images', whiten=True)

            interact(component_inspector, component=BoundedIntText(description='Component', value=0, min=0, max=len(output['components'])-1),
                  threshold=BoundedFloatText(description='Threshold', value=2.0, min=0, max=4, step=.1))

        """
        _, ax = plt.subplots(nrows=3, figsize=(12, 8))
        thresholded = (
            output["components"][component] - output["components"][component].mean()
        ) * (1 / output["components"][component].std())
        thresholded.data[np.abs(thresholded.data) <= threshold] = 0
        plot_stat_map(
            thresholded.to_nifti(),
            cut_coords=range(-40, 70, 10),
            display_mode="z",
            black_bg=True,
            colorbar=True,
            annotate=False,
            draw_cross=False,
            axes=ax[0],
        )
        if isinstance(output["decomposition_object"], (sklearn.decomposition.PCA)):
            var_exp = output["decomposition_object"].explained_variance_ratio_[
                component
            ]
            ax[0].set_title(
                f"Component: {component}/{len(output['components'])}, Variance Explained: {var_exp:2.2}",
                fontsize=18,
            )
        else:
            ax[0].set_title(
                f"Component: {component}/{len(output['components'])}", fontsize=18
            )

        ax[1].plot(output["weights"][:, component], linewidth=2, color="red")
        ax[1].set_ylabel("Intensity (AU)", fontsize=18)
        ax[1].set_title(f"Timecourse (TR={tr})", fontsize=16)
        y = fft(output["weights"][:, component])
        f = fftfreq(len(y), d=tr)
        ax[2].plot(f[f > 0], np.abs(y)[f > 0] ** 2)
        ax[2].set_ylabel("Power", fontsize=18)
        ax[2].set_xlabel("Frequency (Hz)", fontsize=16)

    ipywidgets.interact(
        component_inspector,
        component=ipywidgets.BoundedIntText(
            description="Component", value=0, min=0, max=len(output["components"]) - 1
        ),
        threshold=ipywidgets.BoundedFloatText(
            description="Threshold", value=2.0, min=0, max=4, step=0.1
        ),
    )
