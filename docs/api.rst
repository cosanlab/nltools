:orphan:

.. _api_ref:

API Reference
*************

This reference provides detailed documentation for all modules, classes, and
methods in the current release of Neurolearn.


:mod:`nltools.data`: Data Types
===============================

.. autoclass:: nltools.data.Brain_Data
    :members:

.. autoclass:: nltools.data.Adjacency
    :members:

.. autoclass:: nltools.data.Groupby
    :members:

.. autoclass:: nltools.data.Design_Matrix
    :members:

:mod:`nltools.analysis`: Analysis Tools
=======================================

.. autoclass:: nltools.analysis.Roc
    :members:

:mod:`nltools.stats`: Stats Tools
=================================

.. automodule:: nltools.stats
    :members:

:mod:`nltools.datasets`: Dataset Tools
======================================

.. automodule:: nltools.datasets
    :members:

:mod:`nltools.cross_validation`: Cross-Validation Tools
=======================================================

.. automodule:: nltools.cross_validation
    :members:

.. autoclass:: nltools.cross_validation.KFoldStratified
    :members:

:mod:`nltools.mask`: Mask Tools
===============================

.. automodule:: nltools.mask
    :members:

:mod:`nltools.file_reader`: File Reading
========================================

.. automodule:: nltools.file_reader
    :members:

:mod:`nltools.utils`: Utilities
==============================

.. automodule:: nltools.utils
    :members:

:mod:`nltools.prefs`: Preferences
================================

This module can be used to adjust the default MNI template settings that are used
internally by all `Brain_Data` operations. By default all operations are performed in
**MNI152 2mm space**. Thus any files loaded with be resampled to this space by default.You can control this on a per-file loading basis using the `mask` argument of `Brain_Data`, e.g.

.. code-block::

    from nltools.data import Brain_Data

    # my_brain will be resampled to 2mm
    brain = Brain_Data('my_brain.nii.gz') 

    # my_brain will now be resampled to the same space as my_mask
    brain = Brain_Data('my_brain.nii.gz', mask='my_mask.nii.gz') # will be resampled 

Alternatively this module can be used to switch between 2mm or 3mm MNI spaces with and without ventricles:

.. code-block::

    from nltools.prefs import MNI_Template, resolve_mni_path
    from nltools.data import Brain_Data

    MNI_Template['resolution'] = '3mm'

    # my_brain will be resampled to 3mm and future operation will be in 3mm space
    brain = Brain_Data('my_brain.nii.gz') 

    # get the template nifti files
    resolve_mni_path(MNI_Template)

    # will print like:
    {
        'resolution': '3mm',
        'mask_type': 'with_ventricles',
        'mask': '/Users/Esh/Documents/pypackages/nltools/nltools/resources/MNI152_T1_3mm_brain_mask.nii.gz',
        'plot': '/Users/Esh/Documents/pypackages/nltools/nltools/resources/MNI152_T1_3mm.nii.gz',
        'brain':
        '/Users/Esh/Documents/pypackages/nltools/nltools/resources/MNI152_T1_3mm_brain.nii.gz'
    }

.. automodule:: nltools.prefs
   :members:
   :show-inheritance:

:mod:`nltools.plotting`: Plotting Tools
=======================================

.. automodule:: nltools.plotting
    :members:

:mod:`nltools.simulator`: Simulator Tools
=========================================

.. automodule:: nltools.simulator
    :members:

.. autoclass:: nltools.simulator.Simulator
    :members:


Index
=====

* :ref:`genindex`
* :ref:`modindex`
