'''
NeuroLearn datasets
===================

functions to help download datasets

'''

## Notes:
# Need to figure out how to speed up loading and resampling of data

__all__ = ['download_nifti',
           'get_collection_image_metadata',
           'download_collection',
           'fetch_emotion_ratings',
           'fetch_pain',
           'fetch_localizer']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
import pandas as pd
import numpy as np
from nltools.data import Brain_Data
from nilearn.datasets.utils import (_get_dataset_dir,
                                    _fetch_file,
                                    _fetch_files,
                                    _get_dataset_descr)
from nilearn._utils.compat import _urllib
from sklearn.datasets.base import Bunch
from pynv import Client

# Optional dependencies
try:
    import requests
except ImportError:
    pass


def download_nifti(url, data_dir=None):
    ''' Download a image to a nifti file.'''
    local_filename = url.split('/')[-1]
    if data_dir is not None:
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
    local_filename = os.path.join(data_dir, local_filename)
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return local_filename


def get_collection_image_metadata(collection=None, data_dir=None,
                                  limit=10):
    ''' Get image metadata associated with collection

    Args:
    collection:  (int) collection id
    data_dir:	(str) data directory
    limit:		(int) number of images to increment

    Returns:
    metadata:	(pd.DataFrame) Dataframe with full image metadata from
    collection

    '''

    if os.path.isfile(os.path.join(data_dir, 'metadata.csv')):
        dat = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    else:
        offset = 0
        api = Client()
        i = api.get_collection_images(collection_id=collection, limit=limit, offset=offset)
        dat = pd.DataFrame(columns=i['results'][0].keys())
        while int(offset) < int(i['count']):
            for x in i['results']:
                dat = dat.append(x, ignore_index=True)
            offset = offset + limit
            i = api.get_collection_images(collection_id=collection, limit=limit, offset=offset)
        dat.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)
    return dat


def download_collection(collection=None, data_dir=None, overwrite=False,
                        resume=True, verbose=1):
    ''' Download images and metadata from Neurovault collection

    Args:
    collection:  (int) collection id
    data_dir:	(str) data directory

    Returns:
    metadata:	(pd.DataFrame) Dataframe with full image metadata from
    collection
    files:		(list) list of files of downloaded collection

    '''

    if data_dir is None:
        data_dir = _get_dataset_dir(str(collection), data_dir=data_dir,
                                    verbose=verbose)

    # Get collection Metadata
    metadata = get_collection_image_metadata(collection=collection,
                                             data_dir=data_dir)

    # Get images
    files = []
    for f in metadata['file']:
        files.append(_fetch_file(f, data_dir, resume=resume, verbose=verbose,
                                 overwrite=overwrite))

    return (metadata, files)


def fetch_pain(data_dir=None, resume=True, verbose=1):
    '''Download and loads pain dataset from neurovault

    Args:
        data_dir: (string, optional) Path of the data directory.
                   Used to force data storage in a specified location.
                   Default: None

    Returns:
        out: (Brain_Data) Brain_Data object with downloaded data. X=metadata

    '''

    collection = 504
    dataset_name = 'chang2015_pain'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    metadata, files = download_collection(collection=collection,
                                          data_dir=data_dir, resume=resume,
                                          verbose=verbose)
    return Brain_Data(data=files, X=metadata)


def fetch_emotion_ratings(data_dir=None, resume=True, verbose=1):
    '''Download and loads emotion rating dataset from neurovault

    Args:
        data_dir: (string, optional). Path of the data directory.
                   Used to force data storage in a specified location.
                   Default: None

    Returns:
        out: (Brain_Data) Brain_Data object with downloaded data. X=metadata

    '''

    collection = 1964
    dataset_name = 'chang2015_emotion_ratings'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    metadata, files = download_collection(collection=collection,
                                          data_dir=data_dir, resume=resume,
                                          verbose=verbose)
    return Brain_Data(data=files, X=metadata)

def fetch_localizer(subject_ids=None, get_anats=False, data_type='raw',
                    data_dir=None, url=None, resume=True, verbose=1):
    """ Download and load Brainomics Localizer dataset (94 subjects).
    "The Functional Localizer is a simple and fast acquisition
    procedure based on a 5-minute functional magnetic resonance
    imaging (fMRI) sequence that can be run as easily and as
    systematically as an anatomical scan. This protocol captures the
    cerebral bases of auditory and visual perception, motor actions,
    reading, language comprehension and mental calculation at an
    individual level. Individual functional maps are reliable and
    quite precise. The procedure is decribed in more detail on the
    Functional Localizer page." This code is modified from
    `fetch_localizer_contrasts` from nilearn.datasets.funcs.py.
    (see http://brainomics.cea.fr/localizer/)
    "Scientific results obtained using this dataset are described in
    Pinel et al., 2007" [1]

    Notes:
    It is better to perform several small requests than a big one because the
    Brainomics server has no cache (can lead to timeout while the archive
    is generated on the remote server).  For example, download
    n_subjects=np.array(1,10), then n_subjects=np.array(10,20), etc.

    Args:
        subject_ids: (list) List of Subject IDs (e.g., ['S01','S02'].
                     If None is given, all 94 subjects are used.
        get_anats: (boolean) Whether individual structural images should be
                    fetched or not.
        data_type: (string) type of data to download.
                    Valid values are ['raw','preprocessed']
        data_dir: (string, optional) Path of the data directory.
                    Used to force data storage in a specified location.
        url: (string, optional) Override download URL.
             Used for test only (or if you setup a mirror of the data).
        resume: (bool) Whether to resume download of a partly-downloaded file.
        verbose: (int) Verbosity level (0 means no message).

    Returns:
        data: (Bunch)
            Dictionary-like object, the interest attributes are :
            - 'functional': string list
                Paths to nifti contrast maps
            - 'structural' string
                Path to nifti files corresponding to the subjects structural images

    References
    ----------
    Pinel, Philippe, et al.
    "Fast reproducible identification and large-scale databasing of
    individual functional cognitive networks."
    BMC neuroscience 8.1 (2007): 91.

    """

    if subject_ids is None:
        subject_ids = ['S%02d' % x for x in np.arange(1,95)]
    elif not isinstance(subject_ids, (list)):
        raise ValueError("subject_ids must be a list of subject ids (e.g., ['S01','S02'])")

    if data_type == 'raw':
        dat_type = "raw fMRI"
        dat_label = "raw bold"
        anat_type = "raw T1"
        anat_label = "raw anatomy"
    elif data_type == 'preprocessed':
        dat_type = "preprocessed fMRI"
        dat_label = "bold"
        anat_type = "normalized T1"
        anat_label = "anatomy"
    else:
        raise ValueError("Only ['raw','preprocessed'] data_types are currently supported.")

    root_url = "http://brainomics.cea.fr/localizer/"
    dataset_name = 'brainomics_localizer'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    fdescr = _get_dataset_descr(dataset_name)
    opts = {'uncompress': True}

    bold_files = []; anat_files = [];
    for subject_id in subject_ids:
        base_query = ("Any X,XT,XL,XI,XF,XD WHERE X is Scan, X type XT, "
                  "X concerns S, "
                  "X label XL, X identifier XI, "
                  "X format XF, X description XD, "
                  'S identifier = "%s", ' % (subject_id, ) +
                  'X type IN(%(types)s), X label "%(label)s"')

        file_tarball_url = "%sbrainomics_data.zip?rql=%s&vid=data-zip" % (root_url, _urllib.parse.quote(base_query % {"types": "\"%s\"" % dat_type,  "label": dat_label}, safe=',()'))
        name_aux = str.replace(str.join('_', [dat_type, dat_label]), ' ', '_')
        file_path = os.path.join("brainomics_data", subject_id, "%s.nii.gz" % name_aux)
        bold_files.append(_fetch_files(data_dir, [(file_path, file_tarball_url, opts)], verbose=verbose))

        if get_anats:
            file_tarball_url = "%sbrainomics_data_anats.zip?rql=%s&vid=data-zip" % (root_url, _urllib.parse.quote(base_query % {"types": "\"%s\"" % anat_type, "label": anat_label}, safe=',()'))
            if data_type == 'raw':
                anat_name_aux = "raw_T1_raw_anat_defaced.nii.gz"
            elif data_type == 'preprocessed':
                anat_name_aux = "normalized_T1_anat_defaced.nii.gz"
            file_path = os.path.join("brainomics_data", subject_id, anat_name_aux)
            anat_files.append(_fetch_files(data_dir, [(file_path, file_tarball_url, opts)], verbose=verbose))

    # Fetch subject characteristics (separated in two files)
    if url is None:
        url_csv = ("%sdataset/cubicwebexport.csv?rql=%s&vid=csvexport"
                   % (root_url, _urllib.parse.quote("Any X WHERE X is Subject")))
        url_csv2 = ("%sdataset/cubicwebexport2.csv?rql=%s&vid=csvexport"
                    % (root_url,
                       _urllib.parse.quote("Any X,XI,XD WHERE X is QuestionnaireRun, "
                                    "X identifier XI, X datetime "
                                    "XD", safe=',')))
    else:
        url_csv = "%s/cubicwebexport.csv" % url
        url_csv2 = "%s/cubicwebexport2.csv" % url

    filenames = [("cubicwebexport.csv", url_csv, {}),("cubicwebexport2.csv", url_csv2, {})]
    csv_files = _fetch_files(data_dir, filenames, verbose=verbose)
    metadata = pd.merge(pd.read_csv(csv_files[0], sep=';'), pd.read_csv(csv_files[1], sep=';'), on='"subject_id"')
    metadata.to_csv(os.path.join(data_dir,'metadata.csv'))
    for x in ['cubicwebexport.csv','cubicwebexport2.csv']:
        os.remove(os.path.join(data_dir, x))

    if not get_anats:
        anat_files = None

    return Bunch(functional=bold_files,
                 structural=anat_files,
                 ext_vars=metadata,
                 description=fdescr)
