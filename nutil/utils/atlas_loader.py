import pandas as pd
import numpy as np
import nrrd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_atlas_labels(atlas=None, atlas_name=None):
    if not atlas_name and not atlas:
        raise Exception("Either atlas or atlas name must be specified")
    
    
    atlas_structures = {
        "idx": [i["id"] for i in atlas.structures_list],
        "name": [i["name"] for i in atlas.structures_list],
        "r": [i["rgb_triplet"][0] for i in atlas.structures_list],
        "g": [i["rgb_triplet"][1] for i in atlas.structures_list],
        "b": [i["rgb_triplet"][2] for i in atlas.structures_list],
    }


    atlas_structures["idx"].insert(0, 0)
    atlas_structures["name"].insert(0, "Clear Label")

    logging.info("Atlas structures: %s", atlas_structures["name"][1:10]) 
    atlas_structures["r"].insert(0, 0)
    atlas_structures["g"].insert(0, 0)
    atlas_structures["b"].insert(0, 0)
    atlas_labels = pd.DataFrame(atlas_structures)
    return atlas_labels


def process_atlas_volume(vol):
    """
    Processes the atlas volume by transposing and reversing axes.

    Parameters
    ----------
    vol : numpy.ndarray
        The atlas volume to process.

    Returns
    -------
    numpy.ndarray
        The processed atlas volume.
    """
    return np.transpose(vol, [2, 0, 1])[::-1, ::-1, ::-1]


def load_custom_atlas(atlas_path, hemi_path, label_path):
    """
    Loads a custom atlas from provided file paths.

    Parameters
    ----------
    atlas_path : str
        Path to the custom atlas volume file.
    hemi_path : str or None
        Path to the hemisphere file, if any.
    label_path : str
        Path to the label CSV file for region info.

    Returns
    -------
    numpy.ndarray
        The loaded atlas volume.
    numpy.ndarray or None
        The hemisphere array, or None if hemi_path is not provided.
    pandas.DataFrame
        A dataframe containing atlas labels.
    """
    atlas_volume, _ = nrrd.read(atlas_path)
    if hemi_path:
        hemi_volume, _ = nrrd.read(hemi_path)
    else:
        hemi_volume = None
    atlas_labels = pd.read_csv(label_path)
    print(
        f"Loaded custom atlas with {len(atlas_labels)} regions from {atlas_path} and {hemi_path if hemi_path else 'no hemisphere file'}"
    )
    print(atlas_volume)
    return atlas_volume, hemi_volume, atlas_labels
