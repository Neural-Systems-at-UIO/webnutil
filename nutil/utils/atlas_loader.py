import pandas as pd
import numpy as np
import nrrd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_atlas_labels(
    atlas: Optional[object] = None, atlas_name: Optional[str] = None
) -> pd.DataFrame:
    if not atlas_name and not atlas:
        raise ValueError("Either atlas or atlas name must be specified")
    if not atlas:
        raise ValueError("Atlas object must be provided if atlas_name is not used")
    atlas_structures = {
        "idx": [i["id"] for i in atlas.structures_list],
        "name": [i["name"] for i in atlas.structures_list],
        "r": [i["rgb_triplet"][0] for i in atlas.structures_list],
        "g": [i["rgb_triplet"][1] for i in atlas.structures_list],
        "b": [i["rgb_triplet"][2] for i in atlas.structures_list],
    }
    atlas_structures["idx"].insert(0, 0)
    atlas_structures["name"].insert(0, "Clear Label")
    logger.info("Atlas structures: %s", atlas_structures["name"][1:10])
    atlas_structures["r"].insert(0, 0)
    atlas_structures["g"].insert(0, 0)
    atlas_structures["b"].insert(0, 0)
    return pd.DataFrame(atlas_structures)


def process_atlas_volume(vol: np.ndarray) -> np.ndarray:
    return np.transpose(vol, [2, 0, 1])[::-1, ::-1, ::-1]


def load_custom_atlas(
    atlas_path: str, hemi_path: Optional[str], label_path: str
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    try:
        atlas_volume, _ = nrrd.read(atlas_path)
    except Exception as e:
        logger.error(f"Failed to read atlas volume from {atlas_path}: {e}")
        raise
    hemi_volume = None
    if hemi_path:
        try:
            hemi_volume, _ = nrrd.read(hemi_path)
        except Exception as e:
            logger.error(f"Failed to read hemisphere volume from {hemi_path}: {e}")
            raise
    try:
        atlas_labels = pd.read_csv(label_path)
    except Exception as e:
        logger.error(f"Failed to read atlas labels from {label_path}: {e}")
        raise
    logger.info(
        f"Loaded custom atlas with {len(atlas_labels)} regions from {atlas_path} and {hemi_path if hemi_path else 'no hemisphere file'}"
    )
    return atlas_volume, hemi_volume, atlas_labels
