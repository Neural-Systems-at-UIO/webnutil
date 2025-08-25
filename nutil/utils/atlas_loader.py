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
    atlas_path: str | None, hemi_path: Optional[str], label_path: str | None
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

        # CRITICAL FIX: Remap huge indices to compact range to prevent massive memory allocation
        original_indices = atlas_labels["idx"].values
        max_idx = original_indices.max()
        unique_indices = np.unique(original_indices)

        print(
            f"Atlas indices: min={original_indices.min()}, max={max_idx}, unique={len(unique_indices)}"
        )

        if max_idx > 100000:  # If we have problematically large indices
            print(
                f"WARNING: Large atlas indices detected (max={max_idx}). Analyzing volume usage..."
            )

            # Check which indices actually exist in the volume
            volume_indices = np.unique(atlas_volume)
            volume_max = volume_indices.max()
            print(
                f"Volume indices: min={volume_indices.min()}, max={volume_max}, unique={len(volume_indices)}"
            )

            # Find intersection of label indices and volume indices
            labels_in_volume = np.intersect1d(original_indices, volume_indices)
            print(
                f"Label indices actually used in volume: {len(labels_in_volume)} out of {len(unique_indices)}"
            )

            if volume_max < 100000:
                print("Volume indices are reasonable - no remapping needed for volume")
                # Just ensure labels cover the volume indices
                missing_in_labels = np.setdiff1d(volume_indices, original_indices)
                if len(missing_in_labels) > 0:
                    print(
                        f"Adding {len(missing_in_labels)} missing volume indices to labels"
                    )
                    # Add missing indices as background
                    for missing_idx in missing_in_labels:
                        new_row = {
                            "idx": missing_idx,
                            "name": f"Unknown_{missing_idx}",
                            "r": 0,
                            "g": 0,
                            "b": 0,
                        }
                        atlas_labels = pd.concat(
                            [atlas_labels, pd.DataFrame([new_row])], ignore_index=True
                        )
            else:
                print(
                    "Both labels and volume have large indices - creating lookup table..."
                )
                # Create efficient lookup using numpy indexing
                all_indices = np.union1d(original_indices, volume_indices)
                max_all = all_indices.max()

                # Create lookup table (this is the memory-efficient way)
                lookup = np.arange(
                    max_all + 1, dtype=np.uint32
                )  # Default: identity mapping
                compact_mapping = {
                    old_idx: new_idx for new_idx, old_idx in enumerate(all_indices)
                }

                for old_idx, new_idx in compact_mapping.items():
                    lookup[old_idx] = new_idx

                # Remap volume efficiently
                print("Remapping atlas volume using lookup table...")
                atlas_volume = lookup[atlas_volume]

                # Remap labels
                atlas_labels["original_idx"] = atlas_labels["idx"]
                atlas_labels["idx"] = atlas_labels["idx"].map(compact_mapping)

                print(f"Remapped to range 0-{len(all_indices)-1}")

    except Exception as e:
        logger.error(f"Failed to read atlas labels from {label_path}: {e}")
        raise
    logger.info(
        f"Loaded custom atlas with {len(atlas_labels)} regions from {atlas_path} and {hemi_path if hemi_path else 'no hemisphere file'}"
    )
    return atlas_volume, hemi_volume, atlas_labels
