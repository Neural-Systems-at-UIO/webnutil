import numpy as np
import pandas as pd
import re
from glob import glob


def extract_section_numbers(filenames: list[str], legacy=False):
    """Extract section numbers from a list of filenames."""
    filenames = [f.split("\\")[-1] for f in filenames]
    section_numbers = []
    for filename in filenames:
        if not legacy:
            match = re.findall(r"\_s\d+", filename)
            if not match or len(match) > 1:
                raise ValueError(f"Invalid section number in: {filename}")
            section_numbers.append(int(match[-1][2:]))
        else:
            section_numbers.append(re.sub("[^0-9]", "", filename)[-3:])
    if not section_numbers:
        raise ValueError("No section numbers found in filenames")
    return section_numbers


def find_matching_pixels(segmentation: np.ndarray, id: int):
    """Returns Y and X coordinates of pixels matching the id."""
    mask = np.all(segmentation == id, axis=2)
    return np.where(mask)


scale_positions = lambda id_y, id_x, y_scale, x_scale: (id_y * y_scale, id_x * x_scale)


def calculate_scale_factor(image: np.ndarray, rescaleXY: tuple) -> float:
    """Calculate the scale factor for an image."""
    if rescaleXY:
        return (rescaleXY[0] * rescaleXY[1]) / (image.shape[0] * image.shape[1])
    return False


def get_segmentations(folder):
    """Collect segmentation file paths from the specified folder."""
    types = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dzip"]
    segs = [f for f in glob(folder + "/*") if any(f.endswith(t) for t in types)]
    if not segs:
        raise ValueError(f"No segmentations found in folder {folder}")
    print(f"Found {len(segs)} segmentations in folder {folder}")
    return segs


def get_flat_files(folder: str, use_flat=False) -> tuple:
    """Retrieves flat file paths from the given folder."""
    if use_flat:
        flat_files = [f for f in glob(folder + "/flat_files/*") if f.endswith((".flat", ".seg"))]
        print(f"Found {len(flat_files)} flat files in folder {folder}")
        flat_file_nrs = [int(extract_section_numbers([ff])[0]) for ff in flat_files]
        return flat_files, flat_file_nrs
    return [], []


def get_current_flat_file(seg_nr: int, flat_files, flat_file_nrs, use_flat):
    """Determines the correct flat file for a given section number."""
    if use_flat:
        idx = np.where([f == seg_nr for f in flat_file_nrs])[0]
        return flat_files[idx[0]] if len(idx) > 0 else None
    return None


def process_results(
    points_list,
    centroids_list,
    points_labels,
    centroids_labels,
    points_hemi_labels,
    centroids_hemi_labels,
    points_undamaged_list,
    centroids_undamaged_list,
):
    """Consolidates and organizes results from multiple segmentations."""
    points_len = [len(p) if None not in p else 0 for p in points_list]
    centroids_len = [len(c) if None not in c else 0 for c in centroids_list]
    
    points_list = [p for p in points_list if None not in p and len(p) > 0]
    centroids_list = [c for c in centroids_list if None not in c and len(c) > 0]
    points_labels = [p for p in points_labels if None not in p and len(p) > 0]
    centroids_labels = [c for c in centroids_labels if None not in c and len(c) > 0]
    points_undamaged_list = [p for p in points_undamaged_list if None not in p and len(p) > 0]
    centroids_undamaged_list = [c for c in centroids_undamaged_list if None not in c and len(c) > 0]

    points = np.concatenate(points_list) if points_list else np.array([])
    points_labels = np.concatenate(points_labels) if points_labels else np.array([])
    points_undamaged = np.concatenate(points_undamaged_list) if points_undamaged_list else np.array([])
    points_hemi_labels = np.concatenate(points_hemi_labels) if points_hemi_labels else np.array([])

    centroids = np.concatenate(centroids_list) if centroids_list else np.array([])
    centroids_labels = np.concatenate(centroids_labels) if centroids_labels else np.array([])
    centroids_undamaged = np.concatenate(centroids_undamaged_list) if centroids_undamaged_list else np.array([])
    centroids_hemi_labels = np.concatenate(centroids_hemi_labels) if centroids_hemi_labels else np.array([])

    return (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        points_len,
        centroids_len,
        points_undamaged,
        centroids_undamaged,
    )
