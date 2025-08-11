import numpy as np
import pandas as pd
import struct
import cv2
import os
import gc
import psutil
import logging
from .generate_target_slice import generate_target_slice
from .visualign_deformations import transform_vec
from skimage.transform import resize

# Configure logging for memory debugging
logger = logging.getLogger(__name__)


def log_memory_usage(var_name, array=None, message=""):
    """Log memory usage of an array and system memory."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    if array is not None:
        try:
            if hasattr(array, "nbytes"):
                array_mb = array.nbytes / 1024 / 1024
                logger.info(
                    f"MEMORY: {var_name} = {array_mb:.2f} MB ({array.shape} {array.dtype}) | Process: {memory_mb:.2f} MB | {message}"
                )
            elif hasattr(array, "__len__"):
                logger.info(
                    f"MEMORY: {var_name} = {len(array)} items | Process: {memory_mb:.2f} MB | {message}"
                )
            else:
                logger.info(
                    f"MEMORY: {var_name} = {type(array)} | Process: {memory_mb:.2f} MB | {message}"
                )
        except Exception as e:
            logger.info(
                f"MEMORY: {var_name} = ERROR getting size ({e}) | Process: {memory_mb:.2f} MB | {message}"
            )
    else:
        logger.info(f"MEMORY: Process: {memory_mb:.2f} MB | {message}")


def create_base_counts_dict(with_hemisphere=False, with_damage=False):
    """
    Creates and returns a base dictionary structure for tracking counts.

    Args:
        with_hemisphere (bool): If True, include hemisphere fields.
        with_damage (bool): If True, include damage fields.

    Returns:
        dict: Structure containing count lists for pixels/objects.
    """
    counts = {
        "idx": [],
        "name": [],
        "r": [],
        "g": [],
        "b": [],
        "pixel_count": [],
        "object_count": [],
    }
    if with_damage:
        damage_fields = {
            "undamaged_object_count": [],
            "damaged_object_count": [],
            "undamaged_pixel_count": [],
            "damaged_pixel_counts": [],
        }
        counts.update(damage_fields)
    if with_hemisphere:
        hemisphere_fields = {
            "left_hemi_pixel_count": [],
            "right_hemi_pixel_count": [],
            "left_hemi_object_count": [],
            "right_hemi_object_count": [],
        }
        counts.update(hemisphere_fields)
    if with_damage and with_hemisphere:
        damage_hemisphere_fields = {
            "left_hemi_undamaged_pixel_count": [],
            "left_hemi_damaged_pixel_count": [],
            "right_hemi_undamaged_pixel_count": [],
            "right_hemi_damaged_pixel_count": [],
            "left_hemi_undamaged_object_count": [],
            "left_hemi_damaged_object_count": [],
            "right_hemi_undamaged_object_count": [],
            "right_hemi_damaged_object_count": [],
        }
        counts.update(damage_hemisphere_fields)
    return counts


# related to counting_and_load
def pixel_count_per_region(
    labels_dict_points,
    labeled_dict_centroids,
    current_points_undamaged,
    current_centroids_undamaged,
    current_points_hemi,
    current_centroids_hemi,
    df_label_colours,
    with_damage=False,
):
    """
    Tally object counts by region, optionally tracking damage and hemispheres.

    Args:
        labels_dict_points (dict): Maps points to region labels.
        labeled_dict_centroids (dict): Maps centroids to region labels.
        current_points_undamaged (ndarray): Undamaged-state flags for points.
        current_centroids_undamaged (ndarray): Undamaged-state flags for centroids.
        current_points_hemi (ndarray): Hemisphere tags for points.
        current_centroids_hemi (ndarray): Hemisphere tags for centroids.
        df_label_colours (DataFrame): Region label colors.
        with_damage (bool, optional): Track damage counts if True.

    Returns:
        DataFrame: Summed counts per region."""
    with_hemi = None not in current_points_hemi
    counts_per_label = create_base_counts_dict(
        with_hemisphere=with_hemi, with_damage=with_damage
    )

    # Ensure arrays are boolean and compatible for bitwise operations
    current_points_undamaged = np.asarray(current_points_undamaged, dtype=bool)
    current_centroids_undamaged = np.asarray(current_centroids_undamaged, dtype=bool)
    if with_hemi:
        current_points_hemi = np.asarray(current_points_hemi)
        current_centroids_hemi = np.asarray(current_centroids_hemi)

    if with_hemi and with_damage:
        (
            left_hemi_counted_labels_points_undamaged,
            left_hemi_label_counts_points_undamaged,
        ) = np.unique(
            labels_dict_points[current_points_undamaged & (current_points_hemi == 1)],
            return_counts=True,
        )
        (
            left_hemi_counted_labels_points_damaged,
            left_hemi_label_counts_points_damaged,
        ) = np.unique(
            labels_dict_points[~current_points_undamaged & (current_points_hemi == 1)],
            return_counts=True,
        )
        (
            left_hemi_counted_labels_centroids_undamaged,
            left_hemi_label_counts_centroids_undamaged,
        ) = np.unique(
            labeled_dict_centroids[
                current_centroids_undamaged & (current_centroids_hemi == 1)
            ],
            return_counts=True,
        )
        (
            left_hemi_counted_labels_centroids_damaged,
            left_hemi_label_counts_centroids_damaged,
        ) = np.unique(
            labeled_dict_centroids[
                ~current_centroids_undamaged & (current_centroids_hemi == 1)
            ],
            return_counts=True,
        )
        (
            right_hemi_counted_labels_points_undamaged,
            right_hemi_label_counts_points_undamaged,
        ) = np.unique(
            labels_dict_points[current_points_undamaged & (current_points_hemi == 2)],
            return_counts=True,
        )
        (
            right_hemi_counted_labels_points_damaged,
            right_hemi_label_counts_points_damaged,
        ) = np.unique(
            labels_dict_points[~current_points_undamaged & (current_points_hemi == 2)],
            return_counts=True,
        )
        (
            right_hemi_counted_labels_centroids_undamaged,
            right_hemi_label_counts_centroids_undamaged,
        ) = np.unique(
            labeled_dict_centroids[
                current_centroids_undamaged & (current_centroids_hemi == 2)
            ],
            return_counts=True,
        )
        (
            right_hemi_counted_labels_centroids_damaged,
            right_hemi_label_counts_centroids_damaged,
        ) = np.unique(
            labeled_dict_centroids[
                ~current_centroids_undamaged & (current_centroids_hemi == 2)
            ],
            return_counts=True,
        )
        for index, row in df_label_colours.iterrows():
            # Left hemisphere pixel counts
            if row["idx"] in left_hemi_counted_labels_points_undamaged:
                l_clpu = left_hemi_label_counts_points_undamaged[
                    left_hemi_counted_labels_points_undamaged == row["idx"]
                ][0]
            else:
                l_clpu = 0

            if row["idx"] in left_hemi_counted_labels_points_damaged:
                l_clpd = left_hemi_label_counts_points_damaged[
                    left_hemi_counted_labels_points_damaged == row["idx"]
                ][0]
            else:
                l_clpd = 0

            # Right hemisphere pixel counts
            if row["idx"] in right_hemi_counted_labels_points_undamaged:
                r_clpu = right_hemi_label_counts_points_undamaged[
                    right_hemi_counted_labels_points_undamaged == row["idx"]
                ][0]
            else:
                r_clpu = 0

            if row["idx"] in right_hemi_counted_labels_points_damaged:
                r_clpd = right_hemi_label_counts_points_damaged[
                    right_hemi_counted_labels_points_damaged == row["idx"]
                ][0]
            else:
                r_clpd = 0

            # Left hemisphere object counts
            if row["idx"] in left_hemi_counted_labels_centroids_undamaged:
                l_clcu = left_hemi_label_counts_centroids_undamaged[
                    left_hemi_counted_labels_centroids_undamaged == row["idx"]
                ][0]
            else:
                l_clcu = 0

            if row["idx"] in left_hemi_counted_labels_centroids_damaged:
                l_clcd = left_hemi_label_counts_centroids_damaged[
                    left_hemi_counted_labels_centroids_damaged == row["idx"]
                ][0]
            else:
                l_clcd = 0

            # Right hemisphere object counts
            if row["idx"] in right_hemi_counted_labels_centroids_undamaged:
                r_clcu = right_hemi_label_counts_centroids_undamaged[
                    right_hemi_counted_labels_centroids_undamaged == row["idx"]
                ][0]
            else:
                r_clcu = 0

            if row["idx"] in right_hemi_counted_labels_centroids_damaged:
                r_clcd = right_hemi_label_counts_centroids_damaged[
                    right_hemi_counted_labels_centroids_damaged == row["idx"]
                ][0]
            else:
                r_clcd = 0

            # Skip regions with no counts in any category
            if (
                l_clcd
                == l_clcu
                == l_clpd
                == l_clpu
                == r_clcd
                == r_clcu
                == r_clpd
                == r_clpu
                == 0
            ):
                continue

            # Calculate combined counts
            clpu = l_clpu + r_clpu  # total undamaged pixel count
            clpd = l_clpd + r_clpd  # total damaged pixel count
            clcu = l_clcu + r_clcu  # total undamaged object count
            clcd = l_clcd + r_clcd  # total damaged object count

            # Add to dictionary
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))

            # Total counts
            counts_per_label["pixel_count"].append(clpu + clpd)
            counts_per_label["undamaged_pixel_count"].append(clpu)
            counts_per_label["damaged_pixel_counts"].append(clpd)
            counts_per_label["object_count"].append(clcu + clcd)
            counts_per_label["undamaged_object_count"].append(clcu)
            counts_per_label["damaged_object_count"].append(clcd)

            # Left hemisphere counts
            counts_per_label["left_hemi_pixel_count"].append(l_clpu + l_clpd)
            counts_per_label["left_hemi_undamaged_pixel_count"].append(l_clpu)
            counts_per_label["left_hemi_damaged_pixel_count"].append(l_clpd)
            counts_per_label["left_hemi_object_count"].append(l_clcu + l_clcd)
            counts_per_label["left_hemi_undamaged_object_count"].append(l_clcu)
            counts_per_label["left_hemi_damaged_object_count"].append(l_clcd)

            # Right hemisphere counts
            counts_per_label["right_hemi_pixel_count"].append(r_clpu + r_clpd)
            counts_per_label["right_hemi_undamaged_pixel_count"].append(r_clpu)
            counts_per_label["right_hemi_damaged_pixel_count"].append(r_clpd)
            counts_per_label["right_hemi_object_count"].append(r_clcu + r_clcd)
            counts_per_label["right_hemi_undamaged_object_count"].append(r_clcu)
            counts_per_label["right_hemi_damaged_object_count"].append(r_clcd)

    elif with_damage and (not with_hemi):
        counted_labels_points_undamaged, label_counts_points_undamaged = np.unique(
            labels_dict_points[current_points_undamaged], return_counts=True
        )
        counted_labels_points_damaged, label_counts_points_damaged = np.unique(
            labels_dict_points[~current_points_undamaged], return_counts=True
        )
        counted_labels_centroids_undamaged, label_counts_centroids_undamaged = (
            np.unique(
                labeled_dict_centroids[current_centroids_undamaged], return_counts=True
            )
        )
        counted_labels_centroids_damaged, label_counts_centroids_damaged = np.unique(
            labeled_dict_centroids[~current_centroids_undamaged], return_counts=True
        )
        for index, row in df_label_colours.iterrows():
            if row["idx"] in counted_labels_points_undamaged:
                clpu = label_counts_points_undamaged[
                    counted_labels_points_undamaged == row["idx"]
                ][0]
            else:
                clpu = 0
            if row["idx"] in counted_labels_points_damaged:
                clpd = label_counts_points_damaged[
                    counted_labels_points_damaged == row["idx"]
                ][0]
            else:
                clpd = 0
            if row["idx"] in counted_labels_centroids_undamaged:
                clcu = label_counts_centroids_undamaged[
                    counted_labels_centroids_undamaged == row["idx"]
                ][0]
            else:
                clcu = 0
            if row["idx"] in counted_labels_centroids_damaged:
                clcd = label_counts_centroids_damaged[
                    counted_labels_centroids_damaged == row["idx"]
                ][0]
            else:
                clcd = 0
            if clcd == clcu == clpd == clpu == 0:
                continue
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))
            counts_per_label["pixel_count"].append(clpu + clpd)
            counts_per_label["undamaged_pixel_count"].append(clpu)
            counts_per_label["damaged_pixel_counts"].append(clpd)
            counts_per_label["object_count"].append(clcu + clcd)
            counts_per_label["undamaged_object_count"].append(clcu)
            counts_per_label["damaged_object_count"].append(clcd)

    elif with_hemi and (not with_damage):
        left_hemi_counted_labels_points, left_hemi_label_counts_points = np.unique(
            labels_dict_points[current_points_hemi == 1], return_counts=True
        )
        left_hemi_counted_labels_centroids, left_hemi_label_counts_centroids = (
            np.unique(
                labeled_dict_centroids[current_centroids_hemi == 1], return_counts=True
            )
        )
        right_hemi_counted_labels_points, right_hemi_label_counts_points = np.unique(
            labels_dict_points[current_points_hemi == 2], return_counts=True
        )
        right_hemi_counted_labels_centroids, right_hemi_label_counts_centroids = (
            np.unique(
                labeled_dict_centroids[current_centroids_hemi == 2], return_counts=True
            )
        )

        for index, row in df_label_colours.iterrows():
            # Left hemisphere
            l_clp = (
                left_hemi_label_counts_points[
                    left_hemi_counted_labels_points == row["idx"]
                ][0]
                if row["idx"] in left_hemi_counted_labels_points
                else 0
            )
            l_clc = (
                left_hemi_label_counts_centroids[
                    left_hemi_counted_labels_centroids == row["idx"]
                ][0]
                if row["idx"] in left_hemi_counted_labels_centroids
                else 0
            )
            # Right hemisphere
            r_clp = (
                right_hemi_label_counts_points[
                    right_hemi_counted_labels_points == row["idx"]
                ][0]
                if row["idx"] in right_hemi_counted_labels_points
                else 0
            )
            r_clc = (
                right_hemi_label_counts_centroids[
                    right_hemi_counted_labels_centroids == row["idx"]
                ][0]
                if row["idx"] in right_hemi_counted_labels_centroids
                else 0
            )

            # Skip empty counts
            if l_clp == r_clp == l_clc == r_clc == 0:
                continue

            # Add to dictionary
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))
            counts_per_label["pixel_count"].append(l_clp + r_clp)
            counts_per_label["object_count"].append(l_clc + r_clc)
            counts_per_label["left_hemi_pixel_count"].append(l_clp)
            counts_per_label["right_hemi_pixel_count"].append(r_clp)
            counts_per_label["left_hemi_object_count"].append(l_clc)
            counts_per_label["right_hemi_object_count"].append(r_clc)

    else:
        counted_labels_points, label_counts_points = np.unique(
            labels_dict_points, return_counts=True
        )
        counted_labels_centroids, label_counts_centroids = np.unique(
            labeled_dict_centroids, return_counts=True
        )

        for index, row in df_label_colours.iterrows():
            clp = (
                label_counts_points[counted_labels_points == row["idx"]][0]
                if row["idx"] in counted_labels_points
                else 0
            )
            clc = (
                label_counts_centroids[counted_labels_centroids == row["idx"]][0]
                if row["idx"] in counted_labels_centroids
                else 0
            )
            if clp == 0 and clc == 0:
                continue
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))
            counts_per_label["pixel_count"].append(clp)
            counts_per_label["object_count"].append(clc)

    df_counts_per_label = pd.DataFrame(counts_per_label)
    return df_counts_per_label


def read_flat_file(file):
    """
    Reads a flat file and produces an image array.

    Args:
        file (str): Path to the flat file.

    Returns:
        ndarray: Image array extracted from the file.
    """
    with open(file, "rb") as f:
        b, w, h = struct.unpack(">BII", f.read(9))
        data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))
    image_data = np.array(data)
    image = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            image[y, x] = image_data[x + y * w]
    return image


def read_seg_file(file):
    """
    Reads a segmentation file into an image array.

    Args:
        file (str): Path to the segmentation file.

    Returns:
        ndarray: The segmentation image.
    """
    with open(file, "rb") as f:

        def byte():
            return f.read(1)[0]

        def code():
            c = byte()
            if c < 0:
                raise "!"
            return c if c < 128 else (c & 127) | (code() << 7)

        if "SegRLEv1" != f.read(8).decode():
            raise "Header mismatch"
        atlas = f.read(code()).decode()
        print(f"Target atlas: {atlas}")
        codes = [code() for x in range(code())]
        w = code()
        h = code()
        data = []
        while len(data) < w * h:
            data += [codes[byte() if len(codes) <= 256 else code()]] * (code() + 1)
    image_data = np.array(data)
    image = np.reshape(image_data, (h, w))
    return image


def assign_labels_to_image(image, labelfile):
    """
    Assigns atlas or region labels to an image array.

    Args:
        image (ndarray): Image array to label.
        labelfile (DataFrame): Contains label IDs in the 'idx' column.

    Returns:
        ndarray: Image with assigned labels.
    """
    w, h = image.shape
    allen_id_image = np.zeros((h, w))  # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))

    values = image[coordsy, coordsx]
    lbidx = labelfile["idx"].values

    allen_id_image = lbidx[values.astype(int)]
    return allen_id_image


def count_pixels_per_label(image, scale_factor=False):
    """
    Counts the pixels associated with each label in an image.

    Args:
        image (ndarray): Image array containing labels.
        scale_factor (bool, optional): Apply scaling if True.

    Returns:
        DataFrame: Table of label IDs and pixel counts.
    """
    unique_ids, counts = np.unique(image, return_counts=True)
    if scale_factor:
        counts = counts * scale_factor
    area_per_label = list(zip(unique_ids, counts))
    df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "region_area"])
    return df_area_per_label


def warp_image(image, triangulation, rescaleXY):
    """
    Warps an image using triangulation, applying optional resizing.

    Args:
        image (ndarray): Image array to be warped.
        triangulation (ndarray): Triangulation data for remapping.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        ndarray: The warped image array.
    """
    if rescaleXY is not None:
        w, h = rescaleXY
    else:
        h, w = image.shape
    reg_h, reg_w = image.shape
    oldX, oldY = np.meshgrid(np.arange(reg_w), np.arange(reg_h))
    oldX = oldX.flatten()
    oldY = oldY.flatten()
    h_scale = h / reg_h
    w_scale = w / reg_w
    oldX = oldX * w_scale
    oldY = oldY * h_scale
    newX, newY = transform_vec(triangulation, oldX, oldY)
    newX = newX / w_scale
    newY = newY / h_scale
    newX = newX.reshape(reg_h, reg_w)
    newY = newY.reshape(reg_h, reg_w)
    newX = newX.astype(int)
    newY = newY.astype(int)
    tempX = newX.copy()
    tempY = newY.copy()
    tempX[tempX >= reg_w] = reg_w - 1
    tempY[tempY >= reg_h] = reg_h - 1
    tempX[tempX < 0] = 0
    tempY[tempY < 0] = 0
    new_image = image[tempY, tempX]
    new_image[newX >= reg_w] = 0
    new_image[newY >= reg_h] = 0
    new_image[newX < 0] = 0
    new_image[newY < 0] = 0
    return new_image


def flat_to_dataframe(image, damage_mask, hemi_mask, rescaleXY=None):
    """
    Builds a DataFrame from an image, incorporating optional damage/hemisphere masks.

    Args:
        image (ndarray): Source image with label IDs.
        damage_mask (ndarray): Binary mask indicating damaged areas.
        hemi_mask (ndarray): Binary mask for hemisphere assignment.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        DataFrame: Pixel counts grouped by label.
        ndarray: Scaled label map of the image.
    """
    log_memory_usage(
        "flat_to_dataframe_input", image, "Input image to flat_to_dataframe"
    )
    if damage_mask is not None:
        log_memory_usage("input_damage_mask", damage_mask, "Input damage mask")
    if hemi_mask is not None:
        log_memory_usage("input_hemi_mask", hemi_mask, "Input hemi mask")

    # If rescaleXY is set, resize image and masks to match
    if rescaleXY:
        log_memory_usage("before_rescale", image, "Before rescaling image")
        image = resize(
            image.astype(np.uint16),
            (rescaleXY[1], rescaleXY[0]),
            order=0,
            preserve_range=True,
        )
        log_memory_usage("after_rescale", image, "After rescaling image")
        if hemi_mask is not None:
            hemi_mask = resize(
                hemi_mask.astype(np.uint8),
                (rescaleXY[1], rescaleXY[0]),
                order=0,
                preserve_range=True,
            )
        if damage_mask is not None:
            damage_mask = resize(
                damage_mask.astype(np.uint8),
                (rescaleXY[1], rescaleXY[0]),
                order=0,
                preserve_range=True,
            ).astype(bool)
        scale_factor = False  # No need to scale counts, image is resized
    else:
        scale_factor = False
    df_area_per_label = pd.DataFrame(columns=["idx"])
    if hemi_mask is not None:
        hemi_mask = resize(
            hemi_mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            order=0,
            preserve_range=True,
        )

    if damage_mask is not None:
        damage_mask = resize(
            damage_mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            order=0,
            preserve_range=True,
        ).astype(bool)

    # Build combinations for each scenario
    if (hemi_mask is not None) and (damage_mask is not None):
        combos = [
            (1, 0, "left_hemi_undamaged_region_area"),
            (1, 1, "left_hemi_damaged_region_area"),
            (2, 0, "right_hemi_undamaged_region_area"),
            (2, 1, "right_hemi_damaged_region_area"),
        ]
    elif (hemi_mask is not None) and (damage_mask is None):
        combos = [
            (1, 0, "left_hemi_region_area"),
            (2, 0, "right_hemi_region_area"),
        ]
    elif (hemi_mask is None) and (damage_mask is not None):
        combos = [
            (0, 0, "undamaged_region_area"),
            (0, 1, "damaged_region_area"),
        ]
    else:
        combos = [
            (None, None, "region_area")
        ]  # compute for entire image with no filtering

    # Count pixels for each combo
    for hemi_val, damage_val, col_name in combos:
        mask = np.ones_like(image, dtype=bool)
        if hemi_mask is not None:
            mask &= hemi_mask == hemi_val
        if damage_mask is not None:
            mask &= damage_mask == damage_val
        combo_df = count_pixels_per_label(image[mask], scale_factor)
        combo_df = combo_df.rename(columns={"region_area": col_name})
        df_area_per_label = pd.merge(
            df_area_per_label, combo_df, on="idx", how="outer"
        ).fillna(0)

    # If both masks exist, compute additional columns
    if (hemi_mask is not None) and (damage_mask is not None):
        df_area_per_label["undamaged_region_area"] = (
            df_area_per_label["left_hemi_undamaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["damaged_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_damaged_region_area"]
        )
        df_area_per_label["left_hemi_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["left_hemi_undamaged_region_area"]
        )
        df_area_per_label["right_hemi_region_area"] = (
            df_area_per_label["right_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    if (hemi_mask is not None) and (damage_mask is None):
        df_area_per_label["region_area"] = (
            df_area_per_label["left_hemi_region_area"]
            + df_area_per_label["right_hemi_region_area"]
        )
    if (hemi_mask is None) and (damage_mask is not None):
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    return df_area_per_label


def load_image(file, image_vector, volume, triangulation, rescaleXY, labelfile=None):
    """
    Loads an image from file or transforms a preloaded array, optionally applying warping.

    Args:
        file (str): File path for the source image.
        image_vector (ndarray): Preloaded image data array.
        volume (ndarray): Atlas volume or similar data.
        triangulation (ndarray): Triangulation data for warping.
        rescaleXY (tuple): (width, height) for resizing.
        labelfile (DataFrame, optional): Label definitions.

    Returns:
        ndarray: The loaded or transformed image.
    """
    log_memory_usage("load_image_start", message=f"Loading image: {file}")

    if image_vector is not None and volume is not None:
        log_memory_usage("volume_input", volume, "Volume input to load_image")
        # get the slice of the atlas volume according to the anchorings
        target_slice = generate_target_slice(image_vector, volume)
        log_memory_usage(
            "target_slice", target_slice, "Target slice from generate_target_slice"
        )
        image = target_slice.astype(np.uint32)
        log_memory_usage("image_uint32", image, "Image after uint32 conversion")
    elif file.endswith(".seg"):  # Changed image_path to file
        image = read_seg_file(file)
        log_memory_usage("seg_image", image, "After reading seg file")
        image = assign_labels_to_image(image, labelfile)
        log_memory_usage("labeled_image", image, "After assigning labels")
    else:
        image = read_flat_file(file)
        log_memory_usage("flat_image", image, "After reading flat file")

    if triangulation is not None:
        log_memory_usage("before_warp", image, "Before image warping")
        image = warp_image(image, triangulation, rescaleXY)
        log_memory_usage("after_warp", image, "After image warping")

    log_memory_usage("load_image_final", image, "Final image from load_image")
    return image
