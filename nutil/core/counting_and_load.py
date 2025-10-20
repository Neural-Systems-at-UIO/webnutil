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

logger = logging.getLogger(__name__)

def log_memory_usage(var_name, array=None, message=""):
    """Log memory usage of an array and system memory."""
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    if array is not None:
        try:
            if hasattr(array, "nbytes"):
                logger.info(f"MEMORY: {var_name} = {array.nbytes / 1024 / 1024:.2f} MB ({array.shape} {array.dtype}) | Process: {memory_mb:.2f} MB | {message}")
            elif hasattr(array, "__len__"):
                logger.info(f"MEMORY: {var_name} = {len(array)} items | Process: {memory_mb:.2f} MB | {message}")
            else:
                logger.info(f"MEMORY: {var_name} = {type(array)} | Process: {memory_mb:.2f} MB | {message}")
        except Exception as e:
            logger.info(f"MEMORY: {var_name} = ERROR ({e}) | Process: {memory_mb:.2f} MB | {message}")
    else:
        logger.info(f"MEMORY: Process: {memory_mb:.2f} MB | {message}")


def create_base_counts_dict(with_hemisphere=False, with_damage=False):
    """Creates and returns a base dictionary structure for tracking counts."""
    counts = {"idx": [], "name": [], "r": [], "g": [], "b": [], "pixel_count": [], "object_count": []}
    if with_damage:
        counts.update({"undamaged_object_count": [], "damaged_object_count": [], "undamaged_pixel_count": [], "damaged_pixel_counts": []})
    if with_hemisphere:
        counts.update({"left_hemi_pixel_count": [], "right_hemi_pixel_count": [], "left_hemi_object_count": [], "right_hemi_object_count": []})
    if with_damage and with_hemisphere:
        counts.update({"left_hemi_undamaged_pixel_count": [], "left_hemi_damaged_pixel_count": [], "right_hemi_undamaged_pixel_count": [], 
                      "right_hemi_damaged_pixel_count": [], "left_hemi_undamaged_object_count": [], "left_hemi_damaged_object_count": [], 
                      "right_hemi_undamaged_object_count": [], "right_hemi_damaged_object_count": []})
    return counts


def pixel_count_per_region(labels_dict_points, labeled_dict_centroids, current_points_undamaged, current_centroids_undamaged,
                          current_points_hemi, current_centroids_hemi, df_label_colours, with_damage=False):
    """Tally object counts by region, optionally tracking damage and hemispheres."""
    with_hemi = None not in current_points_hemi
    counts_per_label = create_base_counts_dict(with_hemisphere=with_hemi, with_damage=with_damage)
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
            l_clpu = left_hemi_label_counts_points_undamaged[left_hemi_counted_labels_points_undamaged == row["idx"]][0] if row["idx"] in left_hemi_counted_labels_points_undamaged else 0
            l_clpd = left_hemi_label_counts_points_damaged[left_hemi_counted_labels_points_damaged == row["idx"]][0] if row["idx"] in left_hemi_counted_labels_points_damaged else 0
            r_clpu = right_hemi_label_counts_points_undamaged[right_hemi_counted_labels_points_undamaged == row["idx"]][0] if row["idx"] in right_hemi_counted_labels_points_undamaged else 0
            r_clpd = right_hemi_label_counts_points_damaged[right_hemi_counted_labels_points_damaged == row["idx"]][0] if row["idx"] in right_hemi_counted_labels_points_damaged else 0
            l_clcu = left_hemi_label_counts_centroids_undamaged[left_hemi_counted_labels_centroids_undamaged == row["idx"]][0] if row["idx"] in left_hemi_counted_labels_centroids_undamaged else 0
            l_clcd = left_hemi_label_counts_centroids_damaged[left_hemi_counted_labels_centroids_damaged == row["idx"]][0] if row["idx"] in left_hemi_counted_labels_centroids_damaged else 0
            r_clcu = right_hemi_label_counts_centroids_undamaged[right_hemi_counted_labels_centroids_undamaged == row["idx"]][0] if row["idx"] in right_hemi_counted_labels_centroids_undamaged else 0
            r_clcd = right_hemi_label_counts_centroids_damaged[right_hemi_counted_labels_centroids_damaged == row["idx"]][0] if row["idx"] in right_hemi_counted_labels_centroids_damaged else 0
            
            if l_clcd == l_clcu == l_clpd == l_clpu == r_clcd == r_clcu == r_clpd == r_clpu == 0:
                continue
            
            clpu, clpd = l_clpu + r_clpu, l_clpd + r_clpd
            clcu, clcd = l_clcu + r_clcu, l_clcd + r_clcd
            for key, val in [("idx", row["idx"]), ("name", row["name"]), ("r", int(row["r"])), ("g", int(row["g"])), ("b", int(row["b"])),
                            ("pixel_count", clpu + clpd), ("undamaged_pixel_count", clpu), ("damaged_pixel_counts", clpd),
                            ("object_count", clcu + clcd), ("undamaged_object_count", clcu), ("damaged_object_count", clcd),
                            ("left_hemi_pixel_count", l_clpu + l_clpd), ("left_hemi_undamaged_pixel_count", l_clpu), ("left_hemi_damaged_pixel_count", l_clpd),
                            ("left_hemi_object_count", l_clcu + l_clcd), ("left_hemi_undamaged_object_count", l_clcu), ("left_hemi_damaged_object_count", l_clcd),
                            ("right_hemi_pixel_count", r_clpu + r_clpd), ("right_hemi_undamaged_pixel_count", r_clpu), ("right_hemi_damaged_pixel_count", r_clpd),
                            ("right_hemi_object_count", r_clcu + r_clcd), ("right_hemi_undamaged_object_count", r_clcu), ("right_hemi_damaged_object_count", r_clcd)]:
                counts_per_label[key].append(val)

    elif with_damage and (not with_hemi):
        counted_labels_points_undamaged, label_counts_points_undamaged = np.unique(labels_dict_points[current_points_undamaged], return_counts=True)
        counted_labels_points_damaged, label_counts_points_damaged = np.unique(labels_dict_points[~current_points_undamaged], return_counts=True)
        counted_labels_centroids_undamaged, label_counts_centroids_undamaged = np.unique(labeled_dict_centroids[current_centroids_undamaged], return_counts=True)
        counted_labels_centroids_damaged, label_counts_centroids_damaged = np.unique(labeled_dict_centroids[~current_centroids_undamaged], return_counts=True)
        for index, row in df_label_colours.iterrows():
            clpu = label_counts_points_undamaged[counted_labels_points_undamaged == row["idx"]][0] if row["idx"] in counted_labels_points_undamaged else 0
            clpd = label_counts_points_damaged[counted_labels_points_damaged == row["idx"]][0] if row["idx"] in counted_labels_points_damaged else 0
            clcu = label_counts_centroids_undamaged[counted_labels_centroids_undamaged == row["idx"]][0] if row["idx"] in counted_labels_centroids_undamaged else 0
            clcd = label_counts_centroids_damaged[counted_labels_centroids_damaged == row["idx"]][0] if row["idx"] in counted_labels_centroids_damaged else 0
            if clcd == clcu == clpd == clpu == 0:
                continue
            for key, val in [("idx", row["idx"]), ("name", row["name"]), ("r", int(row["r"])), ("g", int(row["g"])), ("b", int(row["b"])),
                            ("pixel_count", clpu + clpd), ("undamaged_pixel_count", clpu), ("damaged_pixel_counts", clpd),
                            ("object_count", clcu + clcd), ("undamaged_object_count", clcu), ("damaged_object_count", clcd)]:
                counts_per_label[key].append(val)

    elif with_hemi and (not with_damage):
        left_hemi_counted_labels_points, left_hemi_label_counts_points = np.unique(labels_dict_points[current_points_hemi == 1], return_counts=True)
        left_hemi_counted_labels_centroids, left_hemi_label_counts_centroids = np.unique(labeled_dict_centroids[current_centroids_hemi == 1], return_counts=True)
        right_hemi_counted_labels_points, right_hemi_label_counts_points = np.unique(labels_dict_points[current_points_hemi == 2], return_counts=True)
        right_hemi_counted_labels_centroids, right_hemi_label_counts_centroids = np.unique(labeled_dict_centroids[current_centroids_hemi == 2], return_counts=True)

        for index, row in df_label_colours.iterrows():
            l_clp = left_hemi_label_counts_points[left_hemi_counted_labels_points == row["idx"]][0] if row["idx"] in left_hemi_counted_labels_points else 0
            l_clc = left_hemi_label_counts_centroids[left_hemi_counted_labels_centroids == row["idx"]][0] if row["idx"] in left_hemi_counted_labels_centroids else 0
            r_clp = right_hemi_label_counts_points[right_hemi_counted_labels_points == row["idx"]][0] if row["idx"] in right_hemi_counted_labels_points else 0
            r_clc = right_hemi_label_counts_centroids[right_hemi_counted_labels_centroids == row["idx"]][0] if row["idx"] in right_hemi_counted_labels_centroids else 0
            
            if l_clp == r_clp == l_clc == r_clc == 0:
                continue
            
            for key, val in [("idx", row["idx"]), ("name", row["name"]), ("r", int(row["r"])), ("g", int(row["g"])), ("b", int(row["b"])),
                            ("pixel_count", l_clp + r_clp), ("object_count", l_clc + r_clc),
                            ("left_hemi_pixel_count", l_clp), ("right_hemi_pixel_count", r_clp),
                            ("left_hemi_object_count", l_clc), ("right_hemi_object_count", r_clc)]:
                counts_per_label[key].append(val)

    else:
        counted_labels_points, label_counts_points = np.unique(labels_dict_points, return_counts=True)
        counted_labels_centroids, label_counts_centroids = np.unique(labeled_dict_centroids, return_counts=True)
        for index, row in df_label_colours.iterrows():
            clp = label_counts_points[counted_labels_points == row["idx"]][0] if row["idx"] in counted_labels_points else 0
            clc = label_counts_centroids[counted_labels_centroids == row["idx"]][0] if row["idx"] in counted_labels_centroids else 0
            if clp == 0 and clc == 0:
                continue
            for key, val in [("idx", row["idx"]), ("name", row["name"]), ("r", int(row["r"])), ("g", int(row["g"])), ("b", int(row["b"])),
                            ("pixel_count", clp), ("object_count", clc)]:
                counts_per_label[key].append(val)

    return pd.DataFrame(counts_per_label)


def read_flat_file(file: str) -> np.ndarray[int,int]:
    """Reads a flat file and produces an image array."""
    with open(file, "rb") as f:
        b, w, h = struct.unpack(">BII", f.read(9))
        data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))
    image_data = np.array(data)
    image = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            image[y, x] = image_data[x + y * w]
    return image


def read_segmentation_file(file: str) -> np.ndarray:
    """Reads a segmentation file into an image array."""
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

    return np.reshape(np.array(data), (h, w))


def assign_labels_to_image(image: np.ndarray, labelfile: pd.DataFrame) -> np.ndarray:
    """Assigns atlas or region labels to an image array."""
    w, h = image.shape
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
    allen_id_image = labelfile["idx"].values[image[coordsy, coordsx].astype(int)]
    print(f"Created lookup slice")
    return allen_id_image


def count_pixels_per_label(image, scale_factor=1.0):
    """Counts the pixels associated with each label in an image."""
    unique_ids, counts = np.unique(image, return_counts=True)
    counts = counts * scale_factor if scale_factor != 1.0 and scale_factor is not False else counts
    return pd.DataFrame(list(zip(unique_ids, counts)), columns=["idx", "region_area"])


def warp_image(image, triangulation, rescaleXY):
    """Warps an image using triangulation, applying optional resizing."""
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
    """Builds a DataFrame from an image, incorporating optional damage/hemisphere masks."""
    log_memory_usage("flat_to_dataframe_input", image, "Input image to flat_to_dataframe")
    if damage_mask is not None: log_memory_usage("input_damage_mask", damage_mask, "Input damage mask")
    if hemi_mask is not None: log_memory_usage("input_hemi_mask", hemi_mask, "Input hemi mask")

    if rescaleXY:
        scale_factor = (rescaleXY[0] * rescaleXY[1]) / (image.shape[1] * image.shape[0])
        log_memory_usage("scale_calculation", message=f"Segmentation scaling: atlas={image.shape[1]}x{image.shape[0]} -> segmentation={rescaleXY[0]}x{rescaleXY[1]} -> scale_factor={scale_factor:.4f}")
    else:
        scale_factor = 1.0
        log_memory_usage("rescale_disabled", message=f"No scaling applied: keeping {image.shape}")

    if hemi_mask is not None:
        hemi_mask = resize(hemi_mask.astype(np.uint8), (image.shape[1], image.shape[0]), order=0, preserve_range=True)
    if damage_mask is not None:
        damage_mask = resize(damage_mask.astype(np.uint8), (image.shape[1], image.shape[0]), order=0, preserve_range=True).astype(bool)
    
    df_area_per_label = pd.DataFrame(columns=["idx"])

    combos = ([(1, 0, "left_hemi_undamaged_region_area"), (1, 1, "left_hemi_damaged_region_area"), (2, 0, "right_hemi_undamaged_region_area"), (2, 1, "right_hemi_damaged_region_area")] if hemi_mask is not None and damage_mask is not None
              else [(1, 0, "left_hemi_region_area"), (2, 0, "right_hemi_region_area")] if hemi_mask is not None
              else [(0, 0, "undamaged_region_area"), (0, 1, "damaged_region_area")] if damage_mask is not None
              else [(None, None, "region_area")])
    for hemi_val, damage_val, col_name in combos:
        mask = np.ones_like(image, dtype=bool)
        if hemi_mask is not None: mask &= hemi_mask == hemi_val
        if damage_mask is not None: mask &= damage_mask == damage_val
        combo_df = count_pixels_per_label(image[mask], scale_factor).rename(columns={"region_area": col_name})
        df_area_per_label = pd.merge(df_area_per_label, combo_df, on="idx", how="outer").fillna(0)

    if hemi_mask is not None and damage_mask is not None:
        df_area_per_label["undamaged_region_area"] = df_area_per_label["left_hemi_undamaged_region_area"] + df_area_per_label["right_hemi_undamaged_region_area"]
        df_area_per_label["damaged_region_area"] = df_area_per_label["left_hemi_damaged_region_area"] + df_area_per_label["right_hemi_damaged_region_area"]
        df_area_per_label["left_hemi_region_area"] = df_area_per_label["left_hemi_damaged_region_area"] + df_area_per_label["left_hemi_undamaged_region_area"]
        df_area_per_label["right_hemi_region_area"] = df_area_per_label["right_hemi_damaged_region_area"] + df_area_per_label["right_hemi_undamaged_region_area"]
        df_area_per_label["region_area"] = df_area_per_label["undamaged_region_area"] + df_area_per_label["damaged_region_area"]
    elif hemi_mask is not None:
        df_area_per_label["region_area"] = df_area_per_label["left_hemi_region_area"] + df_area_per_label["right_hemi_region_area"]
    elif damage_mask is not None:
        df_area_per_label["region_area"] = df_area_per_label["undamaged_region_area"] + df_area_per_label["damaged_region_area"]
    return df_area_per_label


def load_image(file, image_vector, volume, triangulation, rescaleXY, labelfile=None):
    """Loads an image from file or transforms a preloaded array, optionally applying warping."""
    log_memory_usage("load_image_start", message=f"Loading image: {file}")

    if image_vector is not None and volume is not None:
        log_memory_usage("volume_input", volume, "Volume input to load_image")
        image = generate_target_slice(image_vector, volume).astype(np.uint32)
        log_memory_usage("image_uint32", image, "Image after uint32 conversion")
    elif file.endswith(".seg"):
        image = assign_labels_to_image(read_segmentation_file(file), labelfile)
        log_memory_usage("labeled_image", image, "After reading and assigning labels")
    else:
        image = read_flat_file(file)
        log_memory_usage("flat_image", image, "After reading flat file")

    if triangulation is not None:
        image = warp_image(image, triangulation, rescaleXY)
        log_memory_usage("after_warp", image, "After image warping")

    image = resize(image, rescaleXY, order=0, preserve_range=True, mode="reflect")
    log_memory_usage("load_image_final", image, "Final image from load_image")
    return image
