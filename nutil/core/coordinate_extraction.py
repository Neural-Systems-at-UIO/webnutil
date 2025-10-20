import numpy as np
import pandas as pd
import cv2
import gc
import psutil
import logging
from skimage import measure
from skimage.transform import resize

# custom imports 
from ..utils.read_and_write import load_quint_json
from .counting_and_load import flat_to_dataframe, load_image
from .generate_target_slice import generate_target_slice
from .visualign_deformations import triangulate
from ..utils.reconstruct_dzi import reconstruct_dzi
from .transformations import (
    transform_points_to_atlas_space,
    transform_to_registration,
    get_transformed_coordinates,
)
from .utils import (
    get_flat_files,
    get_segmentations,
    extract_section_numbers,
    scale_positions,
    process_results,
    get_current_flat_file,
)

# Configure logging for memory debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
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


def get_centroids_and_area(segmentation, pixel_cut_off=0):
    """Retrieves centroids, areas, and pixel coordinates of labeled regions."""
    labels_info = [obj for obj in measure.regionprops(measure.label(segmentation)) if obj.area > pixel_cut_off]
    return (
        np.array([obj.centroid for obj in labels_info]),
        np.array([obj.area for obj in labels_info]),
        np.array([obj.coords for obj in labels_info], dtype=object)
    )


update_spacing = lambda anchoring, width, height, grid_spacing: (
    int(width * grid_spacing / np.sqrt(sum(anchoring[i + 3] ** 2 for i in range(3)))),
    int(height * grid_spacing / np.sqrt(sum(anchoring[i + 6] ** 2 for i in range(3))))
)


def folder_to_atlas_space(
    folder: str,
    quint_alignment: str,
    atlas_labels: pd.DataFrame,
    pixel_id: list[int] = [0, 0, 0],
    non_linear: bool = True,
    object_cutoff: int = 0,
    atlas_volume: np.ndarray | None = None,
    hemi_map: np.ndarray | None = None,
    use_flat: bool = False,
    apply_damage_mask: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[pd.DataFrame],
    list[int],
    list[int],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    log_memory_usage(
        "start_folder_to_atlas", message=f"Starting folder_to_atlas_space for {folder}"
    )

    quint_json = load_quint_json(quint_alignment)
    slices = list(quint_json["slices"])  # shallow copy to avoid mutation
    gridspacing = quint_json.get("gridspacing") if apply_damage_mask else None
    if not apply_damage_mask:
        for s in slices:
            s.pop("grid", None)
    segmentations = get_segmentations(folder)
    flat_files, flat_file_nrs = get_flat_files(folder, use_flat)
    n = len(segmentations)

    log_memory_usage("segmentations", message=f"Found {n} segmentations")
    if atlas_volume is not None:
        log_memory_usage("atlas_volume_input", atlas_volume, "Input atlas volume")
    if hemi_map is not None:
        log_memory_usage("hemi_map_input", hemi_map, "Input hemisphere map")
    region_areas_list = [
        pd.DataFrame(
            {
                "idx": [],
                "name": [],
                "r": [],
                "g": [],
                "b": [],
                "region_area": [],
                "pixel_count": [],
                "object_count": [],
                "area_fraction": [],
            }
        )
        for _ in range(n)
    ]
    points_list = [np.array([]) for _ in range(n)]
    points_labels = [np.array([]) for _ in range(n)]
    centroids_list = [np.array([]) for _ in range(n)]
    centroids_labels = [np.array([]) for _ in range(n)]
    per_point_undamaged_list = [np.array([]) for _ in range(n)]
    per_centroid_undamaged_list = [np.array([]) for _ in range(n)]
    points_hemi_labels = [np.array([]) for _ in range(n)]
    centroids_hemi_labels = [np.array([]) for _ in range(n)]
    for index, segmentation_path in enumerate(segmentations):
        log_memory_usage(
            "processing_slice",
            message=f"Processing slice {index+1}/{n}: {segmentation_path}",
        )
        seg_nr = int(extract_section_numbers([segmentation_path])[0])
        idxs = [i for i, s in enumerate(slices) if s["nr"] == seg_nr]
        if not idxs:
            region_areas_list[index] = pd.DataFrame({"idx": []})
            continue
        current_slice = slices[idxs[0]]
        if not current_slice.get("anchoring"):
            region_areas_list[index] = pd.DataFrame({"idx": []})
            continue
        current_flat = get_current_flat_file(
            seg_nr, flat_files, flat_file_nrs, use_flat
        )
        segmentation_to_atlas_space(
            current_slice,
            segmentation_path,
            atlas_labels,
            current_flat,
            pixel_id,
            non_linear,
            points_list,
            centroids_list,
            points_labels,
            centroids_labels,
            region_areas_list,
            per_point_undamaged_list,
            per_centroid_undamaged_list,
            points_hemi_labels,
            centroids_hemi_labels,
            index,
            object_cutoff,
            atlas_volume,
            hemi_map,
            use_flat,
            gridspacing,
        )
        log_memory_usage("after_slice", message=f"After processing slice {index+1}/{n}")
        if index % 5 == 0:  # Log every 5 slices
            gc.collect()
            log_memory_usage(
                "gc_checkpoint", message=f"GC checkpoint at slice {index+1}"
            )
    (
        points,
        centroids,
        points_labels_arr,
        centroids_labels_arr,
        points_hemi_labels_arr,
        centroids_hemi_labels_arr,
        points_len,
        centroids_len,
        per_point_undamaged_arr,
        per_centroid_undamaged_arr,
    ) = process_results(
        points_list,
        centroids_list,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    )
    return (
        points,
        centroids,
        points_labels_arr,
        centroids_labels_arr,
        points_hemi_labels_arr,
        centroids_hemi_labels_arr,
        region_areas_list,
        points_len,
        centroids_len,
        segmentations,
        per_point_undamaged_arr,
        per_centroid_undamaged_arr,
    )





def segmentation_to_atlas_space(
    slice_dict, segmentation_path, atlas_labels, flat_file_atlas=None, pixel_id="auto", non_linear=True,
    points_list=np.ndarray([]), centroids_list=None, points_labels=None, centroids_labels=None,
    region_areas_list=None, per_point_undamaged_list=None, per_centroid_undamaged_list=None,
    points_hemi_labels=None, centroids_hemi_labels=None, index=None, object_cutoff=0,
    atlas_volume=None, hemi_map=None, use_flat=False, grid_spacing=None,
):
    """Transforms a single segmentation file into atlas space."""
    log_memory_usage(
        "start", message=f"Starting segmentation_to_atlas_space for {segmentation_path}"
    )

    segmentation = reconstruct_dzi(segmentation_path) if segmentation_path.endswith(".dzip") else cv2.imread(segmentation_path)
    log_memory_usage("segmentation", segmentation, "After loading segmentation")

    pixel_id = np.array(pixel_id, dtype=np.uint8)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    log_memory_usage(
        "dimensions",
        message=f"seg: {seg_height}x{seg_width}, reg: {reg_height}x{reg_width}",
    )

    triangulation = triangulate(reg_width, reg_height, slice_dict["markers"]) if non_linear and "markers" in slice_dict else None
    if "grid" in slice_dict:
        width, height = slice_dict["width"], slice_dict["height"]
        xspacing, yspacing = update_spacing(slice_dict["anchoring"], width, height, grid_spacing)
        x_coords = np.arange(slice_dict["gridx"], width, xspacing)
        y_coords = np.arange(slice_dict["gridy"], height, yspacing)
        markers = [(x_coords[i % len(x_coords)], y_coords[i // len(x_coords)]) for i in range(len(slice_dict["grid"]))]
        damage_mask = np.ones((len(y_coords), len(x_coords)), dtype=int)
        for i, (x, y) in enumerate(markers):
            if slice_dict["grid"][i] == 4:
                damage_mask[y // yspacing, x // xspacing] = 0
        log_memory_usage(
            "damage_mask_initial", damage_mask, "After creating damage mask"
        )
        damage_mask = resize(
            damage_mask.astype(np.uint8),
            (seg_height, seg_width),
            order=0,
            preserve_range=True,
        ).astype(bool)
        log_memory_usage(
            "damage_mask_resized", damage_mask, "After resizing damage mask"
        )
    else:
        damage_mask = None
    if hemi_map is not None:
        log_memory_usage("hemi_map", hemi_map, "Input hemi_map")
        hemi_mask = generate_target_slice(slice_dict["anchoring"], hemi_map)
        log_memory_usage("hemi_mask", hemi_mask, "After generating hemi mask")
    else:
        hemi_mask = None
    
    # build atlas map and calculate region areas
    reg_width, reg_height = slice_dict["width"], slice_dict["height"]
    log_memory_usage("before_load_image", message=f"Before load_image - seg: {seg_width}x{seg_height}")
    if atlas_volume is not None:
        log_memory_usage("atlas_volume", atlas_volume, "Input atlas volume")
    atlas_map = load_image(flat_file_atlas, slice_dict["anchoring"], atlas_volume, triangulation, (reg_width, reg_height), atlas_labels)
    log_memory_usage("atlas_map_loaded", atlas_map, "Atlas map after load_image")
    region_areas = flat_to_dataframe(atlas_map, damage_mask, hemi_mask, (seg_width, seg_height))
    log_memory_usage("region_areas", message=f"Region areas dataframe: {len(region_areas)} rows")
    log_memory_usage("atlas_map", atlas_map, "After calculating region areas")

    scaled_atlas_map = atlas_map
    atlas_at_original_resolution = False
    y_scale = (reg_height - 1) / (seg_height - 1)
    x_scale = (reg_width - 1) / (seg_width - 1)
    centroids, points = None, None
    scaled_centroidsX, scaled_centroidsY, scaled_x, scaled_y = None, None, None, None

    (
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        scaled_y,
        scaled_x,
        per_centroid_labels,
    ) = get_objects_and_assign_regions_optimized(
        segmentation,
        pixel_id,
        scaled_atlas_map,
        y_scale,
        x_scale,
        object_cutoff,
        tolerance=10,
        atlas_at_original_resolution=atlas_at_original_resolution,
        reg_height=reg_height,
        reg_width=reg_width,
        seg_height=seg_height,
        seg_width=seg_width,
    )

    log_memory_usage(
        "centroids", centroids, "After get_objects_and_assign_regions_optimized"
    )
    log_memory_usage("scaled_coordinates", scaled_x, "scaled_x coordinates")
    log_memory_usage("scaled_coordinates_y", scaled_y, "scaled_y coordinates")

    del segmentation
    log_memory_usage("after_del_segmentation", message="After deleting segmentation")

    # handle case where no pixels detected
    if scaled_y is None or scaled_x is None:
        points_list[index] = centroids_list[index] = np.array([])
        region_areas_list[index] = region_areas
        centroids_labels[index] = per_centroid_undamaged_list[index] = np.array([])
        points_labels[index] = per_point_undamaged_list[index] = np.array([])
        points_hemi_labels[index] = centroids_hemi_labels[index] = np.array([])
        del atlas_map
        if damage_mask is not None: del damage_mask
        if hemi_mask is not None: del hemi_mask
        gc.collect()
        return

    # handle case where pixels exist but no objects (centroids) detected
    if scaled_centroidsX is None or scaled_centroidsY is None:
        centroids_list[index] = centroids_labels[index] = np.array([])
        per_centroid_undamaged_list[index] = centroids_hemi_labels[index] = np.array([])
        # continue processing points below, don't return early

    # assign point labels
    if atlas_at_original_resolution:
        atlas_height, atlas_width = scaled_atlas_map.shape
        atlas_point_y = scaled_y * (atlas_height / reg_height)
        atlas_point_x = scaled_x * (atlas_width / reg_width)
        valid_mask = ((np.round(atlas_point_y).astype(int) >= 0) & (np.round(atlas_point_y).astype(int) < atlas_height) & 
                     (np.round(atlas_point_x).astype(int) >= 0) & (np.round(atlas_point_x).astype(int) < atlas_width))
        per_point_labels = np.zeros(len(scaled_y), dtype=int)
        if np.any(valid_mask):
            per_point_labels[valid_mask] = scaled_atlas_map[np.round(atlas_point_y[valid_mask]).astype(int), np.round(atlas_point_x[valid_mask]).astype(int)]
    else:
        rounded_y, rounded_x = np.round(scaled_y).astype(int), np.round(scaled_x).astype(int)
        y_indices = np.clip(rounded_y, 0, scaled_atlas_map.shape[0] - 1)
        x_indices = np.clip(rounded_x, 0, scaled_atlas_map.shape[1] - 1)
        per_point_labels = scaled_atlas_map[y_indices, x_indices]

    if damage_mask is not None:
        log_memory_usage("damage_mask_before_resize", damage_mask, "Before damage mask resize")
        damage_mask = resize(damage_mask.astype(np.uint8), (scaled_atlas_map.shape[1], scaled_atlas_map.shape[0]), order=0, preserve_range=True).astype(bool)
        log_memory_usage("damage_mask_after_resize", damage_mask, "After damage mask resize")
        per_point_undamaged = damage_mask[np.round(scaled_y * y_scale).astype(int).clip(0, damage_mask.shape[0] - 1), 
                                          np.round(scaled_x * x_scale).astype(int).clip(0, damage_mask.shape[1] - 1)]
        per_centroid_undamaged = (damage_mask[np.round(scaled_centroidsY * y_scale).astype(int).clip(0, damage_mask.shape[0] - 1), 
                                              np.round(scaled_centroidsX * x_scale).astype(int).clip(0, damage_mask.shape[1] - 1)] 
                                 if scaled_centroidsX is not None and scaled_centroidsY is not None else np.array([], dtype=bool))
    else:
        per_point_undamaged = np.ones(scaled_x.shape, dtype=bool)
        per_centroid_undamaged = np.ones(scaled_centroidsX.shape, dtype=bool) if scaled_centroidsX is not None else np.array([], dtype=bool)
    if hemi_mask is not None:
        log_memory_usage("hemi_mask_before_resize", hemi_mask, "Before hemi mask resize")
        hemi_mask = resize(hemi_mask.astype(np.uint8), (scaled_atlas_map.shape[1], scaled_atlas_map.shape[0]), order=0, preserve_range=True)
        log_memory_usage("hemi_mask_after_resize", hemi_mask, "After hemi mask resize")
        per_point_hemi = hemi_mask[np.round(scaled_y).astype(int), np.round(scaled_x).astype(int)][per_point_undamaged]
        per_centroid_hemi = (hemi_mask[np.round(scaled_centroidsY).astype(int), np.round(scaled_centroidsX).astype(int)][per_centroid_undamaged] 
                            if scaled_centroidsX is not None and scaled_centroidsY is not None else np.array([]))
    else:
        per_point_hemi = [None] * len(scaled_x)
        per_centroid_hemi = [None] * len(scaled_centroidsX) if scaled_centroidsX is not None else np.array([])

    per_point_labels = per_point_labels[per_point_undamaged]
    per_centroid_labels = per_centroid_labels[per_centroid_undamaged] if per_centroid_labels is not None and len(per_centroid_labels) > 0 else np.array([])

    # transform coordinates
    if scaled_centroidsX is not None and scaled_centroidsY is not None:
        new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(non_linear, slice_dict, scaled_x[per_point_undamaged], 
                                                                                      scaled_y[per_point_undamaged], scaled_centroidsX[per_centroid_undamaged], 
                                                                                      scaled_centroidsY[per_centroid_undamaged], triangulation)
    else:
        new_x, new_y, _, _ = get_transformed_coordinates(non_linear, slice_dict, scaled_x[per_point_undamaged], scaled_y[per_point_undamaged], 
                                                          np.array([]), np.array([]), triangulation)
        centroids_new_x, centroids_new_y = np.array([]), np.array([])
    
    points, centroids = transform_points_to_atlas_space(slice_dict, new_x, new_y, centroids_new_x, centroids_new_y, reg_height, reg_width)

    log_memory_usage("final_points", points, "Final transformed points")
    log_memory_usage("final_centroids", centroids, "Final transformed centroids")

    del scaled_atlas_map, atlas_map
    if hemi_mask is not None: del hemi_mask
    if damage_mask is not None: del damage_mask
    gc.collect()
    log_memory_usage("after_gc", message="After cleanup and garbage collection")

    points_list[index] = np.array(points if points is not None else [])
    centroids_list[index] = np.array(centroids if centroids is not None else [])
    region_areas_list[index] = region_areas
    centroids_labels[index] = np.array(per_centroid_labels if centroids is not None else [])
    per_centroid_undamaged_list[index] = np.array(per_centroid_undamaged if centroids is not None else [])
    points_labels[index] = np.array(per_point_labels if points is not None else [])
    per_point_undamaged_list[index] = np.array(per_point_undamaged if points is not None else [])
    points_hemi_labels[index] = np.array(per_point_hemi if points is not None else [])
    centroids_hemi_labels[index] = np.array(per_centroid_hemi if points is not None else [])





def get_objects_and_assign_regions_optimized(segmentation, pixel_id, atlas_map, y_scale, x_scale, object_cutoff=0, 
                                             tolerance=10, atlas_at_original_resolution=False, reg_height=None, 
                                             reg_width=None, seg_height=None, seg_width=None):
    """Single-pass object detection, pixel extraction, and region assignment."""
    print(f"Detecting objects with pixel_id: {pixel_id}, tolerance: {tolerance}")
    binary_seg = np.all(np.abs(segmentation.astype(int) - np.array(pixel_id, dtype=int)) <= tolerance, axis=2)
    pixel_y, pixel_x = np.where(binary_seg)
    print(f"Detected {len(pixel_y)} pixels matching the target color")
    if len(pixel_y) == 0:
        return None, None, None, None, None, None

    scaled_y, scaled_x = scale_positions(pixel_y, pixel_x, y_scale, x_scale)
    objects_info = measure.regionprops(measure.label(binary_seg))
    print(f"Total labeled regions (objects): {len(objects_info)}")
    objects_info = [obj for obj in objects_info if obj.area > object_cutoff]
    print(f"Regions after area cutoff ({object_cutoff}): {len(objects_info)}")
    if len(objects_info) == 0:
        return None, None, None, scaled_y, scaled_x, None

    centroids, per_centroid_labels = [], []

    for obj in objects_info:
        centroids.append(obj.centroid)
        scaled_obj_y, scaled_obj_x = scale_positions(obj.coords[:, 0], obj.coords[:, 1], y_scale, x_scale)
        
        if atlas_at_original_resolution:
            atlas_height, atlas_width = atlas_map.shape
            assignment_y = scaled_obj_y * (atlas_height / reg_height)
            assignment_x = scaled_obj_x * (atlas_width / reg_width)
            atlas_bounds_height, atlas_bounds_width = atlas_height, atlas_width
        else:
            assignment_y, assignment_x = scaled_obj_y, scaled_obj_x
            atlas_bounds_height, atlas_bounds_width = atlas_map.shape[0], atlas_map.shape[1]
        
        valid_mask = ((np.round(assignment_y).astype(int) >= 0) & (np.round(assignment_y).astype(int) < atlas_bounds_height) & 
                     (np.round(assignment_x).astype(int) >= 0) & (np.round(assignment_x).astype(int) < atlas_bounds_width))
        
        if not np.any(valid_mask):
            per_centroid_labels.append(0)
            continue
        
        pixel_labels = atlas_map[np.round(assignment_y[valid_mask]).astype(int), np.round(assignment_x[valid_mask]).astype(int)]
        unique_labels, counts = np.unique(pixel_labels, return_counts=True)
        per_centroid_labels.append(unique_labels[np.argmax(counts)])

    if centroids:
        centroids = np.array(centroids)
        scaled_centroidsY, scaled_centroidsX = scale_positions(centroids[:, 0], centroids[:, 1], y_scale, x_scale)
        per_centroid_labels = np.array(per_centroid_labels)
        print(f"Successfully processed {len(per_centroid_labels)} centroids with region assignments")
    else:
        centroids = scaled_centroidsX = scaled_centroidsY = None
        per_centroid_labels = np.array([])

    return centroids, scaled_centroidsX, scaled_centroidsY, scaled_y, scaled_x, per_centroid_labels