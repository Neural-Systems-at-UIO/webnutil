import numpy as np
import pandas as pd
from ..utils.read_and_write import load_quint_json
from .counting_and_load import flat_to_dataframe, load_image
from .generate_target_slice import generate_target_slice
from .visualign_deformations import triangulate
import cv2
import gc
import psutil
import logging
from skimage import measure
from skimage.transform import resize
from ..utils.reconstruct_dzi import reconstruct_dzi
from .transformations import (
    transform_points_to_atlas_space,
    transform_to_registration,
    get_transformed_coordinates,
)
from .utils import (
    get_flat_files,
    get_segmentations,
    number_sections,
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


def get_centroids_and_area(segmentation, pixel_cut_off=0):
    """
    Retrieves centroids, areas, and pixel coordinates of labeled regions.

    Args:
        segmentation (ndarray): Binary segmentation array.
        pixel_cut_off (int, optional): Minimum object size threshold.

    Returns:
        tuple: (centroids, area, coords) of retained objects.
    """
    labels = measure.label(segmentation)
    labels_info = measure.regionprops(labels)
    labels_info = [label for label in labels_info if label.area > pixel_cut_off]
    centroids = np.array([label.centroid for label in labels_info])
    area = np.array([label.area for label in labels_info])
    coords = np.array([label.coords for label in labels_info], dtype=object)
    return centroids, area, coords


def update_spacing(anchoring, width, height, grid_spacing):
    """
    Calculates spacing along width and height from slice anchoring.

    Args:
        anchoring (list): Anchoring transformation parameters.
        width (int): Image width.
        height (int): Image height.
        grid_spacing (int): Grid spacing in image units.

    Returns:
        tuple: (xspacing, yspacing)
    """
    if len(anchoring) != 9:
        print("Anchoring does not have 9 elements.")
    ow = np.sqrt(sum([anchoring[i + 3] ** 2 for i in range(3)]))
    oh = np.sqrt(sum([anchoring[i + 6] ** 2 for i in range(3)]))
    xspacing = int(width * grid_spacing / ow)
    yspacing = int(height * grid_spacing / oh)
    return xspacing, yspacing


def create_damage_mask(section, grid_spacing):
    """
    Creates a binary damage mask from grid information in the given section.

    Args:
        section (dict): Dictionary with slice and grid data.
        grid_spacing (int): Space between grid marks.

    Returns:
        ndarray: Binary mask with damaged areas marked as 0.
    """
    width = section["width"]
    height = section["height"]
    anchoring = section["anchoring"]
    grid_values = section["grid"]
    gridx = section["gridx"]
    gridy = section["gridy"]

    xspacing, yspacing = update_spacing(anchoring, width, height, grid_spacing)
    x_coords = np.arange(gridx, width, xspacing)
    y_coords = np.arange(gridy, height, yspacing)

    num_markers = len(grid_values)
    markers = [
        (x_coords[i % len(x_coords)], y_coords[i // len(x_coords)])
        for i in range(num_markers)
    ]

    binary_image = np.ones((len(y_coords), len(x_coords)), dtype=int)

    for i, (x, y) in enumerate(markers):
        if grid_values[i] == 4:
            binary_image[y // yspacing, x // xspacing] = 0

    return binary_image


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
        seg_nr = int(number_sections([segmentation_path])[0])
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


def load_segmentation(segmentation_path: str):
    """
    Loads segmentation data, handling .dzip files if necessary.

    Args:
        segmentation_path (str): File path.

    Returns:
        ndarray: Image array of the segmentation.
    """
    if segmentation_path.endswith(".dzip"):
        return reconstruct_dzi(segmentation_path)
    else:
        return cv2.imread(segmentation_path)


def detect_pixel_id(segmentation: np.ndarray):
    """
    Infers pixel color from the first non-background region.

    Args:
        segmentation (ndarray): Segmentation array.

    Returns:
        ndarray: Identified pixel color (RGB).
    """
    segmentation_no_background = segmentation[~np.all(segmentation == 0, axis=2)]
    pixel_id = segmentation_no_background[0]
    print("detected pixel_id: ", pixel_id)
    return pixel_id


def get_region_areas(
    use_flat,
    atlas_labels,
    flat_file_atlas,
    seg_width,
    seg_height,
    slice_dict,
    atlas_volume,
    hemi_mask,
    triangulation,
    damage_mask,
):
    """
    Builds the atlas map for a slice and calculates the region areas.

    Args:
        use_flat (bool): If True, uses flat files.
        atlas_labels (DataFrame): DataFrame containing atlas labels.
        flat_file_atlas (str): Path to the flat atlas file.
        seg_width (int): Segmentation image width.
        seg_height (int): Segmentation image height.
        slice_dict (dict): Dictionary with slice metadata (anchoring, etc.).
        atlas_volume (ndarray): 3D atlas volume.
        hemi_mask (ndarray): Hemisphere mask.
        triangulation (ndarray): Triangulation data for non-linear transforms.
        damage_mask (ndarray): Binary damage mask.

    Returns:
        tuple: (DataFrame of region areas, atlas map array).
    """
    log_memory_usage(
        "before_load_image",
        message=f"Before load_image - seg: {seg_width}x{seg_height}",
    )
    if atlas_volume is not None:
        log_memory_usage("atlas_volume", atlas_volume, "Input atlas volume")

    atlas_map = load_image(
        flat_file_atlas,
        slice_dict["anchoring"],
        atlas_volume,
        triangulation,
        (seg_width, seg_height),
        atlas_labels,
    )
    log_memory_usage("atlas_map_loaded", atlas_map, "Atlas map after load_image")

    region_areas = flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    log_memory_usage(
        "region_areas", message=f"Region areas dataframe: {len(region_areas)} rows"
    )
    return region_areas, atlas_map


def segmentation_to_atlas_space(
    slice_dict,
    segmentation_path,
    atlas_labels,
    flat_file_atlas=None,
    pixel_id="auto",
    non_linear=True,
    points_list=np.ndarray([]),
    centroids_list=None,
    points_labels=None,
    centroids_labels=None,
    region_areas_list=None,
    per_point_undamaged_list=None,
    per_centroid_undamaged_list=None,
    points_hemi_labels=None,
    centroids_hemi_labels=None,
    index=None,
    object_cutoff=0,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    grid_spacing=None,
):
    """
    Transforms a single segmentation file into atlas space.

    Args:
        slice_dict (dict): Slice information from alignment JSON.
        segmentation_path (str): Path to the segmentation file.
        atlas_labels (DataFrame): Atlas labels.
        flat_file_atlas (str, optional): Path to flat atlas, if using flat files.
        pixel_id (str or list, optional): Pixel color or 'auto'.
        non_linear (bool, optional): Use non-linear transforms if True.
        points_list (list, optional): Storage for transformed point coordinates.
        centroids_list (list, optional): Storage for transformed centroid coordinates.
        points_labels (list, optional): Storage for assigned point labels.
        centroids_labels (list, optional): Storage for assigned centroid labels.
        region_areas_list (list, optional): Storage for region area data.
        per_point_undamaged_list (list, optional): Track undamaged points.
        per_centroid_undamaged_list (list, optional): Track undamaged centroids.
        points_hemi_labels (list, optional): Hemisphere labels for points.
        centroids_hemi_labels (list, optional): Hemisphere labels for centroids.
        index (int, optional): Index in the lists.
        object_cutoff (int, optional): Minimum object size.
        atlas_volume (ndarray, optional): 3D atlas volume.
        hemi_map (ndarray, optional): Hemisphere mask.
        use_flat (bool, optional): Indicates use of flat files.
        grid_spacing (int, optional): Spacing value for damage mask.

    Returns:        None"""
    log_memory_usage(
        "start", message=f"Starting segmentation_to_atlas_space for {segmentation_path}"
    )

    segmentation = load_segmentation(segmentation_path)
    log_memory_usage("segmentation", segmentation, "After loading segmentation")

    pixel_id = np.array(pixel_id, dtype=np.uint8)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    log_memory_usage(
        "dimensions",
        message=f"seg: {seg_height}x{seg_width}, reg: {reg_height}x{reg_width}",
    )

    triangulation = get_triangulation(slice_dict, reg_width, reg_height, non_linear)
    if "grid" in slice_dict:
        damage_mask = create_damage_mask(slice_dict, grid_spacing)
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
    region_areas, atlas_map = get_region_areas(
        use_flat,
        atlas_labels,
        flat_file_atlas,
        seg_width,  # Use segmentation width for region area scaling
        seg_height,  # Use segmentation height for region area scaling
        slice_dict,
        atlas_volume,
        hemi_mask,
        triangulation,
        damage_mask,
    )
    log_memory_usage("atlas_map", atlas_map, "After get_region_areas")

    # Check if the resize would create a huge array - if so, work at original resolution
    target_size = reg_height * reg_width * 4  # 4 bytes per uint32
    atlas_at_original_resolution = False
    if target_size > 500 * 1024 * 1024:  # If larger than 500MB
        atlas_at_original_resolution = True
        log_memory_usage(
            "large_resize_detected",
            message=f"Large resize detected: {reg_height}x{reg_width} = {target_size/1024/1024:.1f} MB, using original resolution",
        )
        scaled_atlas_map = atlas_map
        # IMPORTANT: Still scale coordinates to registration resolution, not atlas resolution
        # This ensures the coordinate transformation chain remains correct
        y_scale, x_scale = transform_to_registration(
            seg_width, seg_height, reg_width, reg_height
        )
        log_memory_usage(
            "atlas_original_res",
            message=f"Atlas kept at original resolution {atlas_map.shape}, but coordinates scaled to registration {reg_height}x{reg_width}",
        )
    else:
        scaled_atlas_map = resize(
            atlas_map, (reg_height, reg_width), order=0, preserve_range=True
        )
        log_memory_usage(
            "scaled_atlas_map", scaled_atlas_map, "After resizing atlas_map"
        )
        y_scale, x_scale = transform_to_registration(
            seg_width, seg_height, reg_width, reg_height
        )
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
    )

    log_memory_usage(
        "centroids", centroids, "After get_objects_and_assign_regions_optimized"
    )
    log_memory_usage("scaled_coordinates", scaled_x, "scaled_x coordinates")
    log_memory_usage("scaled_coordinates_y", scaled_y, "scaled_y coordinates")

    del segmentation
    log_memory_usage("after_del_segmentation", message="After deleting segmentation")

    # Robustly handle missing color matches
    if (
        scaled_y is None
        or scaled_x is None
        or scaled_centroidsX is None
        or scaled_centroidsY is None
    ):
        points_list[index] = np.array([])
        centroids_list[index] = np.array([])
        region_areas_list[index] = pd.DataFrame()
        centroids_labels[index] = np.array([])
        per_centroid_undamaged_list[index] = np.array([])
        points_labels[index] = np.array([])
        per_point_undamaged_list[index] = np.array([])
        points_hemi_labels[index] = np.array([])
        centroids_hemi_labels[index] = np.array([])

        # Clean up atlas_map early
        del atlas_map
        if damage_mask is not None:
            del damage_mask
        if hemi_mask is not None:
            del hemi_mask
        gc.collect()
        return

    # Assign point labels
    if scaled_y is not None and scaled_x is not None:
        if atlas_at_original_resolution:
            # Map from registration space to atlas space for point assignment
            atlas_height, atlas_width = scaled_atlas_map.shape
            atlas_y_scale = atlas_height / reg_height
            atlas_x_scale = atlas_width / reg_width
            atlas_point_y = scaled_y * atlas_y_scale
            atlas_point_x = scaled_x * atlas_x_scale

            # Bounds checking
            valid_mask = (
                (np.round(atlas_point_y).astype(int) >= 0)
                & (np.round(atlas_point_y).astype(int) < atlas_height)
                & (np.round(atlas_point_x).astype(int) >= 0)
                & (np.round(atlas_point_x).astype(int) < atlas_width)
            )

            if np.any(valid_mask):
                valid_y = np.round(atlas_point_y[valid_mask]).astype(int)
                valid_x = np.round(atlas_point_x[valid_mask]).astype(int)
                per_point_labels = np.zeros(len(scaled_y), dtype=int)
                per_point_labels[valid_mask] = scaled_atlas_map[valid_y, valid_x]
            else:
                per_point_labels = np.zeros(len(scaled_y), dtype=int)
        else:
            # Clamp coordinates to valid bounds to prevent index out of bounds errors
            y_indices = np.clip(
                np.round(scaled_y).astype(int), 0, scaled_atlas_map.shape[0] - 1
            )
            x_indices = np.clip(
                np.round(scaled_x).astype(int), 0, scaled_atlas_map.shape[1] - 1
            )
            per_point_labels = scaled_atlas_map[y_indices, x_indices]
    else:
        per_point_labels = np.array([])

    if damage_mask is not None:
        log_memory_usage(
            "damage_mask_before_resize", damage_mask, "Before damage mask resize"
        )
        # Use the actual scaled_atlas_map shape, not assuming it's huge
        target_shape = (scaled_atlas_map.shape[1], scaled_atlas_map.shape[0])

        damage_mask = resize(
            damage_mask.astype(np.uint8),
            target_shape,
            order=0,
            preserve_range=True,
        ).astype(bool)
        log_memory_usage(
            "damage_mask_after_resize", damage_mask, "After damage mask resize"
        )
        per_point_undamaged = damage_mask[
            np.round(
                scaled_y
                * y_scale
                / (seg_height / atlas_height if "atlas_height" in locals() else 1)
            )
            .astype(int)
            .clip(0, damage_mask.shape[0] - 1),
            np.round(
                scaled_x
                * x_scale
                / (seg_width / atlas_width if "atlas_width" in locals() else 1)
            )
            .astype(int)
            .clip(0, damage_mask.shape[1] - 1),
        ]
        per_centroid_undamaged = damage_mask[
            np.round(
                scaled_centroidsY
                * y_scale
                / (seg_height / atlas_height if "atlas_height" in locals() else 1)
            )
            .astype(int)
            .clip(0, damage_mask.shape[0] - 1),
            np.round(
                scaled_centroidsX
                * x_scale
                / (seg_width / atlas_width if "atlas_width" in locals() else 1)
            )
            .astype(int)
            .clip(0, damage_mask.shape[1] - 1),
        ]
    else:
        per_point_undamaged = np.ones(scaled_x.shape, dtype=bool)
        per_centroid_undamaged = np.ones(scaled_centroidsX.shape, dtype=bool)
    if hemi_mask is not None:
        log_memory_usage(
            "hemi_mask_before_resize", hemi_mask, "Before hemi mask resize"
        )
        hemi_mask = resize(
            hemi_mask.astype(np.uint8),
            (scaled_atlas_map.shape[1], scaled_atlas_map.shape[0]),
            order=0,
            preserve_range=True,
        )
        log_memory_usage("hemi_mask_after_resize", hemi_mask, "After hemi mask resize")

        per_point_hemi = hemi_mask[
            np.round(scaled_y).astype(int),
            np.round(scaled_x).astype(int),
        ]
        per_centroid_hemi = hemi_mask[
            np.round(scaled_centroidsY).astype(int),
            np.round(scaled_centroidsX).astype(int),
        ]
        per_point_hemi = per_point_hemi[per_point_undamaged]
        per_centroid_hemi = per_centroid_hemi[per_centroid_undamaged]
    else:
        per_point_hemi = [None] * len(scaled_x)
        per_centroid_hemi = [None] * len(scaled_centroidsX)

    per_point_labels = per_point_labels[per_point_undamaged]
    per_centroid_labels = per_centroid_labels[per_centroid_undamaged]

    new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(
        non_linear,
        slice_dict,
        scaled_x[per_point_undamaged],
        scaled_y[per_point_undamaged],
        scaled_centroidsX[per_centroid_undamaged],
        scaled_centroidsY[per_centroid_undamaged],
        triangulation,
    )
    points, centroids = transform_points_to_atlas_space(
        slice_dict,
        new_x,
        new_y,
        centroids_new_x,
        centroids_new_y,
        reg_height,
        reg_width,
    )

    log_memory_usage("final_points", points, "Final transformed points")
    log_memory_usage("final_centroids", centroids, "Final transformed centroids")

    del scaled_atlas_map
    if hemi_mask is not None:
        del hemi_mask
    if damage_mask is not None:
        del damage_mask
    del atlas_map
    log_memory_usage("after_cleanup", message="After deleting large arrays")
    gc.collect()
    log_memory_usage("after_gc", message="After garbage collection")

    points_list[index] = np.array(points if points is not None else [])
    centroids_list[index] = np.array(centroids if centroids is not None else [])
    region_areas_list[index] = region_areas
    centroids_labels[index] = np.array(
        per_centroid_labels if centroids is not None else []
    )
    per_centroid_undamaged_list[index] = np.array(
        per_centroid_undamaged if centroids is not None else []
    )
    points_labels[index] = np.array(per_point_labels if points is not None else [])
    per_point_undamaged_list[index] = np.array(
        per_point_undamaged if points is not None else []
    )
    points_hemi_labels[index] = np.array(per_point_hemi if points is not None else [])
    centroids_hemi_labels[index] = np.array(
        per_centroid_hemi if points is not None else []
    )


def get_triangulation(slice_dict, reg_width, reg_height, non_linear):
    """
    Generates triangulation data if non-linear markers exist.

    Args:
        slice_dict (dict): Slice metadata from alignment JSON.
        reg_width (int): Registration width.
        reg_height (int): Registration height.
        non_linear (bool): Whether to use non-linear transform.

    Returns:
        list or None: Triangulation info or None if not applicable.
    """
    if non_linear and "markers" in slice_dict:
        return triangulate(reg_width, reg_height, slice_dict["markers"])
    return None


def get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale, tolerance=10):
    """
    Retrieves pixel coordinates for a specified color and scales them, with tolerance.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel color to match.
        y_scale (float): Vertical scaling factor.
        x_scale (float): Horizontal scaling factor.

    Returns:
        tuple: (scaled_y, scaled_x)
    """
    mask = np.all(
        np.abs(segmentation.astype(int) - np.array(pixel_id, dtype=int)) <= tolerance,
        axis=2,
    )
    id_y, id_x = np.where(mask)
    if len(id_y) == 0:
        return None, None
    scaled_y, scaled_x = scale_positions(id_y, id_x, y_scale, x_scale)
    return scaled_y, scaled_x


def get_objects_and_assign_regions_optimized(
    segmentation,
    pixel_id,
    atlas_map,
    y_scale,
    x_scale,
    object_cutoff=0,
    tolerance=10,
    atlas_at_original_resolution=False,
    reg_height=None,
    reg_width=None,
):
    """
    OPTIMIZED: Single-pass object detection, pixel extraction, and region assignment.

    Replaces the inefficient dual-pass approach with a single comprehensive function
    that does all processing in one go.

    Args:
        segmentation (ndarray): RGB segmentation image
        pixel_id (array): Target pixel color
        atlas_map (ndarray): Atlas region map
        y_scale (float): Vertical scaling factor
        x_scale (float): Horizontal scaling factor
        object_cutoff (int): Minimum object size
        tolerance (int): Color matching tolerance
        atlas_at_original_resolution (bool): Whether atlas is at original resolution
        reg_height (int): Registration height (for coordinate mapping)
        reg_width (int): Registration width (for coordinate mapping)

    Returns:
        tuple: (centroids, scaled_centroidsX, scaled_centroidsY, scaled_y, scaled_x, per_centroid_labels)
    """
    # Create binary mask for target pixels (single operation)
    binary_seg = np.all(
        np.abs(segmentation.astype(int) - np.array(pixel_id, dtype=int)) <= tolerance,
        axis=2,
    )

    # Get all matching pixels for point extraction
    pixel_y, pixel_x = np.where(binary_seg)
    if len(pixel_y) == 0:
        return None, None, None, None, None, None

    # Scale pixel coordinates
    scaled_y, scaled_x = scale_positions(pixel_y, pixel_x, y_scale, x_scale)

    # Single labeling operation for object detection
    labels = measure.label(binary_seg)
    objects_info = measure.regionprops(
        labels
    )  # objects_info.area iterable would be used for the weights
    objects_info = [obj for obj in objects_info if obj.area > object_cutoff]

    if len(objects_info) == 0:
        return None, None, None, scaled_y, scaled_x, None

    # Extract centroids and assign regions
    centroids = []
    per_centroid_labels = []
    # centroid_weights = # For visualization in meshview, like the scale slider

    for obj in objects_info:
        # Get centroid
        centroid = obj.centroid  # (row, col)
        centroids.append(centroid)

        # Get all pixel coordinates for region assignment
        obj_coords = obj.coords  # Shape: (n_pixels, 2) in (row, col) format
        obj_y = obj_coords[:, 0]
        obj_x = obj_coords[:, 1]
        # centroid_weights.append(obj.area)  # Store area for visualization in the meshview
        # this will go into "weights" of the output

        # Scale object coordinates to registration space
        scaled_obj_y, scaled_obj_x = scale_positions(obj_y, obj_x, y_scale, x_scale)

        # If atlas is at original resolution, we need to map coordinates back to atlas space
        if atlas_at_original_resolution:
            # Map from registration space to atlas space
            atlas_height, atlas_width = atlas_map.shape
            atlas_y_scale = atlas_height / reg_height
            atlas_x_scale = atlas_width / reg_width
            atlas_obj_y = scaled_obj_y * atlas_y_scale
            atlas_obj_x = scaled_obj_x * atlas_x_scale

            # Use atlas coordinates for region assignment
            assignment_y = atlas_obj_y
            assignment_x = atlas_obj_x
            atlas_bounds_height = atlas_height
            atlas_bounds_width = atlas_width
        else:
            # Use registration coordinates directly
            assignment_y = scaled_obj_y
            assignment_x = scaled_obj_x
            atlas_bounds_height = atlas_map.shape[0]
            atlas_bounds_width = atlas_map.shape[1]

        # Bounds checking
        valid_mask = (
            (np.round(assignment_y).astype(int) >= 0)
            & (np.round(assignment_y).astype(int) < atlas_bounds_height)
            & (np.round(assignment_x).astype(int) >= 0)
            & (np.round(assignment_x).astype(int) < atlas_bounds_width)
        )

        if not np.any(valid_mask):
            per_centroid_labels.append(0)  # Background
            continue

        # Get region labels for valid pixels and find majority
        valid_y = np.round(assignment_y[valid_mask]).astype(int)
        valid_x = np.round(assignment_x[valid_mask]).astype(int)
        pixel_labels = atlas_map[valid_y, valid_x]

        # Majority voting
        unique_labels, counts = np.unique(pixel_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        per_centroid_labels.append(
            majority_label
        )  # The majority label is placed for the centroid

    # Convert to arrays and scale centroids
    if centroids:
        centroids = np.array(centroids)
        centroidsX = centroids[:, 1]  # Column coordinates
        centroidsY = centroids[:, 0]  # Row coordinates
        scaled_centroidsY, scaled_centroidsX = scale_positions(
            centroidsY, centroidsX, y_scale, x_scale
        )
        per_centroid_labels = np.array(per_centroid_labels)
    else:
        centroids = None
        scaled_centroidsX = None
        scaled_centroidsY = None
        per_centroid_labels = np.array([])

    return (
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        scaled_y,
        scaled_x,
        per_centroid_labels,
    )
