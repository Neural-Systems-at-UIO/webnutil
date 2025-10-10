import numpy as np
import pandas as pd
import cv2
import gc
import psutil
import logging
from dataclasses import dataclass
from typing import Optional
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


@dataclass
class CoordinateSpaceTracker:
    """
    Tracks dimensions and scale factors across different coordinate spaces.

    Coordinate spaces:
    - Segmentation space: Original segmentation image dimensions
    - Registration space: Target alignment dimensions from QuickNII/VisuAlign JSON
    - Atlas space: Atlas map dimensions (may differ if kept at original resolution)

    The registration space is used for warping/deformation transformations.
    """

    # Dimensions
    seg_height: int
    seg_width: int
    reg_height: int
    reg_width: int
    atlas_height: Optional[int] = None
    atlas_width: Optional[int] = None

    # Flags
    atlas_at_original_resolution: bool = False

    def __post_init__(self):
        """Calculate scale factors after initialization."""
        # Segmentation -> Registration scales (used for warping)
        self.seg_to_reg_y_scale = (self.reg_height - 1) / (self.seg_height - 1)
        self.seg_to_reg_x_scale = (self.reg_width - 1) / (self.seg_width - 1)

        # Registration -> Atlas scales (only if atlas at original resolution)
        self.reg_to_atlas_y_scale = None
        self.reg_to_atlas_x_scale = None

        # Combined Segmentation -> Atlas scales
        self.seg_to_atlas_y_scale = None
        self.seg_to_atlas_x_scale = None

    def set_atlas_dimensions(
        self, atlas_height: int, atlas_width: int, at_original_resolution: bool = False
    ):
        """
        Set atlas dimensions and recalculate dependent scale factors.

        Args:
            atlas_height: Height of the atlas map
            atlas_width: Width of the atlas map
            at_original_resolution: True if atlas was kept at original resolution
        """
        self.atlas_height = atlas_height
        self.atlas_width = atlas_width
        self.atlas_at_original_resolution = at_original_resolution

        if at_original_resolution:
            # Calculate reg -> atlas scales
            self.reg_to_atlas_y_scale = atlas_height / self.reg_height
            self.reg_to_atlas_x_scale = atlas_width / self.reg_width

            # Calculate combined seg -> atlas scales
            self.seg_to_atlas_y_scale = (
                self.seg_to_reg_y_scale * self.reg_to_atlas_y_scale
            )
            self.seg_to_atlas_x_scale = (
                self.seg_to_reg_x_scale * self.reg_to_atlas_x_scale
            )
        else:
            # Atlas is at registration resolution
            self.reg_to_atlas_y_scale = 1.0
            self.reg_to_atlas_x_scale = 1.0
            self.seg_to_atlas_y_scale = self.seg_to_reg_y_scale
            self.seg_to_atlas_x_scale = self.seg_to_reg_x_scale

    def transform_seg_to_reg(self, y, x):
        """Transform coordinates from segmentation to registration space."""
        return y * self.seg_to_reg_y_scale, x * self.seg_to_reg_x_scale

    def transform_seg_to_atlas(self, y, x):
        """Transform coordinates from segmentation to atlas space."""
        if self.seg_to_atlas_y_scale is None:
            raise ValueError(
                "Atlas dimensions not set. Call set_atlas_dimensions first."
            )
        return y * self.seg_to_atlas_y_scale, x * self.seg_to_atlas_x_scale

    def transform_reg_to_atlas(self, y, x):
        """Transform coordinates from registration to atlas space."""
        if self.reg_to_atlas_y_scale is None:
            raise ValueError(
                "Atlas dimensions not set. Call set_atlas_dimensions first."
            )
        return y * self.reg_to_atlas_y_scale, x * self.reg_to_atlas_x_scale

    def get_target_space_dims(self):
        """Get the dimensions of the target space (atlas if available, otherwise registration)."""
        if self.atlas_height is not None:
            return self.atlas_height, self.atlas_width
        return self.reg_height, self.reg_width

    def __repr__(self):
        return (
            f"CoordinateSpaceTracker(\n"
            f"  Segmentation: {self.seg_height}x{self.seg_width}\n"
            f"  Registration: {self.reg_height}x{self.reg_width}\n"
            f"  Atlas: {self.atlas_height}x{self.atlas_width} "
            f"(original_res={self.atlas_at_original_resolution})\n"
            f"  Scales: seg->reg=({self.seg_to_reg_y_scale:.3f}, {self.seg_to_reg_x_scale:.3f}), "
            f"seg->atlas=({self.seg_to_atlas_y_scale}, {self.seg_to_atlas_x_scale})\n"
            f")"
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
    reg_width = slice_dict["width"]
    reg_height = slice_dict["height"]
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

    # Initialize coordinate space tracker
    coord_tracker = CoordinateSpaceTracker(
        seg_height=seg_height,
        seg_width=seg_width,
        reg_height=reg_height,
        reg_width=reg_width,
    )

    log_memory_usage(
        "dimensions",
        message=f"seg: {seg_height}x{seg_width}, reg: {reg_height}x{reg_width}",
    )
    print(f"Coordinate spaces initialized:\n{coord_tracker}")

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

    scaled_atlas_map = atlas_map

    # Set atlas dimensions in the coordinate tracker
    coord_tracker.set_atlas_dimensions(
        atlas_height=atlas_map.shape[0],
        atlas_width=atlas_map.shape[1],
        at_original_resolution=False,  # Will be updated if we detect large size
    )

    # For backward compatibility, keep these variables
    atlas_at_original_resolution = False
    y_scale = coord_tracker.seg_to_reg_y_scale
    x_scale = coord_tracker.seg_to_reg_x_scale
    seg_to_reg_y_scale = coord_tracker.seg_to_reg_y_scale
    seg_to_reg_x_scale = coord_tracker.seg_to_reg_x_scale
    atlas_height = coord_tracker.atlas_height
    atlas_width = coord_tracker.atlas_width

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

    # handle case where no pixels or objects are detected
    # region_areas should still be preserved as they're independent of object detection
    if scaled_y is None or scaled_x is None:
        # no pixels detected - set all outputs to empty but preserve region_areas
        # add pixel_count and object_count columns with 0 values for downstream compatibility
        if not region_areas.empty and "pixel_count" not in region_areas.columns:
            region_areas["pixel_count"] = 0
        if not region_areas.empty and "object_count" not in region_areas.columns:
            region_areas["object_count"] = 0

        points_list[index] = np.array([])
        centroids_list[index] = np.array([])
        region_areas_list[index] = region_areas  # preserve calculated region areas
        centroids_labels[index] = np.array([])
        per_centroid_undamaged_list[index] = np.array([])
        points_labels[index] = np.array([])
        per_point_undamaged_list[index] = np.array([])
        points_hemi_labels[index] = np.array([])
        centroids_hemi_labels[index] = np.array([])

        # clean up atlas_map early
        del atlas_map
        if damage_mask is not None:
            del damage_mask
        if hemi_mask is not None:
            del hemi_mask
        gc.collect()
        return

    # handle case where pixels exist but no objects (centroids) detected
    # this happens when all objects are filtered out by object_cutoff
    if scaled_centroidsX is None or scaled_centroidsY is None:
        # pixels detected but no objects - process pixels only, preserve region_areas
        centroids_list[index] = np.array([])
        centroids_labels[index] = np.array([])
        per_centroid_undamaged_list[index] = np.array([])
        centroids_hemi_labels[index] = np.array([])
        # continue processing points below, don't return early

    # Assign point labels
    if scaled_y is not None and scaled_x is not None:
        if coord_tracker.atlas_at_original_resolution:
            # Transform: seg -> reg -> atlas using the tracker
            atlas_point_y, atlas_point_x = coord_tracker.transform_seg_to_atlas(
                scaled_y, scaled_x
            )

            # Bounds checking
            valid_mask = (
                (np.round(atlas_point_y).astype(int) >= 0)
                & (np.round(atlas_point_y).astype(int) < coord_tracker.atlas_height)
                & (np.round(atlas_point_x).astype(int) >= 0)
                & (np.round(atlas_point_x).astype(int) < coord_tracker.atlas_width)
            )

            if np.any(valid_mask):
                valid_y = np.round(atlas_point_y[valid_mask]).astype(int)
                valid_x = np.round(atlas_point_x[valid_mask]).astype(int)
                per_point_labels = np.zeros(len(scaled_y), dtype=int)
                per_point_labels[valid_mask] = scaled_atlas_map[valid_y, valid_x]
            else:
                per_point_labels = np.zeros(len(scaled_y), dtype=int)
        else:
            # Atlas is at registration resolution, transform seg -> reg
            reg_y, reg_x = coord_tracker.transform_seg_to_reg(scaled_y, scaled_x)
            rounded_y = np.round(reg_y).astype(int)
            rounded_x = np.round(reg_x).astype(int)

            y_out_of_bounds = (rounded_y < 0) | (rounded_y >= scaled_atlas_map.shape[0])
            x_out_of_bounds = (rounded_x < 0) | (rounded_x >= scaled_atlas_map.shape[1])
            if np.any(y_out_of_bounds) or np.any(x_out_of_bounds):
                print(
                    f"Point coordinates out of bounds before clamping: {np.sum(y_out_of_bounds)} y out, {np.sum(x_out_of_bounds)} x out"
                )
                print(
                    f"Y: min {rounded_y.min()}, max {rounded_y.max()} (valid 0-{scaled_atlas_map.shape[0]-1})"
                )
                print(
                    f"X: min {rounded_x.min()}, max {rounded_x.max()} (valid 0-{scaled_atlas_map.shape[1]-1})"
                )
            y_indices = np.clip(rounded_y, 0, scaled_atlas_map.shape[0] - 1)
            x_indices = np.clip(rounded_x, 0, scaled_atlas_map.shape[1] - 1)
            per_point_labels = scaled_atlas_map[y_indices, x_indices]
    else:
        per_point_labels = np.array([])

    if damage_mask is not None:
        log_memory_usage(
            "damage_mask_before_resize", damage_mask, "Before damage mask resize"
        )
        # Resize damage mask to match the scaled_atlas_map dimensions
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

        # Use the coordinate tracker to transform coordinates to damage_mask space
        if coord_tracker.atlas_at_original_resolution:
            # damage_mask is at atlas resolution, transform seg -> atlas
            damage_mask_y, damage_mask_x = coord_tracker.transform_seg_to_atlas(
                scaled_y, scaled_x
            )
        else:
            # damage_mask is at registration resolution, transform seg -> reg
            damage_mask_y, damage_mask_x = coord_tracker.transform_seg_to_reg(
                scaled_y, scaled_x
            )

        per_point_undamaged = damage_mask[
            np.round(damage_mask_y).astype(int).clip(0, damage_mask.shape[0] - 1),
            np.round(damage_mask_x).astype(int).clip(0, damage_mask.shape[1] - 1),
        ]

        if scaled_centroidsX is not None and scaled_centroidsY is not None:
            # Same transformation for centroids
            if coord_tracker.atlas_at_original_resolution:
                damage_mask_centroid_y, damage_mask_centroid_x = (
                    coord_tracker.transform_seg_to_atlas(
                        scaled_centroidsY, scaled_centroidsX
                    )
                )
            else:
                damage_mask_centroid_y, damage_mask_centroid_x = (
                    coord_tracker.transform_seg_to_reg(
                        scaled_centroidsY, scaled_centroidsX
                    )
                )

            per_centroid_undamaged = damage_mask[
                np.round(damage_mask_centroid_y)
                .astype(int)
                .clip(0, damage_mask.shape[0] - 1),
                np.round(damage_mask_centroid_x)
                .astype(int)
                .clip(0, damage_mask.shape[1] - 1),
            ]
        else:
            per_centroid_undamaged = np.array([], dtype=bool)
    else:
        per_point_undamaged = np.ones(scaled_x.shape, dtype=bool)
        per_centroid_undamaged = (
            np.ones(scaled_centroidsX.shape, dtype=bool)
            if scaled_centroidsX is not None
            else np.array([], dtype=bool)
        )
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
        if scaled_centroidsX is not None and scaled_centroidsY is not None:
            per_centroid_hemi = hemi_mask[
                np.round(scaled_centroidsY).astype(int),
                np.round(scaled_centroidsX).astype(int),
            ]
            per_centroid_hemi = per_centroid_hemi[per_centroid_undamaged]
        else:
            per_centroid_hemi = np.array([])
        per_point_hemi = per_point_hemi[per_point_undamaged]
    else:
        per_point_hemi = [None] * len(scaled_x)
        per_centroid_hemi = (
            [None] * len(scaled_centroidsX)
            if scaled_centroidsX is not None
            else np.array([])
        )

    per_point_labels = per_point_labels[per_point_undamaged]
    if per_centroid_labels is not None and len(per_centroid_labels) > 0:
        per_centroid_labels = per_centroid_labels[per_centroid_undamaged]
    else:
        per_centroid_labels = np.array([])

    # Scale coordinates to registration space for transformation
    # (they're currently in segmentation space since y_scale=x_scale=1.0)
    scaled_x_for_transform = scaled_x * seg_to_reg_x_scale
    scaled_y_for_transform = scaled_y * seg_to_reg_y_scale
    scaled_centroidsX_for_transform = (
        scaled_centroidsX * seg_to_reg_x_scale
        if scaled_centroidsX is not None
        else None
    )
    scaled_centroidsY_for_transform = (
        scaled_centroidsY * seg_to_reg_y_scale
        if scaled_centroidsY is not None
        else None
    )

    # transform coordinates - handle missing centroids gracefully
    if (
        scaled_centroidsX_for_transform is not None
        and scaled_centroidsY_for_transform is not None
    ):
        new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(
            non_linear,
            slice_dict,
            scaled_x_for_transform[per_point_undamaged],
            scaled_y_for_transform[per_point_undamaged],
            scaled_centroidsX_for_transform[per_centroid_undamaged],
            scaled_centroidsY_for_transform[per_centroid_undamaged],
            triangulation,
        )
    else:
        # only transform points, no centroids
        new_x, new_y, _, _ = get_transformed_coordinates(
            non_linear,
            slice_dict,
            scaled_x_for_transform[per_point_undamaged],
            scaled_y_for_transform[per_point_undamaged],
            np.array([]),
            np.array([]),
            triangulation,
        )
        centroids_new_x = np.array([])
        centroids_new_y = np.array([])
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
    seg_height=None,
    seg_width=None,
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
        seg_height (int): Segmentation height
        seg_width (int): Segmentation width

    Returns:
        tuple: (centroids, scaled_centroidsX, scaled_centroidsY, scaled_y, scaled_x, per_centroid_labels)
    """
    # Create binary mask for target pixels (single operation)
    print(f"Detecting objects with pixel_id: {pixel_id}, tolerance: {tolerance}")
    binary_seg = np.all(
        np.abs(segmentation.astype(int) - np.array(pixel_id, dtype=int)) <= tolerance,
        axis=2,
    )

    # Get all matching pixels for point extraction
    pixel_y, pixel_x = np.where(binary_seg)
    print(f"Detected {len(pixel_y)} pixels matching the target color")
    if len(pixel_y) == 0:
        return None, None, None, None, None, None

    # Scale pixel coordinates
    scaled_y, scaled_x = scale_positions(pixel_y, pixel_x, y_scale, x_scale)

    # Single labeling operation for object detection
    labels = measure.label(
        binary_seg
    )  # measure.label assigns a unique integer label to each connected component (object) in the binary image
    objects_info = measure.regionprops(
        labels
    )  # measure.regionprops returns a list of RegionProperties objects, each containing properties like area, centroid, bounding box, etc. for each labeled region
    total_labeled = len(objects_info)
    print(f"Total labeled regions (objects): {total_labeled}")
    original_objects = objects_info[:]
    objects_info = [obj for obj in objects_info if obj.area > object_cutoff]
    remaining = len(objects_info)
    print(f"Regions after area cutoff ({object_cutoff}): {remaining}")
    filtered_out = total_labeled - remaining
    if filtered_out > 0:
        print(
            f"Filtered out {filtered_out} objects due to area cutoff (area <= {object_cutoff})"
        )

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
            print(
                f"Object {len(per_centroid_labels)} has no valid pixels in atlas map (assigned background label 0)"
            )
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
        print(
            f"Object {len(per_centroid_labels)}: majority label {majority_label} from {len(valid_y)} valid pixels"
        )

    # Convert to arrays and scale centroids
    if centroids:
        centroids = np.array(centroids)
        centroidsX = centroids[:, 1]  # Column coordinates
        centroidsY = centroids[:, 0]  # Row coordinates
        scaled_centroidsY, scaled_centroidsX = scale_positions(
            centroidsY, centroidsX, y_scale, x_scale
        )
        per_centroid_labels = np.array(per_centroid_labels)
        print(
            f"Successfully processed {len(per_centroid_labels)} centroids with region assignments"
        )
        print(f"Centroid labels: {per_centroid_labels}")
    else:
        centroids = None
        scaled_centroidsX = None
        scaled_centroidsY = None
        per_centroid_labels = np.array([])
        print("No centroids found after processing")

    return (
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        scaled_y,
        scaled_x,
        per_centroid_labels,
    )
