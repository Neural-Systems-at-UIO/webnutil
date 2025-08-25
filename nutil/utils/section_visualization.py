"""
Section visualization utilities for creating colored atlas slice images.
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ..core.generate_target_slice import generate_target_slice
from ..core.transformations import image_to_atlas_space
from .read_and_write import load_segmentation


def create_colored_atlas_slice(
    slice_dict: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_path: str,
    segmentation_path: Optional[str] = None,
    objects_data: Optional[List[Dict]] = None,
    scale_factor: float = 0.5,
) -> None:
    """
    Create a colored atlas slice image showing regions with their atlas colors
    and optionally overlay detected objects with region IDs.

    Args:
        slice_dict: Dictionary containing slice information including anchoring vector
        atlas_volume: 3D atlas volume
        atlas_labels: DataFrame containing atlas region information with colors
        output_path: Path to save the output image
        segmentation_path: Optional path to segmentation image for overlay
        objects_data: Optional list of object dictionaries with coordinates and region IDs
        scale_factor: Factor to scale the output image size
    """
    atlas_slice = generate_target_slice(slice_dict["anchoring"], atlas_volume)

    color_map = create_atlas_color_map(atlas_labels)

    colored_slice = create_colored_image_from_slice(atlas_slice, color_map)

    target_width = colored_slice.shape[1]
    target_height = colored_slice.shape[0]

    # If segmentation exists, use its resolution as base (so outputs match segmentation size)
    segmentation_img = None
    if segmentation_path and os.path.exists(segmentation_path):
        try:
            segmentation_img = load_segmentation(segmentation_path)
            seg_height, seg_width = segmentation_img.shape[:2]
            target_width, target_height = seg_width, seg_height
        except Exception as e:
            print(f"Warning: Could not load segmentation for sizing: {e}")

    # If no segmentation, fall back to registration dimensions from slice_dict
    # There are alignments with eg 20 images registered, despite only having 10 segmentations
    if segmentation_img is None:
        try:
            reg_w = int(slice_dict.get("width", target_width))
            reg_h = int(slice_dict.get("height", target_height))
            if reg_w > 0 and reg_h > 0:
                target_width, target_height = reg_w, reg_h
        except Exception:
            pass

    # First scale to base target size (segmentation or registration)
    if (colored_slice.shape[1], colored_slice.shape[0]) != (
        target_width,
        target_height,
    ):
        colored_slice = cv2.resize(
            colored_slice,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    # Then apply extra scale factor if requested (uniform scaling)
    if scale_factor != 1.0:
        new_width = max(1, int(target_width * scale_factor))
        new_height = max(1, int(target_height * scale_factor))
        colored_slice = cv2.resize(
            colored_slice, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

    pil_image = Image.fromarray(colored_slice)

    # Overlay segmentation if provided (resize segmentation to current image size)
    if segmentation_path and os.path.exists(segmentation_path):
        overlay_segmentation(pil_image, segmentation_path)

    # Overlay object locations and region IDs if provided
    if objects_data:
        overlay_objects_with_region_ids(
            pil_image, objects_data, slice_dict, font_size=16
        )

    # Save the image
    pil_image.save(output_path)


def create_atlas_color_map(
    atlas_labels: pd.DataFrame,
) -> Dict[int, Tuple[int, int, int]]:
    """
    Create a color mapping from atlas labels DataFrame.

    Args:
        atlas_labels: DataFrame containing 'idx', 'r', 'g', 'b' columns

    Returns:
        Dictionary mapping region IDs to RGB colors
    """
    color_map = {0: (0, 0, 0)}  # Background

    for _, row in atlas_labels.iterrows():
        if "idx" in row and "r" in row and "g" in row and "b" in row:
            region_id = int(row["idx"])
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            color_map[region_id] = (r, g, b)

    return color_map


def create_colored_image_from_slice(
    atlas_slice: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Create a colored RGB image from an atlas slice using the color mapping.

    Args:
        atlas_slice: 2D array with region IDs
        color_map: Dictionary mapping region IDs to RGB colors

    Returns:
        RGB image array
    """
    height, width = atlas_slice.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for region_id, color in color_map.items():
        mask = atlas_slice == region_id
        colored_image[mask] = color

    # For unmapped regions, use a default gray color
    unmapped_mask = np.isin(atlas_slice, list(color_map.keys()), invert=True)
    colored_image[unmapped_mask] = (128, 128, 128)

    return colored_image


def overlay_segmentation(
    pil_image: Image.Image,
    segmentation_path: str,
    alpha: float = 0.3,
) -> None:
    """
    Overlay segmentation contours on the atlas slice image.

    Args:
        pil_image: PIL Image to overlay on
        segmentation_path: Path to segmentation image
        slice_dict: Slice dictionary with transformation info
        scale_factor: Scale factor applied to the image
        alpha: Transparency for the overlay
    """
    try:
        # Load segmentation
        segmentation = load_segmentation(segmentation_path)

        # Resize segmentation to the current PIL image size for overlay
        img_width, img_height = pil_image.size
        segmentation = cv2.resize(
            segmentation, (img_width, img_height), interpolation=cv2.INTER_NEAREST
        )

        # Convert to grayscale if needed
        if len(segmentation.shape) == 3:
            segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(
            segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create overlay image
        overlay = Image.fromarray(np.zeros_like(np.array(pil_image)))
        draw = ImageDraw.Draw(overlay)

        # Draw contours
        for contour in contours:
            points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            if len(points) > 2:
                draw.polygon(points, outline=(255, 255, 0), width=2)

        # Blend with original image
        pil_image.paste(Image.blend(pil_image, overlay, alpha))

    except Exception as e:
        print(f"Warning: Could not overlay segmentation from {segmentation_path}: {e}")


def overlay_objects_with_region_ids(
    pil_image: Image.Image,
    objects_data: List[Dict],
    slice_dict: Dict,
    font_size: int = 8,
) -> None:
    """
    Overlay object locations with their region IDs on the image.

    Args:
        pil_image: PIL Image to overlay on
        objects_data: List of dictionaries with object information
        slice_dict: Slice dictionary with transformation info
        scale_factor: Scale factor applied to the image
        font_size: Font size for region ID labels
    """
    draw = ImageDraw.Draw(pil_image)

    try:
        try:
            # Scale font size with image vs. registration size ratio
            img_w, img_h = pil_image.size
            reg_w = max(1, int(slice_dict.get("width", img_w)))
            reg_h = max(1, int(slice_dict.get("height", img_h)))
            scale_x = img_w / reg_w
            scale_y = img_h / reg_h
            scale = max(1.0, float(min(scale_x, scale_y)))
            font = ImageFont.truetype("arial.ttf", max(6, int(font_size * scale)))
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
    except:
        font = None

    objects_rendered = 0
    objects_failed_transform = 0
    objects_out_of_bounds = 0
    objects_background_skipped = 0

    for obj_data in objects_data:
        if (
            "triplets" in obj_data
            and "idx" in obj_data
            and len(obj_data["triplets"]) >= 3
        ):
            triplets = obj_data["triplets"]
            region_id = obj_data["idx"]

            # Skip background/clear label (but count them)
            # if region_id == 0:
            #    objects_background_skipped += 1
            #    continue # This skip will wait for a while, and we now intend to keep it

            # Convert atlas coordinates back to image coordinates
            # Process triplets in groups of 3 (x, y, z coordinates)
            for i in range(0, len(triplets), 3):
                if i + 2 < len(triplets):
                    atlas_coords = np.array(
                        [triplets[i], triplets[i + 1], triplets[i + 2]]
                    )

                    # Transform from atlas space back to image space
                    image_coords = transform_atlas_to_image_coords_improved(
                        atlas_coords, slice_dict, scale_factor=1.0
                    )

                    if image_coords is not None:
                        # Scale coordinates to current image size
                        img_width, img_height = pil_image.size
                        reg_width = max(1, int(slice_dict.get("width", img_width)))
                        reg_height = max(1, int(slice_dict.get("height", img_height)))
                        scale_x = img_width / reg_width
                        scale_y = img_height / reg_height
                        x = int(image_coords[0] * scale_x)
                        y = int(image_coords[1] * scale_y)

                        # Ensure coordinates are within image bounds (inclusive of edges)
                        img_width, img_height = pil_image.size
                        if 0 <= x <= img_width - 1 and 0 <= y <= img_height - 1:
                            # Draw a red dot at the object location
                            # Scale marker size with image scale
                            base_radius = 6
                            scale = max(scale_x, scale_y)
                            circle_radius = max(
                                3, int(base_radius * scale)
                            )  # At min renders at 3
                            draw.ellipse(
                                [
                                    x - circle_radius,
                                    y - circle_radius,
                                    x + circle_radius,
                                    y + circle_radius,
                                ],
                                fill=(255, 0, 0),  # Red dot
                                outline=(255, 255, 255),  # White border
                                width=1,
                            )

                            # Draw region ID
                            text = str(region_id)
                            if font:
                                # Get text size
                                bbox = draw.textbbox((0, 0), text, font=font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]
                            else:
                                text_width, text_height = len(text) * 4, 8

                            # Position text near the circle
                            text_x = x + circle_radius + 1
                            text_y = y - text_height // 2

                            # Ensure text is within bounds
                            if text_x + text_width > img_width:
                                text_x = x - circle_radius - text_width - 1
                            if text_y < 0:
                                text_y = 0
                            elif text_y + text_height > img_height:
                                text_y = img_height - text_height

                            # Draw text background (small)
                            # This is for visibility
                            # draw.rectangle(
                            #    [
                            #        text_x - 1,
                            #        text_y - 1,
                            #        text_x + text_width + (1 * scale),
                            #        text_y + text_height + (1 * scale),
                            #    ],
                            #    fill=(255, 255, 255, 220),
                            #    outline=(0, 0, 0),
                            #    width=1,
                            # )

                            # Draw text
                            draw.text(
                                (text_x, text_y), text, fill=(255, 255, 255), font=font
                            )
                            objects_rendered += 1

                            # Optional: Remove or increase limit for debugging
                            # if objects_rendered >= 500:  # Increased limit
                            #    break
                        else:
                            objects_out_of_bounds += 1
                    else:
                        objects_failed_transform += 1

                # Removed the arbitrary 100 object limit to draw all points
                # if objects_rendered >= 100:
                #     break

    print(f"Rendered {objects_rendered} objects on image")
    print(f"Failed transformations: {objects_failed_transform}")
    print(f"Out of bounds: {objects_out_of_bounds}")
    print(f"Background objects skipped: {objects_background_skipped}")
    total_objects = sum(
        len(obj.get("triplets", [])) // 3 for obj in objects_data if "triplets" in obj
    )
    print(f"Total objects processed: {total_objects}")


def transform_atlas_to_image_coords_improved(
    atlas_coords: np.ndarray, slice_dict: Dict, scale_factor: float = 1.0
) -> Optional[Tuple[float, float]]:
    """
    Improved transformation from atlas coordinates back to image coordinates.
    Uses a more robust inverse transformation approach.

    Args:
        atlas_coords: 3D atlas coordinates [x, y, z]
        slice_dict: Slice dictionary with anchoring vector
        scale_factor: Scale factor applied to the image

    Returns:
        2D image coordinates or None if transformation fails
    """
    try:
        # Extract anchoring vector
        anchoring = slice_dict["anchoring"]
        ox, oy, oz = anchoring[0:3]
        ux, uy, uz = anchoring[3:6]
        vx, vy, vz = anchoring[6:9]

        # Get image dimensions
        reg_height = slice_dict["height"]
        reg_width = slice_dict["width"]

        # Create vectors
        o = np.array([ox, oy, oz])
        u = np.array([ux, uy, uz])
        v = np.array([vx, vy, vz])

        # Difference vector from origin to point
        diff = atlas_coords - o

        # Solve the system: diff = s*u + t*v for s and t
        # This is done using least squares if the system is overdetermined
        A = np.column_stack([u, v])
        try:
            # Use least squares to solve for [s, t]
            st, residuals, rank, s_values = np.linalg.lstsq(A, diff, rcond=None)
            s, t = st[0], st[1]
        except np.linalg.LinAlgError:
            # Fallback to simpler projection method
            u_norm_sq = np.dot(u, u)
            v_norm_sq = np.dot(v, v)

            if u_norm_sq > 0 and v_norm_sq > 0:
                s = np.dot(diff, u) / u_norm_sq
                t = np.dot(diff, v) / v_norm_sq
            else:
                return None

        # Convert to normalized image coordinates (0 to 1)
        # The solution s, t gives us the parametric coordinates
        # s=0 corresponds to left edge, s=1 to right edge
        # t=0 corresponds to top edge, t=1 to bottom edge

        # Convert to image pixel coordinates directly
        # s and t are already in the correct parametric space [0,1]
        image_x = s * reg_width * scale_factor
        image_y = t * reg_height * scale_factor

        # Debug: Check if coordinates are reasonable
        if not (
            -reg_width <= image_x <= 2 * reg_width
            and -reg_height <= image_y <= 2 * reg_height
        ):
            # Coordinates are way outside expected range - might indicate transformation issue
            return None

        return (image_x, image_y)

    except Exception as e:
        # More specific error logging for debugging
        print(f"Warning: Coordinate transformation failed for {atlas_coords}: {e}")
        return None


def transform_atlas_to_image_coords(
    atlas_coords: np.ndarray, slice_dict: Dict, scale_factor: float = 1.0
) -> Optional[Tuple[float, float]]:
    """
    Transform atlas coordinates back to image coordinates.
    This is a simplified inverse transformation.

    Args:
        atlas_coords: 3D atlas coordinates [x, y, z]
        slice_dict: Slice dictionary with anchoring vector
        scale_factor: Scale factor applied to the image

    Returns:
        2D image coordinates or None if transformation fails
    """
    try:
        # Extract anchoring vector
        anchoring = slice_dict["anchoring"]
        ox, oy, oz = anchoring[0:3]
        ux, uy, uz = anchoring[3:6]
        vx, vy, vz = anchoring[6:9]

        # This is a simplified inverse transformation
        # In practice, this might need more sophisticated calculation

        # Get image dimensions
        reg_height = slice_dict["height"]
        reg_width = slice_dict["width"]

        # Calculate approximate image coordinates
        # This assumes a linear relationship which may not be entirely accurate
        o = np.array([ox, oy, oz])
        u = np.array([ux, uy, uz])
        v = np.array([vx, vy, vz])

        # Solve for s and t such that atlas_coords ≈ o + s*u + t*v
        # This is a simplified approximation
        diff = atlas_coords - o

        # Project onto u and v vectors
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        if u_norm > 0 and v_norm > 0:
            s = np.dot(diff, u) / (u_norm * u_norm)
            t = np.dot(diff, v) / (v_norm * v_norm)

            # Convert to image coordinates
            image_x = s * reg_width * scale_factor
            image_y = t * reg_height * scale_factor

            return (image_x, image_y)

    except Exception as e:
        print(f"Warning: Could not transform coordinates: {e}")

    return None


def create_section_visualizations(
    segmentation_folder: str,
    alignment_json: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_folder: str,
    objects_per_section: Optional[List[List[Dict]]] = None,
    scale_factor: float = 0.5,
) -> None:
    """
    Create visualization images for all sections in the analysis.

    Args:
        segmentation_folder: Path to folder containing segmentation images
        alignment_json: Alignment JSON data
        atlas_volume: 3D atlas volume
        atlas_labels: DataFrame with atlas region information
        output_folder: Output folder for visualizations
        objects_per_section: Optional list of object data per section
        scale_factor: Scale factor for output images
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_folder, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Get list of slices from alignment JSON
    slices = alignment_json.get("slices", [])

    for i, slice_dict in enumerate(slices):
        try:
            # Find corresponding segmentation file
            filename = slice_dict.get("filename", "")
            if filename:
                # Look for segmentation file
                base_name = os.path.splitext(filename)[0]
                seg_files = [
                    f"{base_name}.png",
                    f"{base_name}_Seg.png",
                    f"{base_name}_Simple_Seg.png",
                    f"{base_name}_resize_Simple_Seg.png",
                ]

                segmentation_path = None
                for seg_file in seg_files:
                    potential_path = os.path.join(segmentation_folder, seg_file)
                    if os.path.exists(potential_path):
                        segmentation_path = potential_path
                        break

                # Get objects data for this section
                section_objects = None
                if objects_per_section and i < len(objects_per_section):
                    section_objects = objects_per_section[i]

                # Create output filename
                output_filename = f"section_{slice_dict.get('nr', i):03d}_{base_name}_atlas_colored.png"
                output_path = os.path.join(viz_dir, output_filename)

                # Create the colored slice visualization
                create_colored_atlas_slice(
                    slice_dict,
                    atlas_volume,
                    atlas_labels,
                    output_path,
                    segmentation_path,
                    section_objects,
                    scale_factor,
                )

                print(f"Created visualization: {output_filename}")

        except Exception as e:
            print(f"Error creating visualization for slice {i}: {e}")
