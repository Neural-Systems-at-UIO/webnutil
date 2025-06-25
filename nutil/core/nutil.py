import json
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import logging
import sys

from ..utils.atlas_loader import load_custom_atlas
from .data_analysis import quantify_labeled_points
from ..utils.file_operations import save_analysis_output
from .coordinate_extraction import folder_to_atlas_space

# --- Setup Logger ---
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of messages

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)  # Console handler
f_handler = logging.FileHandler("nutil.log")  # File handler
c_handler.setLevel(logging.INFO)  # Console shows INFO and above
f_handler.setLevel(logging.DEBUG)  # File logs everything

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)
# --- End Setup Logger ---


class Nutil:
    def __init__(
        self,
        segmentation_folder: Optional[str] = None,
        alignment_json: Optional[str] = None,
        colour: Optional[list[int]] = None,
        atlas_path: Optional[str] = None,
        label_path: Optional[str] = None,
        hemi_path: Optional[str] = None,
        custom_region_path: Optional[str] = None,
    ) -> None:
        logger.info("Initializing Nutil class.")
        logger.debug(
            f"Parameters: segmentation_folder={segmentation_folder}, alignment_json={alignment_json}, colour={colour}, atlas_path={atlas_path}, label_path={label_path}, hemi_path={hemi_path}, custom_region_path={custom_region_path}"
        )
        try:
            self.segmentation_folder: Optional[str] = segmentation_folder
            self.alignment_json: Optional[str] = alignment_json
            self.colour: Optional[list[int]] = colour
            self.custom_region_path: Optional[str] = custom_region_path
            self._validate_atlas_params(atlas_path, label_path)
            self.atlas_path: Optional[str] = atlas_path
            self.label_path: Optional[str] = label_path
            self.hemi_path: Optional[str] = hemi_path
            self.atlas_volume: Optional[np.ndarray]
            self.hemi_map: Optional[np.ndarray]
            self.atlas_labels: pd.DataFrame
            self.atlas_volume, self.hemi_map, self.atlas_labels = load_custom_atlas(
                atlas_path, hemi_path, label_path
            )
            logger.info("Custom atlas loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Error loading atlas files: {e}", exc_info=True)
            raise ValueError(f"Error loading atlas files: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from atlas files: {e}", exc_info=True)
            raise ValueError(f"Error decoding JSON from atlas files: {e}")
        except ValueError as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected initialization error: {e}", exc_info=True)
            raise ValueError(f"Unexpected initialization error: {e}")

    def _validate_atlas_params(
        self, atlas_path: Optional[str], label_path: Optional[str]
    ) -> None:
        if not atlas_path:
            logger.error("atlas_path parameter is required but not provided.")
            raise ValueError("The atlas_path parameter is required.")
        if not label_path:
            logger.error("label_path parameter is required but not provided.")
            raise ValueError("The label_path parameter is required.")
        logger.debug("Atlas parameters validated successfully.")

    def get_coordinates(
        self,
        non_linear: bool = True,
        object_cutoff: int = 0,
        use_flat: bool = False,
        apply_damage_mask: bool = True,
    ) -> None:
        logger.info("Starting coordinate extraction.")
        logger.debug(
            f"Parameters: non_linear={non_linear}, object_cutoff={object_cutoff}, use_flat={use_flat}, apply_damage_mask={apply_damage_mask}"
        )
        if self.segmentation_folder is None:
            logger.error("Segmentation folder not provided for coordinate extraction.")
            raise ValueError("Segmentation folder must be provided to get_coordinates.")
        if self.alignment_json is None:
            logger.error("Alignment JSON not provided for coordinate extraction.")
            raise ValueError("Alignment JSON must be provided to get_coordinates.")
        if self.colour is None:
            logger.error("Colour not provided for coordinate extraction.")
            raise ValueError("Colour must be provided to get_coordinates.")
        try:
            (
                self.pixel_points,
                self.centroids,
                self.points_labels,
                self.centroids_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.region_areas_list,
                self.points_len,
                self.centroids_len,
                self.segmentation_filenames,
                self.per_point_undamaged,
                self.per_centroid_undamaged,
            ) = folder_to_atlas_space(
                self.segmentation_folder,
                self.alignment_json,
                self.atlas_labels,
                self.colour,
                non_linear,
                object_cutoff,
                self.atlas_volume,
                self.hemi_map,
                use_flat,
                apply_damage_mask,
            )
            self.apply_damage_mask = apply_damage_mask
            logger.info("Coordinate extraction completed successfully.")
            logger.debug(
                f"Number of pixel points: {len(self.pixel_points) if self.pixel_points is not None else 0}"
            )
            logger.debug(
                f"Number of centroids: {len(self.centroids) if self.centroids is not None else 0}"
            )
        except ValueError as e:
            logger.error(f"ValueError during coordinate extraction: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting coordinates: {e}", exc_info=True)
            raise ValueError(f"Unexpected error extracting coordinates: {e}")

    def quantify_coordinates(self) -> None:
        logger.info("Starting coordinate quantification.")
        if not hasattr(self, "pixel_points") or not hasattr(self, "centroids"):
            logger.error("Attempted to quantify coordinates before extraction.")
            raise ValueError(
                "Please run get_coordinates before running quantify_coordinates."
            )
        try:
            (self.label_df, self.per_section_df) = quantify_labeled_points(
                self.points_len,
                self.centroids_len,
                self.region_areas_list,
                self.points_labels,
                self.centroids_labels,
                self.atlas_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.per_point_undamaged,
                self.per_centroid_undamaged,
                self.apply_damage_mask,
            )
            logger.info("Coordinate quantification completed successfully.")
            logger.debug(
                f"Label_df shape: {self.label_df.shape if hasattr(self, 'label_df') else 'Not generated'}"
            )
        except Exception as e:
            logger.error(f"Error quantifying coordinates: {e}", exc_info=True)
            raise ValueError(f"Error quantifying coordinates: {e}")

    def save_analysis(self, output_folder: str) -> None:
        logger.info(f"Attempting to save analysis to: {output_folder}")
        if not hasattr(self, "label_df") or not hasattr(self, "per_section_df"):
            logger.error("Attempted to save analysis before quantification.")
            raise ValueError(
                "Please run quantify_coordinates before saving the analysis."
            )
        if not hasattr(self, "pixel_points"):
            logger.error("Attempted to save analysis before coordinate extraction.")
            raise ValueError("Please run get_coordinates before saving the analysis.")
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured output folder exists or created: {output_folder}")
            save_analysis_output(
                pixel_points=self.pixel_points,
                centroids=self.centroids,
                label_df=self.label_df,
                per_section_df=self.per_section_df,
                labeled_points=self.points_labels,
                labeled_points_centroids=self.centroids_labels,
                points_hemi_labels=self.points_hemi_labels,
                centroids_hemi_labels=self.centroids_hemi_labels,
                points_len=self.points_len,
                centroids_len=self.centroids_len,
                segmentation_filenames=self.segmentation_filenames,
                atlas_labels=self.atlas_labels,
                output_folder=output_folder,
                segmentation_folder=getattr(self, "segmentation_folder", None),
                alignment_json=getattr(self, "alignment_json", None),
                colour=getattr(self, "colour", None),
                atlas_name=None,
                custom_region_path=getattr(self, "custom_region_path", None),
                atlas_path=getattr(self, "atlas_path", None),
                label_path=getattr(self, "label_path", None),
                settings_file=getattr(self, "settings_file", None),
                prepend="",
            )
            logger.info(f"Analysis results saved successfully to: {output_folder}")
        except FileNotFoundError as e:
            logger.error(f"Error saving analysis - file not found: {e}", exc_info=True)
            raise ValueError(f"Error saving analysis - file not found: {e}")
        except PermissionError as e:
            logger.error(
                f"Error saving analysis - permission denied for {output_folder}: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Error saving analysis - permission denied for {output_folder}: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error saving analysis: {e}", exc_info=True)
            raise ValueError(f"Unexpected error saving analysis: {e}")

    def get_region_summary(self) -> pd.DataFrame:
        logger.info("Attempting to get region summary.")
        if not hasattr(self, "label_df"):
            logger.error("Attempted to get region summary before quantification.")
            raise ValueError(
                "Please run quantify_coordinates before getting region summary."
            )
        logger.info("Region summary retrieved successfully.")
        return self.label_df
