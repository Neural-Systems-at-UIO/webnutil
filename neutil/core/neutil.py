import json
from ..utils.atlas_loader import load_custom_atlas
from .data_analysis import (
    quantify_labeled_points,
)
from ..utils.file_operations import save_analysis_output
from .coordinate_extraction import folder_to_atlas_space


class Neutil:
    """
    A class to perform brain-wide quantification and spatial analysis of serial section images.

    Methods
    -------
    constructor(...)
        Initialize the Neutil class with segmentation, alignment, atlas and region settings.
    get_coordinates(...)
        Extract and transform pixel coordinates from segmentation files.
    quantify_coordinates()
        Quantify pixel and centroid counts by atlas regions.
    save_analysis(output_folder)
        Save the analysis output to the specified directory.
    """

    def __init__(
        self,
        segmentation_folder=None,
        alignment_json=None,
        colour=None,
        atlas_name=None,
        atlas_path=None,
        label_path=None,
        hemi_path=None,
        custom_region_path=None,
    ):
        """
        Initializes the Neutil class with the given parameters.

        Parameters
        ----------
        segmentation_folder : str, optional
            The folder containing the segmentation files (default is None).
        alignment_json : str, optional
            The path to the alignment JSON file (default is None).
        colour : list, optional
            The RGB colour of the object to be quantified in the segmentation (default is None).
        atlas_name : str, optional
            The name of the atlas in the brainglobe api to be used for quantification (default is None).
        atlas_path : str, optional
            The path to the custom atlas volume file, only specify if you don't want to use brainglobe (default is None).
        label_path : str, optional
            The path to the custom atlas label file, only specify if you don't want to use brainglobe (default is None).
        custom_region_path : str, optional
            The path to a custom region id file. This can be found

        Raises
        ------
        KeyError
            If the settings file does not contain the required keys.
        ValueError
            If both atlas_path and atlas_name are specified or if neither is specified.
        """
        try:

            self.segmentation_folder = segmentation_folder
            self.alignment_json = alignment_json
            self.colour = colour
            self.atlas_name = atlas_name
            self.custom_region_path = custom_region_path
            if (atlas_path or label_path) and atlas_name:
                raise ValueError(
                    "Please specify either atlas_path and label_path or atlas_name. Atlas and label paths are only used for loading custom atlases."
                )

            if atlas_path and label_path:
                self.atlas_path = atlas_path
                self.label_path = label_path
                self.hemi_path = hemi_path
                self.atlas_volume, self.hemi_map, self.atlas_labels = load_custom_atlas(
                    atlas_path, hemi_path, label_path
                )

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading settings file: {e}")
        except Exception as e:
            raise ValueError(f"Initialization error: {e}")

    def _check_atlas_name(self):
        if not self.atlas_name:
            raise ValueError(
                "When atlas_path and label_path are not specified, atlas_name must be specified."
            )

    def get_coordinates(
        self, non_linear=True, object_cutoff=0, use_flat=False, apply_damage_mask=True
    ):
        """
        Retrieves pixel and centroid coordinates from segmentation data,
        applies atlas-space transformations, and optionally uses a damage
        mask if specified.

        Args:
            non_linear (bool, optional): Enable non-linear transformation.
            object_cutoff (int, optional): Minimum object size.
            use_flat (bool, optional): Use flat maps if True.
            apply_damage_mask (bool, optional): Apply damage mask if True.

        Returns:
            None: Results are stored in class attributes.
        """
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

        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")

        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")

    def quantify_coordinates(self):
        """
        Quantifies and summarizes pixel and centroid coordinates by atlas region,
        storing the aggregated results in class attributes.

        Attributes:
            label_df (pd.DataFrame): Contains aggregated label information.
            per_section_df (list of pd.DataFrame): DataFrames with section-wise statistics.
            custom_label_df (pd.DataFrame): Label data enriched with custom regions if custom regions is set.
            custom_per_section_df (list of pd.DataFrame): Section-wise stats for custom regions if custom regions is set.

        Raises:
            ValueError: If required attributes are missing or computation fails.

        Returns:
            None
        """
        if not hasattr(self, "pixel_points") and not hasattr(self, "centroids"):
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

        except Exception as e:
            raise ValueError(f"Error quantifying coordinates: {e}")

    def save_analysis(self, output_folder):
        """
        Saves the pixel coordinates and pixel counts to different files in the specified output folder.

        Parameters
        ----------
        output_folder : str
            The folder where the analysis output will be saved.

        Returns
        -------
        None
        """
        try:
            save_analysis_output(
                self.pixel_points,
                self.centroids,
                self.label_df,
                self.per_section_df,
                self.points_labels,
                self.centroids_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.points_len,
                self.centroids_len,
                self.segmentation_filenames,
                self.atlas_labels,
                output_folder,
                segmentation_folder=self.segmentation_folder,
                alignment_json=self.alignment_json,
                colour=self.colour,
                atlas_name=getattr(self, "atlas_name", None),
                custom_region_path=getattr(self, "custom_region_path", None),
                atlas_path=getattr(self, "atlas_path", None),
                label_path=getattr(self, "label_path", None),
                settings_file=getattr(self, "settings_file", None),
                prepend="",
            )

            print(f"Saved output to {output_folder}")
        except Exception as e:
            raise ValueError(f"Error saving analysis: {e}")
