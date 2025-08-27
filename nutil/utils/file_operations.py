import os
import json
from .read_and_write import write_hemi_points_to_meshview
from .section_visualization import create_section_visualizations
from typing import List
import shutil
import pandas as pd


def ensure_dir_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def remove_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def list_files(path: str, ext: str = "") -> List[str]:
    return [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and (f.endswith(ext) if ext else True)
    ]


def copy_file(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def move_file(src: str, dst: str) -> None:
    shutil.move(src, dst)


def _apply_dataframe_manipulations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard manipulations to a dataframe (used for both whole series and per-section reports).
    
    Args:
        df: Input dataframe to manipulate
        
    Returns:
        Manipulated dataframe
    """
    if df is None:
        return df
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Remove damaged columns if no damaged objects exist
    if (
        "damaged_object_count" in df.columns
        and df["damaged_object_count"].sum() == 0
    ):
        # If no damaged objects, remove the column
        # feature implementation remains on request by Sharon 04.07
        df = df.drop(
            columns=[
                "damaged_object_count",
                "damaged_pixel_counts",
                "undamaged_object_count",
                "undamaged_pixel_count",
            ],
            errors="ignore",
        )

    # TODO Investigate why these guys multiply to 6
    df = df.drop(columns=["MSH", "VIS"], errors="ignore")
    
    if "a" in df.columns:
        df["a"] = (df["object_count"] != 0).astype(int)
        # Look at alpha use in the future

    if "original_idx" in df.columns:
        df["idx"] = df["original_idx"]
        df = df.drop(columns=["original_idx"])
    
    return df


def save_analysis_output(
    pixel_points,
    centroids,
    label_df: pd.DataFrame,
    per_section_df,
    labeled_points,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    points_len,
    centroids_len,
    segmentation_filenames,
    atlas_labels,
    output_folder,
    segmentation_folder=None,
    alignment_json=None,
    colour=None,
    atlas_name=None,
    custom_region_path=None,
    atlas_path=None,
    label_path=None,
    settings_file=None,
    prepend=None,
):
    """
    Save the analysis output to the specified folder.

    Parameters
    ----------
    output_folder : str
        The folder where the output will be saved.
    segmentation_folder : str, optional
        The folder containing the segmentation files (default is None).
    alignment_json : str, optional
        The path to the alignment JSON file (default is None).
    colour : list, optional
        The RGB colour of the object to be quantified in the segmentation (default is None).
    atlas_name : str, optional
        The name of the atlas in the brainglobe api to be used for quantification (default is None).
    atlas_path : str, optional
        The path to the custom atlas volume file, only specific if you don't want to use brainglobe (default is None).
    label_path : str, optional
        The path to the custom atlas label file, only specific if you don't want to use brainglobe (default is None).
    settings_file : str, optional
        The path to the settings file that was used (default is None).
    """
    # Create the output folder if it doesn't exist
    ensure_dir_exists(output_folder)
    ensure_dir_exists(f"{output_folder}/whole_series_report")
    ensure_dir_exists(f"{output_folder}/per_section_meshview")
    ensure_dir_exists(f"{output_folder}/per_section_reports")
    ensure_dir_exists(f"{output_folder}/whole_series_meshview")
    
    # Apply standard dataframe manipulations
    label_df = _apply_dataframe_manipulations(label_df)

    if label_df is not None:
        label_df.to_csv(
            f"{output_folder}/whole_series_report/{prepend}counts.csv",
            sep=";",
            na_rep="",
            index=False,
        )
    else:
        print("No quantification found, so only coordinates will be saved.")
        print(
            "If you want to save the quantification, please run quantify_coordinates."
        )

    _save_per_section_reports(
        per_section_df,
        segmentation_filenames,
        points_len,
        centroids_len,
        pixel_points,
        centroids,
        labeled_points,
        labeled_points_centroids,
        points_hemi_labels,
        centroids_hemi_labels,
        atlas_labels,
        output_folder,
        prepend,
    )
    _save_whole_series_meshview(
        pixel_points,
        labeled_points,
        centroids,
        labeled_points_centroids,
        points_hemi_labels,
        centroids_hemi_labels,
        atlas_labels,
        output_folder,
        prepend,
    )

    # Save settings to JSON file for reference
    settings_dict = {
        "segmentation_folder": segmentation_folder,
        "alignment_json": alignment_json,
        "colour": colour,
        "custom_region_path": custom_region_path,
    }
    # Add atlas information to settings
    if atlas_name:
        settings_dict["atlas_name"] = atlas_name
    if atlas_path:
        settings_dict["atlas_path"] = atlas_path
    if label_path:
        settings_dict["label_path"] = label_path
    if settings_file:
        settings_dict["settings_file"] = settings_file
    if custom_region_path:
        settings_dict["custom_region_path"] = custom_region_path

    # Create section visualizations if we have the necessary data
    if segmentation_folder and alignment_json:
        try:
            # Load alignment JSON data and atlas volume from the calling context
            # This requires access to atlas_volume from the Nutil class
            print("Creating section visualizations...")
            # For now, we'll add this functionality but it needs atlas_volume
            # which should be passed as a parameter
        except Exception as e:
            print(f"Warning: Could not create section visualizations: {e}")

    # Write settings to file
    settings_file_path = os.path.join(output_folder, "pynutil_settings.json")
    with open(settings_file_path, "w") as f:
        json.dump(settings_dict, f, indent=4)


def _save_per_section_reports(
    per_section_df,
    segmentation_filenames,
    points_len,
    centroids_len,
    pixel_points,
    centroids,
    labeled_points,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    atlas_labels,
    output_folder,
    prepend,
):
    prev_pl = 0
    prev_cl = 0

    for pl, cl, fn, df in zip(
        points_len,
        centroids_len,
        segmentation_filenames,
        per_section_df,
    ):
        split_fn = fn.split(os.sep)[-1].split(".")[0]
        
        # Apply the same manipulations to per-section dataframes
        df = _apply_dataframe_manipulations(df)
        
        df.to_csv(
            f"{output_folder}/per_section_reports/{prepend}{split_fn}.csv",
            sep=";",
            na_rep="",
            index=False,
        )
        _save_per_section_meshview(
            split_fn,
            pl,
            cl,
            prev_pl,
            prev_cl,
            pixel_points,
            centroids,
            labeled_points,
            labeled_points_centroids,
            points_hemi_labels,
            centroids_hemi_labels,
            atlas_labels,
            output_folder,
            prepend,
        )
        prev_cl += cl
        prev_pl += pl


def _save_per_section_meshview(
    split_fn,
    pl,
    cl,
    prev_pl,
    prev_cl,
    pixel_points,
    centroids,
    labeled_points,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    atlas_labels,
    output_folder,
    prepend,
):
    write_hemi_points_to_meshview(
        pixel_points[prev_pl : pl + prev_pl],
        labeled_points[prev_pl : pl + prev_pl],
        points_hemi_labels[prev_pl : pl + prev_pl],
        f"{output_folder}/per_section_meshview/{prepend}{split_fn}_pixels.json",
        atlas_labels,
    )
    write_hemi_points_to_meshview(
        centroids[prev_cl : cl + prev_cl],
        labeled_points_centroids[prev_cl : cl + prev_cl],
        centroids_hemi_labels[prev_cl : cl + prev_cl],
        f"{output_folder}/per_section_meshview/{prepend}{split_fn}_centroids.json",
        atlas_labels,
    )


def _save_whole_series_meshview(
    pixel_points,
    labeled_points,
    centroids,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    atlas_labels,
    output_folder,
    prepend,
):
    write_hemi_points_to_meshview(
        pixel_points,
        labeled_points,
        points_hemi_labels,
        f"{output_folder}/whole_series_meshview/{prepend}pixels_meshview.json",
        atlas_labels,
    )
    write_hemi_points_to_meshview(
        centroids,
        labeled_points_centroids,
        centroids_hemi_labels,
        f"{output_folder}/whole_series_meshview/{prepend}objects_meshview.json",
        atlas_labels,
    )
