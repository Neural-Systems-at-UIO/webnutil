from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from .counting_and_load import pixel_count_per_region


AGGREGATABLE_COLUMNS = [
    "pixel_count",
    "undamaged_pixel_count",
    "damaged_pixel_counts",
    "region_area",
    "undamaged_region_area",
    "damaged_region_area",
    "object_count",
    "undamaged_object_count",
    "damaged_object_count",
    "left_hemi_pixel_count",
    "left_hemi_undamaged_pixel_count",
    "left_hemi_damaged_pixel_count",
    "left_hemi_region_area",
    "left_hemi_undamaged_region_area",
    "left_hemi_damaged_region_area",
    "left_hemi_object_count",
    "left_hemi_undamaged_object_count",
    "left_hemi_damaged_object_count",
    "right_hemi_pixel_count",
    "right_hemi_undamaged_pixel_count",
    "right_hemi_damaged_pixel_count",
    "right_hemi_region_area",
    "right_hemi_undamaged_region_area",
    "right_hemi_damaged_region_area",
    "right_hemi_object_count",
    "right_hemi_undamaged_object_count",
    "right_hemi_damaged_object_count",
]


def map_to_custom_regions(
    custom_regions_dict: Dict, points_labels: np.ndarray
) -> np.ndarray:
    """
    Reassigns atlas-region labels into user-defined custom regions.

    Args:
        custom_regions_dict: Mapping containing custom region definitions
        points_labels: Array of atlas region labels to remap

    Returns:
        Array with updated region assignments
    """
    custom_points_labels = np.zeros_like(points_labels)

    for region_id in np.unique(points_labels):
        matches = np.where(
            [
                region_id in subregions
                for subregions in custom_regions_dict["subregion_ids"]
            ]
        )[0]

        if len(matches) > 1:
            raise ValueError(
                f"Region ID {region_id} appears in multiple custom regions"
            )
        if len(matches) == 0:
            continue

        custom_id = custom_regions_dict["custom_ids"][matches[0]]
        custom_points_labels[points_labels == region_id] = int(custom_id)

    return custom_points_labels


def _create_mappings(custom_regions_dict: Dict) -> Tuple[Dict, Dict, Dict]:
    """Create ID, name, and RGB mappings from custom regions dictionary."""
    id_mapping = {}
    name_mapping = {}
    rgb_mapping = {}

    for cid, cname, rgb, subregions in zip(
        custom_regions_dict["custom_ids"],
        custom_regions_dict["custom_names"],
        custom_regions_dict["rgb_values"],
        custom_regions_dict["subregion_ids"],
    ):
        for sid in subregions:
            id_mapping[sid] = cid
            name_mapping[sid] = cname
            rgb_mapping[sid] = rgb

    return id_mapping, name_mapping, rgb_mapping


def _add_rgb_columns(df: pd.DataFrame, rgb_mapping: Dict) -> pd.DataFrame:
    """Add RGB color columns to dataframe."""
    for color_idx, color in enumerate(["r", "g", "b"]):
        df[color] = df["idx"].map(
            lambda x: rgb_mapping[x][color_idx] if x in rgb_mapping else None
        )
    return df


def _calculate_area_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate area fractions for all hemisphere and damage combinations."""
    fraction_configs = [
        ("pixel_count", "region_area", "area_fraction"),
        ("undamaged_pixel_count", "undamaged_region_area", "undamaged_area_fraction"),
        ("left_hemi_pixel_count", "left_hemi_region_area", "left_hemi_area_fraction"),
        (
            "right_hemi_pixel_count",
            "right_hemi_region_area",
            "right_hemi_area_fraction",
        ),
        (
            "left_hemi_undamaged_pixel_count",
            "left_hemi_undamaged_region_area",
            "left_hemi_undamaged_area_fraction",
        ),
        (
            "right_hemi_undamaged_pixel_count",
            "right_hemi_undamaged_region_area",
            "right_hemi_undamaged_area_fraction",
        ),
    ]

    for pixel_col, area_col, fraction_col in fraction_configs:
        if pixel_col in df.columns and area_col in df.columns:
            df[fraction_col] = df[pixel_col] / df[area_col]

    return df


def apply_custom_regions(
    df: pd.DataFrame, custom_regions_dict: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies custom region definitions to dataframe region labels.

    Args:
        df: DataFrame with region data to remap
        custom_regions_dict: Custom region definitions

    Returns:
        Tuple of (grouped_df, original_df) with applied custom regions
    """
    id_mapping, name_mapping, rgb_mapping = _create_mappings(custom_regions_dict)

    df["custom_region_name"] = df["idx"].map(name_mapping).fillna("")
    temp_df = df.copy()

    temp_df = _add_rgb_columns(temp_df, rgb_mapping)
    temp_df["idx"] = temp_df["idx"].map(id_mapping)

    # Aggregate columns that exist in the dataframe
    agg_dict = {col: "sum" for col in AGGREGATABLE_COLUMNS if col in temp_df.columns}
    agg_dict.update({"r": "first", "g": "first", "b": "first"})

    grouped_df = (
        temp_df[temp_df["custom_region_name"] != ""]
        .groupby("custom_region_name", dropna=True)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"custom_region_name": "name"})
    )

    grouped_df = _calculate_area_fractions(grouped_df)

    # Reorder columns to match original
    common_columns = [col for col in df.columns if col in grouped_df.columns]
    remaining_columns = [col for col in grouped_df.columns if col not in common_columns]
    grouped_df = grouped_df.reindex(columns=common_columns + remaining_columns)

    return grouped_df, df


def quantify_labeled_points(
    points_len: List[int],
    centroids_len: List[int],
    region_areas_list: List,
    labeled_points: np.ndarray,
    labeled_points_centroids: np.ndarray,
    atlas_labels: pd.DataFrame,
    points_hemi_labels: np.ndarray,
    centroids_hemi_labels: np.ndarray,
    per_point_undamaged: np.ndarray,
    per_centroid_undamaged: np.ndarray,
    apply_damage_mask: bool,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Aggregates labeled points into a summary table.

    Args:
        points_len: List of point counts per section
        centroids_len: List of centroid counts per section
        region_areas_list: List of region areas per section
        labeled_points: Array of labeled point data
        labeled_points_centroids: Array of labeled centroid data
        atlas_labels: DataFrame containing atlas region labels
        points_hemi_labels: Hemisphere labels for points
        centroids_hemi_labels: Hemisphere labels for centroids
        per_point_undamaged: Damage status per point
        per_centroid_undamaged: Damage status per centroid
        apply_damage_mask: Whether to include damage analysis

    Returns:
        Tuple of (combined_df, per_section_dfs)
    """
    per_section_df = _quantify_per_section(
        labeled_points,
        labeled_points_centroids,
        points_len,
        centroids_len,
        region_areas_list,
        atlas_labels,
        per_point_undamaged,
        per_centroid_undamaged,
        points_hemi_labels,
        centroids_hemi_labels,
        apply_damage_mask,
    )

    label_df = _combine_slice_reports(per_section_df, atlas_labels)

    if not apply_damage_mask:
        damage_cols = [col for col in label_df.columns if "damage" in col]
        label_df = label_df.drop(columns=damage_cols, errors="ignore")
        per_section_df = [
            df.drop(columns=damage_cols, errors="ignore") for df in per_section_df
        ]

    return label_df, per_section_df


def _quantify_per_section(
    labeled_points: np.ndarray,
    labeled_points_centroids: np.ndarray,
    points_len: List[int],
    centroids_len: List[int],
    region_areas_list: List,
    atlas_labels: pd.DataFrame,
    per_point_undamaged: np.ndarray,
    per_centroid_undamaged: np.ndarray,
    points_hemi_labels: np.ndarray,
    centroids_hemi_labels: np.ndarray,
    apply_damage_mask: bool = False,
) -> List[pd.DataFrame]:
    """
    Quantifies labeled points per section.

    Args:
        labeled_points (ndarray): Array of labeled points.
        labeled_points_centroids (ndarray): Array of labeled centroids.
        points_len (list): List of lengths of points per section.
        centroids_len (list): List of lengths of centroids per section.
        region_areas_list (list): List of region areas per section.
        atlas_labels (DataFrame): DataFrame with atlas labels.

    Returns:
        list: List of DataFrames for each section.
    """
    prev_pl = 0
    prev_cl = 0
    per_section_df = []

    for pl, cl, ra in zip(points_len, centroids_len, region_areas_list):
        current_centroids = labeled_points_centroids[prev_cl : prev_cl + cl]
        current_points = labeled_points[prev_pl : prev_pl + pl]
        current_points_undamaged = per_point_undamaged[prev_pl : prev_pl + pl]
        current_centroids_undamaged = per_centroid_undamaged[prev_cl : prev_cl + cl]
        current_points_hemi = points_hemi_labels[prev_pl : prev_pl + pl]
        current_centroids_hemi = centroids_hemi_labels[prev_cl : prev_cl + cl]
        current_df = pixel_count_per_region(
            current_points,
            current_centroids,
            current_points_undamaged,
            current_centroids_undamaged,
            current_points_hemi,
            current_centroids_hemi,
            atlas_labels,
            apply_damage_mask,
        )
        current_df_new = _merge_dataframes(current_df, ra, atlas_labels)
        per_section_df.append(current_df_new)
        prev_pl += pl
        prev_cl += cl

    return per_section_df


def _merge_dataframes(
    current_df: pd.DataFrame, ra: pd.DataFrame, atlas_labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges current DataFrame with region areas and atlas labels.

    Args:
        current_df: Current section's pixel count data
        ra: Region areas DataFrame
        atlas_labels: Atlas labels DataFrame

    Returns:
        Merged DataFrame with area fractions calculated
    """
    try:
        if ra.empty or "idx" not in ra.columns:
            current_df_new = atlas_labels.copy()
            current_df_new["pixel_count"] = 0
            current_df_new["object_count"] = 0
            return current_df_new

        # Merge region areas
        cols_to_use = ra.columns.difference(atlas_labels.columns)
        all_region_df = atlas_labels.merge(
            ra[["idx", *cols_to_use]], on="idx", how="left"
        )

        if current_df.empty or "idx" not in current_df.columns:
            return all_region_df

        # Merge current section data
        cols_to_use = current_df.columns.difference(all_region_df.columns)
        current_df_new = all_region_df.merge(
            current_df[["idx", *cols_to_use]], on="idx", how="left"
        )

    except KeyError as e:
        print(f"Warning: Merge failed ({e}), returning atlas labels with zero counts")
        current_df_new = atlas_labels.copy()
        current_df_new["pixel_count"] = 0
        current_df_new["object_count"] = 0

    current_df_new = _calculate_area_fractions(current_df_new)
    current_df_new.fillna(0, inplace=True)

    return current_df_new


def _combine_slice_reports(
    per_section_df: List[pd.DataFrame], atlas_labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines slice reports into a single DataFrame.

    Args:
        per_section_df: List of DataFrames for each section
        atlas_labels: DataFrame with atlas labels

    Returns:
        Combined DataFrame with calculated area fractions
    """
    label_df = (
        pd.concat(per_section_df)
        .groupby(["idx", "name", "r", "g", "b"])
        .sum()
        .reset_index()
        .drop(columns=["area_fraction"], errors="ignore")
    )

    label_df = _calculate_area_fractions(label_df)
    label_df.fillna(0, inplace=True)

    # Reindex to match atlas labels order
    label_df = (
        label_df.set_index("idx").reindex(index=atlas_labels["idx"]).reset_index()
    )

    return label_df
