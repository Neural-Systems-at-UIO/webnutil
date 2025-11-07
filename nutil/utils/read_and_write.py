import json
import numpy as np
import struct
import pandas as pd
import os
import re
import cv2
import logging
from typing import Any, Dict, List, Tuple, Union
from .propagation import propagate
from .reconstruct_dzi import reconstruct_dzi
from .atlas_loader import load_atlas_labels

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def open_custom_region_file(path: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if len(df.columns) < 2:
        raise ValueError("Expected at least two columns in the file.")
    custom_region_names = df.columns[1:].to_list()
    rgb_values = df.iloc[0, :].values[1:]
    try:
        rgb_values = [list(map(int, rgb.split(";"))) for rgb in rgb_values]
    except Exception:
        logger.error("Non-integer value found in rgb list")
        raise
    atlas_ids = df.iloc[1:, 1:].T.values
    atlas_ids = [[int(j) for j in i if not pd.isna(j)] for i in atlas_ids]
    new_ids, new_id = [], 1
    for ids in atlas_ids:
        if 0 in ids:
            new_ids.append(0)
        else:
            new_ids.append(new_id)
            new_id += 1
    if 0 not in new_ids:
        new_ids.append(0)
        custom_region_names.append("unlabeled")
        rgb_values.append([0, 0, 0])
        atlas_ids.append([0])
    custom_region_dict = {
        "custom_ids": new_ids,
        "custom_names": custom_region_names,
        "rgb_values": rgb_values,
        "subregion_ids": atlas_ids,
    }
    df = pd.DataFrame(
        {
            "idx": custom_region_dict["custom_ids"],
            "name": custom_region_dict["custom_names"],
            "r": [c[0] for c in custom_region_dict["rgb_values"]],
            "g": [c[1] for c in custom_region_dict["rgb_values"]],
            "b": [c[2] for c in custom_region_dict["rgb_values"]],
        }
    )
    if df["name"].duplicated().any():
        raise ValueError("Duplicate region names found in custom region file.")
    return custom_region_dict, df


def read_flat_file(file: str) -> np.ndarray:
    with open(file, "rb") as f:
        b, w, h = struct.unpack(">BII", f.read(9))
        data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))
    arr = np.array(data).reshape((h, w), order="C")
    return arr


def read_seg_file(file: str) -> np.ndarray:
    with open(file, "rb") as f:

        def byte() -> int:
            return f.read(1)[0]

        def code() -> int:
            c = byte()
            if c < 0:
                raise ValueError("Negative code")
            return c if c < 128 else (c & 127) | (code() << 7)

        if "SegRLEv1" != f.read(8).decode():
            raise ValueError("Header mismatch")
        atlas = f.read(code()).decode()
        logger.info(f"Target atlas: {atlas}")
        codes = [code() for _ in range(code())]
        w, h = code(), code()
        data = []
        while len(data) < w * h:
            data += [codes[byte() if len(codes) <= 256 else code()]] * (code() + 1)
    arr = np.array(data).reshape((h, w))
    return arr


def load_segmentation(segmentation_path: str) -> np.ndarray:
    if segmentation_path.endswith(".dzip"):
        return reconstruct_dzi(segmentation_path)
    return cv2.imread(segmentation_path)


def load_quint_json(
    filename: str, propagate_missing_values: bool = True
) -> Dict[str, Any]:
    with open(filename) as f:
        vafile = json.load(f)
    if filename.endswith(".waln") or filename.endswith("wwrp"):
        # handle both "sections" (newer format) and "slices" (older format)
        if "sections" in vafile:
            slices = vafile["sections"]
            vafile["slices"] = slices
        elif "slices" in vafile:
            slices = vafile["slices"]
        else:
            raise KeyError(f"JSON file {filename} must contain either 'sections' or 'slices' key")
        for s in slices:
            s["nr"] = int(re.search(r"_s(\d+)", s["filename"]).group(1))
            if "ouv" in s:
                s["anchoring"] = s["ouv"]
    else:
        if "slices" not in vafile:
            raise KeyError(f"JSON file {filename} must contain 'slices' key")
        slices = vafile["slices"]
    if len(slices) > 1 and propagate_missing_values:
        slices = propagate(slices)
    vafile["slices"] = slices
    return vafile


def create_region_dict(
    points: np.ndarray, regions: np.ndarray
) -> Dict[int, List[float]]:
    return {
        int(region): points[regions == region].flatten().tolist()
        for region in np.unique(regions)
    }


def write_points(
    points_dict: Dict[int, List[float]], filename: str, info_file: pd.DataFrame
) -> None:
    with open(filename, "w") as f:
        f.write("[")
        first = True
        for idx, region_id in enumerate(points_dict):
            if not first:
                f.write(",")
            else:
                first = False
            triplets = points_dict[region_id]
            count = len(triplets) // 3
            match = info_file[info_file["idx"] == region_id]
            if not match.empty:
                region_name = str(match["name"].values[0])
                r_val = int(match["r"].values[0])
                g_val = int(match["g"].values[0])
                b_val = int(match["b"].values[0])
            else:
                region_name = f"Region ID {region_id}"
                r_val, g_val, b_val = 128, 128, 128
            f.write(
                f"{{\n  \"idx\": {idx},\n  \"count\": {count},\n  \"name\": \"{region_name}\",\n  \"triplets\": [{','.join(map(str, triplets))}],\n  \"r\": {r_val},\n  \"g\": {g_val},\n  \"b\": {b_val}\n}}"
            )
        f.write("]")


def write_hemi_points_to_meshview(
    points: np.ndarray,
    point_names: np.ndarray,
    hemi_label: np.ndarray,
    filename: str,
    info_file: pd.DataFrame,
) -> None:
    if not (hemi_label == None).all():
        split_fn_left = filename.split("/")
        split_fn_left[-1] = "left_hemisphere_" + split_fn_left[-1]
        outname_left = os.sep.join(split_fn_left)
        write_points_to_meshview(
            points[hemi_label == 1],
            point_names[hemi_label == 1],
            outname_left,
            info_file,
        )
        split_fn_right = filename.split("/")
        split_fn_right[-1] = "right_hemisphere_" + split_fn_right[-1]
        outname_right = os.sep.join(split_fn_right)
        write_points_to_meshview(
            points[hemi_label == 2],
            point_names[hemi_label == 2],
            outname_right,
            info_file,
        )
    write_points_to_meshview(points, point_names, filename, info_file)


def write_points_to_meshview(
    points: np.ndarray,
    point_ids: np.ndarray,
    filename: str,
    info_file: Union[pd.DataFrame, str],
) -> None:
    if isinstance(info_file, str):
        info_file = load_atlas_labels(info_file)
    region_dict = create_region_dict(points, point_ids)
    write_points(region_dict, filename, info_file)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_lines(lines: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def dir_exists(path: str) -> bool:
    return os.path.isdir(path)


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Failed to remove file {path}: {e}")
