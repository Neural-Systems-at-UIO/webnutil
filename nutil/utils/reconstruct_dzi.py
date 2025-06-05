import cv2
import numpy as np
import os
import zipfile
import xmltodict
import logging
import sys

import os
import sys
import zipfile
import logging
import xmltodict
import numpy as np
import cv2
import re
from typing import Optional, Tuple

try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False
    print("Warning: pyvips not available. Falling back to OpenCV-based reconstruction.")

# --- Setup Logger ---
# Create a logger for this module
logger = logging.getLogger(__name__)
# Set default logging level ; can be overridden by the application using this module
logger.setLevel(logging.DEBUG)

# Check if handlers are already added to avoid duplication if this module is reloaded
if not logger.handlers:
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    c_handler.setLevel(logging.INFO)  # Console shows INFO and above

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
# --- End Setup Logger ---


def reconstruct_dzi(zip_file_path: str):
    """
    Reconstructs a Deep Zoom Image (DZI) from a zip file using the most efficient method available.
    
    This function serves as the main entry point for DZI reconstruction. It automatically
    uses PyVIPS for memory-efficient reconstruction when available, falling back to 
    OpenCV-based reconstruction if PyVIPS is not installed.
    
    Parameters
    ----------
    zip_file_path : str
        Path to the zip file containing the DZI tiles.

    Returns
    -------
    np.ndarray
        The reconstructed DZI as a NumPy array (BGR format) if successful.
        
    Raises
    ------
    FileNotFoundError
        If the zip file doesn't exist.
    ValueError
        If the DZI format is invalid or no tiles are found.
    RuntimeError
        If reconstruction fails due to processing errors.
    """
    if PYVIPS_AVAILABLE:
        return reconstruct_dzi_pyvips(zip_file_path)
    else:
        return reconstruct_dzi_opencv(zip_file_path)


def reconstruct_dzi_pyvips(zip_file_path: str):
    """
    Reconstructs a Deep Zoom Image (DZI) from a zip file using PyVIPS for memory efficiency.
    
    Parameters
    ----------
    zip_file_path : str
        Path to the zip file containing the DZI tiles.

    Returns
    -------
    np.ndarray or None
        The reconstructed DZI as a NumPy array (BGR format) if successful, None otherwise.
    """
    if not PYVIPS_AVAILABLE:
        logger.warning("PyVIPS not available, falling back to OpenCV reconstruction")
        return reconstruct_dzi_opencv(zip_file_path)
    
    logger.info(f"Starting PyVIPS DZI reconstruction for: {zip_file_path}")

    if not os.path.exists(zip_file_path):
        logger.error(f"DZI zip file not found: {zip_file_path}")
        raise FileNotFoundError(f"DZI zip file not found: {zip_file_path}")

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            logger.debug(f"Successfully opened zip file: {zip_file_path}")

            # Find the .dzi XML descriptor file
            dzi_file_name = None
            for name in zip_file.namelist():
                if name.endswith(".dzi"):
                    dzi_file_name = name
                    break
            
            if not dzi_file_name:
                logger.error(f"No .dzi file found in zip: {zip_file_path}")
                raise ValueError(f"No .dzi file found in zip: {zip_file_path}")
            logger.debug(f"Found .dzi file: {dzi_file_name}")

            # Read and parse the DZI XML file
            try:
                with zip_file.open(dzi_file_name) as dzi_xml_file:
                    xml_content = dzi_xml_file.read()
                dzi_data = xmltodict.parse(xml_content)
                logger.debug("Successfully parsed .dzi XML content.")
            except Exception as e:
                logger.error(f"Error parsing DZI XML: {e}")
                raise ValueError(f"Invalid DZI XML format: {e}")

            tile_size = int(dzi_data["Image"]["@TileSize"])
            width = int(dzi_data["Image"]["Size"]["@Width"])
            height = int(dzi_data["Image"]["Size"]["@Height"])
            img_format = dzi_data["Image"]["@Format"]
            logger.info(f"DZI properties: Width={width}, Height={height}, TileSize={tile_size}, Format={img_format}")

            # Find the highest resolution level
            max_level = 0
            files_dir_name = os.path.splitext(dzi_file_name)[0] + "_files"
            
            for item_name in zip_file.namelist():
                if item_name.startswith(files_dir_name + "/"):
                    parts = item_name.split('/')
                    if len(parts) >= 3:  # files_dir/level/tile.ext
                        try:
                            level = int(parts[1])
                            max_level = max(max_level, level)
                        except ValueError:
                            continue

            highest_level_dir = f"{files_dir_name}/{max_level}"
            logger.debug(f"Highest level directory: {highest_level_dir}")
            
            # Get all tiles for the highest level
            tile_files = [
                f for f in zip_file.namelist()
                if f.startswith(highest_level_dir + "/") and f.endswith(f".{img_format}")
            ]

            if not tile_files:
                logger.error(f"No tile files found in {highest_level_dir}")
                raise ValueError(f"No highest level tile files found")

            # Parse tile coordinates and organize them
            tile_pattern = re.compile(r'(\d+)_(\d+)\.' + re.escape(img_format))
            tiles_dict = {}
            max_x, max_y = 0, 0
            
            for tile_file in tile_files:
                basename = os.path.basename(tile_file)
                match = tile_pattern.match(basename)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    tiles_dict[(x, y)] = tile_file
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

            tiles_across = max_x + 1
            tiles_down = max_y + 1
            logger.info(f"Found tiles grid: {tiles_across} x {tiles_down}")

            # Extract tiles to temporary files for PyVIPS
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_files = []
            
            try:
                # Create a grid of PyVIPS images
                tile_images = []
                
                for y in range(tiles_down):
                    # Most of the handling of images is done 
                    row_images = []
                    for x in range(tiles_across):
                        if (x, y) in tiles_dict:
                            # Extract tile to temp file
                            tile_data = zip_file.read(tiles_dict[(x, y)])
                            temp_tile_path = os.path.join(temp_dir, f"tile_{x}_{y}.{img_format}")
                            with open(temp_tile_path, 'wb') as temp_file:
                                temp_file.write(tile_data)
                            temp_files.append(temp_tile_path)
                            
                            # Load with PyVIPS
                            tile_img = pyvips.Image.new_from_file(temp_tile_path, access="sequential")
                            row_images.append(tile_img)
                        else:
                            # Create blank tile if missing
                            blank_tile = pyvips.Image.black(tile_size, tile_size, bands=3)
                            row_images.append(blank_tile)
                    
                    # Join row horizontally
                    if row_images:
                        row_joined = pyvips.Image.arrayjoin(row_images, across=len(row_images))
                        tile_images.append(row_joined)

                # Join all rows vertically
                if tile_images:
                    final_image = pyvips.Image.arrayjoin(tile_images, across=1)
                    
                    # Crop to actual size if needed
                    if final_image.width > width or final_image.height > height:
                        final_image = final_image.crop(0, 0, min(width, final_image.width), min(height, final_image.height))
                    
                    # Convert to numpy array
                    # PyVIPS uses RGB, OpenCV expects BGR
                    np_array = np.ndarray(buffer=final_image.write_to_memory(),
                                        dtype=np.uint8,
                                        shape=[final_image.height, final_image.width, final_image.bands])
                    
                    # Convert RGB to BGR for OpenCV compatibility
                    if np_array.shape[2] == 3:
                        np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
                    
                    logger.info(f"PyVIPS DZI reconstruction completed: {np_array.shape}")
                    return np_array
                
            finally:
                # Clean up temp files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass

    except Exception as e:
        logger.error(f"PyVIPS reconstruction failed: {e}")
        raise RuntimeError(f"Failed to reconstruct DZI with PyVIPS: {e}")


def reconstruct_dzi_opencv(zip_file_path: str):
    """
    Reconstructs a Deep Zoom Image (DZI) from a zip file containing the tiles.
    This function currently reconstructs the entire image from the highest resolution tiles.
    For a sliding window approach, this function could be adapted to:
    1.  Load only necessary tiles for a given window. This would require
        parsing the DZI XML to understand the tile layout and then selectively
        extracting and decoding those specific tiles.
    2.  Alternatively, the full image is reconstructed here, and the sliding window
        logic is applied to the resulting numpy array in a subsequent step.
        This is simpler if memory allows for the full image reconstruction.

    Parameters
    ----------
    zip_file_path : str
        Path to the zip file containing the DZI tiles.

    Returns
    -------
    np.ndarray or None
        The reconstructed DZI as a NumPy array (BGR format) if successful, None otherwise.
    """
    logger.info(f"Starting DZI reconstruction for: {zip_file_path}")

    if not os.path.exists(zip_file_path):
        logger.error(f"DZI zip file not found: {zip_file_path}")
        raise FileNotFoundError(f"DZI zip file not found: {zip_file_path}")

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            logger.debug(f"Successfully opened zip file: {zip_file_path}")

            # Find the .dzi XML descriptor file
            dzi_file_name = None
            for name in zip_file.namelist():
                if name.endswith(".dzi"):
                    dzi_file_name = name
                    break
            
            if not dzi_file_name:
                logger.error(f"No .dzi file found in zip: {zip_file_path}")
                raise ValueError(f"No .dzi file found in zip: {zip_file_path}")
            logger.debug(f"Found .dzi file: {dzi_file_name}")

            # Read and parse the DZI XML file
            try:
                with zip_file.open(dzi_file_name) as dzi_xml_file:
                    xml_content = dzi_xml_file.read()
                dzi_data = xmltodict.parse(xml_content)
                logger.debug("Successfully parsed .dzi XML content.")
            except xmltodict.expat.ExpatError as e:
                logger.error(f"Error parsing DZI XML content from {dzi_file_name} in {zip_file_path}: {e}", exc_info=True)
                raise ValueError(f"Invalid DZI XML format in {zip_file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading or parsing {dzi_file_name} in {zip_file_path}: {e}", exc_info=True)
                raise

            tile_size = int(dzi_data["Image"]["@TileSize"])
            width = int(dzi_data["Image"]["Size"]["@Width"])
            height = int(dzi_data["Image"]["Size"]["@Height"])
            img_format = dzi_data["Image"]["@Format"] # e.g., 'png', 'jpeg'
            logger.info(f"DZI properties: Width={width}, Height={height}, TileSize={tile_size}, Format={img_format}")

            # Determine the directory containing the highest resolution tiles
            # DZI typically stores tiles in subdirectories named by level.
            # e.g., 'myimage_files/13/0_0.png' where '13' is the level.
            max_level = 0
            tile_files_by_level = {} # {level: [filepaths]}
            
            for item_name in zip_file.namelist():
                parts = item_name.split('/')
                if len(parts) > 1 and parts[-1].startswith("0_0."): # Heuristic: check for top-left tile
                    try:
                        level = int(parts[-2])
                        if level > max_level:
                            max_level = level
                    except ValueError:
                        continue # Not a level directory

            if max_level == 0 and len(dzi_data["Image"]["Size"]["@Width"]) > tile_size : # if no levels, check if it's a single image
                logger.warning(f"Could not determine max level, or it's a single tile image. Assuming level 0 for {zip_file_path}")
                # Fallback or specific logic for single large tile might be needed
                # For now, we'll try to find tiles in the root of the _files dir if it exists
                files_dir_name = os.path.splitext(dzi_file_name)[0] + "_files"
                if not files_dir_name in [os.path.dirname(f) for f in zip_file.namelist()]: # if no _files dir, then it's a single tile
                     files_dir_name = ""


            highest_level_dir_name = os.path.splitext(dzi_file_name)[0] + "_files" + (f"/{max_level}" if max_level > 0 else "")
            logger.debug(f"Highest level directory identified as: {highest_level_dir_name}")
            
            highest_level_tile_files = [
                f for f in zip_file.namelist()
                if f.startswith(highest_level_dir_name + "/") and (f.endswith(f".{img_format}"))
            ]

            if not highest_level_tile_files:
                # Attempt to find files if the structure is flat (no level folders)
                # This might happen for single-level DZIs or non-standard ones.
                files_dir_name_flat = os.path.splitext(dzi_file_name)[0] + "_files"
                highest_level_tile_files = [
                    f for f in zip_file.namelist()
                    if f.startswith(files_dir_name_flat + "/") and (f.endswith(f".{img_format}"))
                ]
                if highest_level_tile_files:
                    logger.info(f"Found tiles in flat structure: {files_dir_name_flat}")
                else:
                    logger.error(f"No tile files found for format '{img_format}' in expected highest level directory: {highest_level_dir_name} or {files_dir_name_flat} within {zip_file_path}")
                    raise ValueError(f"No highest level tile files found in {zip_file_path}")

            logger.info(f"Found {len(highest_level_tile_files)} tiles for the highest level.")

            # Create an empty image (assuming 3 channels, BGR for OpenCV)
            # If the DZI format is grayscale, this might need adjustment, but imdecode usually handles it.
            reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
            logger.debug(f"Created empty image canvas of size ({height}, {width}, 3)")

            # Fill in the image with the highest level tiles
            for tile_file_path in highest_level_tile_files:
                try:
                    with zip_file.open(tile_file_path) as tile_f:
                        tile_contents = tile_f.read()
                    
                    # Decode the image tile data
                    tile_image = cv2.imdecode(np.frombuffer(tile_contents, np.uint8), cv2.IMREAD_COLOR)
                    if tile_image is None:
                        logger.warning(f"Failed to decode tile: {tile_file_path} in {zip_file_path}. Skipping.")
                        continue
                    
                    # Expected filename format like '0_0.png', '1_0.png' etc.
                    tile_filename = os.path.basename(tile_file_path)
                    tile_coords_str = os.path.splitext(tile_filename)[0]
                    col_str, row_str = tile_coords_str.split("_")
                    col, row = int(col_str), int(row_str)
                    
                    x_offset, y_offset = col * tile_size, row * tile_size
                    
                    # Calculate the region in the full image to place this tile
                    y_start, y_end = y_offset, min(y_offset + tile_size, height)
                    x_start, x_end = x_offset, min(x_offset + tile_size, width)
                    
                    # Calculate the portion of the tile to use (in case of edge tiles smaller than tile_size)
                    tile_h, tile_w = tile_image.shape[:2]
                    tile_y_end = y_end - y_start
                    tile_x_end = x_end - x_start

                    reconstructed_image[y_start:y_end, x_start:x_end, :] = tile_image[:tile_y_end, :tile_x_end, :]
                    logger.debug(f"Placed tile {tile_file_path} at ({x_offset}, {y_offset})")

                except Exception as e:
                    logger.error(f"Error processing tile {tile_file_path} in {zip_file_path}: {e}", exc_info=True)
                    # Decide if one bad tile should stop the whole process or just be skipped
                    continue # Skipping problematic tile

            logger.info(f"DZI reconstruction completed for: {zip_file_path}")
            return reconstructed_image

    except zipfile.BadZipFile as e:
        logger.error(f"Invalid or corrupted zip file: {zip_file_path} - {e}", exc_info=True)
        raise ValueError(f"Invalid or corrupted zip file: {zip_file_path} - {e}")
    except FileNotFoundError: # Already caught and raised above, but good to be explicit
        raise
    except ValueError: # Catch ValueErrors raised internally (e.g. no .dzi, no tiles)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during DZI reconstruction for {zip_file_path}: {e}", exc_info=True)
        # Instead of returning None, re-raise a more generic exception or the original one
        # This makes it clearer to the caller that something went wrong.
        raise RuntimeError(f"Failed to reconstruct DZI from {zip_file_path}: {e}")

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Setup a basic logger for the script if run directly
    if not logger.handlers: # Ensure handlers are set up if __main__ is the entry point
        main_c_handler = logging.StreamHandler(sys.stdout)
        main_c_handler.setLevel(logging.DEBUG)
        main_c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        main_c_handler.setFormatter(main_c_format)
        logger.addHandler(main_c_handler)
        logger.propagate = False # Prevent duplicate messages if root logger also has handlers

    logger.info("reconstruct_dzi.py executed as main script.")
    # Create a dummy DZI zip for testing
    # This part would require creating actual DZI files or having a sample one.
    # For now, this just demonstrates the logger setup.
    # try:
    #     # Replace with a path to an actual DZI zip file to test
    #     # img = reconstruct_dzi("path_to_your_dzi_file.zip")
    #     # if img is not None:
    #     #     logger.info(f"Reconstructed image shape: {img.shape}")
    #     #     cv2.imwrite("reconstructed_output.png", img) # Save if needed
    #     # else:
    #     #     logger.error("Reconstruction failed.")
    #     logger.warning("No DZI file provided for testing in __main__ block.")
    # except Exception as e:
    #     logger.error(f"Error in __main__ example: {e}", exc_info=True)
