import aiohttp
import aiofiles
import os
import json
import shutil
from fastapi import HTTPException
import asyncio
from typing import Optional
from enum import Enum

# Shared constants
DOWNLOAD_TIMEOUT = 3600  # 1 hour timeout
CHUNK_SIZE = 8192
MAX_CONCURRENT_TASKS = 3  # This is a similar approach to the DeepZoom
TASK_BASE_DIR = "/data/nutil_tasks"  # Base directory for all tasks
TASK_TEMP_DIR = "/data/nutil_temp"  # Directory for temporary files


class TaskStatus(Enum):
    PENDING = "pending"
    DOWNLOADING_JSON = "downloading json"  # Downloading the alignment file
    DOWNLOADING = "downloading segments"  # Downloading n images and the alignment file
    PROCESSING = "quantifying"  # The full process from PyNutil
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


# Extent of the statuses we can have for Pynutil


class Task:
    def __init__(self):
        self.status = TaskStatus.PENDING
        self.message = ""
        self.error = None


async def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


async def download_directory(directory_path: str, token: str, task_id: str):
    print(f"Downloading images from: {directory_path}")

    # Split path into bucket and prefix
    parts = directory_path.strip("/").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # Ensure prefix ends with slash
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    # Get directory listing
    url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}?prefix={prefix}&delimiter=/&limit=1000"
    # TODO Do we expect 1000+ images?
    headers = {"Authorization": f"Bearer {token}"}

    # Create local directory
    dir_name = os.path.basename(directory_path.rstrip("/")) or "downloads"
    local_dir = f"{TASK_BASE_DIR}/{task_id}/downloads/{dir_name}/"
    await ensure_directory(local_dir)

    downloaded_paths = []

    async with aiohttp.ClientSession() as session:
        # Get directory listing
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Failed to list directory contents: {response.status}",
                )

            data = await response.json()
            objects = data.get("objects", [])

            # Filter for image files
            image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dzip")
            # Currently disabling type checkss
            # Supported images are the ilastik outputs, often .dzip internally, otherwise jpeg
            files_to_download = [
                obj for obj in objects if not obj.get("name", "").endswith("/")
            ]

            print(f"Found {len(files_to_download)} files to download")

            # Download each file
            for i, file in enumerate(files_to_download):
                file_path = file.get("name")
                if not file_path:
                    continue

                try:
                    print(
                        f"[{i+1}/{len(files_to_download)}] Downloading {file_path.split('/')[-1]}..."
                    )

                    # Get pre-signed URL
                    file_url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}/{file_path}?redirect=false"
                    local_path = f"{local_dir}{os.path.basename(file_path)}"

                    async with session.get(file_url, headers=headers) as file_response:
                        if file_response.status != 200:
                            print(f"Failed to get download URL for {file_path}")
                            continue

                        data = await file_response.json()
                        download_url = data.get("url")
                        if not download_url:
                            print(f"No download URL for {file_path}")
                            continue

                        # Download the file
                        async with session.get(download_url) as download_response:
                            if download_response.status != 200:
                                print(f"Failed to download {file_path}")
                                continue

                            await ensure_directory(os.path.dirname(local_path))
                            async with aiofiles.open(local_path, "wb") as f:
                                async for (
                                    chunk
                                ) in download_response.content.iter_chunked(CHUNK_SIZE):
                                    await f.write(chunk)

                            downloaded_paths.append(local_path)
                            print(f"✓ {os.path.basename(file_path)}")

                except Exception as e:
                    print(f"Error downloading {file_path}: {str(e)}")

    print(f"Downloaded {len(downloaded_paths)} images for task {task_id}")
    return downloaded_paths


async def get_json(
    path: str, token: str, task_id: str, local_dir: Optional[str] = None
):
    """
    Download JSON data from an EBRAINS bucket and return its contents

    Args:
        path (str): Path to the JSON file in the bucket (bucket/project/brain/jsons/alignment.waln)
        token (str): Authorization token for the bucket
        task_id (str): Task ID for the current task
        local_dir (Optional[str]): Directory to save the JSON file, defaults to task's download dir

    Returns:
        tuple: (JSON data as dict, local file path with .json extension)
    """
    print(f"Downloading JSON: {path}")

    # Whole path is passed down, getting the file name
    parts = path.strip("/").split("/", 1)
    file_path = parts[1] if len(parts) > 1 else ""

    headers = {"Authorization": f"Bearer {token}"}

    # Set default local directory in tasks if not provided
    if local_dir is None:
        local_dir = f"{TASK_BASE_DIR}/{task_id}/downloads/json/"

    await ensure_directory(local_dir)

    # Extract base filename but always save with .json extension
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    local_path = f"{local_dir}{base_filename}.json"  # Ensure .json extension so it works with PyNutil. We can safely persist the use of .waln like this

    async with aiohttp.ClientSession() as session:
        try:
            # Get pre-signed URL
            file_url = (
                f"https://data-proxy.ebrains.eu/api/v1/buckets/{path}?redirect=false"
            )

            async with session.get(file_url, headers=headers) as file_response:
                if file_response.status != 200:
                    error_msg = f"Failed to get download URL for JSON: {path} (Status: {file_response.status})"
                    print(error_msg)
                    raise HTTPException(
                        status_code=file_response.status, detail=error_msg
                    )

                data = await file_response.json()
                download_url = data.get("url")
                if not download_url:
                    error_msg = f"No download URL for JSON: {path}"
                    print(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)

                # Download the file
                async with session.get(download_url) as download_response:
                    if download_response.status != 200:
                        error_msg = f"Failed to download JSON: {path} (Status: {download_response.status})"
                        print(error_msg)
                        raise HTTPException(
                            status_code=download_response.status, detail=error_msg
                        )

                    # Save the file - MODIFIED: Download as text first
                    await ensure_directory(os.path.dirname(local_path))

                    # Read as text content instead of directly parsing as JSON
                    content_text = await download_response.text()

                    # Try to parse as JSON
                    try:
                        json_content = json.loads(content_text)
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse JSON from {path}: {str(e)}"
                        print(error_msg)
                        print(f"Content preview: {content_text[:200]}")
                        raise HTTPException(status_code=400, detail=error_msg)

                    # Print JSON content for debugging
                    print(f"Alignment content preview (first 500 chars):")
                    print(
                        str(json_content)[:500]
                        + ("..." if len(str(json_content)) > 500 else "")
                    )

                    # Save to file with .json extension
                    async with aiofiles.open(local_path, "w") as f:
                        json_str = json.dumps(json_content, indent=2)
                        await f.write(json_str)

                    print(
                        f"✓ Alignment downloaded and saved to {local_path} (converted to .json)"
                    )

                    # Return the parsed JSON content and the local path with .json extension
                    return json_content, local_path

        except Exception as e:
            error_msg = f"Error downloading JSON {path}: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)


async def upload_directory(directory_path: str, upload_base_path: str, token: str):
    """
    Upload directory contents to EBrains bucket
    """
    headers = {"Authorization": f"Bearer {token}"}

    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail="Directory not found")

    # Upload each file in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Create relative path for upload
            relative_path = os.path.relpath(file_path, directory_path)
            upload_path = f"{upload_base_path}/{relative_path}"

            url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{upload_path}"

            async with aiohttp.ClientSession() as session:
                # Get upload URL
                async with session.put(url, headers=headers) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to get upload URL for {relative_path}",
                        )
                    data = await response.json()
                    upload_url = data.get("url")
                    if not upload_url:
                        raise HTTPException(
                            status_code=400,
                            detail="Upload URL not provided in response",
                        )

                    # Upload the file
                    async with aiofiles.open(file_path, "rb") as f:
                        file_content = await f.read()
                        async with session.put(
                            upload_url, data=file_content
                        ) as upload_response:
                            if upload_response.status not in (200, 201):
                                raise HTTPException(
                                    status_code=upload_response.status,
                                    detail=f"Failed to upload {relative_path}",
                                )
