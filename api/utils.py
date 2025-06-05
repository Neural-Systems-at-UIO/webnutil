import os
import json
import shutil
import requests
from fastapi import HTTPException
from typing import Optional
from enum import Enum

# Shared constants
DOWNLOAD_TIMEOUT = 3600  # 1 hour timeout
CHUNK_SIZE = 8192
MAX_CONCURRENT_TASKS = 3
TASK_BASE_DIR = "/data/nutil_tasks"
TASK_TEMP_DIR = "/data/nutil_temp"


class TaskStatus(Enum):
    PENDING = "pending"
    DOWNLOADING_JSON = "downloading json"
    DOWNLOADING = "downloading segments"
    PROCESSING = "quantifying"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    def __init__(self):
        self.status = TaskStatus.PENDING
        self.message = ""
        self.error = None


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def download_directory(directory_path: str, token: str, task_id: str):
    """
    Download image files from a directory hosted in an EBRAINS bucket.
    Returns a list of local file paths for the downloaded images.
    """
    print(f"Downloading images from: {directory_path}")
    parts = directory_path.strip("/").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}?prefix={prefix}&delimiter=/&limit=1000"
    headers = {"Authorization": f"Bearer {token}"}
    dir_name = os.path.basename(directory_path.rstrip("/")) or "downloads"
    local_dir = f"{TASK_BASE_DIR}/{task_id}/downloads/{dir_name}/"
    ensure_directory(local_dir)
    downloaded_paths = []
    resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to list directory contents: {resp.status_code}",
        )
    data = resp.json()
    objects = data.get("objects", [])
    files_to_download = [
        obj for obj in objects if not obj.get("name", "").endswith("/")
    ]
    print(f"Found {len(files_to_download)} files to download")
    for i, file in enumerate(files_to_download):
        file_path = file.get("name")
        if not file_path:
            continue
        try:
            print(
                f"[{i+1}/{len(files_to_download)}] Downloading {file_path.split('/')[-1]}..."
            )
            file_url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}/{file_path}?redirect=false"
            local_path = f"{local_dir}{os.path.basename(file_path)}"
            file_response = requests.get(
                file_url, headers=headers, timeout=DOWNLOAD_TIMEOUT
            )
            if file_response.status_code != 200:
                print(f"Failed to get download URL for {file_path}")
                continue
            data = file_response.json()
            download_url = data.get("url")
            if not download_url:
                print(f"No download URL for {file_path}")
                continue
            download_response = requests.get(
                download_url, stream=True, timeout=DOWNLOAD_TIMEOUT
            )
            if download_response.status_code != 200:
                print(f"Failed to download {file_path}")
                continue
            ensure_directory(os.path.dirname(local_path))
            with open(local_path, "wb") as f:
                for chunk in download_response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            downloaded_paths.append(local_path)
            print(f"✓ {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error downloading {file_path}: {str(e)}")
    print(f"Downloaded {len(downloaded_paths)} images for task {task_id}")
    return downloaded_paths


def get_json(path: str, token: str, task_id: str, local_dir: Optional[str] = None):
    """
    Download JSON data from an EBRAINS bucket and return its contents and local path.
    """
    print(f"Downloading JSON: {path}")
    parts = path.strip("/").split("/", 1)
    file_path = parts[1] if len(parts) > 1 else ""
    headers = {"Authorization": f"Bearer {token}"}
    if local_dir is None:
        local_dir = f"{TASK_BASE_DIR}/{task_id}/downloads/json/"
    ensure_directory(local_dir)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    local_path = f"{local_dir}{base_filename}.json"
    file_url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{path}?redirect=false"
    try:
        file_response = requests.get(
            file_url, headers=headers, timeout=DOWNLOAD_TIMEOUT
        )
        if file_response.status_code != 200:
            error_msg = f"Failed to get download URL for JSON: {path} (Status: {file_response.status_code})"
            print(error_msg)
            raise HTTPException(status_code=file_response.status_code, detail=error_msg)
        data = file_response.json()
        download_url = data.get("url")
        if not download_url:
            error_msg = f"No download URL for JSON: {path}"
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        download_response = requests.get(download_url, timeout=DOWNLOAD_TIMEOUT)
        if download_response.status_code != 200:
            error_msg = f"Failed to download JSON: {path} (Status: {download_response.status_code})"
            print(error_msg)
            raise HTTPException(
                status_code=download_response.status_code, detail=error_msg
            )
        ensure_directory(os.path.dirname(local_path))
        content_text = download_response.text
        try:
            json_content = json.loads(content_text)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON from {path}: {str(e)}"
            print(error_msg)
            print(f"Content preview: {content_text[:200]}")
            raise HTTPException(status_code=400, detail=error_msg)
        print(f"Alignment content preview (first 500 chars):")
        print(str(json_content)[:500] + ("..." if len(str(json_content)) > 500 else ""))
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)
        print(f"✓ Alignment downloaded and saved to {local_path} (converted to .json)")
        return json_content, local_path
    except Exception as e:
        error_msg = f"Error downloading JSON {path}: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


def upload_directory(directory_path: str, upload_base_path: str, token: str):
    """
    Upload directory contents to EBrains bucket
    """
    headers = {"Authorization": f"Bearer {token}"}
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail="Directory not found")
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory_path)
            upload_path = f"{upload_base_path}/{relative_path}"
            url = f"https://data-proxy.ebrains.eu/api/v1/buckets/{upload_path}"
            response = requests.put(url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get upload URL for {relative_path}",
                )
            data = response.json()
            upload_url = data.get("url")
            if not upload_url:
                raise HTTPException(
                    status_code=400, detail="Upload URL not provided in response"
                )
            with open(file_path, "rb") as f:
                file_content = f.read()
                upload_response = requests.put(
                    upload_url, data=file_content, timeout=DOWNLOAD_TIMEOUT
                )
                if upload_response.status_code not in (200, 201):
                    raise HTTPException(
                        status_code=upload_response.status_code,
                        detail=f"Failed to upload {relative_path}",
                    )
