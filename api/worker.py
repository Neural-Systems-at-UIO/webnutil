import os
import sys
import time
import traceback
import json
from redis import Redis
from utils import download_directory, get_json, upload_directory, TaskStatus
from nutil import Nutil

redis = Redis(host="localhost", port=6379, db=0, decode_responses=True)

QUEUE_NAME = "nutil_tasks_queue"


# Worker main loop
def process_task(task_id):
    task_key = f"nutil_task:{task_id}"
    task = redis.hgetall(task_key)
    print(f"DEBUG: task type: {type(task)}, value: {task}")
    if not isinstance(task, dict) or not task:
        print(f"Task {task_id} not found in Redis or not a dict.")
        return
    try:
        # Update status: DOWNLOADING
        redis.hset(
            task_key,
            mapping={
                "status": TaskStatus.DOWNLOADING.value,
                "message": "Downloading input files...",
            },
        )
        # Download segmentation directory
        seg_paths = download_directory(
            str(task["segmentation_path"]), str(task["token"]), task_id
        )
        # Download alignment JSON
        alignment_json, alignment_json_path = get_json(
            str(task["alignment_json_path"]), str(task["token"]), task_id
        )
        # Update status: PROCESSING
        redis.hset(
            task_key,
            mapping={
                "status": TaskStatus.PROCESSING.value,
                "message": "Running quantification...",
            },
        )
        # Always parse colour as JSON list
        colour = task["colour"]
        if isinstance(colour, str):
            try:
                colour = json.loads(colour)
            except Exception:
                colour = [
                    int(x) for x in colour.strip("[]").split(",") if x.strip().isdigit()
                ]
        nutil = Nutil(
            segmentation_folder=os.path.dirname(seg_paths[0]) if seg_paths else None,
            alignment_json=alignment_json_path,
            colour=colour,
            atlas_path=str(task["atlas_path"]),
            label_path=str(task["label_path"]),
        )
        nutil.get_coordinates(object_cutoff=0, use_flat=False)
        nutil.quantify_coordinates()
        output_dir = f"/data/nutil_tasks/{task_id}/output"
        os.makedirs(output_dir, exist_ok=True)
        nutil.save_analysis(output_dir)
        # Update status: UPLOADING
        redis.hset(
            task_key,
            mapping={
                "status": TaskStatus.UPLOADING.value,
                "message": "Uploading results...",
            },
        )
        upload_directory(output_dir, str(task["upload_to"]), str(task["token"]))
        # Update status: COMPLETED
        redis.hset(
            task_key,
            mapping={
                "status": TaskStatus.COMPLETED.value,
                "message": "Task completed successfully.",
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        redis.hset(
            task_key,
            mapping={"status": TaskStatus.FAILED.value, "message": str(e), "error": tb},
        )
        print(f"Task {task_id} failed: {e}\n{tb}")


def main():
    print("Worker started. Waiting for tasks...")
    while True:
        result = redis.blpop([QUEUE_NAME], timeout=10)
        print(f"DEBUG: blpop result type: {type(result)}, value: {result}")
        if result:
            task_id = (
                result[1]
                if isinstance(result, (list, tuple)) and len(result) > 1
                else result
            )
            print(f"Processing task {task_id}")
            process_task(task_id)
        else:
            time.sleep(2)


if __name__ == "__main__":
    main()
