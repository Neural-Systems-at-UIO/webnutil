from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from pydantic import BaseModel
import uuid 
import json # Ensure json is imported



# nutil request model
class NutilRequest(BaseModel):
    # DATA proxy locations for the files
    segmentation_path: str 
    alignment_json_path: str
    colour: list[int] # BGR format, eg RED [0, 0, 255]
    atlas_name: str  # 
    output_path: str  # Data proxy location for the output folder
    token: str  # Data Proxy token
    # atlas_path: str
    # label_path: str

# Init redis connection

QUEUE_NAME = "nutil_tasks_queue" # For consistency with worker
ALL_TASKS_SET_KEY = "nutil:all_tasks_set" # Key for the set of all task IDs

atlas_configurations = { 
    # Atlases offered for now with the QUINT Online service
    # Accessible for api.py for resolving atlas names to paths
    "ABA_Mouse_CCFv3_2017_25um" : {
        "path": "/app/atlases/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
        "labels" : "/app/atlases/allen_mouse_2017_atlas/allen2017_colours.csv",
        "resolution": "25um",
        "name" : "Allen Mouse Brain Atlas CCFv3 2017 25um"

    },
    "WHS_SD_Rat_v3_39um" : {
        "path": "/app/atlases/pynutil-waxholm_atlases/waxholm_v3.01.nrrd",
        "labels" : "/app/atlases/pynutil-waxholm_atlases/waxholm_v3.01.label",
        "resolution": "39um",
        "name" : "Waxholm Space Atlas of the Sprague Dawley rat v4"
    }, 
    "WHS_SD_Rat_v4_39um" : {
        "path": "/app/atlases/pynutil-waxholm_atlases/waxholm_v4.01.nrrd",
        "labels" : "/app/atlases/pynutil-waxholm_atlases/waxholm_v4.01.label",
        "resolution": "39um",
        "name" : "Waxholm Space Atlas of the Sprague Dawley rat v3"
    },
}

redis = Redis(host="nutil-redis", port=6379, db=0, decode_responses=True)

app = FastAPI()


@app.router.get("/")
def root():
    return {"message": "Welcome to the Nutil API!"}
    
@app.get("/atlases" )
def get_atlases():
    """
    Returns a list of available atlases with their details.
    """
    return atlas_configurations

@app.post("/schedule-task")
def schedule_task(request: NutilRequest):

    # Check who is calling

    # Schedule to the queue
    task_id = str(uuid.uuid4())

    # Looking up the preset configurations for the atlas
    # The nrrd lives in the temp storage of the worker pod
    task_info = {
        "task_id": task_id,
        "segmentation_path": request.segmentation_path,
        "alignment_json_path": request.alignment_json_path,
        "colour": json.dumps(request.colour),  # Store colour as JSON string
        "atlas_path": atlas_configurations[request.atlas_name]["path"],
        "label_path": atlas_configurations[request.atlas_name]["labels"],
        "upload_to": request.output_path,
        "token": request.token, # Token will be stored but not returned in status checks
        "status": "scheduled", # Initial status
        "message": "Task has been scheduled successfully."
    }

    # Store the task details in a Redis hash
    redis.hmset(f"nutil_task:{task_id}", task_info)
    
    # Push only the task_id to the worker queue
    redis.rpush(QUEUE_NAME, task_id)

    # Add task_id to a set for get_all_tasks
    redis.sadd(ALL_TASKS_SET_KEY, task_id)

    return {"message": "Task scheduled successfully.", 
            "task_id": task_id}

@app.get("/task-status/{task_id}")
def get_task(task_id: str):
    """
    Returns the status of a specific task by its ID.
    """
    task_details = redis.hgetall(f"nutil_task:{task_id}")
    
    if task_details:
        # Obfuscate token before returning
        task_details.pop("token", None) 
        
        # Convert colour back to list if it exists and is a string
        if 'colour' in task_details and isinstance(task_details['colour'], str):
            try:
                task_details['colour'] = json.loads(task_details['colour'])
            except json.JSONDecodeError:
                # If it's not valid JSON, leave it as is or handle error
                # For now, we'll leave it as the string it was in Redis
                pass 
        return {"task": task_details}
    
    return {"message": "Task not found."}

@app.get("/all-tasks")
def get_all_tasks():
    task_ids_bytes = redis.smembers(ALL_TASKS_SET_KEY)
    tasks_details_list = []
    for task_id_bytes in task_ids_bytes:
        task_id = task_id_bytes.decode('utf-8') # smembers returns bytes
        task_details = redis.hgetall(f"nutil_task:{task_id}")
        if task_details:
            # Obfuscate token before returning
            task_details.pop("token", None)

            # Convert colour back to list if it exists and is a string
            if 'colour' in task_details and isinstance(task_details['colour'], str):
                try:
                    task_details['colour'] = json.loads(task_details['colour'])
                except json.JSONDecodeError:
                    pass # Leave as string if not valid JSON
            tasks_details_list.append(task_details)
    return {"tasks": tasks_details_list}

# TODO Add download for images and json
# TODO import the utils