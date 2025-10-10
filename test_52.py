import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nutil import Nutil

script_dir = os.path.dirname(os.path.abspath(__file__))

# TODO Sliding window for object detection and quantification
# TODO Strip non-custom atlas code


nt = Nutil(
    segmentation_folder=os.path.join(script_dir, "./tests/real_data/segmentations"),
    alignment_json=os.path.join(
        script_dir,
        "./tests/real_data/tTA_2877_NOP_horizontal_final_2017_new_28_05_25.json",
    ),
    colour=[0, 0, 255],
    atlas_path=os.path.join(
        script_dir,
        "./tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    ),
    label_path=os.path.join(
        script_dir, "./tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv"
    ),
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()

nt.save_analysis("./test_result/real_world_use_case")
