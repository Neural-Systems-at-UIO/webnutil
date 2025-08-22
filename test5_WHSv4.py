import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nutil import Nutil

script_dir = os.path.dirname(os.path.abspath(__file__))

# TODO Sliding window for object detection and quantification
# TODO Strip non-custom atlas code


nt = Nutil(
    segmentation_folder=os.path.join(
        script_dir, "./tests/test_data/segmentations_test5_WHSv4/segmentations/"
    ),
    alignment_json=os.path.join(
        script_dir, "./tests/test_data/segmentations_test5_WHSv4/test5_WHSv4.json"
    ),
    colour=[255, 0, 0],
    atlas_path=os.path.join(
        script_dir,
        "./atlases/pynutil-waxholm_atlases/waxholm_v4.nrrd",
    ),
    label_path=os.path.join(
        script_dir, "./atlases/pynutil-waxholm_atlases/waxholm_v4_label.csv"
    ),
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()
print(nt.get_region_summary().sort_values(by="object_count", ascending=False))
nt.save_analysis("./test_result/test5_WHSv4")