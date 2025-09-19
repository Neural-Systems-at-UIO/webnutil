import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nutil import Nutil

script_dir = os.path.dirname(os.path.abspath(__file__))

nt = Nutil(
    segmentation_folder=os.path.join(
        script_dir, "./tests/test_data/Test8_synthetic_sagittal/Segmentations/"
    ),
    alignment_json=os.path.join(
        script_dir,
        "./tests/test_data/Test8_synthetic_sagittal/Test8_synthetic_sagittal_nonlin.json",
    ),
    colour=[0, 0, 0],
    atlas_path=os.path.join(
        script_dir,
        "./tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    ),
    label_path=os.path.join(
        script_dir, "./tests/test_data/allen_mouse_2017_atlas//allen2017_colours.csv"
    ),
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()


nt.save_analysis("./test_result/Test8_synthetic_sagittal_12_09_25")
